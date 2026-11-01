"""Color-channel misregistration from slanted edges.

The method is Burns and Williams, *Using Slanted Edge Analysis for Color
Registration Measurement* (Proc. IS&T PICS, 51-53, 1999) -- the first steps of the
ISO 12233 procedure, run once per color record on the **same** ROI:

    linearize -> per-row LSF centroid -> fit a line -> evaluate at the ROI center

The difference between two records' edge locations is their displacement along the
edge normal, in pixels. Nothing after the line fit is needed: no oversampled
profile, no window, no transform. The steps up to that point are already
implemented for SFR and are imported from there rather than repeated.

FADGI states the criterion and not the method. Section 2.4.7 of the 3rd Edition
says only that the parameter "measures the spatial dislocation of red, green, and
blue light in terms of pixels of misregistration", and gives limits of 1.2 / 0.8 /
0.5 / 0.33 px for 1 to 4 stars. It does not say how a field of edges is reduced,
nor whether the limit applies per axis or to the displacement vector. ISO 19264-1
defines the quantity in the same words and is equally silent on reduction. Those
choices are therefore parameters here, and every reading is reported whichever one
drives the verdict.

**Why opposite edges are paired.** A square feature presents two edges of each
orientation, of opposite polarity. A displacement between two records moves both
the same way; anything that changes the feature's apparent *size* in one record
moves them oppositely. So the pair's mean is the translation -- which is what the
standard defines -- and half their difference is a separate quantity that would
otherwise be added into the answer at full strength.

That term is not hypothetical. On the ISA Golden Thread capture the translation is
0.003-0.028 px while the antisymmetric term is 0.03-0.048 px, present at every
field position including the center, and confined to the pairs involving blue.
A magnification difference between records large enough to explain it would put
0.25 px of translation at the corner modules, and there is none, so it is not
lateral chromatic aberration; the estimator is ruled out by synthetic tests. What
is left is a wavelength-dependent property of the illumination or the printed edge
-- flare, whose signature this capture already shows in its tonal ends. Reported
under its own name rather than as misregistration it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from deltachrome.imaging.sfr import (
    EdgeFit,
    SFRError,
    fit_edge_line,
    orient_edge,
    row_edge_positions,
)

#: The device's own color records, in file order.
CHANNELS = ("R", "G", "B")

#: The three differences OpenDICE plots, in its order and with its sign, so a
#: Deltachrome table can be read straight against a GoldenThread or OpenDICE screen
#: without mentally negating anything. Only two of the three are independent; the
#: third is shown because the extremes of the spectrum are often the worst pair and
#: a reader should not have to derive it.
CHANNEL_PAIRS = (("G", "B"), ("G", "R"), ("B", "R"))

Pairing = Literal["opposite_edges", "per_edge"]


def pair_name(pair: Sequence[str]) -> str:
    return f"{pair[0]}-{pair[1]}"


@dataclass
class ChannelPosition:
    """Where one color record puts the edge, and how well that is known."""

    channel: str
    #: Along the edge normal, at the ROI's center row, in pixels of the ROI's frame.
    location_px: float
    angle_deg: float
    rows_used: int
    #: Scatter of the located row positions about the fitted line.
    scatter_px: float

    @property
    def standard_error_px(self) -> float:
        """Standard error of the fitted location -- a **lower bound** on uncertainty.

        It describes only the random scatter of the row positions. A term common to
        every row -- flare, a chromatic difference in the printed edge itself -- is
        systematic, does not shrink with row count, and is invisible to this figure.
        """
        return self.scatter_px / math.sqrt(max(self.rows_used, 1))


@dataclass
class EdgeRegistration:
    """One edge, located independently in each color record."""

    edge_id: str
    region_id: str
    #: The image axis the displacement lies along: a vertical edge can only reveal
    #: an x displacement, a horizontal edge only a y one.
    axis: str
    #: "rising" if the signal increases along +axis, "falling" if it decreases.
    #: What makes an opposite-edge pair an opposite-*polarity* pair.
    polarity: str
    center_px: tuple[float, float]
    channels: dict[str, ChannelPosition]
    linearization: str
    settings: dict[str, Any] = field(default_factory=dict)

    def shift(self, a: str, b: str) -> float:
        return self.channels[a].location_px - self.channels[b].location_px

    @property
    def shifts(self) -> dict[str, float]:
        return {pair_name(p): self.shift(*p) for p in CHANNEL_PAIRS}

    @property
    def spread_px(self) -> float:
        """Distance between the extreme records -- the largest of the three pairs."""
        locations = [c.location_px for c in self.channels.values()]
        return max(locations) - min(locations)

    @property
    def standard_error_px(self) -> float:
        """Standard error of the worst channel *pair* -- two locations differenced.

        The two largest per-record errors combined in quadrature, since the pair
        that reports the largest shift is not known in advance.
        """
        errors = sorted(c.standard_error_px for c in self.channels.values())
        return math.hypot(errors[-1], errors[-2])


@dataclass
class FieldPoint:
    """One position and axis at which the records' displacement is reported.

    Built from an opposite-edge pair where the feature provides one, and from a
    single edge where it does not -- in which case ``antisymmetric`` is empty and
    the translation carries whatever polarity-dependent term the capture has.
    """

    region_id: str
    axis: str
    edge_ids: list[str]
    center_px: tuple[float, float]
    #: Signed displacement per channel pair, in pixels along ``axis``.
    translation: dict[str, float]
    #: Half the difference between the pair's readings: an apparent size difference
    #: between the records, not a displacement. Empty when unpaired.
    antisymmetric: dict[str, float] = field(default_factory=dict)
    standard_error_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def paired(self) -> bool:
        return len(self.edge_ids) == 2

    @property
    def spread_px(self) -> float:
        """The worst of the three channel pairs at this point."""
        return max(abs(v) for v in self.translation.values()) if self.translation else 0.0


# -- the measurement --------------------------------------------------------


def measure_edge_registration(
    roi: np.ndarray,
    *,
    edge_id: str = "edge",
    region_id: str = "",
    center_px: tuple[float, float] = (0.0, 0.0),
    linearize_channels: Callable[[np.ndarray], np.ndarray] | None = None,
    linearization: str = "none",
    centroid_threshold: float = 0.10,
    edge_fit: EdgeFit = "linear",
) -> EdgeRegistration:
    """Locate one slanted edge separately in each color record.

    ``roi`` is an (H, W, 3) array holding exactly one edge. ``linearize_channels``
    must map encoded RGB to a signal linear in luminance **without mixing the
    channels** -- see :meth:`~deltachrome.imaging.icc.ICCProfile.channel_linearizer`.
    A 3x3 to XYZ would blend the records and average the displacement away.
    """
    roi = np.asarray(roi, dtype=np.float64)
    if roi.ndim != 3 or roi.shape[-1] < 3:
        raise SFRError(
            f"Channel registration needs a three-record ROI; got shape {roi.shape}"
        )
    if linearize_channels is not None:
        roi = np.asarray(linearize_channels(roi[..., :3]), dtype=np.float64)

    # The orientation decision is made once, on the luminance combination, so every
    # record is projected onto the same axis. Deciding per record would let a noisy
    # channel transpose itself and be differenced against a different quantity.
    luminance = roi[..., :3] @ np.array([0.213, 0.715, 0.072])
    _, transposed = orient_edge(luminance)

    positions: dict[str, ChannelPosition] = {}
    for index, channel in enumerate(CHANNELS):
        plane = roi[..., index]
        plane = plane.T if transposed else plane
        located = row_edge_positions(plane, centroid_threshold)
        fitted, coefficients, rows_used = fit_edge_line(located, edge_fit)
        positions[channel] = ChannelPosition(
            channel=channel,
            location_px=float(np.polyval(coefficients, (plane.shape[0] - 1) / 2.0)),
            angle_deg=math.degrees(math.atan(float(coefficients[-2]))),
            rows_used=rows_used,
            scatter_px=float(np.nanstd(located - fitted)),
        )

    oriented = luminance.T if transposed else luminance
    n = max(4, oriented.shape[1] // 8)
    rising = float(np.median(oriented[:, -n:])) > float(np.median(oriented[:, :n]))

    return EdgeRegistration(
        edge_id=edge_id,
        region_id=region_id,
        axis="y" if transposed else "x",
        polarity="rising" if rising else "falling",
        center_px=(float(center_px[0]), float(center_px[1])),
        channels=positions,
        linearization=linearization,
        settings={
            "centroid_threshold": centroid_threshold,
            "edge_fit": edge_fit,
            "linearized": linearize_channels is not None,
            "transposed": transposed,
        },
    )


def measure_edges(
    edges: Iterable[Any],
    *,
    centroid_threshold: float = 0.10,
    edge_fit: EdgeFit = "linear",
    linearize: bool = True,
) -> tuple[list[EdgeRegistration], list[str]]:
    """Measure every :class:`~deltachrome.imaging.edges.EdgeROI` that can be measured.

    Returns the measurements and, separately, the edges that could not be: an edge
    the SFR analysis rejected is not a reason to report no registration at all.
    """
    measured: list[EdgeRegistration] = []
    failures: list[str] = []
    for edge in edges:
        linearizer = getattr(edge, "linearize_channels", None) if linearize else None
        try:
            measured.append(
                measure_edge_registration(
                    edge.pixels,
                    edge_id=edge.edge_id,
                    region_id=edge.region_id,
                    center_px=edge.center_px,
                    linearize_channels=linearizer,
                    linearization=getattr(edge, "linearization", "none")
                    if linearizer is not None
                    else "none",
                    centroid_threshold=centroid_threshold,
                    edge_fit=edge_fit,
                )
            )
        except SFRError as exc:
            failures.append(f"{edge.edge_id}: {exc}")
    return measured, failures


# -- reduction --------------------------------------------------------------


def field_points(
    measurements: Sequence[EdgeRegistration], *, pairing: Pairing = "opposite_edges",
    declared: Mapping[tuple[str, str], int] | None = None,
) -> list[FieldPoint]:
    """Reduce measured edges to the points at which displacement is reported.

    With ``opposite_edges`` (the default), the two edges a square feature presents
    on one axis are combined: their mean is the translation, half their difference
    the antisymmetric term. With ``per_edge`` every edge stands alone, which is what
    a tool applying the criterion edge by edge would report -- larger, because the
    antisymmetric term is then inside the number.

    ``declared`` counts the edges the chart offers per ``(region, axis)``, measured
    or not. A lone edge is only worth a note where the chart printed a pair and
    one of them failed: the ISA FADGI 19264 prints one edge per axis per module by
    design, and noting that on every run said nothing the operator could act on.
    """
    groups: dict[tuple[str, str], list[EdgeRegistration]] = {}
    for m in measurements:
        groups.setdefault((m.region_id, m.axis), []).append(m)

    points: list[FieldPoint] = []
    for (region_id, axis), group in sorted(groups.items()):
        group = sorted(group, key=lambda m: m.edge_id)
        if pairing == "opposite_edges" and len(group) == 2:
            a, b = group
            notes: list[str] = []
            points.append(
                FieldPoint(
                    region_id=region_id,
                    axis=axis,
                    edge_ids=[a.edge_id, b.edge_id],
                    center_px=(
                        (a.center_px[0] + b.center_px[0]) / 2.0,
                        (a.center_px[1] + b.center_px[1]) / 2.0,
                    ),
                    translation={
                        pair_name(p): (a.shift(*p) + b.shift(*p)) / 2.0
                        for p in CHANNEL_PAIRS
                    },
                    antisymmetric={
                        pair_name(p): (a.shift(*p) - b.shift(*p)) / 2.0
                        for p in CHANNEL_PAIRS
                    },
                    standard_error_px=math.hypot(
                        a.standard_error_px, b.standard_error_px
                    ) / 2.0,
                    notes=notes,
                )
            )
            continue
        for m in group:
            note = []
            if (
                pairing == "opposite_edges" and declared is not None
                and declared.get((region_id, axis), 0) >= 2
            ):
                note.append(f"Edge not measured on {region_id}.")
            points.append(
                FieldPoint(
                    region_id=region_id,
                    axis=axis,
                    edge_ids=[m.edge_id],
                    center_px=m.center_px,
                    translation=m.shifts,
                    standard_error_px=m.standard_error_px,
                    notes=note,
                )
            )
    return points


def field_vectors(points: Sequence[FieldPoint]) -> dict[str, dict[str, Any]]:
    """Per region, the displacement as a 2-D vector, where both axes were measured.

    The dislocation at a point is a vector; a per-axis reading is its projection
    onto one axis and is never larger. A displacement of 0.4 px on the diagonal
    reads 0.28 px on each axis, so a per-axis criterion passes it at a limit the
    dislocation itself fails. Reported alongside rather than instead: the per-axis
    figures are what OpenDICE plots and what the thresholds have been read against.
    """
    by_region: dict[str, dict[str, FieldPoint]] = {}
    for point in points:
        by_region.setdefault(point.region_id, {})[point.axis] = point

    out: dict[str, dict[str, Any]] = {}
    for region, axes in sorted(by_region.items()):
        if not {"x", "y"} <= set(axes):
            continue
        vectors = {}
        for pair in CHANNEL_PAIRS:
            name = pair_name(pair)
            dx = axes["x"].translation.get(name, 0.0)
            dy = axes["y"].translation.get(name, 0.0)
            vectors[name] = {
                "dx": dx,
                "dy": dy,
                "magnitude_px": math.hypot(dx, dy),
            }
        out[region] = {
            "center_px": [
                (axes["x"].center_px[0] + axes["y"].center_px[0]) / 2.0,
                (axes["x"].center_px[1] + axes["y"].center_px[1]) / 2.0,
            ],
            "pairs": vectors,
            "worst_magnitude_px": max(v["magnitude_px"] for v in vectors.values()),
        }
    return out


def rotation_between_records(
    measurements: Sequence[EdgeRegistration],
) -> dict[str, float]:
    """Mean and scatter of the per-record edge angle, in degrees.

    Burns and Williams note that the fitted line carries rotation as well as
    translation: two records whose edges are not parallel are rotated relative to
    one another, which a translation figure cannot show. A tri-linear scanner whose
    arrays are not square to the travel is the case that produces it.
    """
    out: dict[str, float] = {}
    for a, b in (("R", "G"), ("B", "G")):
        deltas = [
            m.channels[a].angle_deg - m.channels[b].angle_deg
            for m in measurements
            if a in m.channels and b in m.channels
        ]
        if deltas:
            out[f"{a}-{b}_mean_deg"] = float(np.mean(deltas))
            out[f"{a}-{b}_std_deg"] = float(np.std(deltas))
    return out
