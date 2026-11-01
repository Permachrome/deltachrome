"""Resolving the capture's sampling rate at the target plane.

Shared because two metrics need the same number and must not disagree about it:
``sampling_frequency`` reports it directly, and MTF50 in line pairs per millimetre
is a frequency in cycles per *pixel* until a pixel size is known.

The order of preference is the point. A chart that prints a ruler carries its own
dimensional reference and reading it beats everything else -- the answer comes from
the same capture as the measurement it scales, so no per-model constant has to be
declared and no per-unit measurement has to be trusted. Failing that, an extent the
operator measured on this specimen. Failing that, the extent in the target
definition, which is a property of a batch rather than of the card in front of the
camera.

The file header is deliberately not in that list. It can say anything: the sample
ISA 4x5 capture claims 600 ppi where the chart's own rule measures 2002.
"""

from __future__ import annotations

from dataclasses import dataclass

from deltachrome.metrics.base import MetricContext

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class SamplingRate:
    """A resolved sampling rate, and where it came from."""

    ppi: float
    source: str
    notes: tuple[str, ...] = ()
    detail: dict | None = None

    @property
    def pixels_per_mm(self) -> float:
        return self.ppi / MM_PER_INCH

    @property
    def nyquist_lp_per_mm(self) -> float:
        """The most any system sampling at this rate can resolve, however sharp.

        A line pair needs two samples, so the ceiling is half the sampling rate.
        A requirement in lp/mm above this is unreachable by a better lens; it needs
        a higher sampling rate.
        """
        return self.pixels_per_mm / 2.0


def resolve_sampling_ppi(ctx: MetricContext) -> tuple[SamplingRate | None, str]:
    """The sampling rate at the target plane, or the reason there isn't one.

    Returns ``(rate, "")`` or ``(None, reason)``, where the reason is written to be
    shown to an operator: it says what to do about it.
    """
    placement = ctx.extras.get("placement")
    if placement is None:
        return None, "No chart placement was supplied to the metric context."

    readings = ctx.extras.get("rulers") or []
    if readings:
        from deltachrome.imaging.rulers import sampling_rate

        measured = sampling_rate(readings)
        notes = [
            "Scale measured from the chart's own printed ruler, not read from the "
            "file header and not derived from a declared physical size."
        ]
        if measured.get("warning"):
            notes.append(measured["warning"])
        return SamplingRate(
            ppi=float(measured["ppi"]), source="printed ruler",
            notes=tuple(notes), detail=measured,
        ), ""

    override = ctx.extras.get("fiducial_extent_mm")
    size_mm = override or ctx.target.fiducial_extent_mm
    if not size_mm:
        reference = ctx.target.dimensional_reference or {}
        if reference.get("available") is False:
            return None, (
                f"{ctx.target.display_name} carries no printed scale or other "
                "dimensional reference, so its physical size cannot be established "
                "from the capture. "
                + (reference.get("notes") or "")
                + " To supply one: use a target that carries a scale, or measure this "
                "specimen and pass --fiducial-extent WxH (in mm), which is per unit "
                "and not transferable to another."
            )
        return None, (
            f"Target {ctx.target.target_id!r} does not record the physical size of the "
            "area its fiducials bound, so pixels per inch cannot be derived. Measure "
            "the area the fiducials enclose (not the whole card) and set "
            "'fiducial_extent_mm' in the target definition, or pass --fiducial-extent "
            "WxH for a single run. Reporting a rate computed from the card size "
            "instead would under-state it and could fail a capture that conforms."
        )

    value = placement.effective_ppi(size_mm)
    if value is None:
        return None, "Placement did not yield a usable sampling rate."

    if override:
        source = "operator-supplied extent"
        notes = [
            "Physical extent was supplied for this run rather than taken from the "
            "target definition. It describes the specimen measured, and does not "
            "transfer to another unit of the same model."
        ]
    else:
        source = "target definition extent"
        provenance = (ctx.target.dimensional_reference or {}).get("summary")
        notes = [f"Extent from the target definition: {provenance}"] if provenance else []

    return SamplingRate(
        ppi=float(value), source=source, notes=tuple(notes),
        detail={"fiducial_extent_mm": list(size_mm)},
    ), ""
