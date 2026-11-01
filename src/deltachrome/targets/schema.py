"""Target definition schema.

One format describes any target -- color-only, SFR-only, or hybrid. Three
decisions from the primer are load-bearing here:

* ``reference_value`` is never stored in a definition. A definition is a reusable
  template for a *model* of chart; the numbers belong to a *physical unit* and
  arrive from an imported CGATS file at analysis time.
* ``capabilities`` / ``not_capable_of`` let a partial target (color-only, SFR-only)
  work with the same engine. A profile skips what a target cannot measure instead
  of the engine growing special cases.
* Geometry is relative to fiducials, never absolute pixels, so manual placement
  and future auto-detection share the same maths.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from deltachrome.errors import TargetError

#: Roles the engine understands. Metric plugins select patches by role rather than
#: by hard-coded patch id, so a new target needs no plugin changes.
#: Region name used when a target does not declare regions explicitly.
DEFAULT_REGION = "main"

#: What a region contains. Only ``patch_grid`` yields sampleable patches today;
#: the others are declared so a target can describe features it carries before the
#: metrics that use them exist, rather than the definition pretending they are absent.
REGION_KINDS = frozenset({"patch_grid", "slanted_edge", "marker", "scale"})

#: Every capability a target may claim, or disclaim. Closed on purpose.
#:
#: A capability name is the join between a target definition and a profile's
#: metrics, and an unrecognised one fails silently in the worst possible way: the
#: target matches no metric, every one of them reports N/A, and the run passes on
#: whatever is left. That reads as a target limitation rather than as a typo. With
#: three definitions it can be eyeballed; a growing database of them cannot.
#:
#: Adding a capability here is deliberate: it means some metric plugin can consume
#: it and some profile can require it.
KNOWN_CAPABILITIES = frozenset(
    {
        "color_accuracy",        # Delta E against a reference, needs color patches
        "tonal_response",        # an OECF ramp of known neutrals
        "white_balance",         # neutrals whose rendered chroma can be judged
        "lightness_uniformity",  # repeated neutrals spread over the chart's area
        "sfr",                   # slanted edges
        "color_registration",    # slanted edges, measured per color record
        "reproduction_scale",    # a printed scale, or another dimensional reference
        "geometric_distortion",  # a grid or ruled field spanning the frame
        "flat_field",            # fills the frame, so the *plane* can be measured
    }
)

KNOWN_ROLES = frozenset(
    {
        "achromatic_ramp",  # the tonal ramp / OECF steps
        "achromatic_border",  # repeated near-identical neutrals, used for uniformity
        "achromatic_other",  # neutral, but neither ramp nor uniformity cluster
        "chromatic",  # everything with meaningful chroma
        "unassigned",  # role not yet resolved (pre-reference-import state)
    }
)


@dataclass(frozen=True)
class Patch:
    """One addressable patch on a target."""

    patch_id: str
    row: int
    col: int
    role: str = "unassigned"
    ramp_position: int | None = None
    notes: str | None = None
    #: Which region of the target this patch belongs to. Targets that are a single
    #: array leave this at the default; a target built from several separated groups
    #: names the group, because row/col only mean anything within one lattice.
    region: str = DEFAULT_REGION
    #: What the chart prints beside this patch, where that differs from
    #: :attr:`patch_id`. The ISA FADGI - ISO 19264 numbers its patches 1-168 in
    #: reading order and *prints* a ragged A-N by 1-16 grid, and the per-unit
    #: measurements published for it use the printed labels -- so a reference file
    #: and a definition can name the same patch two ways. Declared per patch rather
    #: than generated, because the layout is ragged and no formula produces it.
    printed_id: str | None = None

    def with_role(self, role: str, ramp_position: int | None = None) -> "Patch":
        if role not in KNOWN_ROLES:
            raise TargetError(f"Unknown patch role {role!r}; expected one of {sorted(KNOWN_ROLES)}")
        return Patch(
            patch_id=self.patch_id,
            row=self.row,
            col=self.col,
            role=role,
            ramp_position=ramp_position if ramp_position is not None else self.ramp_position,
            notes=self.notes,
            region=self.region,
            printed_id=self.printed_id,
        )


@dataclass(frozen=True)
class PatchGrid:
    """Regular grid geometry in target-relative coordinates.

    Coordinates run 0..1 across the fiducial quadrilateral, origin top-left.
    """

    rows: int
    columns: int
    row_labels: list[Any]
    col_labels: list[str]
    patch_id_format: str
    patch_size_relative: tuple[float, float]
    patch_spacing_relative: tuple[float, float]
    grid_origin_relative: tuple[float, float]
    provisional: bool = False
    provisional_reason: str | None = None

    def patch_id(self, row: int, col: int) -> str:
        return self.patch_id_format.format(
            column_letter=self.col_labels[col],
            row_number=self.row_labels[row],
            row_label=self.row_labels[row],
            col_label=self.col_labels[col],
        )

    def center_relative(self, row: int, col: int) -> tuple[float, float]:
        """Center of a patch in target-relative coordinates."""
        ou, ov = self.grid_origin_relative
        su, sv = self.patch_spacing_relative
        wu, wv = self.patch_size_relative
        return (ou + col * su + wu / 2.0, ov + row * sv + wv / 2.0)

    def corners_relative(
        self, row: int, col: int, inset: float = 1.0
    ) -> tuple[tuple[float, float], ...]:
        """The patch bounding box, optionally shrunk about its center by ``inset``."""
        cu, cv = self.center_relative(row, col)
        wu, wv = self.patch_size_relative
        hu, hv = (wu * inset) / 2.0, (wv * inset) / 2.0
        return (
            (cu - hu, cv - hv),
            (cu + hu, cv - hv),
            (cu + hu, cv + hv),
            (cu - hu, cv + hv),
        )

    def is_perimeter(self, row: int, col: int) -> bool:
        return row in (0, self.rows - 1) or col in (0, self.columns - 1)

    @property
    def extent_relative(self) -> tuple[float, float]:
        """How much of the 0..1 space the grid actually spans."""
        su, sv = self.patch_spacing_relative
        wu, wv = self.patch_size_relative
        return ((self.columns - 1) * su + wu, (self.rows - 1) * sv + wv)


@dataclass(frozen=True)
class Region:
    """One addressable area of a target.

    A ColorChecker SG is a single uniform lattice, and modelling it needed nothing
    more than one grid. Real device-level targets are not like that: an ISA Golden
    Thread carries four patch strips of differing length at four positions, plus
    five slanted-edge modules, separated by printed text. Row and column indices
    only mean something *within* one lattice, so each group gets its own.

    Non-grid kinds (``slanted_edge``, ``marker``, ``scale``) carry no patches. They
    exist so a target definition can describe what the chart physically has before
    the metrics that read them are built -- silence would misrepresent the chart as
    lacking the feature.
    """

    region_id: str
    kind: str = "patch_grid"
    grid: PatchGrid | None = None
    bounds_relative: tuple[float, float, float, float] | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in REGION_KINDS:
            raise TargetError(
                f"Region {self.region_id!r} has kind {self.kind!r}; "
                f"expected one of {sorted(REGION_KINDS)}"
            )
        if self.kind == "patch_grid" and self.grid is None:
            raise TargetError(f"Region {self.region_id!r} is a patch_grid but has no grid.")

    @property
    def holds_patches(self) -> bool:
        return self.kind == "patch_grid"


@dataclass(frozen=True)
class Fiducials:
    """How the target is located in the image."""

    method: str
    corner_ids: list[str]
    corner_positions: list[tuple[float, float]]
    notes: str | None = None


@dataclass
class TargetDefinition:
    """A complete, reusable description of a model of test target."""

    target_id: str
    display_name: str
    manufacturer: str | None
    target_type: str
    physical_size_mm: tuple[float, float] | None
    capabilities: frozenset[str]
    not_capable_of: frozenset[str]
    fiducials: Fiducials
    regions: list[Region]
    patches: list[Patch]
    reference_data: dict[str, Any] = field(default_factory=dict)
    #: Where and how this model's units are published on an OpenQualia
    #: measurement host: ``host``, ``manufacturer`` and ``target_type``. Empty
    #: for the targets nobody publishes, which is most of them, and a lookup by
    #: serial is offered only where all three are declared -- OpenQualia has no
    #: central index, so the host is a fact about the model like the other two. Recorded rather than
    #: derived, because the host matches these case- and whitespace-agnostically
    #: but not dash-agnostically, and its own web slugs are not its API
    #: spellings -- ``OLT-1x`` browses and 404s where ``OLT1x`` resolves.
    openqualia: dict[str, Any] = field(default_factory=dict)
    metric_defaults: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    source_path: Path | None = None
    #: Physical size of the area the fiducials bound, in mm. Distinct from
    #: ``physical_size_mm``, which is the whole card: on a chart whose fiducials are
    #: the corners of its patch array, the two differ by the border, and using the
    #: card size to derive sampling rate under-states it.
    fiducial_extent_mm: tuple[float, float] | None = None
    #: How (or whether) this model of target can establish physical scale at all.
    #: A target with a printed scale can be verified per unit; one without cannot,
    #: and a figure taken from one specimen would be an assumption about every other
    #: batch and cut.
    dimensional_reference: dict[str, Any] = field(default_factory=dict)

    # -- reference data -----------------------------------------------------

    @property
    def nominal_reference(self) -> dict[str, Any]:
        """The manufacturer's aim values for this model, if the definition ships any.

        A per-unit measurement is always preferred, and is used whenever the operator
        supplies one. But some targets are seldom sold measured and are impractical to
        measure yourself -- the ISA Golden Thread's patches are too small for a hand
        spectrophotometer -- so refusing to run at all would put the target out of
        reach entirely. Where the manufacturer states aim values, those are carried
        beside the definition and used as a declared, flagged fallback.
        """
        return self.reference_data.get("nominal_values") or {}

    @property
    def nominal_reference_path(self) -> Path | None:
        """Where the nominal file sits: beside the definition that declares it.

        Relative to the definition's own directory, so a custom target and its aim
        values travel together in one folder.
        """
        name = self.nominal_reference.get("file")
        if not name or self.source_path is None:
            return None
        return (self.source_path.parent / name).resolve()

    # -- lookups ------------------------------------------------------------

    def __post_init__(self) -> None:
        self._regions = {r.region_id: r for r in self.regions}
        if len(self._regions) != len(self.regions):
            raise TargetError(f"Target {self.target_id!r} declares a region id twice.")
        unknown = sorted({p.region for p in self.patches} - set(self._regions))
        if unknown:
            raise TargetError(
                f"Target {self.target_id!r} has patches in undeclared region(s) {unknown}. "
                f"Declared: {sorted(self._regions)}"
            )
        self._by_id = {p.patch_id: p for p in self.patches}
        if len(self._by_id) != len(self.patches):
            counts = Counter(p.patch_id for p in self.patches)
            dupes = sorted(pid for pid, n in counts.items() if n > 1)
            raise TargetError(f"Target {self.target_id!r} defines duplicate patch ids: {dupes}")
        overlap = self.capabilities & self.not_capable_of
        if overlap:
            raise TargetError(
                f"Target {self.target_id!r} lists {sorted(overlap)} in both "
                "capabilities and not_capable_of."
            )
        unknown = sorted((self.capabilities | self.not_capable_of) - KNOWN_CAPABILITIES)
        if unknown:
            raise TargetError(
                f"Target {self.target_id!r} names unknown capability/capabilities "
                f"{unknown}. Known: {sorted(KNOWN_CAPABILITIES)}. A name no profile "
                "recognizes does not disable one metric -- it silently disables every "
                "metric that would have matched, and the run then passes on what is "
                "left while looking like a target limitation."
            )
        self._check_corner_winding()

    def _check_corner_winding(self) -> None:
        """The declared corners must go round the quad, not across it.

        ``corner_positions`` is paired positionally with the four points the operator
        places, so listing them TL, TR, BL, BR instead of TL, TR, BR, BL swaps two
        corners and the homography maps the chart onto a bow tie. Nothing raises: the
        placement is a valid projective transform, every ROI lands somewhere inside
        the image, and each one reads roughly the frame's average -- so a run comes
        back complete, plausible and uniformly grey, with 140 patches agreeing on one
        colour. That is the most expensive kind of wrong this file can be, and it
        costs one cross product to refuse.

        Tested by the shoelace sum, which is positive or negative for a simple
        quadrilateral and near zero for one that crosses itself.
        """
        points = self.fiducials.corner_positions
        if len(points) != 4:
            return
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = points
        quad = [(ax, ay), (bx, by), (cx, cy), (dx, dy)]
        area = 0.0
        for (x0, y0), (x1, y1) in zip(quad, quad[1:] + quad[:1]):
            area += x0 * y1 - x1 * y0
        if abs(area) < 1e-9:
            raise TargetError(
                f"Target {self.target_id!r} lists its fiducial corners "
                f"{self.fiducials.corner_ids} in an order that crosses the chart "
                "rather than going round it. They are matched one-for-one against the "
                "four points an operator places, so this silently maps the chart onto "
                "a bow tie: every region still lands inside the image and every patch "
                "reads about the frame's average. List them in order around the quad, "
                "conventionally TL, TR, BR, BL."
            )

    # -- regions ------------------------------------------------------------

    @property
    def grid_regions(self) -> list[Region]:
        return [r for r in self.regions if r.holds_patches]

    @property
    def patch_grid(self) -> PatchGrid | None:
        """The grid, for targets that are a single lattice.

        Kept so the common case reads simply. A target built from several separated
        groups returns ``None`` here on purpose -- there is no one grid, and code
        that assumed otherwise should be asking a patch for its own region instead.
        """
        grids = self.grid_regions
        return grids[0].grid if len(grids) == 1 else None

    def region_for(self, patch: Patch) -> Region:
        try:
            return self._regions[patch.region]
        except KeyError as exc:
            raise TargetError(
                f"Patch {patch.patch_id!r} names region {patch.region!r}, which "
                f"{self.target_id!r} does not declare."
            ) from exc

    def grid_for(self, patch: Patch) -> PatchGrid:
        region = self.region_for(patch)
        if region.grid is None:
            raise TargetError(
                f"Patch {patch.patch_id!r} sits in region {region.region_id!r}, which "
                f"is a {region.kind!r} and has no patch geometry."
            )
        return region.grid

    def patch_corners_relative(
        self, patch: Patch, inset: float = 1.0
    ) -> tuple[tuple[float, float], ...]:
        """The patch's bounding box in target-relative coordinates."""
        return self.grid_for(patch).corners_relative(patch.row, patch.col, inset=inset)

    @property
    def geometry_is_provisional(self) -> bool:
        return any(r.grid.provisional for r in self.grid_regions if r.grid)

    def provisional_reasons(self) -> list[str]:
        return [
            f"{r.region_id}: {r.grid.provisional_reason}"
            for r in self.grid_regions
            if r.grid and r.grid.provisional and r.grid.provisional_reason
        ]

    def __getitem__(self, patch_id: str) -> Patch:
        try:
            return self._by_id[patch_id]
        except KeyError as exc:
            raise TargetError(f"Target {self.target_id!r} has no patch {patch_id!r}") from exc

    def get(self, patch_id: str) -> Patch | None:
        return self._by_id.get(patch_id)

    @property
    def patch_ids(self) -> list[str]:
        return [p.patch_id for p in self.patches]

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def by_role(self, role: str) -> list[Patch]:
        return [p for p in self.patches if p.role == role]

    def ramp(self) -> list[Patch]:
        """Ramp patches ordered by ``ramp_position``."""
        patches = [p for p in self.patches if p.role == "achromatic_ramp"]
        return sorted(patches, key=lambda p: (p.ramp_position is None, p.ramp_position, p.patch_id))

    def with_patches(self, patches: Iterable[Patch]) -> "TargetDefinition":
        """Return a copy carrying newly resolved patch roles."""
        clone = TargetDefinition(
            target_id=self.target_id,
            display_name=self.display_name,
            manufacturer=self.manufacturer,
            target_type=self.target_type,
            physical_size_mm=self.physical_size_mm,
            capabilities=self.capabilities,
            not_capable_of=self.not_capable_of,
            fiducials=self.fiducials,
            regions=self.regions,
            patches=list(patches),
            reference_data=self.reference_data,
            openqualia=self.openqualia,
            metric_defaults=self.metric_defaults,
            notes=self.notes,
            source_path=self.source_path,
            fiducial_extent_mm=self.fiducial_extent_mm,
            dimensional_reference=self.dimensional_reference,
        )
        return clone

    def center_relative(self, patch: Patch) -> tuple[float, float]:
        return self.grid_for(patch).center_relative(patch.row, patch.col)

    def is_perimeter(self, patch: Patch) -> bool:
        """Whether the patch sits on the edge of its own region's lattice."""
        return self.grid_for(patch).is_perimeter(patch.row, patch.col)


# -- (de)serialisation ------------------------------------------------------


def target_from_dict(data: dict[str, Any], source_path: Path | None = None) -> TargetDefinition:
    """Build a :class:`TargetDefinition` from parsed JSON.

    Accepts both shapes: a top-level ``patch_grid`` (a target that is one lattice)
    and a ``regions`` list (a target built from several separated groups). The
    former is read as a single region named ``main``, so existing definitions keep
    working and simple targets stay simple to write.
    """
    try:
        regions = _regions_from_dict(data)

        fid = data["fiducials"]
        markers = fid.get("corner_markers", [])
        fiducials = Fiducials(
            method=fid.get("method", "chart_corners"),
            corner_ids=[m["id"] for m in markers],
            corner_positions=[tuple(m["relative_position"]) for m in markers],  # type: ignore[misc]
            notes=fid.get("notes"),
        )

        default_region = next(
            (r.region_id for r in regions if r.holds_patches), DEFAULT_REGION
        )
        patches = [
            Patch(
                patch_id=p["patch_id"],
                row=p["grid_position"]["row"],
                col=p["grid_position"]["col"],
                role=p.get("role", "unassigned"),
                ramp_position=p.get("ramp_position"),
                notes=p.get("notes"),
                region=p.get("region", default_region),
                printed_id=p.get("printed_id"),
            )
            for p in data.get("patches", [])
        ]

        size = data.get("physical_size_mm")
        return TargetDefinition(
            target_id=data["target_id"],
            display_name=data.get("display_name", data["target_id"]),
            manufacturer=data.get("manufacturer"),
            target_type=data.get("target_type", "unknown"),
            physical_size_mm=(size["width"], size["height"]) if size else None,
            capabilities=frozenset(data.get("capabilities", [])),
            not_capable_of=frozenset(data.get("not_capable_of", [])),
            fiducials=fiducials,
            regions=regions,
            patches=patches,
            reference_data=data.get("reference_data", {}),
            openqualia=data.get("openqualia", {}),
            metric_defaults=data.get("metric_defaults", {}),
            notes=data.get("notes"),
            source_path=source_path,
            fiducial_extent_mm=(
                (extent["width"], extent["height"])
                if (extent := data.get("fiducial_extent_mm"))
                else None
            ),
            dimensional_reference=data.get("dimensional_reference", {}),
        )
    except KeyError as exc:
        raise TargetError(f"Target definition is missing required key {exc}") from exc


def _regions_from_dict(data: dict[str, Any]) -> list[Region]:
    declared = data.get("regions")
    if declared:
        return [
            Region(
                region_id=r["region_id"],
                kind=r.get("kind", "patch_grid"),
                grid=_grid_from_dict(r["patch_grid"]) if r.get("patch_grid") else None,
                bounds_relative=(
                    tuple(r["bounds_relative"]) if r.get("bounds_relative") else None
                ),
                geometry=_geometry_from_dict(r.get("geometry", {})),
                notes=r.get("notes"),
            )
            for r in declared
        ]

    grid_data = data.get("patch_grid")
    if not grid_data:
        return []
    return [
        Region(
            region_id=DEFAULT_REGION,
            kind="patch_grid",
            grid=_grid_from_dict(grid_data),
            notes=grid_data.get("notes"),
        )
    ]


#: Geometry keys whose spelling changed when the vocabulary was settled on
#: American English. A target definition is a file a user can write, and several
#: ship in this repository, so a definition written against the older spelling
#: keeps working rather than silently declaring a region with no geometry --
#: which would read as a chart that prints no such feature.
GEOMETRY_ALIASES = {"centre": "center"}


def _geometry_from_dict(geometry: dict) -> dict:
    if not isinstance(geometry, dict):
        return {}
    return {GEOMETRY_ALIASES.get(k, k): v for k, v in geometry.items()}


def _grid_from_dict(g: dict[str, Any]) -> PatchGrid:
    return PatchGrid(
        rows=g["rows"],
        columns=g["columns"],
        row_labels=g.get("row_labels", list(range(1, g["rows"] + 1))),
        col_labels=g.get("col_labels", []),
        patch_id_format=g.get("patch_id_format", "{column_letter}{row_number}"),
        patch_size_relative=(g["patch_size_relative"]["width"], g["patch_size_relative"]["height"]),
        patch_spacing_relative=(
            g["patch_spacing_relative"]["horizontal"],
            g["patch_spacing_relative"]["vertical"],
        ),
        grid_origin_relative=tuple(g["grid_origin_relative"]),  # type: ignore[arg-type]
        provisional=g.get("provisional", False),
        provisional_reason=g.get("provisional_reason"),
    )


def load_target_file(path: str | Path) -> TargetDefinition:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TargetError(f"{path}: invalid JSON -- {exc}") from exc
    return target_from_dict(data, source_path=path)
