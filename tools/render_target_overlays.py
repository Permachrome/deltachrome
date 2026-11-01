"""Draw every shipped target's measured geometry over its own capture.

Every target here has its geometry measured off one capture and written into a
definition as relative coordinates, and the whole class of error that produces is
invisible in the numbers: a patch grid shifted by a third of a patch still samples
140 regions and returns 140 plausible Delta E values. Looking at the ROIs is the only
check, and until this tool there was no way to do it for all of them at once.

The placements below were each measured with the tool named beside them, not clicked
by hand -- so re-running this is reproducible and a change to a definition shows up
as a change to a picture.

Run from the repository root:

    python tools/render_target_overlays.py

``--target`` renders one. ``--refine`` snaps each placement onto the chart's printed
features first, which is what an operator would do and what shows whether refinement
is helping. ``--labels`` writes each patch id beside its ROI, which is unreadable on
a 264-patch chart and exactly what is wanted on a twelve-step wedge.

**The captures are not in the repository** -- they belong to the organisations
that published them -- so this runs against whatever is present and says what it
skipped. Each placement names its capture by filename, found anywhere under
``data/`` or under the folder ``DELTACHROME_CAPTURES`` names. The overlays go to ``data/overlays``, which is
gitignored for the same reason and for a sharper one: an overlay is the capture
with lines drawn on it, so committing one would redistribute the capture that the
ignore rule beside it exists to keep out.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = Path(os.environ.get("DELTACHROME_CAPTURES") or ROOT / "data")


@lru_cache(maxsize=None)
def capture_path(name: str) -> Path:
    """Where a capture named by filename is, under :data:`CAPTURES`.

    An absolute path is taken as it is. Otherwise the file is looked for directly
    under the captures folder and then anywhere beneath it, so the placements below
    need say nothing about how somebody's folder of captures is arranged.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    direct = CAPTURES / path
    if direct.exists() or not CAPTURES.is_dir():
        return direct
    return next(CAPTURES.rglob(path.name), direct)


@dataclass(frozen=True)
class Placement:
    """One capture, the target it holds, and where the fiducials are in it."""

    target_id: str
    capture: str
    corners: tuple[tuple[float, float], ...]
    #: Which tool measured these corners, so the number has a provenance rather
    #: than being a magic constant somebody typed.
    measured_by: str
    reference: str | None = None
    note: str = ""
    #: Which chart in the frame, where the capture holds more than one. Only the
    #: QA-62 does: it is a small card read at one field position and is shot four to
    #: a board, so four entries share one capture and would otherwise share one
    #: output filename as well.
    unit: str = ""

    @property
    def path(self) -> Path:
        return capture_path(self.capture)

    @property
    def name(self) -> str:
        stem = f"{self.target_id}__{Path(self.capture).stem.replace(' ', '_')}"
        return f"{stem}__{self.unit}" if self.unit else stem


#: The four QA-62 cards laid on one board, measured on the capture that gave the
#: definition its geometry. Four entries rather than one because a run is one chart:
#: the frame is read as four placements, four runs and one session.
QA62_BOARD = tuple(
    Placement(
        "applied_image_qa62_sfr_v1", "QA62_2.jp2", corners,
        "locate_qa62_geometry.py",
        unit=unit,
        note=("Corners are the four printed crosshairs bounding this card's "
              "measurement field, not the card. All four units agree on a square "
              "quad to 0.07%." if unit == "TL" else ""),
    )
    for unit, corners in (
        ("TL", ((982.5, 990.6), (2561.9, 967.1), (2588.5, 2545.4), (1009.9, 2569.8))),
        ("TR", ((8796.9, 1104.0), (10378.0, 1065.3), (10410.9, 2645.5), (8831.9, 2682.2))),
        ("BL", ((1061.6, 5033.2), (2639.2, 5032.5), (2639.5, 6609.5), (1061.2, 6610.2))),
        ("BR", ((8874.9, 4889.6), (10453.5, 4847.4), (10496.0, 6424.8), (8917.9, 6466.7))),
    )
)


PLACEMENTS = QA62_BOARD + (
    # Captures supplied by Digital Transitions.
    # The two Revision A charts share one frame and sit a quarter turn from upright,
    # ten columns across and thirteen rows down with the QR at the bottom right. The
    # corners below are the chart's own TL TR BR BL through that rotation, which is
    # why they read anticlockwise in image coordinates.
    Placement(
        "dt_nexus_transmissive_ntpt_rev_a_v1",
        "20250103 - DTNTPT (TP24010) and DTNTVT (TV24017) - tight mask - 177310.tif",
        ((10899.1, 864.6), (11387.2, 6257.2), (7433.8, 6747.5), (6952.7, 1820.1)),
        "locate_patch_lattice.py --rows 10 --columns 13 --shape circle",
        note="Unit TP24010, the right-hand chart of the pair. Identified by measurement "
             "rather than by the filename: sampled and compared against both published "
             "references it reads 2.44 dE2000 against TP24010 and 41.9 against TV24017. "
             "Corners are the centres of circles A-1, L-0, M-8 and C-9, the four every DT "
             "Nexus chart places on: one corner cell is opaque on every unit, so the "
             "lattice's own corners are not all visible. Re-derived through the placement "
             "that put the lattice's corners at the old quad, so no ROI moved.",
    ),
    Placement(
        "dt_nexus_transmissive_ntvt_rev_a_v1",
        "20250103 - DTNTPT (TP24010) and DTNTVT (TV24017) - tight mask - 177310.tif",
        ((4965.2, 894.2), (5442.2, 6320.0), (1459.9, 6794.2), (1001.2, 1869.5)),
        "locate_patch_lattice.py --rows 10 --columns 13 --shape circle",
        note="Unit TV24017, the left-hand chart of the same frame -- 2.77 dE2000 against "
             "its own reference and 42.6 against the other chart's, which is what "
             "identifies it.",
    ),
    Placement(
        "dt_nexus_transmissive_ntpt_rev_b_v1",
        "NTPT_TP26008_Front.tif",
        ((3343.2, 2874.4), (10723.5, 2240.5), (11405.3, 7662.5), (4705.1, 8301.4)),
        "locate_patch_lattice.py --rows 10 --columns 13 --shape circle",
        note="Unit TP26008, upright and alone in its frame. The only capture in the set "
             "carrying a scale -- 2133 ppi -- which is where this revision's physical "
             "extent comes from. No capture of a Revision B Validation target exists, so "
             "that definition borrows this lattice and says so.",
    ),
    Placement(
        "isa_golden_thread_dlt_v1", "ISA_Device_Level_Target.tif",
        ((6100.8, 2666.7), (8238.5, 3870.1), (5816.1, 5712.2), (3372.1, 4206.6)),
        "locate_printed_fiducials.py discs",
        note="Corners are the four printed discs, taken top, right, bottom, left -- a "
             "DIAMOND rather than a rectangle, which is what the definition now says a "
             "placement is made on and what the Golden Thread software keys on. The "
             "relative frame is still the four corner modules' resolution fields, so "
             "every drawn ROI is at the same coordinate it was before.",
    ),
    Placement(
        "isa_transmissive_4x5_v1", "ISA_4x5.tif",
        ((1550.7, 759.8), (10565.7, 745.1), (10574.3, 7756.0), (1559.8, 7768.1)),
        "locate_printed_fiducials.py frame",
        note="Corners are the printed keyline's four line intersections. The relative "
             "frame is still the quarter-disc apex midpoints, so the amber quad now "
             "ENCLOSES the chart instead of sitting inside it, and every ROI is at the "
             "coordinate it always was.",
    ),
    Placement(
        "isa_transmissive_35mm_v1", "ISA_35mm.tif",
        ((460.3, 466.1), (5796.4, 481.9), (5787.1, 3934.2), (451.2, 3918.8)),
        "locate_printed_fiducials.py frame",
        note="Corners are the printed keyline's four line intersections, as on the "
             "4x5 and the 120. This is the chart whose keyline the frame finder's "
             "darkness threshold used to just miss.",
    ),
    Placement(
        "isa_transmissive_120_v1", "120_target.tif",
        ((1714.4, 638.9), (8771.6, 694.9), (8718.8, 7748.8), (1659.8, 7694.2)),
        "locate_printed_fiducials.py frame",
        note="Corners are the printed keyline's four line intersections, as on the 4x5.",
    ),
    Placement(
        "image_engineering_utt_a3_v1",
        "qa62_LWF_BC_001_20211126_SH01_0411.jp2",
        ((2457.5, 1359.8), (7576.6, 1326.4), (7605.6, 4806.4), (2480.1, 4843.3)),
        "locate_utt_geometry.py",
    ),
    Placement(
        "isa_fadgi_19264_commercial_v1",
        "FADGI19264_Commercial_forV3.0.tif",
        ((711.4, 649.3), (3318.3, 656.0), (3315.9, 4662.3), (704.7, 4660.6)),
        "locate_printed_fiducials.py crosshairs",
        note="Corners are the four printed plus marks just inside the chart's own "
             "corners. The relative frame is still the corner modules' dpi fields, "
             "which is why the quad drawn here ENCLOSES the patch array rather than "
             "sitting inside it as it used to.",
    ),
    Placement(
        "microform_35mm_positive_v1", "Positive 35mm.tif",
        ((1570.7, 842.8), (4727.9, 843.1), (4727.0, 4001.5), (1569.3, 4000.6)),
        "locate_film_target_geometry.py microform",
    ),
    Placement(
        "colorchecker_sg_v1", "Calibrite_ColorChecker_SG.jp2",
        ((1020.9, 1918.5), (6963.5, 1933.5), (6955.2, 6162.9), (1008.2, 6148.7)),
        "locate_printed_fiducials.py crosshairs",
        note="The crosshairs sit just outside the array's corners, which is what the "
             "definition says to place on -- so this is not the patch array's own "
             "bounding box and the ROIs landing right is a real check.",
    ),
    Placement(
        "avian_ngt_v1", "NGT.tif",
        ((317.93, 289.01), (3082.73, 289.08), (3089.56, 2419.49), (321.15, 2421.63)),
        "locate_patch_lattice.py --rows 10 --columns 13",
        note="This chart prints no registration marks at all, so the definition "
             "places on the patch array's own bounding box -- which is what the "
             "lattice tool measures, so the ROIs landing right is weaker evidence "
             "here than elsewhere.",
    ),
    Placement(
        "it8_7_1_transmissive_v1", "IT8-7.1.tif",
        ((542.83, 290.95), (5686.27, 290.95), (5686.27, 3099.47), (542.83, 3099.47)),
        "locate_it8_lattice.py",
    ),
    Placement(
        "it8_7_2_reflective_v1", "IT8-7.2.tif",
        ((410.33, 299.86), (3136.56, 299.86), (3136.56, 1780.15), (410.33, 1780.15)),
        "locate_it8_lattice.py",
    ),
    Placement(
        "it8_7_2_reflective_v1", "WolfFaust_IT8-7-2.tif",
        ((3209.52, 2149.55), (5831.29, 2149.55), (5831.29, 3571.58), (3209.52, 3571.58)),
        "locate_it8_lattice.py --frame 255,155,3125,1825, on a crop at (2830, 1870)",
        note="A Wolf Faust unit, charge R151101, on Kodak paper, a third of a frame it "
             "shares with a ColorChecker Passport and masking tape. The second vendor's "
             "IT8.7/2 measured here, and the evidence that the generic definition -- "
             "grey strip included -- fits more than LaserSoft's print of the layout. "
             "The tool's own search for the keyline lands on patch boundaries on this "
             "chart, hence the frame given by hand; every boundary then sits on the "
             "comb, 23 of 23 across and 13 of 13 down.",
    ),
    Placement(
        "isa_fadgi_19264_v1", "FADGI_19264.tif",
        ((237.7, 225.0), (2326.0, 222.8), (2328.4, 3231.9), (239.4, 3234.3)),
        "locate_fadgi19264_geometry.py",
        note="Corners are the three black corner squares and, at the bottom right, "
             "registration dot 8. Re-derived through the placement that put the corner "
             "modules' dpi fields at the old quad, so the relative frame -- and every "
             "ROI -- is unchanged.",
    ),
    Placement(
        "lasersoft_iso12641_2_advanced_v1", "LaserSoft_4x5.tif",
        ((795.6, 999.0), (7612.3, 950.6), (7646.8, 5619.7), (829.4, 5666.8)),
        "locate_printed_fiducials.py frame",
        note="The quad is the black frame's four line intersections, which enclose "
             "the printed row and column labels as well as the 864 patches -- so "
             "the array is visibly inset from the amber quad, and should be.",
    ),
    Placement(
        "lasersoft_iso12641_2_advanced_v1", "LaserSoft_120.tif",
        ((585.6, 698.2), (6700.6, 720.3), (6688.1, 4908.1), (565.8, 4885.5)),
        "locate_printed_fiducials.py frame, on a crop inside the film holder",
        note="A 120 unit, serial T1710030, printed 'ISO 12641-2 Candidate 2017:10', "
             "in a film holder whose mask leaks light along the top and bottom. Measured "
             "on a crop because the tool's left and right fits caught the mask and the "
             "vertical 'SilverFast' text beside the keyline; detection on the whole "
             "frame agrees to 0.1 px. A second placement on one definition, as the "
             "Object Level 1X has, because it is the evidence that one definition "
             "serves both formats. A test export rendered to sRGB: good for "
             "geometry, not for judging the chart's colour.",
    ),
    Placement(
        "avian_dt_ngt_v2", "DT-NGT-v2.tif",
        ((155.4, 163.0), (3892.4, 167.9), (3882.9, 3062.4), (135.1, 3058.9)),
        "locate_printed_fiducials.py crosshairs",
        note="The four printed pluses sit diagonally outside the array's corners, "
             "so as with the SG this is not the array's own bounding box.",
    ),
    Placement(
        "isa_object_level_1x_v1", "ObjectDICE.tif",
        ((315.5, 279.2), (3703.0, 253.3), (3703.7, 352.9), (316.3, 378.9)),
        "locate_printed_fiducials.py strip",
        note="A 34:1 quad, and the older unit -- three discs, no yellow border. The "
             "definition places on the patch strip's own corners, so as with the "
             "Avian NGT the patch ROIs landing right is weaker evidence than "
             "elsewhere; the module and the two rules are the real check, and they "
             "are placed from the patches alone.",
    ),
    Placement(
        "isa_object_level_1x_v1", "ObjectLevel_2.tif",
        ((227.9, 201.4), (3621.4, 204.9), (3621.3, 304.7), (227.8, 301.2)),
        "locate_printed_fiducials.py strip",
        note="The current unit -- two discs, yellow border, serial M000390. Shipped "
             "as a second placement rather than replacing the first because this is "
             "the only target here measured on two physical specimens.",
    ),
    Placement(
        "microform_35mm_negative_v1",
        "Negative_35mm_ICC (1).tif",
        ((1303.4, 487.4), (4303.7, 485.6), (4316.7, 3474.9), (1315.7, 3477.8)),
        "locate_film_target_geometry.py microform",
        note="Grayscale: one colour record, tagged Gray Gamma 2.2.",
    ),
)


#: How far a patch's mean may move when its ROI is grown, in normalized code value,
#: before the geometry is suspect.
#:
#: The reason this exists: a picture shows a *gross* error, and the error that
#: matters here is a subtle one -- a lattice whose pitch is half a percent out
#: samples every patch and drifts into the neighbours only at the ends of a long
#: row. Growing each ROI turns that into a number. A well-centred sample on a flat
#: patch barely moves; one already near a boundary moves as soon as it crosses.
#: 0.02 is about five levels of an 8-bit encoding.
SENSITIVITY_LIMIT = 0.02
#: The insets compared. The engine samples at 0.55.
INSETS = (0.40, 0.55, 0.75)


def _inset_margin(image, placement, target) -> dict:
    """How much each patch's reading moves when its ROI is grown and shrunk.

    The numeric half of this tool, and the more sensitive half. A patch whose ROI
    sits centrally on a flat region reads the same at 0.40, 0.55 and 0.75 of the
    declared cell; one whose ROI is off-centre or whose cell is mis-sized starts
    taking in its neighbour as it grows, and says so long before the misalignment
    is visible.
    """
    from deltachrome.imaging.sampling import sample_patches

    readings: dict[str, list[float]] = {}
    for inset in INSETS:
        result = sample_patches(image, placement, target, inset=inset,
                                measure_noise=False)
        for patch_id, sample in result.samples.items():
            if sample.usable:
                readings.setdefault(patch_id, []).append(float(sample.rgb_mean.mean()))

    spreads = {pid: max(v) - min(v) for pid, v in readings.items() if len(v) == len(INSETS)}
    if not spreads:
        return {}
    worst = max(spreads, key=lambda k: spreads[k])
    import statistics

    return {
        "worst": spreads[worst],
        "worst_patch": worst,
        "median": statistics.median(spreads.values()),
        "suspect": sum(1 for v in spreads.values() if v > SENSITIVITY_LIMIT),
        "spreads": spreads,
    }


def render(entry: Placement, out: Path, *, refine: bool, labels: bool) -> str:
    from deltachrome.imaging.edges import extract_edges
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.loader import load_image
    from deltachrome.imaging.refine import refine_placement
    from deltachrome.imaging.rulers import read_rulers
    from deltachrome.imaging.sampling import sample_patches
    from deltachrome.reference import read_reference
    from deltachrome.reporting.overlay import render_geometry
    from deltachrome.targets.registry import load_target
    from deltachrome.targets.roles import resolve_roles

    image = load_image(entry.path)
    target = load_target(entry.target_id)

    # Roles are resolved from the reference so that the patches carry the same
    # roles the run would give them; without it a ramp is "unassigned" and the
    # picture is of a slightly different set of regions from the one measured.
    if target.patches and target.nominal_reference_path:
        target = resolve_roles(target, read_reference(target.nominal_reference_path)).target

    placement = place_chart(
        list(entry.corners), image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )
    line = ""
    if refine:
        result = refine_placement(image, placement, target)
        if result.applied:
            placement = place_chart(
                result.corners, image_size=(image.width, image.height),
                corner_positions_relative=target.fiducials.corner_positions or None,
            )
            line = (f"  refined {result.features_used} features as a {result.model}, "
                    f"moved {max(result.moved_px):.1f} px, residual "
                    f"{result.residual_px:.2f} px")
        else:
            line = f"  not refined: {result.reason}"

    sampling = (sample_patches(image, placement, target, measure_noise=False)
                if target.grid_regions else None)
    edges = (extract_edges(image, placement, target)
             if any(r.kind == "slanted_edge" for r in target.regions) else [])
    rulers = ([] if not any(r.kind == "scale" for r in target.regions)
              else read_rulers(image, placement, target)[0])

    picture = render_geometry(
        image, placement, target, sampling=sampling, edges=edges, rulers=rulers,
        label_patches=labels,
    )
    path = out / f"{entry.name}{'_refined' if refine else ''}.png"
    cv2.imwrite(str(path), picture)

    margin = _inset_margin(image, placement, target) if sampling else None
    flagged = sum(1 for s in sampling.samples.values() if s.flags) if sampling else 0
    scales = sum(1 for r in target.regions if r.kind == "scale")
    summary = (
        f"{path.name}\n"
        f"  {len(sampling) if sampling else 0} patches ({flagged} flagged), "
        f"{len(edges)} edge ROIs, {len(rulers)} of {scales} rules read"
    )
    if margin:
        summary += (
            f"\n  inset sensitivity: worst {margin['worst']:.4f} at "
            f"{margin['worst_patch']}, median {margin['median']:.4f}"
            + (f"  -- {margin['suspect']} patch(es) over "
               f"{SENSITIVITY_LIMIT:.2f}" if margin["suspect"] else "")
        )
    return summary + ("\n" + line if line else "")


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "overlays")
    parser.add_argument("--target", default=None, help="render only this target id")
    parser.add_argument("--refine", action="store_true",
                        help="snap each placement onto the printed features first")
    parser.add_argument("--labels", action="store_true",
                        help="write each patch id beside its ROI")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    missing = []
    for entry in PLACEMENTS:
        if args.target and entry.target_id != args.target:
            continue
        if not (entry.path).exists():
            missing.append(entry.capture)
            continue
        print(render(entry, args.out, refine=args.refine, labels=args.labels))
        if entry.note:
            print(f"  note: {entry.note}")

    if missing:
        print("\nnot present, so not rendered:")
        for path in missing:
            print(f"  {path}")
        print("  The captures are gitignored -- they belong to the Library of "
              "Congress and to the operator, not to this project.")


if __name__ == "__main__":
    main()
