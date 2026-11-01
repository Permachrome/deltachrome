"""The DT Nexus transmissive targets, and what it means to ship geometry unmeasured.

These are the first definitions here whose geometry did not come from a capture.
Every other target was measured off real pixels by a named tool, because
spec-sheet arithmetic is not precise enough for tight ROI placement -- so what
this module pins is mostly the reasoning that made an exception defensible, and
the facts about the charts that were established from published data rather than
assumed from the drawing.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pytest

from deltachrome.reference import read_reference
from deltachrome.reference.sample_ids import reconcile_sample_ids
from deltachrome.targets.registry import available_targets, load_target
from deltachrome.targets.roles import resolve_roles

TARGETS = [
    "dt_nexus_transmissive_ntpt_rev_a_v1",
    "dt_nexus_transmissive_ntpt_rev_b_v1",
    "dt_nexus_transmissive_ntvt_rev_a_v1",
    "dt_nexus_transmissive_ntvt_rev_b_v1",
]


@pytest.mark.parametrize("target_id", TARGETS)
def test_all_four_ship(target_id):
    assert target_id in available_targets()


@pytest.mark.parametrize("target_id", TARGETS)
def test_the_qr_cell_is_not_a_patch(target_id):
    """M-0 holds the OpenQualia label, so there are 129 patches and not 130. DT
    warn implementers about exactly this: some libraries invent a patch there,
    and some shift the whole first row up by one."""
    target = load_target(target_id)
    assert len(target.patches) == 129
    assert target.get("M-0") is None
    assert target.get("L-0") is not None
    assert target.get("M-1") is not None


@pytest.mark.parametrize("target_id", TARGETS)
def test_rows_are_zero_indexed_and_ids_carry_a_dash(target_id):
    """Both are DT's deliberate choices and both are things implementations get
    wrong -- the dash has been read as a minus sign."""
    ids = {p.patch_id for p in load_target(target_id).patches}
    assert "A-0" in ids and "M-9" in ids
    assert not any(re.fullmatch(r"[A-M]10", i) for i in ids)


@pytest.mark.parametrize("target_id", TARGETS)
def test_the_placement_is_the_patch_array_never_the_frame(target_id):
    """Not a preference. The border is opaque anodised aluminium and vanishes in a
    properly masked transmissive capture, and Revision B's drawing states that the
    glass assembly floats in the frame so the outside dimensions vary."""
    target = load_target(target_id)
    assert target.fiducials.method == "patch_array_corners"


def test_every_dt_chart_places_on_the_same_four_visible_circles():
    """A-1, L-0, M-8 and C-9 on all four, so nobody has to remember two placements.

    Not the lattice's corners. The frame is black in a transmissive capture, and on
    every Profiling target the A-9 corner is the opaque end of the neutral ladder --
    L* 0, the frame's own level -- while the Validation unit captured here reads L* 1
    at M-9. These four read L* 42 or more on all three captured units. Each point is
    the CENTRE of its circle, which is where a ring-shaped handle is put most
    precisely by eye."""
    for target_id in TARGETS:
        target = load_target(target_id)
        grid = target.grid_regions[0].grid
        columns = "ABCDEFGHIJKLM"
        expected = [grid.center_relative(row, columns.index(col))
                    for col, row in (("A", 1), ("L", 0), ("M", 8), ("C", 9))]
        assert np.allclose(target.fiducials.corner_positions, expected, atol=1e-4)
        assert target.fiducials.corner_ids == ["TL", "TR", "BR", "BL"]


@pytest.mark.parametrize("target_id", TARGETS)
def test_the_geometry_was_measured_from_a_capture(target_id):
    """These first shipped derived from DT's published drawing, because no capture
    existed. One arrived, and the lattice was re-measured off the real chart.

    The correction is worth a test of its own: the drawing-derived *spacing* held
    to 0.6-0.8%, but the drawing-derived *patch size* was 2% too large on
    Revision A and 10% on Revision B, because it assumed the stated 0.254 in
    aperture survived the 2026 revision while the pitch shrank. It did not.
    """
    grid = load_target(target_id).patch_grid
    assert grid.patch_size_relative[0] < 0.06, "the drawing's 0.254 in was too large"
    assert 0.72 < grid.patch_size_relative[0] / grid.patch_spacing_relative[0] < 0.74


def test_no_dt_lattice_is_flagged_provisional():
    """Three of the four were measured on their own model and revision. The
    Validation target's Revision B was not -- no capture of one exists here -- and
    carries the Profiling target's lattice, which DT state is the same in every
    dimension. That is assumed rather than warned about on every run; the borrowing
    is recorded in the definition, and confirming it is a beta-testing item."""
    for target_id in TARGETS:
        assert not load_target(target_id).geometry_is_provisional

    import json
    from deltachrome.targets.registry import DEFINITIONS_DIR

    raw = json.loads(
        (DEFINITIONS_DIR / "dt_nexus_transmissive_ntvt_rev_b_v1.json").read_text()
    )
    assert "TP26008" in raw["patch_grid"]["measured_from"]["unit"]


@pytest.mark.parametrize("target_id", TARGETS)
def test_no_nominal_values_are_shipped(target_id):
    """Every unit is individually measured and published, so a per-unit file is
    always obtainable and a batch mean would be a worse answer that looks like a
    better one."""
    target = load_target(target_id)
    assert target.reference_data["required"] is True
    assert target.nominal_reference_path is None


@pytest.mark.parametrize("target_id", TARGETS)
def test_each_declares_where_its_units_are_published(target_id):
    from deltachrome.reference.openqualia import label_for_serial, publishes_measurements

    target = load_target(target_id)
    assert publishes_measurements(target)
    serial = "TP24010" if "_ntpt_" in target_id else "TV24017"
    label = label_for_serial(target, serial)
    assert label is not None and label.manufacturer == "DT"


def test_the_two_revisions_differ_where_it_actually_matters():
    """The obvious reading of the revision is wrong, and the test records which.

    Revision B's pitch is about 8% tighter in real units, which sounds like a chart
    that must not be measured on the other's lattice. It is not: both definitions'
    frame is the patch array's own bounding box, so both lattices fill their frame
    and the relative geometry barely moves -- the far column's centre shifts 0.23%
    of the quad, some 5 px on a 2175 px array against a 64 px patch radius.

    What does differ is the physical size of that quad, 110.5 mm against 102.5.
    Sampling frequency scales directly with it, so the wrong revision misreports it
    by 7.3% -- enough to move a verdict sitting near a threshold. That is why both
    ship.
    """
    a = load_target("dt_nexus_transmissive_ntpt_rev_a_v1")
    b = load_target("dt_nexus_transmissive_ntpt_rev_b_v1")

    far_a = 12 * a.patch_grid.patch_spacing_relative[0] + a.patch_grid.patch_size_relative[0] / 2
    far_b = 12 * b.patch_grid.patch_spacing_relative[0] + b.patch_grid.patch_size_relative[0] / 2
    assert abs(far_a - far_b) < 0.005, "relative geometry is nearly shared"

    wide_a = a.dimensional_reference["fiducial_extent_mm"]["width"]
    wide_b = b.dimensional_reference["fiducial_extent_mm"]["width"]
    assert 0.90 < wide_b / wide_a < 0.95, "the physical extent is what moves"


def test_both_models_declare_a_neutral_series_measured_from_published_units():
    """Both series were established from the published measurements, and the
    Validation target's nearly was not.

    The Profiling target's is column A, ten steps, identical across six units
    spanning both revisions -- easy. The Validation target's is nine scattered
    patches, and resolving neutrals by chroma alone gives a different answer per
    unit: nine on the Revision A units, six on TV26001, five on TV26011. The first
    reading of that was that the neutral set changes between production runs. It
    does not. The same nine positions are the series on all five published units;
    what changes is how chromatic they are, Revision A's worst sitting at C*ab
    3.71 and Revision B's at 5.20, so a 4.0 cut-off keeps all nine on one revision
    and drops three or four on the other.

    That is the case for declaring rather than inferring: the positions are a
    property of the design, and where they fall against an arbitrary threshold is
    not.
    """
    ntvt = ["H-9", "C-7", "I-9", "K-4", "J-9", "K-9", "F-2", "L-9", "M-9"]
    for target_id in TARGETS:
        target = load_target(target_id)
        ramp = sorted((p for p in target.patches if p.role == "achromatic_ramp"),
                      key=lambda p: p.ramp_position)
        ids = [p.patch_id for p in ramp]
        assert ids == ([f"A-{n}" for n in range(10)] if "_ntpt_" in target_id else ntvt)
        assert {"tonal_response", "white_balance"} <= set(target.capabilities)


# -- against real published measurements ----------------------------------

UNITS = sorted(glob.glob("out/openqualia-units/DT*/*.oqm.txt"))


def _published(prefix):
    """Published units held in the local cache, if any have been fetched."""
    return [p for p in UNITS if os.path.basename(p).startswith(prefix)]


@pytest.mark.skipif(not _published("TP"), reason="no cached NTPT measurements")
def test_a_published_measurement_covers_every_patch():
    target = load_target("dt_nexus_transmissive_ntpt_rev_a_v1")
    reference, _ = reconcile_sample_ids(read_reference(_published("TP")[0]), target)
    present, missing = reference.coverage(target.patch_ids)
    assert not missing
    assert len(present) == 129


@pytest.mark.skipif(not _published("TP"), reason="no cached NTPT measurements")
def test_the_ramp_resolves_from_a_real_file():
    target = load_target("dt_nexus_transmissive_ntpt_rev_a_v1")
    reference, _ = reconcile_sample_ids(read_reference(_published("TP")[0]), target)
    resolution = resolve_roles(target, reference)
    assert resolution.counts.get("achromatic_ramp") == 10
