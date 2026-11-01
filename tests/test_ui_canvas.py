"""The placement canvas.

The invariant under test throughout: **a handle's position is an image pixel
coordinate**, at every zoom, and is therefore exactly what ``analyze(corners=...)``
expects. No conversion step exists to get backwards, and this is where that is
pinned down.

The round trip through the CLI's own argument parser is the most valuable test
here. It is what stops the two front ends drifting into describing the same
geometry differently -- a GUI that wrote its corners in a subtly different order
or scale would produce reports that disagree with the command line about the same
capture, which is precisely what a shared engine exists to prevent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui.canvas import CORNER_LABELS, PlacementCanvas  # noqa: E402
from test_ui_imageproxy import make_image  # noqa: E402
from deltachrome_ui.imageproxy import ImageProxy  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def canvas(qt_app):
    view = PlacementCanvas()
    view.set_proxy(ImageProxy(make_image(1200, 900)))
    view.resize(600, 500)
    return view


@pytest.fixture
def big_canvas(qt_app):
    """Large enough that the overview is decimated, so detail tiles are in play."""
    view = PlacementCanvas()
    view.set_proxy(ImageProxy(make_image(6000, 4500)))
    view.resize(600, 500)
    return view


def settle_sampling(widget, timeout_s: float = 60.0):
    """Wait for the overlay job, which runs off the UI thread.

    The work is the real thing -- the edge ROIs drawn are the ones the
    measurement cuts, refinement included -- and that is most of a second on the
    largest charts here, so it cannot run on the thread that has to keep the
    corner handles moving. These tests therefore drive the threaded path rather
    than a synchronous stand-in for it: what would be pinned by a stand-in is a
    code path the application never takes.
    """
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    while (widget._sample_running or widget._sample_pending
           or widget._sample_timer.isActive()):
        QApplication.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("the sampling overlay never settled")
        time.sleep(0.002)
    QApplication.processEvents()
    return widget


def draw_regions(widget):
    """Ask for the overlay and wait for it, as one step."""
    widget._refresh_sample_regions()
    return settle_sampling(widget)


def flat(points):
    """Corner lists flattened for comparison.

    ``pytest.approx`` does not apply its tolerance to a list of tuples -- it
    compares them exactly and, when they differ, reports "0 mismatched elements",
    which is a confusing way to fail. Flattening makes the tolerance real.
    """
    return [value for point in points for value in point]


SQUARE = [(100.0, 80.0), (900.0, 80.0), (900.0, 700.0), (100.0, 700.0)]


def test_corners_round_trip_in_image_pixels(canvas):
    canvas.set_corners(SQUARE)
    assert flat(canvas.corners()) == pytest.approx(flat(SQUARE))


def test_corners_are_image_pixels_at_any_zoom(canvas):
    """Zoom changes the view transform and must not touch the geometry."""
    canvas.set_corners(SQUARE)
    for factor in (0.05, 1.0, 8.0):
        canvas.zoom_to(factor)
        assert flat(canvas.corners()) == pytest.approx(flat(SQUARE)), factor


def test_the_placement_parses_back_through_the_cli_argument_parser(canvas):
    """One geometry, described one way, whichever front end a person uses."""
    from deltachrome.inputs import parse_corners

    canvas.set_corners(SQUARE)
    assert flat(parse_corners(canvas.corners_text())) == pytest.approx(flat(SQUARE))


def test_a_corner_dragged_off_frame_is_held_on_the_last_pixel(canvas):
    canvas.set_corners([(-500.0, -500.0), (1e6, 0.0), (1e6, 1e6), (0.0, 1e6)])
    for x, y in canvas.corners():
        assert 0.0 <= x <= 1199.0
        assert 0.0 <= y <= 899.0


def test_the_seeded_rectangle_is_inside_the_frame_and_correctly_ordered(canvas):
    canvas.seed_default()
    states = []
    canvas.validityChanged.connect(lambda s, r: states.append(s))
    canvas.set_corners(canvas.corners())
    assert states[-1] == "ok"
    assert len(canvas.corners()) == 4


def _severity(canvas, corners) -> tuple[str, str]:
    seen: list[tuple[str, str]] = []
    canvas.validityChanged.connect(lambda s, r: seen.append((s, r)))
    canvas.set_corners(corners)
    return seen[-1]


def test_a_correctly_ordered_quad_is_silent(canvas):
    assert _severity(canvas, SQUARE)[0] == "ok"


def test_a_rotated_capture_is_silent(canvas):
    """Corners are the *chart's*, and geometry is relative to the fiducial quad,
    so a chart photographed upside down needs no special handling. Rotation
    preserves winding; flagging it would train people to ignore the warning."""
    rotated = [(900.0, 700.0), (100.0, 700.0), (100.0, 80.0), (900.0, 80.0)]
    assert _severity(canvas, rotated)[0] == "ok"


def test_a_crossed_quad_is_an_error(canvas):
    crossed = [(100.0, 80.0), (900.0, 80.0), (100.0, 700.0), (900.0, 700.0)]
    severity, reason = _severity(canvas, crossed)
    assert severity == "error"
    assert "cross" in reason.lower()


def test_a_mirrored_traversal_is_a_caution_not_an_error(canvas):
    """Anticlockwise means a mirrored capture, which is normal for film shot
    through the base. The tool says what it sees and leaves the judgement to
    someone who knows which way round the original was."""
    mirrored = [(100.0, 80.0), (100.0, 700.0), (900.0, 700.0), (900.0, 80.0)]
    severity, reason = _severity(canvas, mirrored)
    assert severity == "caution"
    assert "mirror" in reason.lower()


def test_the_handles_are_labelled_in_the_order_the_engine_expects(canvas):
    assert CORNER_LABELS == ("TL", "TR", "BR", "BL")
    assert [h.label for h in canvas._handles] == list(CORNER_LABELS)


def test_a_placement_seeded_from_elsewhere_needs_no_special_path(canvas):
    """Manual entry, a saved run and (later) a detector all arrive the same way.
    Phase 4 changes what seeds the handles, not this widget."""
    from_record = [(11.0, 22.0), (1100.0, 33.0), (1150.0, 800.0), (44.0, 850.0)]
    canvas.set_corners(from_record)
    assert flat(canvas.corners()) == pytest.approx(flat(from_record))


def test_an_image_smaller_than_the_overview_never_needs_a_detail_read(canvas):
    """The overview is already every pixel there is, so zooming reads nothing."""
    assert canvas.proxy.overview_stride == 1
    canvas.zoom_to(8.0)
    canvas._refresh_detail()
    assert canvas._detail_item is None or not canvas._detail_item.isVisible()


def test_zooming_past_the_overview_produces_a_finer_read(big_canvas):
    """The whole reason the canvas is usable on a 100 MP master: the overview is
    for framing and detail tiles are read only for what is on screen."""
    canvas = big_canvas
    canvas.zoom_to(4.0)
    canvas._refresh_detail()
    assert canvas._detail_item is not None and canvas._detail_item.isVisible()
    assert canvas._detail_item.scale() < canvas.proxy.overview_stride


def test_zoomed_out_needs_no_detail_read_at_all(big_canvas):
    canvas = big_canvas
    canvas.zoom_to(4.0)
    canvas._refresh_detail()
    canvas.fit()
    canvas._refresh_detail()
    assert canvas._detail_item is None or not canvas._detail_item.isVisible()


# -- refining a placement ----------------------------------------------------
#
# The button exists because a placement can look right, pass the aspect check,
# produce good figures across eighteen metrics and silently disable a nineteenth.
# So what it must never do is move the corners without saying so.


@pytest.fixture
def view(qt_app, tmp_path):
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path))), tmp_path / "chart.tif")
    # The shell's job, stood in for here: nothing is drawn over a capture until
    # something has said which chart it is.
    widget.settle_target()
    widget.resize(900, 600)
    return widget


def test_refining_moves_the_handles_onto_the_features(view):
    from test_refine_placement import TRUE_CORNERS, shift, worst_error

    view.set_corners(shift(TRUE_CORNERS, 24, 18))
    view.refine_button.click()
    assert worst_error(view.canvas.corners()) < 3.0


def test_refining_says_how_far_it_moved_and_on_what_evidence(view):
    """Silence would be the worst outcome here: every measurement that follows
    uses the new placement.

    A pill and not the paragraph it was. How far it moved is the pill; on what
    evidence is the detail behind it, because the corner readout directly above
    already shows the placement having changed, and four lines restating that is
    what a fluent reader learns to skip."""
    from test_refine_placement import TRUE_CORNERS, shift

    view.set_corners(shift(TRUE_CORNERS, 24, 18))
    view.refine_button.click()
    assert view.refine_pill.isVisibleTo(view)
    assert "Refined" in view.refine_pill.text()
    assert "px" in view.refine_pill.text()
    assert "5 printed features" in view.refine_pill.toolTip()


def test_refining_an_already_correct_placement_says_nothing_moved(view):
    """Distinct from declining. The features were found and they confirm the
    placement, which is a different fact from being unable to look -- and it is
    not an improvement either, so it does not read in the same tone."""
    from deltachrome_ui.status import TONE_PASS
    from test_refine_placement import TRUE_CORNERS

    view.set_corners(TRUE_CORNERS)
    view.refine_button.click()
    assert view.refine_pill.text() == "Already Refined"
    assert "Nothing moved" in view.refine_pill.toolTip()
    assert TONE_PASS not in view.refine_pill.styleSheet()


def test_refining_twice_is_stable(view):
    from test_refine_placement import TRUE_CORNERS, shift

    view.set_corners(shift(TRUE_CORNERS, 28, 20))
    view.refine_button.click()
    first = flat(view.canvas.corners())
    view.refine_button.click()
    assert flat(view.canvas.corners()) == pytest.approx(first, abs=1.0)


def test_the_button_is_disabled_for_a_chart_that_prints_nothing_usable(view):
    """A ColorChecker SG has no features to snap to, and a button that fails on
    every press teaches people to ignore it."""
    from deltachrome.targets.registry import load_target

    view.set_target(load_target("colorchecker_sg_v1"))
    assert not view.refine_button.isEnabled()
    assert "No marks" in view.refine_button.toolTip()


def test_the_button_is_offered_for_a_chart_whose_marks_are_quarter_discs(view):
    """The regression. This gate used to be computed here and counted rotated
    squares only, so the three ISA transmissive charts -- ten disc features apiece,
    and the charts refinement recovers every corner of to 0.44 px -- were shown a
    dead button claiming they print no marks to snap to."""
    from deltachrome.targets.registry import load_target

    view.set_target(load_target("isa_transmissive_4x5_v1"))
    assert view.refine_button.isEnabled()
    assert view.refine_button.toolTip() == ""


def test_the_button_is_withheld_from_a_chart_that_declares_too_few(view):
    """Enabled is not the only way to be wrong. Both Object Level Targets declare
    their one edge as ``straight_edges``, which yields no centre to correspond, so
    this button used to be live on a chart that refuses on every press."""
    from deltachrome.targets.registry import load_target

    view.set_target(load_target("isa_object_level_1x_v1"))
    assert not view.refine_button.isEnabled()
    assert "No marks" in view.refine_button.toolTip()


def test_the_sampling_button_is_disabled_only_when_there_is_nothing_to_draw(view):
    """A frame-filling white sheet has neither patches nor edges, and asking the
    sampler for patches raises rather than returning nothing -- so this used to
    fail on every press.

    Both kinds count. A chart that measures only sharpness declares no patch grid
    at all, and its edges are the whole of what an operator has to check."""
    from deltachrome.targets.registry import load_target
    from test_refine_placement import module_target

    sharpness_only = module_target()
    assert not sharpness_only.grid_regions, "the fixture is the shape being tested"

    view.set_target(load_target("flat_field_v1"))
    assert not view.check_button.isEnabled()
    assert "No regions" in view.check_button.toolTip()

    view.set_target(sharpness_only)
    assert view.check_button.isEnabled()


def test_moving_a_corner_clears_a_now_stale_sampling_overlay(qt_app, tmp_path):
    """The ROIs shown were computed from a placement that no longer exists.

    They come back, once the corners stop moving: sampling every region costs a
    fraction of a second, which is fine once and hopeless per mouse-move, so the
    overlay is cleared immediately and recomputed on a delay."""
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import render

    widget = PlacementView()
    widget.set_target(load_target("isa_golden_thread_dlt_v1"))
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="dlt.tif"))),
                     tmp_path / "dlt.tif")
    widget.settle_target()
    widget.resize(900, 600)
    draw_regions(widget)
    assert widget.canvas.showing_sample_regions
    widget.set_corners([(x + 5, y + 5) for x, y in widget.canvas.corners()])
    assert not widget.canvas.showing_sample_regions
    draw_regions(widget)  # what the timer fires
    assert widget.canvas.showing_sample_regions


def test_a_chart_with_only_edges_still_draws_its_regions(view):
    """The case this generalisation is for. A chart that reads only sharpness
    declares no patch grid at all, so an overlay built out of patch ROIs would
    show it nothing -- and its edge ROIs are the whole of what an operator has to
    check before believing a number.

    An edge ROI fails the same way a patch ROI does and as invisibly: it can
    straddle the wrong transition or catch a neighbouring printed feature, and
    either returns a confident answer."""
    assert not view.target.grid_regions
    draw_regions(view)
    assert view.canvas.showing_sample_regions


def test_the_edge_rois_drawn_are_the_ones_the_measurement_cuts(view):
    """Not a cheaper approximation of them. A picture of a region that is not the
    region being read would confirm a placement the run does not use."""
    from deltachrome.imaging.edges import extract_edges
    from deltachrome.imaging.geometry import place_chart

    draw_regions(view)
    placement = place_chart(
        view.canvas.corners(),
        image_size=(view.proxy.width, view.proxy.height),
        corner_positions_relative=view.target.fiducials.corner_positions or None,
    )
    cut = extract_edges(view.proxy.image, placement, view.target)
    assert cut, "the fixture chart should offer edges to cut"
    assert len(view.canvas._roi_items) == len(cut)


def test_the_sampling_regions_are_drawn_without_being_asked_for(qt_app, tmp_path):
    """They are the one thing on this screen that says whether the placement is
    good enough to measure from. A button offering to reveal them asks the
    operator to suspect a problem before showing them one.

    Once the chart is known — see the test below for the other half."""
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import render

    widget = PlacementView()
    widget.set_target(load_target("isa_golden_thread_dlt_v1"))
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="dlt.tif"))),
                     tmp_path / "dlt.tif")
    widget.settle_target()
    draw_regions(widget)
    assert widget.canvas.showing_sample_regions
    assert widget.check_button.text() == "Hide Sampling"


def test_the_button_label_cannot_disagree_with_what_is_drawn(qt_app, tmp_path):
    """It did, and the failure was the worst kind: the label was set in whichever
    branch of the toggle last ran, so reusing this view for a second capture left
    it reading "Hide Sampling" over a canvas with nothing on it -- and the first
    press then appeared to do the opposite of what the button said."""
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import render

    widget = PlacementView()
    widget.set_target(load_target("isa_golden_thread_dlt_v1"))
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="dlt.tif"))),
                     tmp_path / "dlt.tif")
    widget.settle_target()
    draw_regions(widget)

    widget._toggle_sample_regions()
    settle_sampling(widget)
    assert not widget.canvas.showing_sample_regions
    assert widget.check_button.text() == "Check Sampling"

    widget._toggle_sample_regions()
    settle_sampling(widget)
    assert widget.canvas.showing_sample_regions
    assert widget.check_button.text() == "Hide Sampling"

    # A second capture through the same view, which is how they came apart.
    widget.set_target(load_target("colorchecker_sg_v1"))
    assert widget.check_button.text() == "Hide Sampling"
    widget.set_target(load_target("flat_field_v1"))
    assert widget.check_button.text() == "Check Sampling", (
        "a chart with no regions to draw cannot offer to hide them"
    )


# -- the sampling overlay runs off the UI thread ------------------------------
#
# Measured on the committed placements before this moved: 898 ms on the ISA
# transmissive 4x5, of which 818 is extract_edges snapping ten quarter discs;
# 326 ms on the ColorChecker SG's 140 patches. Inline, that landed on the one
# gesture this screen exists for -- the overlay rebuilds 250 ms after the corners
# stop moving, so a careful placement put a near-second freeze between every
# nudge and the next and the handle jumped to catch up afterwards.


def test_the_overlay_is_not_computed_on_the_ui_thread(view):
    """The whole point. If this ever runs inline the assertion below passes by
    accident, so what is pinned is that the request has *not* finished when it
    returns."""
    view._refresh_sample_regions()
    assert view._sample_running
    assert not view.canvas.showing_sample_regions
    settle_sampling(view)
    assert view.canvas.showing_sample_regions


def test_only_one_sampling_job_runs_at_a_time(view):
    """A drag outruns this by a wide margin, so a job per request would pile up
    threads all but one of whose answers is stale before it is computed."""
    view._refresh_sample_regions()
    assert view._sample_running
    for _ in range(5):
        view._refresh_sample_regions()
    assert view._sample_pending
    assert len(getattr(view, "_threads", {})) == 1
    settle_sampling(view)


def test_a_result_for_corners_that_have_since_moved_is_not_drawn(view):
    """The overlay is cleared the instant a handle moves because it is then a
    picture of a placement that no longer exists. A result computed for those same
    old corners is the same picture arriving late, and must not be drawn either.
    """
    from deltachrome_ui.placement_view import sample_regions

    stale = view.canvas.corners()
    # Far enough that the two answers genuinely differ. A small shift does not
    # prove anything here: a slanted edge self-locates, so refinement pulls the
    # ROI back onto the same printed square and both placements cut the same
    # pixels -- which is exactly why SFR survives a placement error.
    inward = (180, 180), (-180, 180), (-180, -180), (180, -180)
    moved = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(stale, inward)]
    _p, _f, was = sample_regions(view.proxy.image, view.target, stale)
    _p, _f, edges = sample_regions(view.proxy.image, view.target, moved)
    assert [roi[1] for roi in was] != [roi[1] for roi in edges]

    view._refresh_sample_regions()
    assert view._sample_running
    view.canvas.set_corners(moved)
    settle_sampling(view)

    assert view.canvas.showing_sample_regions
    drawn = [item.rect() for item in view.canvas._roi_items]
    assert len(drawn) == len(edges)
    for item, (_id, (x0, y0, x1, y1), _flag) in zip(drawn, edges):
        assert (item.left(), item.top()) == pytest.approx((x0, y0))


def test_sampling_leaves_the_capture_untouched(view):
    """It reads the pixels from another thread while the loupe reads them from
    this one. Concurrent reads are safe and a write would not be, so what is
    pinned is that there is no write."""
    import numpy as np

    before = np.array(view.proxy.image.pixels, copy=True)
    draw_regions(view)
    assert np.array_equal(before, view.proxy.image.pixels)


# -- drawing a region for the search to work in -------------------------------
#
# The case: a chart that is a small part of a busy frame. `chart_extent` finds
# the chart as the largest region unlike the frame's border, which is the whole
# frame when nothing separates them -- and every finder's size gate is a fraction
# of that region, so the whole-frame answer puts the smallest admissible patch
# above the largest the chart prints. On the Smithsonian Odonata captures the
# Object Level Target refuses with "Only 0 colored patches found"; given one
# loosely dragged rectangle it places in 0.2 s.


def band(canvas, x0, y0, x1, y1):
    """Drive Qt's rubber band over a scene rectangle, then release it."""
    from PySide6.QtCore import QPointF, QRect

    corner = canvas.mapFromScene(QPointF(x0, y0))
    far = canvas.mapFromScene(QPointF(x1, y1))
    canvas.rubberBandChanged.emit(
        QRect(corner, far).normalized(), QPointF(), QPointF()
    )
    canvas.rubberBandChanged.emit(QRect(), QPointF(), QPointF())


def test_a_dragged_rectangle_comes_back_in_image_pixels(canvas):
    """The invariant this whole class is built on: a scene coordinate is an image
    pixel. A region reported in anything else would be searched somewhere else."""
    seen = []
    canvas.regionDrawn.connect(seen.append)
    canvas.begin_region_search()
    assert canvas.searching_region
    band(canvas, 200, 150, 900, 700)

    assert len(seen) == 1
    x0, y0, x1, y1 = seen[0]
    # To within the zoom's own rounding: Qt reports the band in whole viewport
    # pixels, and one of those is several image pixels at a fit-to-window zoom.
    # That is the right accuracy for the job -- a search region only has to be
    # correct to a few percent, which is what the margin sweep measured.
    tolerance = 2.0 / canvas.zoom
    assert (x0, y0) == pytest.approx((200, 150), abs=tolerance)
    assert (x1, y1) == pytest.approx((900, 700), abs=tolerance)


def test_the_region_mode_lasts_one_gesture(canvas):
    """Panning is how the operator reaches the corner they are about to check, so
    a canvas that stayed in this mode would take it away until they found the
    control again."""
    canvas.begin_region_search()
    band(canvas, 200, 150, 900, 700)
    assert not canvas.searching_region


def test_a_drag_too_small_to_be_a_rectangle_is_abandoned(canvas):
    """A slipped click must not start a search of eleven pixels. Its own refusal
    would be perfectly accurate and completely unhelpful."""
    drawn, abandoned = [], []
    canvas.regionDrawn.connect(drawn.append)
    canvas.regionAbandoned.connect(lambda: abandoned.append(1))
    canvas.begin_region_search()
    band(canvas, 400, 400, 404, 404)

    assert not drawn
    assert abandoned
    assert not canvas.searching_region


def test_the_region_is_not_taken_for_a_placement(canvas):
    """It would be far easier to treat the four dragged corners as the chart's
    own, and wrong in the way this project cares most about: a hand-drawn box is
    out by tens of pixels, which reads every ROI off the edge of its patch and
    returns a full set of confident numbers."""
    before = canvas.corners()
    canvas.begin_region_search()
    band(canvas, 200, 150, 900, 700)
    assert canvas.corners() == before


def test_the_panel_button_unsets_itself_after_the_gesture(view):
    """The button says which mode the canvas is in, so it cannot stay pressed
    over a canvas that has gone back to panning."""
    view.region_button.setChecked(True)
    view._toggle_region_search()
    assert view.canvas.searching_region

    asked = []
    view.regionSearched.connect(asked.append)
    band(view.canvas, 100, 100, 800, 700)
    assert asked
    assert not view.region_button.isChecked()


def test_an_abandoned_gesture_puts_the_button_back(view):
    view.region_button.setChecked(True)
    view._toggle_region_search()
    band(view.canvas, 100, 100, 103, 103)
    assert not view.region_button.isChecked()


def test_the_detector_searches_a_region_when_given_one(monkeypatch):
    """The region takes the single-placement path, because a region means the
    operator has already said which chart is wanted."""
    from deltachrome_ui import workers

    calls = {}

    def fake_near(image, target, corners, **kwargs):
        calls["corners"] = corners
        return "placed"

    monkeypatch.setattr(
        "deltachrome.imaging.detect.detect_placement_near", fake_near
    )
    monkeypatch.setattr(
        "deltachrome.imaging.detect.detect_placements",
        lambda *a, **k: pytest.fail("a drawn region must not scan the frame"),
    )
    found = []
    worker = workers.PlacementDetector(
        image=object(), target=object(), region=(10, 20, 110, 220)
    )
    worker.finished.connect(found.append)
    worker.run()

    assert found == [["placed"]]
    assert calls["corners"] == [(10, 20), (110, 20), (110, 220), (10, 220)]


# -- nothing is drawn until something says which chart this is ----------------


def test_nothing_is_drawn_over_a_capture_whose_chart_is_unknown(qt_app, tmp_path):
    """The combo always holds a value, so a capture nothing could be identified in
    opens showing whichever target happens to be first in the list. Drawing that
    chart's ROIs over the capture is not a neutral default — it is the screen
    asserting a chart it has no evidence for, in the one place the operator is
    meant to be checking evidence."""
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="u.tif"))),
                     tmp_path / "u.tif")
    widget.resize(900, 600)

    assert not widget.target_settled
    draw_regions(widget)
    assert not widget.canvas.showing_sample_regions

    # And the moment somebody says what it is, they appear without being asked.
    widget.settle_target()
    settle_sampling(widget)
    assert widget.canvas.showing_sample_regions


def test_unsettling_takes_the_regions_back_off(qt_app, tmp_path):
    """Opening a second capture must not leave the first one's regions drawn over
    it while its own chart is still unknown."""
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="v.tif"))),
                     tmp_path / "v.tif")
    widget.settle_target()
    draw_regions(widget)
    assert widget.canvas.showing_sample_regions

    widget.unsettle_target()
    assert not widget.canvas.showing_sample_regions
    assert not widget.target_settled


def test_the_region_search_asks_for_an_identification(qt_app):
    """It identifies rather than only placing, which is the difference between a
    tool for a chart you have already named and a tool for a capture nothing could
    be found in. Identification has always taken the region to look in."""
    from deltachrome_ui import workers

    seen = {}

    def fake_identify(image, target_ids=None, *, extent=None):
        seen["extent"] = extent
        return "named"

    monkey = pytest.MonkeyPatch()
    monkey.setattr("deltachrome.imaging.detect.identify_target", fake_identify)
    monkey.setattr(
        "deltachrome.imaging.detect.identify_targets",
        lambda *a, **k: pytest.fail("a drawn region must not scan the whole frame"),
    )
    try:
        from deltachrome.imaging.loader import load_image
        from test_refine_placement import render
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmp:
            image = load_image(render(pathlib.Path(tmp), name="r.tif"))
            found = []
            worker = workers.TargetIdentifier(image=image, region=(100, 100, 900, 900))
            worker.finished.connect(found.append)
            worker.run()
    finally:
        monkey.undo()

    assert found == [["named"]]
    # The rectangle reached the search as the region to look in, padded.
    assert seen["extent"] is not None
    x0, y0, x1, y1 = seen["extent"].box
    assert x0 < 100 and y0 < 100 and x1 > 900 and y1 > 900


def test_a_chart_added_by_hand_draws_nothing_until_it_is_moved(qt_app, tmp_path):
    """It starts on the canvas's default inset rectangle — the middle of the frame,
    over pixels that are almost certainly not a chart — so its ROIs there are a
    picture of something nobody has claimed. Same fault as drawing an unidentified
    chart's, and it clears differently: *which chart* is settled by naming it,
    *where it is* by somebody putting it somewhere."""
    from PySide6.QtCore import QPointF
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="a.tif"))),
                     tmp_path / "a.tif")
    widget.settle_target()
    widget.unsettle_placement()
    widget.resize(900, 600)

    assert widget.target_settled and not widget.placement_settled
    assert not widget.drawing_regions
    draw_regions(widget)
    assert not widget.canvas.showing_sample_regions

    # A hand on a handle is what it was waiting for.
    handle = widget.canvas._handles[0]
    handle.setPos(QPointF(handle.scenePos().x() + 40, handle.scenePos().y() + 30))
    settle_sampling(widget)
    assert widget.placement_settled
    assert widget.canvas.showing_sample_regions


def test_both_halves_must_hold_before_anything_is_drawn(qt_app, tmp_path):
    """Two unknowns, two flags. Either one open withholds the overlay."""
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="b.tif"))),
                     tmp_path / "b.tif")
    widget.resize(900, 600)

    for target, placement in ((False, False), (True, False), (False, True)):
        widget.settle_target() if target else widget.unsettle_target()
        widget.settle_placement() if placement else widget.unsettle_placement()
        draw_regions(widget)
        assert not widget.canvas.showing_sample_regions, (target, placement)

    widget.settle_target()
    widget.settle_placement()
    draw_regions(widget)
    assert widget.canvas.showing_sample_regions


def test_a_refinement_settles_where_the_chart_is(qt_app, tmp_path):
    """Refinement puts the corners on the printed features, which is a placement
    however the corners got there."""
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import TRUE_CORNERS, module_target, render, shift

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="c.tif"))),
                     tmp_path / "c.tif")
    widget.settle_target()
    widget.set_corners(shift(TRUE_CORNERS, 18, 12))
    widget.unsettle_placement()
    assert not widget.drawing_regions

    widget.refine_button.click()
    assert widget.placement_settled


# -- a capture nobody could identify shows a clean canvas ---------------------


def test_no_reticles_either_while_nothing_is_named(qt_app, tmp_path):
    """Manual placement is the last resort, not the default. Four handles on a
    rectangle in the middle of the frame are an invitation to drag them — and on a
    capture where nothing has been named they are a placement of nothing. The
    offer on that screen should be Search a Region."""
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy
    from deltachrome_ui.placement_view import PlacementView
    from test_refine_placement import module_target, render

    widget = PlacementView()
    widget.set_target(module_target())
    widget.set_proxy(ImageProxy(load_image(render(tmp_path, name="n.tif"))),
                     tmp_path / "n.tif")
    widget.resize(900, 600)

    widget.unsettle_target()
    assert not widget.canvas.placement_visible
    draw_regions(widget)
    assert not widget.canvas.showing_sample_regions

    widget.settle_target()
    assert widget.canvas.placement_visible


def test_the_target_list_opens_blank(qt_app):
    """A combo always holds a value, so whichever target is first in the list read
    as an answer. Blank reads as a question, which is what it is."""
    from deltachrome_ui.choices import NO_TARGET, ChoicesPanel

    panel = ChoicesPanel()
    assert panel.target_id == NO_TARGET
    assert not panel.has_target
    assert not panel.ready, "a run cannot start without a chart to run against"

    assert panel.select_target("colorchecker_sg_v1")
    assert panel.has_target
    panel.clear_target()
    assert not panel.has_target


def test_a_search_shows_a_moving_bar_and_not_only_a_word(view):
    """A region search on a hundred-megapixel master runs for seconds over a canvas
    that does not change, which is indistinguishable from a press that did
    nothing."""
    assert not view.busy_strip.isVisibleTo(view)
    view.set_busy("Searching the Region…")
    assert view.busy_strip.isVisibleTo(view)
    assert "Searching" in view.busy_strip.label.text()
    view._clear_busy()
    assert not view.busy_strip.isVisibleTo(view)


def test_every_target_number_is_visible_without_scrolling(view):
    """A popup that scrolls is the gesture that has already caused trouble here."""
    from deltachrome_ui.placement_view import ChartOffer

    view.offer_charts([ChartOffer() for _ in range(12)], 0)
    assert view.chart_combo.count() == 12
    assert view.chart_combo.maxVisibleItems() >= 12


def test_a_short_popup_has_no_scroll_arrows(view):
    """`setMaxVisibleItems` is not enough and the reason is circular: Qt's popup
    container takes the arrows' height out of the *view* rather than adding it to
    the container, so a two-item popup got a view one and a half rows tall — which
    made it scrollable, which is the condition for showing the arrows. Measured on
    the chart selector before this: container 66 px, view 39, one visible 175x15
    arrow strip, scroll range 1, on a list of two."""
    from deltachrome_ui.placement_view import ChartOffer

    view.offer_charts([ChartOffer(), ChartOffer()], 1)
    combo = view.chart_combo
    combo.showPopup()
    try:
        inner = combo.view()
        container = inner.parentWidget()
        assert inner.verticalScrollBar().maximum() == 0, "nothing to scroll to"
        assert not any(
            child.isVisible() for child in container.children()
            if hasattr(child, "isVisible") and child is not inner
        ), "the scroll arrows are hidden"
        # Every row fits, which is what takes the scroll range to zero.
        rows = sum(inner.sizeHintForRow(r) for r in range(combo.count()))
        assert inner.height() >= rows
    finally:
        combo.hidePopup()


def test_a_long_popup_is_left_alone(qt_app):
    """Only a list that fits loses its arrows. A long one genuinely needs them, so
    it keeps Qt's own sizing and pixel scrolling — where a momentum gesture is the
    point rather than a control fighting the hand.

    Asserted on the policy rather than on the rendered scrollbar: the popup's real
    height depends on the screen it opens against, which a headless test has not
    got."""
    from deltachrome_ui.widgets import ComboBox
    from PySide6.QtWidgets import QAbstractItemView

    combo = ComboBox()
    combo.addItems([f"Item {n}" for n in range(40)])
    combo.show()
    combo.showPopup()
    try:
        assert combo.maxVisibleItems() == ComboBox.FITS_WITHOUT_SCROLLING
        assert (combo.view().verticalScrollMode()
                == QAbstractItemView.ScrollMode.ScrollPerPixel)
        # And its height was not pinned to the row count, which is what would
        # make a forty-item list open as a forty-item wall.
        assert combo.view().maximumHeight() > combo.view().sizeHintForRow(0) * 40
    finally:
        combo.hidePopup()


# -- a scroll gesture scrolls -------------------------------------------------


def wheel(canvas, dx=0, dy=0, modifiers=None):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    return QWheelEvent(
        QPointF(canvas.viewport().rect().center()),
        canvas.mapToGlobal(canvas.viewport().rect().center()),
        QPoint(0, 0), QPoint(dx, dy),
        Qt.MouseButton.NoButton,
        modifiers or Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )


def test_a_vertical_swipe_zooms(big_canvas):
    """This canvas is an inspection surface, not a document. The gesture it is
    asked for most is "look closer at that corner", which is what CAD, 3D, web
    maps and the image views in Lightroom and Capture One put on the wheel."""
    big_canvas.zoom_to(1.0)
    big_canvas.show()

    big_canvas.wheelEvent(wheel(big_canvas, dy=120))
    assert big_canvas.zoom > 1.0
    big_canvas.wheelEvent(wheel(big_canvas, dy=-120))
    assert big_canvas.zoom == pytest.approx(1.0, rel=1e-6)


def test_a_horizontal_swipe_pans_sideways(big_canvas):
    """Its delta used to be thrown away: only `angleDelta().y()` was read."""
    big_canvas.zoom_to(4.0)
    big_canvas.show()
    bar = big_canvas.horizontalScrollBar()
    bar.setValue(bar.minimum())
    before, zoom = bar.value(), big_canvas.zoom

    big_canvas.wheelEvent(wheel(big_canvas, dx=-360))

    assert bar.value() != before, "it pans"
    assert big_canvas.zoom == pytest.approx(zoom), "and does not magnify"


def test_a_slanted_swipe_does_one_thing_or_the_other(big_canvas):
    """A swipe is never exactly straight, so acting on both deltas would zoom and
    pan at once from one flick and neither by the amount asked for. The larger
    delta decides what the gesture was; the other is dropped."""
    big_canvas.zoom_to(4.0)
    big_canvas.show()

    # Mostly vertical: it zooms, and by what the vertical delta alone asks for --
    # which is how "the sideways component is dropped" is visible. The scroll bar
    # moves either way, because an anchored zoom re-centres the view.
    big_canvas.zoom_to(2.0)
    big_canvas.wheelEvent(wheel(big_canvas, dx=20, dy=240))
    slanted = big_canvas.zoom
    big_canvas.zoom_to(2.0)
    big_canvas.wheelEvent(wheel(big_canvas, dy=240))
    assert slanted == pytest.approx(big_canvas.zoom, rel=1e-9)

    # Mostly sideways: it pans, and the magnification is untouched.
    big_canvas.zoom_to(4.0)
    bar = big_canvas.horizontalScrollBar()
    bar.setValue((bar.minimum() + bar.maximum()) // 2)
    before = bar.value()
    big_canvas.wheelEvent(wheel(big_canvas, dx=240, dy=20))
    assert big_canvas.zoom == pytest.approx(4.0, rel=1e-9)
    assert bar.value() != before


def test_zooming_is_gentler_than_it_was(big_canvas):
    """1.25 a notch is sensible for a wheel that clicks and wrong for anything
    Apple sells: a high-resolution delta arrives many times a second, so the
    exponent comes as a stream of fractions and compounds."""
    from deltachrome_ui.canvas import PlacementCanvas

    assert PlacementCanvas.ZOOM_STEP < 1.25
    big_canvas.zoom_to(1.0)
    big_canvas.wheelEvent(wheel(big_canvas, dy=120))
    assert big_canvas.zoom == pytest.approx(PlacementCanvas.ZOOM_STEP, rel=1e-6)


def test_a_pinch_zooms_about_the_pointer(big_canvas):
    """A trackpad reports one as a native gesture rather than as a wheel, and its
    value is already the fractional change for this step."""
    big_canvas.zoom_to(1.0)
    big_canvas._zoom_by(1.0 + 0.2)
    assert big_canvas.zoom == pytest.approx(1.2, rel=1e-6)


# -- the frame's other placements ------------------------------------------
#
# A board of four cards showed one quadrilateral at a time, so ``Target 3 of 4``
# named a chart the operator could not locate on the frame in front of them.
# The others are drawn behind the live one now. What matters is that they are
# *only* context: they carry no handles, they sit under everything the operator
# is actually checking, and they leave completely when asked.

GHOST = [(100.0, 100.0), (400.0, 100.0), (400.0, 380.0), (100.0, 380.0)]
OTHER = [(700.0, 500.0), (1000.0, 500.0), (1000.0, 780.0), (700.0, 780.0)]


def test_other_placements_are_drawn_and_cleared(canvas):
    before = len(canvas.scene().items())
    canvas.show_other_placements([(1, GHOST), (2, OTHER)])
    assert len(canvas.scene().items()) > before
    canvas.clear_other_placements()
    assert canvas._ghost_items == []
    assert len(canvas.scene().items()) == before


def test_clearing_twice_is_harmless(canvas):
    canvas.show_other_placements([(1, GHOST)])
    canvas.clear_other_placements()
    canvas.clear_other_placements()
    assert canvas._ghost_items == []


def test_showing_again_replaces_rather_than_accumulates(canvas):
    """Stepping between charts redraws the ghosts on every move."""
    canvas.show_other_placements([(1, GHOST), (2, OTHER)])
    first = len(canvas._ghost_items)
    canvas.show_other_placements([(1, GHOST), (2, OTHER)])
    assert len(canvas._ghost_items) == first


def test_a_degenerate_placement_is_skipped_rather_than_drawn(canvas):
    """A chart added by hand and never moved may hold nothing worth drawing."""
    canvas.show_other_placements([(1, []), (2, [(1.0, 2.0)]), (3, GHOST)])
    canvas.show_other_placements([(3, GHOST)])
    only = len(canvas._ghost_items)
    canvas.show_other_placements([(1, []), (2, [(1.0, 2.0)]), (3, GHOST)])
    assert len(canvas._ghost_items) == only


def test_ghosts_stay_under_the_live_quad_and_its_handles(canvas):
    """A ghost is context. Nothing about it may sit over the chart being checked."""
    canvas.show_other_placements([(1, GHOST)])
    ghost_top = max(item.zValue() for item in canvas._ghost_items)
    assert ghost_top < canvas._quad.zValue()
    assert all(ghost_top < handle.zValue() for handle in canvas._handles)


def test_a_new_capture_forgets_the_previous_frame(canvas):
    """``set_proxy`` clears the scene, so the list must not keep dead items."""
    canvas.show_other_placements([(1, GHOST), (2, OTHER)])
    canvas.set_proxy(ImageProxy(make_image(800, 600)))
    assert canvas._ghost_items == []
    # And the canvas is still usable afterwards.
    canvas.show_other_placements([(1, GHOST)])
    assert canvas._ghost_items
