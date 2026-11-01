"""Where each chart of a capture sits, drawn on the capture.

A board of four QA-62 cards is four runs called ``Target 1 of 4`` through
``Target 4 of 4``, and that ordinal is the order the extent search returned the
regions in -- not a position, and nothing an operator can predict. On four cards
of one model it identifies nothing, which bites hardest on the combined verdict,
the screen whose job is comparing them.

What is pinned here is the three claims the feature rests on.

**The outlines come from the record, not from the pixels.** A capture reopened
from the library holds every chart's ``geometry.corners_px`` and no pixels at
all, and decoding a preservation master is five to sixteen seconds. The map has
to draw without one.

**One frame, or no map.** Charts read from different captures are not a frame,
and a fingerprint is what says so -- not a path, which changes when a file moves,
and not its absence, since two records carrying no fingerprint are not thereby
one capture.

**A verdict never renders in the pass tone without a conformance claim.** Every
QA-62 and every ISA target runs on nominal values, so an advisory pass is the
ordinary result here and the outline must say so.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from deltachrome_ui.frame_map import (  # noqa: E402
    ChartInFrame, FrameInfo, frame_from_records,
)
from deltachrome_ui.status import TONE_PASS, TONE_QUALIFIED, overall_status  # noqa: E402

QUADS = [
    [(982.5, 990.6), (2561.9, 967.1), (2588.5, 2545.4), (1009.9, 2569.8)],
    [(8796.9, 1104.0), (10378.0, 1065.3), (10410.9, 2645.5), (8831.9, 2682.2)],
    [(1061.6, 5033.2), (2639.2, 5032.5), (2639.5, 6609.5), (1061.2, 6610.2)],
]


def fingerprint(digest: str = "a1b2c3") -> dict:
    """The shape ``file_fingerprint`` actually writes.

    Faked as a bare string while this was being built, which is what let a
    ``TypeError`` -- the mapping is unhashable -- reach the running application
    with every test passing. Test data that is not the record's own shape tests
    a record nobody has.
    """
    return {"algorithm": "blake2b-128/size+1048576B-ends",
            "size_bytes": 302848000, "digest": digest}


def record(index: int, *, digest: str = "a1b2c3") -> dict:
    return {
        "run_id": f"run{index}",
        "image": {"fingerprint": fingerprint(digest),
                  "width": 11600, "height": 8700},
        "geometry": {"corners_px": [list(point) for point in QUADS[index]]},
        "target": {"target_id": "applied_image_qa62_sfr_v1",
                   "display_name": "Applied Image QA-62"},
    }


def statuses(count: int, *, claim: bool = False) -> list:
    return [overall_status("PASS", claim, evidence_sound=claim)
            for _ in range(count)]


def test_frame_is_built_from_records_alone():
    """No pixels anywhere in this test, and a complete map comes out."""
    frame = frame_from_records([record(0), record(1)], statuses(2))
    assert frame is not None and frame.usable
    assert frame.image is None
    assert (frame.width, frame.height) == (11600, 8700)
    assert [chart.ordinal for chart in frame.charts] == [1, 2]
    assert [chart.run_id for chart in frame.charts] == ["run0", "run1"]
    assert frame.charts[0].corners[0] == (982.5, 990.6)


def test_charts_from_different_captures_are_not_a_frame():
    mixed = [record(0), record(1, digest="different")]
    assert frame_from_records(mixed, statuses(2)) is None


def test_records_without_a_fingerprint_are_not_matched_to_each_other():
    """Absence is not agreement. Older records carry no fingerprint, and two of
    those are not thereby one capture."""
    bare = [
        {**record(0), "image": {"width": 11600, "height": 8700}},
        {**record(1), "image": {"width": 11600, "height": 8700}},
    ]
    assert frame_from_records(bare, statuses(2)) is None


def test_a_single_chart_is_not_worth_drawing():
    """Its quad is very nearly the whole frame, so the outline is a border round
    a picture and answers a question nobody asked."""
    assert frame_from_records([record(0)], statuses(1)) is None


def test_a_chart_with_no_recorded_placement_makes_the_frame_unusable():
    missing = [record(0), {**record(1), "geometry": {}}]
    assert frame_from_records(missing, statuses(2)) is None


def test_no_pass_tone_without_a_conformance_claim():
    """The rule `status.py` states, carried onto the outlines."""
    advisory = frame_from_records([record(0), record(1)], statuses(2, claim=False))
    assert advisory is not None
    assert {chart.status.tone for chart in advisory.charts} == {TONE_QUALIFIED}

    measured = frame_from_records([record(0), record(1)], statuses(2, claim=True))
    assert measured is not None
    assert {chart.status.tone for chart in measured.charts} == {TONE_PASS}


def test_notes_are_optional_and_positional():
    frame = frame_from_records(
        [record(0), record(1)], statuses(2), ["nominal reference"])
    assert frame is not None
    assert frame.charts[0].note == "nominal reference"
    assert frame.charts[1].note == ""


def test_usable_needs_a_size():
    """A frame of no known dimensions cannot be drawn to its own aspect."""
    frame = FrameInfo(width=0, height=0, charts=[
        ChartInFrame(ordinal=n + 1, corners=QUADS[n],
                     status=overall_status("PASS", False), target_name="x")
        for n in range(2)
    ])
    assert not frame.usable


def test_the_key_is_the_digest_and_not_the_whole_mapping():
    """So a record written by another version still matches its own capture.

    The algorithm string is part of what ``file_fingerprint`` writes and is not
    part of what identifies the file. Comparing the mappings whole would fail
    exactly when a library holds records from two versions -- and comparing a
    mapping at all is what broke first: it is unhashable.
    """
    from deltachrome_ui.frame_map import capture_key

    assert capture_key(fingerprint("dead00")) == "dead00"
    restated = {**fingerprint("dead00"), "algorithm": "blake2b-128/something-else"}
    assert capture_key(restated) == capture_key(fingerprint("dead00"))
    assert capture_key(None) is None
    assert capture_key({}) is None
    assert capture_key({"algorithm": "x", "size_bytes": 1}) is None


def test_two_charts_of_one_capture_are_matched_through_the_real_shape():
    """The end-to-end version of the bug: real fingerprints, one frame out."""
    frame = frame_from_records([record(0), record(1)], statuses(2))
    assert frame is not None and frame.usable


# -- the map as a control ----------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def built(charts: int = 3):
    frame = frame_from_records(
        [record(n) for n in range(charts)], statuses(charts),
        ["nominal reference"] * charts)
    assert frame is not None
    return frame


def centre_of(widget, index: int):
    """Where a chart's outline actually landed, through the map's own transform."""
    from PySide6.QtCore import QPointF

    box = widget._canvas()
    polygon = widget._polygon(widget._frame.charts[index].corners, box)
    points = [polygon.at(i) for i in range(polygon.count())]
    return QPointF(sum(p.x() for p in points) / len(points),
                   sum(p.y() for p in points) / len(points))


def test_clicking_an_outline_asks_for_that_target_s_report(qt_app):
    """The map's whole interaction. Pinned through the real transform rather
    than a guessed coordinate, so a change to the projection fails here."""
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    from deltachrome_ui.frame_map import FrameMap

    widget = FrameMap(built())
    widget.resize(800, 600)
    asked: list[str] = []
    widget.chartActivated.connect(asked.append)

    for index in (0, 2, 1):
        point = centre_of(widget, index)
        event = QMouseEvent(QEvent.Type.MouseButtonRelease, point,
                            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        widget.mouseReleaseEvent(event)
    assert asked == ["run0", "run2", "run1"]


def test_pointing_between_the_outlines_asks_for_nothing(qt_app):
    """The board between two cards is not a target, and a press there is not a
    request for whichever one happens to be nearest."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent

    from deltachrome_ui.frame_map import FrameMap

    widget = FrameMap(built())
    widget.resize(800, 600)
    asked: list[str] = []
    widget.chartActivated.connect(asked.append)

    middle = widget._canvas().center()
    widget.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, middle, Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert asked == []
    assert widget._at(middle) == -1


def test_the_map_and_the_list_light_each_other(qt_app):
    """One selection shared between a picture and a list, which is the
    arrangement the report already uses for its plots and readings tables."""
    from deltachrome_ui.frame_map import FrameBlock

    block = FrameBlock(built())
    block.resize(800, 700)

    block._light(1)
    assert block.map._hover == 1
    assert [row._lit for row in block._rows] == [False, True, False]

    block._light(-1)
    assert block.map._hover == -1
    assert not any(row._lit for row in block._rows)


def test_clicking_a_row_asks_for_the_same_report_as_its_outline(qt_app):
    from deltachrome_ui.frame_map import FrameBlock

    block = FrameBlock(built())
    asked: list[str] = []
    block.chartActivated.connect(asked.append)
    block._activate(2)
    assert asked == ["run2"]


def test_the_map_paints_with_and_without_pixels(qt_app):
    """A paint-time error is invisible until somebody runs the application: Qt
    swallows the traceback to stderr and draws nothing. Both states are painted
    here because they take different branches -- one draws the capture, the
    other the frame's own outline and its size."""
    from PySide6.QtGui import QImage

    from deltachrome_ui.frame_map import FrameBlock

    block = FrameBlock(built())
    block.resize(800, 700)
    bare = block.grab()
    assert not bare.isNull()

    pixels = QImage(64, 48, QImage.Format.Format_RGB888)
    pixels.fill(0x336699)
    block.set_image(pixels)
    assert block.map._frame.image is pixels
    filled = block.grab()
    assert not filled.isNull()
    assert filled.toImage() != bare.toImage(), "the capture was not drawn"

    # And a hovered outline draws its fill without upsetting anything.
    block._light(0)
    assert not block.grab().isNull()


def test_a_press_lands_correctly_before_the_first_paint_and_after_a_resize(qt_app):
    """The bug this found. Hit-testing used the outlines ``paintEvent`` had
    cached, so a press was only correct after a repaint -- nothing at all before
    the first paint, and tested against the previous size's outlines after a
    resize, which on a frame of four cards can answer with the wrong one."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent

    from deltachrome_ui.frame_map import FrameMap

    def press(widget, index):
        asked: list[str] = []
        widget.chartActivated.connect(asked.append)
        widget.mouseReleaseEvent(QMouseEvent(
            QEvent.Type.MouseButtonRelease, centre_of(widget, index),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))
        widget.chartActivated.disconnect()
        return asked

    widget = FrameMap(built())
    widget.resize(800, 600)
    assert press(widget, 1) == ["run1"], "nothing was painted yet"

    widget.grab()                      # now it has been
    widget.resize(500, 950)            # and the projection has changed under it
    assert press(widget, 1) == ["run1"]
    assert press(widget, 2) == ["run2"]
