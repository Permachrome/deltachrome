"""The capture panel's own geometry.

One claim, tested numerically because it is the kind that looks right in code and
is wrong on screen: **the point of the image under the pointer stays under the
pointer** when the wheel turns.

It did not. The new centre was solved against the *widget's* width and height,
where the invariant is in image pixels -- image extent times scale -- and those
two agree only on the axis the fit scale was set by. Everything on the other axis
crept, by the ratio between the widget's extent and the letterboxed image's, and
it compounded per notch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui.capture_dock import FrameOverview  # noqa: E402
from deltachrome_ui.charts import ChartSelection  # noqa: E402
from deltachrome_ui.imageproxy import ImageProxy  # noqa: E402

#: Panel-shaped, so the letterboxing under test is the letterboxing that happens.
PANEL = (384, 240)

#: Three shapes, because the bug was invisible on one of them: a capture whose
#: aspect matches the panel's is the case where widget extent and image extent
#: agree on both axes, and the wrong formula is right there by accident.
SHAPES = [(8000, 6000), (6000, 8000), (12000, 3000)]


class _Flat:
    """Pixels enough to build a proxy from; nothing here reads them."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height, self.scale = width, height, 1.0
        self.pixels = np.full((height, width, 3), 0.5, dtype=np.float32)
        # None is a real value here, not a stand-in: ImageProxy's raw display
        # path is what this class exists to exercise cheaply.
        self.profile = None


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def overview(qt_app, size: tuple[int, int]) -> FrameOverview:
    view = FrameOverview(ImageProxy(_Flat(*size)), ChartSelection(), {})
    view.resize(*PANEL)
    return view


def turn(view: FrameOverview, at: tuple[float, float], notches: float) -> None:
    view.wheelEvent(QWheelEvent(
        QPointF(*at), QPointF(*at), QPoint(0, 0), QPoint(0, int(120 * notches)),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))


def image_point_under(view: FrameOverview, at: tuple[float, float]):
    scale, dx, dy = view._placement()
    return ((at[0] - dx) / scale, (at[1] - dy) / scale)


@pytest.mark.parametrize("size", SHAPES)
@pytest.mark.parametrize("at", [(120, 80), (300, 170), (250, 60), (140, 190)])
def test_the_point_under_the_pointer_stays_under_the_pointer(qt_app, size, at):
    """Started already zoomed in, which is where this was worst and where the
    clamp is not the thing being measured: at low zoom the view legitimately
    cannot move, because there is nothing off screen to move to."""
    view = overview(qt_app, size)
    view._zoom, view._center = 4.0, [0.5, 0.5]

    before = image_point_under(view, at)
    turn(view, at, +1)
    after = image_point_under(view, at)
    assert after == pytest.approx(before, abs=0.5)


@pytest.mark.parametrize("size", SHAPES)
def test_it_holds_over_a_run_of_notches_rather_than_one(qt_app, size):
    """The error compounded, which is why it was reported as 'worse the further
    in you already are' rather than as a jump."""
    view = overview(qt_app, size)
    # Zoomed in far enough that both axes have somewhere to go on every shape
    # here. On the 4:1 capture the whole height is still on screen at zoom 2.5,
    # so below that the vertical centre is pinned and what would be measured is
    # the clamp doing its job rather than the anchoring.
    view._zoom, view._center = 4.0, [0.5, 0.5]
    at = (300, 170)

    before = image_point_under(view, at)
    for _ in range(4):
        turn(view, at, +1)
    assert image_point_under(view, at) == pytest.approx(before, abs=1.0)


@pytest.mark.parametrize("size", SHAPES)
def test_the_view_never_leaves_the_image(qt_app, size):
    """The other half of the same arithmetic. How much of an axis is on screen
    is the widget's extent over the image's, and that is *not* ``1 / zoom``: the
    fit is set by whichever axis is tighter, so the other is letterboxed and
    shows more of itself at every zoom."""
    view = overview(qt_app, size)
    for at in [(2, 2), (382, 238), (2, 238), (382, 2)]:
        view._zoom, view._center = 1.0, [0.5, 0.5]
        for _ in range(8):
            turn(view, at, +1)
        scale, dx, dy = view._placement()
        width = view.proxy.width * scale
        height = view.proxy.height * scale
        # Either the axis fills the panel and no edge is inside it, or the whole
        # of the axis is visible and it is centred.
        assert dx <= 0.5 or width <= view.width() + 0.5
        assert dx + width >= view.width() - 0.5 or width <= view.width() + 0.5
        assert dy <= 0.5 or height <= view.height() + 0.5
        assert dy + height >= view.height() - 0.5 or height <= view.height() + 0.5


@pytest.mark.parametrize("size", SHAPES)
def test_zooming_all_the_way_out_recentres(qt_app, size):
    view = overview(qt_app, size)
    view._zoom, view._center = 6.0, [0.8, 0.2]
    for _ in range(20):
        turn(view, (300, 60), -1)
    assert view._zoom == pytest.approx(view.ZOOM_RANGE[0])
    assert view._center == pytest.approx([0.5, 0.5])
