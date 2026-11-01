"""The application mark.

Three things are worth pinning and the rest is drawing.

**The aspect must agree with the geometry.** :data:`~deltachrome_ui.mark.ASPECT`
is a hand-written constant sitting beside hand-written polygons, and the whole
point of it is that callers size boxes by it. If someone nudges a coordinate and
not the constant, every mark in the application stretches by a little and
nothing else complains.

**A square box must not stretch the mark.** The delta already carries a
deliberate 117.6% horizontal stretch from the artwork; a second, accidental one
is a different logo. :func:`~deltachrome_ui.mark.fitted` is what prevents it,
and it is the function every painter goes through.

**The chi must actually be ramped and the delta must not.** That is the
construction the mark is built on -- the delta is the difference and the chi is
the colour -- and it is the one property a careless refactor would quietly lose
by filling both from one brush.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui import mark  # noqa: E402
from deltachrome_ui.theme import BG_INPUT, MARK_INK, RAMP  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_aspect_matches_the_polygons():
    """The constant is the geometry's own ratio, not an independent guess."""
    xs = [x for poly in (mark._DELTA_OUTER, mark._CHI) for x, _ in poly]
    ys = [y for poly in (mark._DELTA_OUTER, mark._CHI) for _, y in poly]
    # The polygons are normalised to their joint bounding box, so this is 1x1.
    assert (min(xs), min(ys)) == pytest.approx((0.0, 0.0), abs=1e-4)
    assert (max(xs), max(ys)) == pytest.approx((1.0, 1.0), abs=1e-4)
    # Which makes the traced box's own aspect the thing ASPECT has to carry.
    assert mark.ASPECT == pytest.approx(1355.8 / 1237.8, rel=1e-3)


def test_the_counter_is_inside_the_delta():
    """An even-odd fill only leaves a hole if the hole is enclosed."""
    xs = [x for x, _ in mark._DELTA_OUTER]
    ys = [y for _, y in mark._DELTA_OUTER]
    for x, y in mark._DELTA_COUNTER:
        assert min(xs) < x < max(xs)
        assert min(ys) < y < max(ys)


@pytest.mark.parametrize("box", [(0, 0, 200, 200), (0, 0, 400, 100), (0, 0, 60, 300)])
def test_fitted_never_stretches(box):
    area = mark.fitted(QRectF(*box))
    assert area.width() / area.height() == pytest.approx(mark.ASPECT, rel=1e-6)
    # And it stays inside what it was given, centred.
    outer = QRectF(*box)
    assert area.width() <= outer.width() + 1e-6
    assert area.height() <= outer.height() + 1e-6
    assert area.center().x() == pytest.approx(outer.center().x(), abs=1e-6)
    assert area.center().y() == pytest.approx(outer.center().y(), abs=1e-6)


def _painted(size=256):
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(QColor(BG_INPUT))
    painter = QPainter(image)
    mark.paint_mark(painter, QRectF(0, 0, size, size), MARK_INK, RAMP)
    painter.end()
    return image


def test_the_chi_is_ramped_and_the_delta_is_not(qt_app):
    """Sample both letters and check each carries the fill it is supposed to.

    The delta's apex and the chi's two ends, in the mark's own normalised
    coordinates, mapped through the same ``fitted`` box the painter used.
    """
    size = 256
    image = _painted(size)
    area = mark.fitted(QRectF(0, 0, size, size))

    def at(x, y):
        return QColor(image.pixel(
            int(area.x() + x * area.width()), int(area.y() + y * area.height())))

    # A point inside the delta's left leg: flat ink, and no ramp anywhere in it.
    upper = at(0.16, 0.62)
    lower = at(0.06, 0.72)
    assert upper.name() == QColor(MARK_INK).name()
    assert lower.name() == QColor(MARK_INK).name()

    # The chi's top-right arm and its bottom-right leg: the ramp's two ends.
    top = at(0.93, 0.32)
    bottom = at(0.95, 0.96)
    assert top.red() > 200 and top.green() < 120
    assert bottom.red() > 200 and bottom.green() > 170
    # And it really is a ramp rather than two flat halves.
    assert bottom.green() - top.green() > 80


def test_app_icon_offers_every_size_it_claims(qt_app):
    icon = mark.app_icon(MARK_INK, RAMP, BG_INPUT)
    available = {s.width() for s in icon.availableSizes()}
    assert {16, 32, 64, 128, 256, 512, 1024} <= available
    # Not empty tiles: the ground is painted, so the centre is never transparent.
    for size in (16, 512):
        pixmap = icon.pixmap(size, size)
        assert not pixmap.isNull()
        centre = QColor(pixmap.toImage().pixel(size // 2, size // 2))
        assert centre.alpha() == 255


def test_a_macos_bundle_keeps_its_own_icon(monkeypatch):
    """Setting a window icon replaces the bundle's .icns while the application
    runs, and macOS drew the replacement soft. Only a packaged macOS
    build has an icon of its own; source runs and Windows still need the call."""
    from deltachrome_ui.app import bundle_supplies_icon

    for platform, frozen, expected in (
        ("darwin", True, True),
        ("darwin", False, False),
        ("win32", True, False),
    ):
        monkeypatch.setattr("sys.platform", platform)
        monkeypatch.setattr("sys.frozen", frozen, raising=False)
        assert bundle_supplies_icon() is expected
