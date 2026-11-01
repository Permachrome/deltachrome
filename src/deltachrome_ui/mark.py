"""The Deltachrome mark, as geometry rather than as an asset.

Painted, for the same reason :func:`~deltachrome_ui.widgets.draw_glyph` paints
the interface's icons: an asset directory beside the executable is a second
thing to install and a second thing to get out of step. Here there is a further
reason -- the mark is a *pair* of shapes with different fills, one flat and one
ramped, and a single PNG could not follow the ink colour of whatever it is
sitting on.

**The coordinates are traced, not redrawn.** They come from the two text layers
of ``deltachrome.psd`` via their layer masks: the delta is a four-point outline
with a three-point counter, the chi is twelve points, and the polygons here
differ from those masks by 0.7% of the ink -- which is the antialiased boundary
and nothing else. Both letters are Helvetica Neue Bold, the delta carrying a
117.6% horizontal and 103.8% vertical layer transform, and that transform is
baked into these numbers rather than applied at paint time.

Everything is a fraction of the mark's own bounding box, so the same code serves
a 26 px header and a 1024 px application icon. :data:`ASPECT` is the box's
width over its height; a caller that ignores it gets a stretched mark, so
:func:`paint_mark` fits itself into whatever rectangle it is given rather than
trusting one.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
)

#: Width over height of the mark's bounding box. The delta's own stretch is in
#: here, which is why it is not the ratio of two round numbers.
ASPECT = 1.0953

#: The application icon's rounded square, as a fraction of its canvas. macOS
#: draws every icon on one grid -- 824 px of a 1024 px canvas for a square
#: shape -- and the dock sizes the canvas, not the artwork, so a tile filling
#: its own bitmap sits visibly larger than everything around it.
TILE_FRACTION = 824 / 1024

#: The delta: an outline and its counter, wound so an even-odd fill leaves the
#: middle empty. Never filled with the ramp -- see the module docstring for
#: :mod:`deltachrome_ui.theme`.
_DELTA_OUTER = [
    (0.29149, 0.00000), (0.00000, 0.74770),
    (0.75269, 0.74794), (0.46600, 0.00016),
]
_DELTA_COUNTER = [
    (0.37778, 0.18420), (0.52751, 0.60931), (0.22238, 0.60979),
]

#: The chi. One contour, and it is painted *after* the delta because in the
#: artwork it is the upper layer: where its top-left arm crosses the delta's
#: right leg the ramp runs unbroken over the white. That crossing is what makes
#: the two letters a monogram instead of two letters standing next to each
#: other, so the order is not a detail.
_CHI = [
    (0.98112, 0.29488), (0.83641, 0.29504), (0.75439, 0.49645),
    (0.67008, 0.29504), (0.52109, 0.29528), (0.68484, 0.62248),
    (0.50561, 1.00000), (0.65260, 0.99976), (0.74952, 0.74794),
    (0.85079, 0.99952), (1.00000, 0.99976), (0.81959, 0.61739),
]

#: Where the ramp starts and ends down the mark's box. The chi's own top and
#: bottom -- the gradient is the letter's, not the bounding box's, so a mark
#: drawn beside a taller sibling still ramps over the same ink.
_RAMP_TOP = 0.29488
_RAMP_BOTTOM = 1.00000


def _polygon(points: list[tuple[float, float]], box: QRectF) -> QPainterPath:
    path = QPainterPath()
    for i, (x, y) in enumerate(points):
        p = QPointF(box.x() + x * box.width(), box.y() + y * box.height())
        path.moveTo(p) if i == 0 else path.lineTo(p)
    path.closeSubpath()
    return path


def fitted(box: QRectF) -> QRectF:
    """The largest correctly-proportioned mark that fits inside ``box``.

    A caller passing a square gets a mark centred in it rather than a stretched
    one. The delta already carries a deliberate horizontal stretch and a second
    accidental one would read as a different logo.
    """
    if box.width() / max(box.height(), 1e-6) > ASPECT:
        h = box.height()
        w = h * ASPECT
    else:
        w = box.width()
        h = w / ASPECT
    return QRectF(box.x() + (box.width() - w) / 2,
                  box.y() + (box.height() - h) / 2, w, h)


def paint_mark(painter: QPainter, box: QRectF, ink: str,
               ramp: tuple[str, str]) -> None:
    """Draw the mark into ``box``: the delta in ``ink``, the chi in ``ramp``.

    ``ink`` follows the surface -- it is the text colour on a dark panel and the
    ground colour on a light one -- which is the whole reason the delta is not
    part of the gradient. ``ramp`` is the product's own two stops, so a sibling
    product changes one argument and nothing else.
    """
    area = fitted(box)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)

    delta = _polygon(_DELTA_OUTER, area)
    delta.setFillRule(Qt.FillRule.OddEvenFill)
    delta.addPath(_polygon(_DELTA_COUNTER, area))
    painter.fillPath(delta, QColor(ink))

    gradient = QLinearGradient(
        area.x(), area.y() + _RAMP_TOP * area.height(),
        area.x(), area.y() + _RAMP_BOTTOM * area.height(),
    )
    gradient.setColorAt(0.0, QColor(ramp[0]))
    gradient.setColorAt(1.0, QColor(ramp[1]))
    painter.fillPath(_polygon(_CHI, area), QBrush(gradient))
    painter.restore()


def app_icon(ink: str, ramp: tuple[str, str], ground: str) -> QIcon:
    """The application icon: the mark on the house ground, in a rounded square.

    On the ground rather than knocked out of the ramp, because flattening both
    letters into one colour throws away the distinction the mark is built on --
    the delta is the difference and the chi is the colour, and only one of them
    is supposed to carry a gradient.

    Several sizes, because the platform picks: macOS wants 1024 for the Finder
    and 32 for the dock badge, and letting Qt downsample one bitmap to 16 px
    closes the delta's counter into a grey triangle.

    **The tile does not fill its canvas**, and that is the whole of
    :data:`TILE_FRACTION`. Every macOS application icon is drawn on one grid --
    a rounded square 824 px wide on a 1024 px canvas, the rest transparent
    margin -- and the dock lays out the *canvas* rather than the artwork. An
    icon painted edge to edge is therefore not a bolder icon, it is a bigger
    one, standing about a fifth larger than every neighbour beside it. Which is
    what it did.
    """
    icon = QIcon()
    for size in (16, 32, 64, 128, 256, 512, 1024):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ground))
        tile = size * TILE_FRACTION
        margin = (size - tile) / 2
        # Apple's own corner radius is a little under a quarter of the tile.
        painter.drawRoundedRect(QRectF(margin, margin, tile, tile),
                                tile * 0.2246, tile * 0.2246)
        # Less margin as the tile gets smaller. A constant fraction is right
        # for the Finder's 512 and wrong for a 16 px menu bar, where 17% of the
        # tile is nearly three pixels and leaves the mark too small to read the
        # counter out of at all.
        inset = tile * (0.17 if size >= 64 else 0.10)
        paint_mark(painter, QRectF(margin + inset, margin + inset,
                                   tile - inset * 2, tile - inset * 2),
                   ink, ramp)
        painter.end()
        icon.addPixmap(pixmap)
    return icon
