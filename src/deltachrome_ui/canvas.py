"""The placement canvas: four corners, dragged onto a chart.

This is the half of an existing feature that was deferred. Placement is not
automated yet (roadmap phase 4), so today the corners are supplied by hand -- and
by hand, through a CLI, means typing four pixel coordinates for a chart in an
11600 x 8700 master. There is no way to make that tolerable in a terminal.

Two decisions shape everything here.

**Scene coordinates are full-resolution image pixels.** Not proxy pixels, not
normalized, at any zoom. What a handle sits on *is* the number
:func:`deltachrome.pipeline.analyze` wants for ``corners``. There is no scaling
step anywhere to get backwards, which is the kind of bug that produces a plausible
wrong answer rather than a crash -- the failure mode this project is built to
avoid.

**Corners are seeded state, never typed input.** :meth:`PlacementCanvas.set_corners`
does not care where the four points came from: a default rectangle, a previous
run's saved record, or -- when phase 4 arrives -- a detector. Automation then
changes what seeds the handles and nothing about this widget. Building four
coordinate boxes instead would have meant throwing the screen away.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from deltachrome_ui.imageproxy import ImageProxy

#: The order the engine expects, and the order the CLI documents. Displayed on the
#: handles themselves because getting it wrong is the most common placement error
#: -- and the resulting homography is wrong without being obviously wrong.
#:
#: These are the **chart's** corners, not the image's. Target geometry is relative
#: to the fiducial quadrilateral with its own origin at top-left, so a chart
#: photographed rotated needs no special handling: TL simply sits wherever the
#: chart's top-left landed in the frame, and the homography absorbs the rest.
CORNER_LABELS = ("TL", "TR", "BR", "BL")

#: Fraction of the frame left outside the seeded rectangle. A chart shot to fill
#: the frame sits roughly here, so the handles start near the work rather than
#: stacked in a corner.
SEED_MARGIN = 0.08

_ACCENT = QColor(255, 214, 0)
_ACCENT_DIM = QColor(255, 214, 0, 120)
_WARN = QColor(255, 96, 80)
_OK = QColor(110, 230, 130, 200)
#: Slanted-edge ROIs. A different color because they are a different kind of
#: region, read a different way: a patch ROI is averaged and an edge ROI is a
#: transition profile, and one drawn like the other invites the question "why
#: is that patch so long and thin".
_EDGE = QColor(140, 185, 255, 210)
#: The other charts in the frame, while one of them is being checked. Spent
#: rather than dim accent: a ghost is context and must not read as a second
#: thing that could be dragged, and the accent is what says "this one".
_GHOST = QColor(84, 84, 94, 230)


class CornerHandle(QGraphicsObject):
    """One draggable corner, drawn at a constant size on screen."""

    moved = Signal(int, QPointF)
    grabbed = Signal(int)

    RADIUS = 11.0

    def __init__(self, index: int, bounds: QRectF, label: str | None = None) -> None:
        super().__init__()
        self.index = index
        self.label = label or CORNER_LABELS[index]
        self._bounds = bounds
        self._hover = False
        self._ok = True
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
            # Constant screen size at every zoom. A handle that grew with the
            # image would cover the pixels it is being placed on.
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setZValue(20)

    def boundingRect(self) -> QRectF:
        r = self.RADIUS + 2
        return QRectF(-r, -r - 18, 2 * r + 34, 2 * r + 20)

    def set_ok(self, ok: bool) -> None:
        if ok != self._ok:
            self._ok = ok
            self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = _ACCENT if self._ok else _WARN
        r = self.RADIUS

        # A ring with a gap at the center, so the pixel being placed stays visible.
        painter.setPen(QPen(QColor(0, 0, 0, 180), 3.5))
        painter.drawEllipse(QPointF(0, 0), r, r)
        painter.setPen(QPen(color, 1.8 if not self._hover else 2.6))
        painter.drawEllipse(QPointF(0, 0), r, r)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            painter.drawLine(
                QPointF(dx * 3.0, dy * 3.0), QPointF(dx * (r - 2), dy * (r - 2))
            )

        font = QFont()
        font.setPointSizeF(9.5)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
        painter.drawText(QPointF(r + 3, -r + 2), self.label)
        painter.setPen(QPen(color, 1))
        painter.drawText(QPointF(r + 3, -r + 2), self.label)

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.grabbed.emit(self.index)
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = QPointF(value)
            # Held inside the frame. A chart photographed to the edge has its
            # corner on the last pixel, so the bound is inclusive.
            point.setX(min(max(point.x(), self._bounds.left()), self._bounds.right()))
            point.setY(min(max(point.y(), self._bounds.top()), self._bounds.bottom()))
            return point
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self.moved.emit(self.index, self.scenePos())
        return super().itemChange(change, value)


class Loupe(QWidget):
    """A 1:1 window on the pixels under the active handle.

    The overview cannot resolve a chart edge -- at stride 5 a corner is a smear --
    and zooming the whole view to 1:1 to place one corner, then back out to reach
    the next, is four round trips per capture. The loupe is the cheaper answer:
    it reads a small region at full resolution regardless of the canvas zoom.
    """

    SIZE = 168
    HALF_WIDTH = 52

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._pixmap: QPixmap | None = None
        self._point: tuple[float, float] | None = None
        self.hide()

    def show_at(self, proxy: ImageProxy, x: float, y: float) -> None:
        tile = proxy.loupe(x, y, half_width=self.HALF_WIDTH)
        self._pixmap = QPixmap.fromImage(tile.image)
        self._point = (x - tile.region[0], y - tile.region[1])
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        if self._pixmap is None or self._point is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        inner = QRectF(2, 2, self.SIZE - 4, self.SIZE - 4)
        scale = inner.width() / max(1, self._pixmap.width())
        painter.fillRect(self.rect(), QColor(12, 12, 14))
        painter.drawPixmap(inner, self._pixmap, QRectF(self._pixmap.rect()))

        cx = inner.left() + self._point[0] * scale
        cy = inner.top() + self._point[1] * scale
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.drawLine(QPointF(cx, inner.top()), QPointF(cx, inner.bottom()))
        painter.drawLine(QPointF(inner.left(), cy), QPointF(inner.right(), cy))
        painter.setPen(QPen(_ACCENT, 1))
        painter.drawLine(QPointF(cx, inner.top()), QPointF(cx, inner.bottom()))
        painter.drawLine(QPointF(inner.left(), cy), QPointF(inner.right(), cy))
        painter.setPen(QPen(QColor(70, 70, 76), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class PlacementCanvas(QGraphicsView):
    """Pan/zoom view of a master with four corner handles on it."""

    cornersChanged = Signal(list)
    #: A handle was dragged by the operator. Distinct from ``cornersChanged``,
    #: which also fires when a placement arrives from a detector, a saved run or
    #: a refinement -- and the difference is load-bearing: a placement stops
    #: being automatic when a *person* moves a corner, not when the corners
    #: move. Deriving that from the order of two calls, which is what listening
    #: to ``cornersChanged`` amounts to, is a rule nothing states and nothing
    #: checks.
    cornersDragged = Signal()
    #: ``("ok" | "caution" | "error", explanation)``. Two severities, because two
    #: genuinely different things can be wrong -- see :meth:`_check_validity`.
    validityChanged = Signal(str, str)
    #: A rectangle the operator dragged round a chart, in image pixels as
    #: ``(x0, y0, x1, y1)``. Not a placement and never treated as one -- it says
    #: where to look. See :meth:`begin_region_search`.
    regionDrawn = Signal(tuple)
    #: The region gesture ended without a region: a click, or a drag too small to
    #: be a rectangle. Said out loud so the panel can put its own state back
    #: rather than being left armed with nothing having happened.
    regionAbandoned = Signal()

    #: Zoom per wheel notch.
    #: Zoom per classic wheel notch. Was 1.25, which is a sensible figure for a
    #: wheel that clicks and a wrong one for anything Apple sells: a Magic Mouse
    #: or a trackpad reports a *high-resolution* delta many times a second, so
    #: the exponent arrives in a stream of fractions and 1.25 compounds to
    #: several times magnification in a single flick.
    ZOOM_STEP = 1.12
    #: Delay before a pan or zoom triggers a detail read, so dragging does not
    #: queue a tile per mouse event.
    DETAIL_DELAY_MS = 90

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.proxy: ImageProxy | None = None
        self._handles: list[CornerHandle] = []
        self._quad: QGraphicsPolygonItem | None = None
        self._overview_item: QGraphicsPixmapItem | None = None
        #: The target's own corner ids, when a target has been chosen. Most
        #: declare TL/TR/BR/BL, but they are the chart's names for its corners
        #: and the definition is what says so.
        self._labels: tuple[str, ...] = CORNER_LABELS
        self._detail_item: QGraphicsPixmapItem | None = None
        self._roi_items: list[QGraphicsPolygonItem] = []
        #: The frame's other placements, drawn behind the live one.
        self._ghost_items: list[QGraphicsItem] = []
        self._severity = "ok"
        #: While set, the view re-fits on every resize. Cleared the moment the
        #: user zooms, so their framing is never yanked out from under them.
        #: This exists because ``set_proxy`` runs before the widget has been
        #: shown at its real size -- the loading screen is still in front of it --
        #: so the fit computed there is against a viewport that no longer applies.
        self._auto_fit = True
        #: Raised while corners are being placed programmatically. The loupe is a
        #: response to a hand on the mouse, not to the corners changing -- seeding
        #: a placement from a saved run should not pop a magnifier over an image
        #: nobody has touched yet.
        self._placing = False
        #: While set, a drag draws a search region instead of panning. Qt draws
        #: the rectangle itself in this mode; what is kept here is the last
        #: non-empty one, because the signal's final emission is always empty.
        self._region_rect = None

        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(24, 24, 27)))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self.rubberBandChanged.connect(self._on_rubber_band)

        self.loupe = Loupe(self)
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(self.DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._refresh_detail)

    # -- setup -------------------------------------------------------------

    def set_proxy(self, proxy: ImageProxy) -> None:
        self.proxy = proxy
        scene = self.scene()
        scene.clear()
        self._handles.clear()
        self._detail_item = None
        self._roi_items.clear()
        # Cleared rather than removed: ``scene.clear()`` has already deleted them.
        self._ghost_items.clear()

        bounds = QRectF(0, 0, proxy.width, proxy.height)
        scene.setSceneRect(bounds)

        self._overview_item = scene.addPixmap(QPixmap.fromImage(proxy.overview))
        # Scaled so that one overview pixel spans `overview_stride` scene pixels,
        # which is what keeps scene coordinates equal to image coordinates.
        self._overview_item.setScale(proxy.overview_stride)
        self._overview_item.setZValue(-20)
        self._overview_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

        self._quad = QGraphicsPolygonItem()
        self._quad.setPen(QPen(_ACCENT_DIM, 0))  # cosmetic: constant width on screen
        self._quad.setZValue(10)
        scene.addItem(self._quad)

        for index in range(4):
            handle = CornerHandle(index, bounds, self._label_for(index))
            handle.moved.connect(self._on_handle_moved)
            handle.grabbed.connect(self._on_handle_grabbed)
            scene.addItem(handle)
            self._handles.append(handle)

        self.seed_default()
        self.fit()  # re-applied on the first real resize, see _auto_fit

    def show_sample_regions(self, polygons, flagged=(), edges=()) -> None:
        """Draw every region this run would read, of both kinds.

        With provisional geometry this is not a nicety. It is the only way to
        confirm the ROIs sit where they are meant to before believing a single
        number -- a placement can be plausible, produce a clean homography, and
        still have every ROI straddling two patches.

        **Edges are drawn for the same reason and are not the same region.** A
        patch ROI is an area averaged to one color; an edge ROI is a strip
        straddling a transition, and what has to be true of it is different --
        it must hold the whole transition plus flat ground on both sides, and it
        must not have caught a neighbouring printed feature. Both failures are
        obvious in a picture and invisible in a number, and on a chart that
        measures only sharpness the edges are the whole of what there is to
        check. So they are drawn in their own color rather than folded in.

        ``edges`` is ``(edge_id, (x0, y0, x1, y1), flagged)``: an axis-aligned
        box, because that is what was actually cut from the frame. The edge
        inside it is slanted, which is the point of it.
        """
        self.clear_sample_regions()
        flagged_ids = set(flagged)
        for patch_id, polygon in polygons:
            item = QGraphicsPolygonItem(
                QPolygonF([QPointF(float(x), float(y)) for x, y in polygon])
            )
            color = _WARN if patch_id in flagged_ids else _OK
            item.setPen(QPen(color, 0))
            item.setZValue(5)
            self.scene().addItem(item)
            self._roi_items.append(item)
        for _edge_id, (x0, y0, x1, y1), is_flagged in edges:
            item = QGraphicsRectItem(
                QRectF(QPointF(float(x0), float(y0)), QPointF(float(x1), float(y1)))
            )
            item.setPen(QPen(_WARN if is_flagged else _EDGE, 0))
            item.setZValue(6)
            self.scene().addItem(item)
            self._roi_items.append(item)

    #: Radius of a ghost's ordinal badge, in screen pixels: these items ignore
    #: the view transform, so the marker stays the same size at every zoom.
    GHOST_BADGE = 11

    def show_other_placements(self, placements) -> None:
        """Every *other* chart in this frame, behind the one being checked.

        A board of four cards used to show one quadrilateral at a time, so
        ``Target 3 of 4`` named a chart the operator could not locate -- the
        ordinal is the detection's order and nothing about the frame. With the
        others drawn, stepping is a move around a capture.

        They carry no verdict tone, and that is not an oversight: nothing has
        been measured yet on this screen. The only thing a ghost asserts is that
        a placement exists there and which number it answers to.

        ``placements`` is ``(ordinal, corners)``. The live chart is the caller's
        to leave out; this draws exactly what it is given.
        """
        self.clear_other_placements()
        scene = self.scene()
        if scene is None:
            return
        for ordinal, corners in placements:
            points = [QPointF(float(x), float(y)) for x, y in (corners or [])]
            if len(points) < 3:
                continue
            quad = QGraphicsPolygonItem(QPolygonF(points))
            pen = QPen(_GHOST, 0)  # cosmetic: constant width on screen
            pen.setStyle(Qt.PenStyle.DashLine)
            quad.setPen(pen)
            # Under the live quad, its handles and the sampled regions, all of
            # which are about the chart the operator is actually checking.
            quad.setZValue(4)
            scene.addItem(quad)
            self._ghost_items.append(quad)
            self._ghost_items.extend(self._ghost_badge(scene, points[0], ordinal))

    def _ghost_badge(self, scene, corner: QPointF, ordinal: int) -> list[QGraphicsItem]:
        """The chart's number, outside its own quadrilateral.

        Outside rather than in: a badge inside the quad sits on the chart's own
        printed corner, which on several targets is the fiducial the placement
        was made on and the one part of it worth looking at.
        """
        radius = self.GHOST_BADGE
        # Up and to the left of the chart's top-left corner, in screen pixels --
        # the item ignores the view transform, so this offset is constant at
        # every zoom instead of drifting off the chart as the capture grows.
        centre = -radius - 3
        disc = QGraphicsEllipseItem(
            centre - radius, centre - radius, radius * 2, radius * 2)
        disc.setBrush(QBrush(QColor(26, 26, 29, 235)))
        disc.setPen(QPen(_GHOST, 1))
        disc.setPos(corner)
        disc.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        disc.setZValue(4)
        scene.addItem(disc)

        text = QGraphicsSimpleTextItem(str(ordinal), disc)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        text.setFont(font)
        text.setBrush(QBrush(_GHOST))
        box = text.boundingRect()
        text.setPos(centre - box.width() / 2, centre - box.height() / 2)
        # The text is a child of the disc, so it is removed with it.
        return [disc]

    def clear_other_placements(self) -> None:
        scene = self.scene()
        for item in self._ghost_items:
            if scene is not None and item.scene() is scene:
                scene.removeItem(item)
        self._ghost_items.clear()

    def clear_sample_regions(self) -> None:
        for item in self._roi_items:
            self.scene().removeItem(item)
        self._roi_items.clear()

    @property
    def showing_sample_regions(self) -> bool:
        return bool(self._roi_items)

    def _label_for(self, index: int) -> str:
        return self._labels[index] if index < len(self._labels) else CORNER_LABELS[index]

    def set_corner_labels(self, labels) -> None:
        """Name the handles from the target definition.

        Called before :meth:`set_proxy` so the handles are built with them; called
        after, it relabels in place.
        """
        cleaned = tuple(str(label) for label in labels if str(label).strip())
        self._labels = cleaned if len(cleaned) == 4 else CORNER_LABELS
        for index, handle in enumerate(self._handles):
            handle.label = self._label_for(index)
            handle.update()

    # -- corners -----------------------------------------------------------

    def seed_default(self) -> None:
        """An inset rectangle, as a starting position rather than a guess.

        Deliberately not called a detection. It is where a chart shot to fill the
        frame roughly sits, and it exists so the handles start near the work.
        """
        if self.proxy is None:
            return
        mx = self.proxy.width * SEED_MARGIN
        my = self.proxy.height * SEED_MARGIN
        x0, y0 = mx, my
        x1, y1 = self.proxy.width - mx, self.proxy.height - my
        self.set_corners([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def set_corners(self, corners) -> None:
        """Place the handles. Source-agnostic -- see the module docstring."""
        if self.proxy is None or len(corners) != 4:
            return
        self._placing = True
        try:
            for handle, (x, y) in zip(self._handles, corners):
                handle.setPos(QPointF(*self.proxy.clamp(float(x), float(y))))
        finally:
            self._placing = False
        self._update_quad()
        self.cornersChanged.emit(self.corners())

    def corners(self) -> list[tuple[float, float]]:
        """The four points in image pixels, in TL TR BR BL order."""
        return [(h.scenePos().x(), h.scenePos().y()) for h in self._handles]

    def corners_text(self) -> str:
        """The same points as the CLI's ``--corners`` argument.

        Worth having: it makes a placement made in the GUI reproducible on the
        command line, and a run scripted from the CLI reviewable in the GUI.
        """
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in self.corners())

    # -- validity ----------------------------------------------------------

    def _check_validity(self) -> None:
        """Flag a bad quadrilateral while it is being made.

        The engine warns after the fact, when the placed aspect ratio disagrees
        with the chart's real proportions. That catches a lot, but not a
        correctly-proportioned quad whose corners are in the wrong order -- and
        the homography from one of those is wrong without looking wrong, which is
        precisely the failure this project treats as the dangerous one. Ordering
        is cheap to check here, while the user still has a hand on the mouse.

        Two severities, because two different things can be wrong:

        * **Crossed** is always an error. No placement of a real chart produces a
          self-intersecting quadrilateral.
        * **Anticlockwise** is a caution, not an error. Corners are given in the
          chart's own order, and rotation preserves winding -- so a chart
          photographed upside down still traverses clockwise in the frame. What
          flips the winding is a *mirrored* capture, which is a real thing for
          film shot through the base. So this says what it sees and leaves the
          judgement to someone who knows which way round the original was.
        """
        points = np.array(self.corners(), dtype=float)
        crosses = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            u, v = b - a, c - b
            # Written out rather than via np.cross: NumPy 2 removed the 2-D form,
            # and the z component of a planar cross product is this scalar.
            crosses.append(float(u[0] * v[1] - u[1] * v[0]))
        signs = {c > 0 for c in crosses if abs(c) > 1e-9}

        if len(signs) > 1:
            severity = "error"
            reason = (
                "These four points cross over. No placement of a real chart does "
                "that — the corners must go round it in the order TL, TR, BR, BL."
            )
        elif crosses and crosses[0] < 0:
            # Image coordinates put y downwards, which flips the usual intuition:
            # TL(0,0) TR(1,0) BR(1,1) BL(0,1) traverses clockwise on screen and
            # gives a *positive* z, so the correct order is the positive one and
            # a negative z is the mirrored traversal.
            severity = "caution"
            reason = (
                "These corners wind anticlockwise, which means the capture is "
                "mirrored — normal for film shot through the base, wrong if it "
                "isn't. Remember TL is the chart's top-left, not the image's: a "
                "rotated chart is fine and needs no special handling."
            )
        else:
            severity, reason = "ok", ""

        self._severity = severity
        for handle in self._handles:
            handle.set_ok(severity != "error")
        self._quad.setPen(QPen(_ACCENT_DIM if severity == "ok" else _WARN, 0))
        # Emitted every time rather than only on change. A listener that connects
        # after a placement has settled would otherwise never learn the state, and
        # the cost of a redundant signal on four handles is nothing.
        self.validityChanged.emit(severity, reason)

    def _update_quad(self) -> None:
        if self._quad is None:
            return
        self._quad.setPolygon(QPolygonF([QPointF(x, y) for x, y in self.corners()]))
        self._check_validity()

    def _on_handle_moved(self, index: int, point: QPointF) -> None:
        self._update_quad()
        if self.proxy is not None and not self._placing:
            self._position_loupe()
            self.loupe.show_at(self.proxy, point.x(), point.y())
            self.cornersDragged.emit()
        self.cornersChanged.emit(self.corners())

    def _on_handle_grabbed(self, index: int) -> None:
        if self.proxy is None:
            return
        point = self._handles[index].scenePos()
        self._position_loupe()
        self.loupe.show_at(self.proxy, point.x(), point.y())

    def _position_loupe(self) -> None:
        margin = 12
        self.loupe.move(
            self.viewport().width() - self.loupe.width() - margin, margin
        )

    # -- zoom, pan and detail ---------------------------------------------

    @property
    def severity(self) -> str:
        """``"ok"``, ``"caution"`` or ``"error"`` for the current placement."""
        return self._severity

    @property
    def zoom(self) -> float:
        """Screen pixels per image pixel."""
        return float(self.transform().m11())

    def fit(self) -> None:
        if self.proxy is None:
            return
        self._auto_fit = True
        #: Raised while corners are being placed programmatically. The loupe is a
        #: response to a hand on the mouse, not to the corners changing -- seeding
        #: a placement from a saved run should not pop a magnifier over an image
        #: nobody has touched yet.
        self._placing = False
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._detail_timer.start()

    def zoom_to(self, factor: float) -> None:
        if self.proxy is None:
            return
        self._auto_fit = False
        self.setTransform(self.transform().fromScale(factor, factor))
        self._detail_timer.start()

    def set_placement_visible(self, visible: bool) -> None:
        """Show or hide the four handles and the quad they describe.

        **A clean canvas is the point of hiding them.** Manual placement is meant
        to be the last resort: the chart is found automatically, and where it is
        not the operator draws a region for the search to work in. Four handles
        sitting on a default rectangle in the middle of the frame are an invitation
        to drag them, which is the thing to reach for last -- and they are also a
        placement of nothing, on a capture where nothing has been named.
        """
        visible = bool(visible)
        for handle in self._handles:
            handle.setVisible(visible)
        if self._quad is not None:
            self._quad.setVisible(visible)
        if not visible:
            self.loupe.hide()

    @property
    def placement_visible(self) -> bool:
        return bool(self._handles) and self._handles[0].isVisible()

    # -- drawing a search region ------------------------------------------

    #: Smallest drag, as a fraction of the frame's shorter side, taken as a
    #: rectangle rather than a slipped click. A chart small enough to fall under
    #: this could not be placed anyway: every finder's size gates are fractions
    #: of the region, and the smallest chart in any capture here -- one QA-62 card
    #: of four on a board -- is 11% of its frame's short side.
    MIN_REGION_FRACTION = 0.02

    @property
    def searching_region(self) -> bool:
        """Whether the next drag draws a rectangle instead of panning."""
        return self.dragMode() == QGraphicsView.DragMode.RubberBandDrag

    def begin_region_search(self) -> None:
        """Arm the next drag to draw a rectangle round a chart.

        Qt's own rubber band does the drawing, which is the whole reason this is
        a handful of lines: it rubber-bands in view coordinates, tracks the drag,
        and reports the rectangle in scene coordinates -- and a scene coordinate
        here *is* an image pixel, which is the invariant the rest of this class
        is built on.

        The mode is one gesture long. A canvas that stayed in it would have taken
        panning away until somebody found the control again, and panning is how
        the operator gets to the corner they are about to check.
        """
        if self.proxy is None:
            return
        self._region_rect = None
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_region_search(self) -> None:
        self._region_rect = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.unsetCursor()

    def _on_rubber_band(self, rect, from_point, to_point) -> None:
        """Qt reports the band as it is dragged, and empty when it is released.

        So the last non-empty rectangle is the answer, and the empty one says the
        gesture is over. There is no separate release event to listen to in
        rubber-band mode -- the view consumes the mouse itself.

        **Connected, not overridden.** ``rubberBandChanged`` is a *signal* on
        ``QGraphicsView``, so a method of that name on a subclass replaces it
        rather than extending it: Qt emits from C++ into a signal that no longer
        exists, the handler never runs, and the only thing that ever calls it is
        a test calling it directly. Which is exactly what happened, and is why
        the tests below drive the signal rather than the method.

        The rectangle arrives in *viewport* coordinates and is mapped to the
        scene, where a coordinate is an image pixel.
        """
        del from_point, to_point  # the band's own rectangle is the whole answer
        if not self.searching_region:
            return
        if not rect.isNull():
            self._region_rect = self.mapToScene(rect).boundingRect()
            return

        band, self._region_rect = self._region_rect, None
        self.cancel_region_search()
        if band is None or self.proxy is None:
            self.regionAbandoned.emit()
            return
        floor = self.MIN_REGION_FRACTION * min(self.proxy.width, self.proxy.height)
        if band.width() < floor or band.height() < floor:
            self.regionAbandoned.emit()
            return
        x0, y0 = self.proxy.clamp(band.left(), band.top())
        x1, y1 = self.proxy.clamp(band.right(), band.bottom())
        self.regionDrawn.emit((float(x0), float(y0), float(x1), float(y1)))

    # -- zoom, pan and detail ---------------------------------------------

    def wheelEvent(self, event) -> None:
        """Vertical zooms, horizontal pans, and the axis that moved most decides.

        **This canvas is an inspection surface, not a document.** The gesture it
        is asked for most is "look closer at that corner", and wheel-vertical-to-
        zoom is what the software it resembles does -- CAD, 3D, web maps, and the
        image views in Lightroom and Capture One. Document viewers scroll on a
        wheel and take a modifier to zoom, which is right for reading a page and
        wrong for reading a chart.

        **The axis decides which of the two the gesture was, never both.** A swipe
        is never exactly straight, so reading both deltas as both meanings would
        zoom *and* pan from one flick and do neither by the amount asked for. The
        larger delta settles it. Where it settles on zoom the sideways component is
        dropped; where it settles on pan the whole event goes to Qt, because a pan
        that also drifts a little vertically is still just a pan.

        Panning up and down is by dragging the canvas, which is what
        ``ScrollHandDrag`` has always made the empty parts of it do, and by the
        arrow keys. That is the trade this makes: the zoom gesture is the frequent
        one and it is the one that gets the wheel.

        The report's capture panel keeps wheel-to-zoom on both axes. It is a
        painted widget with no scroll bars, a drag that pans and a clamped zoom,
        so it has no horizontal scroll for a sideways swipe to reach.
        """
        if self.proxy is None:
            return
        delta = event.angleDelta()
        if abs(delta.y()) >= abs(delta.x()):
            notches = delta.y() / 120.0
            if notches:
                self._zoom_by(self.ZOOM_STEP**notches)
            event.accept()
            return
        # Sideways. Qt's own handling scrolls the bar that exists.
        self._auto_fit = False
        super().wheelEvent(event)

    def _zoom_by(self, factor: float) -> None:
        """Magnify about the pointer, which ``AnchorUnderMouse`` is set for."""
        if self.proxy is None or factor <= 0:
            return
        self._auto_fit = False
        self.scale(factor, factor)
        self._detail_timer.start()

    def event(self, event) -> bool:
        """Pinch to zoom, where the hardware has a pinch.

        A trackpad reports one as a native gesture rather than as a wheel, and its
        ``value`` is already the fractional change for this step -- so it composes
        over a gesture without a rate of its own to choose, which is why nothing
        here scales it.
        """
        if event.type() == QEvent.Type.NativeGesture and (
            event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
        ):
            self._zoom_by(1.0 + event.value())
            return True
        return super().event(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit and self.proxy is not None:
            self.fitInView(
                self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
        self._position_loupe()
        self._detail_timer.start()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._detail_timer.start()

    def _refresh_detail(self) -> None:
        """Read the visible region at whatever stride the current zoom resolves.

        Skipped entirely while the overview is already finer than the screen,
        which is most of the time -- so panning around at fit-to-window costs
        nothing, and only a genuine zoom-in touches the master.
        """
        if self.proxy is None or self._overview_item is None:
            return
        zoom = self.zoom
        if zoom * self.proxy.overview_stride <= 1.0:
            self._clear_detail()
            return

        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        region = (
            int(np.floor(visible.left())),
            int(np.floor(visible.top())),
            int(np.ceil(visible.right())),
            int(np.ceil(visible.bottom())),
        )
        target_px = max(self.viewport().width(), 1)
        stride = self.proxy.stride_for(region, target_px)
        if stride >= self.proxy.overview_stride:
            self._clear_detail()
            return

        tile = self.proxy.tile(region, stride=stride)
        if self._detail_item is None:
            self._detail_item = self.scene().addPixmap(QPixmap.fromImage(tile.image))
            self._detail_item.setZValue(-10)
        else:
            self._detail_item.setPixmap(QPixmap.fromImage(tile.image))
        self._detail_item.setScale(tile.stride)
        self._detail_item.setPos(QPointF(tile.region[0], tile.region[1]))
        self._detail_item.setTransformationMode(
            # Past 1:1 the honest thing is to show the pixels, not to interpolate
            # a smoother edge than the file contains.
            Qt.TransformationMode.FastTransformation
            if zoom >= 1.0
            else Qt.TransformationMode.SmoothTransformation
        )
        self._detail_item.setVisible(True)

    def _clear_detail(self) -> None:
        if self._detail_item is not None:
            self._detail_item.setVisible(False)
