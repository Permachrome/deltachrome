"""The camera profile screen: the chart's patches, which of them to use, and colprof.

Reached from the placement screen once a chart is placed, and a separate path
from QC rather than a mode of it: the capture usually carries no profile -- the
profile is what is being made -- so nothing here is colour-managed and nothing
here is judged.

**The panel says what was detected and does not explain it.** `Linear`, the
number of patches, how many the sampler flagged: facts about this capture, in
chips. What Held Out means, or why a shaper beat a matrix, belongs in the guide.

**A patch is clicked in or out on the capture itself**, which is the one control
``colprof`` has never had. Every deselection belongs to the profile it was made
for and is remembered nowhere else. Flagged patches start out excluded, and can
be clicked back in like any other: the operator's answer is the whole answer.

**Automatic is a default only where it is quick.** It builds every candidate
six times over -- once in full and once per held-out fold -- so it costs what
the slowest candidate does. Matrix and shaper builds are seconds, and on the SG
the whole of it is about fifteen. A chart with patches enough to try tables
(:data:`~deltachrome.profiling.automatic.TABLE_MIN_PATCHES`) adds twenty-four
table builds, three minutes on the LaserSoft, and that is not a wait to put
behind the button pressed by default. There the type opens on
:data:`LARGE_CHART_DEFAULT`, which builds in seconds and still reports its
held-out error, and *Auto Calculate* is the same search asked for. It leaves
the winner chosen in the menus, already built.

**Any change after a build puts the button back to Build Profile.** A saved
profile and the panel beside it must describe the same build, so Save is offered
only while nothing has changed since.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGraphicsPathItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsSimpleTextItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from deltachrome.errors import ProfilingError
from deltachrome.profiling.automatic import TABLE_MIN_PATCHES
from deltachrome.profiling.colprof import (
    PROFILE_TYPE_LABELS,
    QUALITIES,
    TABLE_TYPES,
    ColprofOptions,
    profile_type_label,
)
from deltachrome.profiling.patches import MIN_PATCHES
from deltachrome.profiling.scale import SCALE_FOR
from deltachrome_ui import paths
from deltachrome_ui.canvas import PlacementCanvas
from deltachrome_ui.charts import segment_style
from deltachrome_ui.imageproxy import ImageProxy
from deltachrome_ui.theme import ACCENT, BAR_HEIGHT, BORDER, TEXT, TEXT_DIM
from deltachrome_ui.widgets import (
    BusyStrip,
    ComboBox,
    ElidedLabel,
    IconButton,
    Pill,
    caveat,
    divider,
    elided_subtitle,
    figures_font,
    heading,
)

#: The panel is the placement screen's width, so moving between the two does not
#: move the capture.
PANEL_WIDTH = 380

AUTOMATIC = "automatic"

#: The type and quality a chart large enough to try tables opens on. A shaper
#: takes out the offset stray light leaves in the blacks, which a matrix cannot,
#: and extrapolates past the chart's gamut without a table's guesswork.
LARGE_CHART_DEFAULT = ("shaper", "medium")

WHITE_LABELS = {
    "absolute": "Absolute",
    "chart": "Chart White",
    "chart-scaled": "Chart White, Scaled",
}
SCALE_FOR_LABELS = {"white": "White", "ramp": "Full Ramp"}

_USED = QColor(110, 230, 130, 200)
_FLAGGED = QColor(255, 96, 80)
_SHADE = QColor(26, 26, 29, 200)
_CROSS = QColor(TEXT)
#: A patch printing a colour the chart repeats, while merging is on. Tinted,
#: outlined heavier and numbered by family: a hairline changing from green to
#: blue was the whole of it, and nobody could see it change.
_REPEAT = QColor(120, 190, 255, 255)

_HOVER = QColor(ACCENT)
_HOVER_FILL = QColor(ACCENT)
_HOVER_FILL.setAlpha(60)

#: A sampled region is the middle 55% of its patch, so a click is taken anywhere
#: on the patch by growing the region back out about its centre.
_HIT_GROWTH = 1.0 / 0.55

#: How far the pointer may move between press and release and still be a click
#: rather than a pan, in screen pixels.
_CLICK_SLOP = 4


def _family_badge(number: int) -> QGraphicsEllipseItem:
    """A repeat family's number, at a fixed screen size and centred on its patch."""
    size = 18.0
    badge = QGraphicsEllipseItem(-size / 2, -size / 2, size, size)
    badge.setPen(QPen(Qt.PenStyle.NoPen))
    badge.setBrush(QBrush(_REPEAT))
    badge.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
    badge.setZValue(7)
    label = QGraphicsSimpleTextItem(str(number), badge)
    font = QFont()
    font.setPixelSize(11)
    font.setBold(True)
    label.setFont(font)
    label.setBrush(QBrush(QColor(_SHADE.red(), _SHADE.green(), _SHADE.blue())))
    box = label.boundingRect()
    label.setPos(-box.width() / 2, -box.height() / 2)
    return badge


class PatchCanvas(PlacementCanvas):
    """The capture with every patch on it, each one clicked in or out.

    The placement canvas with its handles put away: the chart is already placed,
    and panning and zooming are wanted exactly as they were.
    """

    patchToggled = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._patch_items: dict[str, list] = {}
        self._hit: dict[str, QPolygonF] = {}
        self._shapes: dict[str, QPolygonF] = {}
        #: Patch id to every id of its repeat family, while merging is on.
        self._family: dict[str, tuple[str, ...]] = {}
        self._hover_items: list = []
        self._hovered: str | None = None
        self._press = None
        self.setMouseTracking(True)

    def set_proxy(self, proxy: ImageProxy) -> None:
        self._patch_items.clear()
        self._hit.clear()
        self._shapes.clear()
        self._hover_items = []
        self._hovered = None
        super().set_proxy(proxy)
        self.set_placement_visible(False)

    def swap_proxy(self, proxy: ImageProxy) -> None:
        """The same capture drawn another way, keeping the view where it is."""
        self.proxy = proxy
        self._overview_item.setPixmap(QPixmap.fromImage(proxy.overview))
        if self._detail_item is not None:
            self._detail_item.setVisible(False)
        self._detail_timer.start()

    def show_patches(self, regions: dict, excluded: set, flagged: set,
                     families: list[tuple[str, ...]] = ()) -> None:
        """Draw every patch: used outlined, left out shaded and crossed.

        ``families`` are the repeats merging would pool; their members are
        outlined in their own colour, and pointing at one lights all of them.
        """
        scene = self.scene()
        for items in self._patch_items.values():
            for item in items:
                scene.removeItem(item)
        self._patch_items.clear()
        self._hit.clear()
        self._shapes.clear()
        self._family = {pid: family for family in families for pid in family}
        numbers = {family: n for n, family in enumerate(families, start=1)}
        for patch_id, polygon in regions.items():
            points = [QPointF(float(x), float(y)) for x, y in polygon]
            if len(points) < 3:
                continue
            centre = np.mean(np.asarray(polygon, dtype=float), axis=0)
            self._hit[patch_id] = QPolygonF([
                QPointF(*(centre + (np.array([p.x(), p.y()]) - centre) * _HIT_GROWTH))
                for p in points
            ])
            shape = QPolygonF(points)
            self._shapes[patch_id] = shape
            outline = QGraphicsPolygonItem(shape)
            outline.setZValue(5)
            outline.setToolTip(patch_id)
            items = [outline]
            if patch_id in excluded:
                outline.setPen(QPen(Qt.PenStyle.NoPen))
                outline.setBrush(QBrush(_SHADE))
                path = QPainterPath()
                path.moveTo(points[0])
                path.lineTo(points[2])
                path.moveTo(points[1])
                path.lineTo(points[3 % len(points)])
                cross = QGraphicsPathItem(path)
                cross.setPen(QPen(_FLAGGED if patch_id in flagged else _CROSS, 0))
                cross.setZValue(6)
                cross.setToolTip(patch_id)
                items.append(cross)
            elif patch_id in self._family and patch_id not in flagged:
                outline.setPen(QPen(_REPEAT, 0))
                # The whole patch outlined, not only its sampled middle, so a
                # family reads as a set of patches from across the chart.
                frame = QGraphicsPolygonItem(self._hit[patch_id])
                pen = QPen(_REPEAT, 3)
                pen.setCosmetic(True)
                frame.setPen(pen)
                frame.setZValue(6)
                frame.setToolTip(patch_id)
                items.append(frame)
                # The family's number on a badge of its own colour, readable on
                # a white patch and a black one alike, so which patches pool
                # together is read off the chart rather than found by pointing.
                badge = _family_badge(numbers[self._family[patch_id]])
                badge.setPos(QPointF(float(centre[0]), float(centre[1])))
                badge.setToolTip(patch_id)
                items.append(badge)
            else:
                outline.setPen(QPen(_FLAGGED if patch_id in flagged else _USED, 0))
            for item in items:
                scene.addItem(item)
            self._patch_items[patch_id] = items
        hovered, self._hovered = self._hovered, None
        self._set_hover(hovered)

    def _set_hover(self, patch_id: str | None) -> None:
        """Light the patch under the pointer, and its repeats while merging."""
        if patch_id == self._hovered:
            return
        scene = self.scene()
        for item in self._hover_items:
            scene.removeItem(item)
        self._hover_items = []
        self._hovered = patch_id
        if patch_id is None or patch_id not in self._shapes:
            return
        for member in self._family.get(patch_id, (patch_id,)):
            shape = self._shapes.get(member)
            if shape is None:
                continue
            item = QGraphicsPolygonItem(shape)
            pen = QPen(_HOVER, 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(_HOVER_FILL))
            item.setZValue(7)
            item.setToolTip(member)
            scene.addItem(item)
            self._hover_items.append(item)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        if self._press is not None:
            return
        patch_id = self.patch_at(self.mapToScene(event.position().toPoint()))
        self._set_hover(patch_id)
        # A crosshair says "this is a thing to pick" where the open hand says
        # "drag to pan"; both still do what they say.
        if patch_id is None:
            self.viewport().unsetCursor()
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def leaveEvent(self, event) -> None:
        self._set_hover(None)
        super().leaveEvent(event)

    def patch_at(self, scene_point: QPointF) -> str | None:
        for patch_id, polygon in self._hit.items():
            if polygon.containsPoint(scene_point, Qt.FillRule.OddEvenFill):
                return patch_id
        return None

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        press, self._press = self._press, None
        if press is None or event.button() != Qt.MouseButton.LeftButton:
            return
        moved = event.position() - press
        if abs(moved.x()) + abs(moved.y()) > _CLICK_SLOP:
            return
        patch_id = self.patch_at(self.mapToScene(event.position().toPoint()))
        if patch_id is not None:
            self.patchToggled.emit(patch_id)


class ProfileView(QWidget):
    """Build a camera input profile from the chart already placed."""

    backRequested = Signal()
    #: The build to run, as the keyword arguments of
    #: :class:`~deltachrome_ui.workers.ProfileBuilder`. The shell owns the worker
    #: threads, so it runs the build and hands the result back.
    buildRequested = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setup = None
        self.target = None
        self.image_path: Path | None = None
        #: Patches the operator has left out, flagged ones included.
        self._excluded: set[str] = set()
        self._flagged: set[str] = set()
        self._built = None
        self._building = False
        #: Whether this chart searches by button rather than by default.
        self._large = False
        #: Whether the build running is Auto Calculate, whose winner the menus take.
        self._finding = False
        self._proxies: dict[bool, ImageProxy] = {}
        #: Raised once every control exists; a combo filled during construction
        #: signals before the controls its handler reads have been made.
        self._constructed = False

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_header())

        row = QWidget()
        shell.addWidget(row, 1)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = PatchCanvas()
        self.canvas.patchToggled.connect(self._toggle_patch)
        layout.addWidget(self.canvas, 1)

        panel = QScrollArea()
        panel.setFixedWidth(PANEL_WIDTH)
        panel.setWidgetResizable(True)
        panel.setFrameShape(QScrollArea.Shape.NoFrame)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setStyleSheet(f"QScrollArea {{ border-left: 1px solid {BORDER}; }}")
        layout.addWidget(panel)
        contents = QWidget()
        panel.setWidget(contents)
        self.panel_layout = QVBoxLayout(contents)
        self.panel_layout.setContentsMargins(20, 20, 20, 20)
        self.panel_layout.setSpacing(12)
        self._build_panel()
        self._constructed = True
        self._on_options_changed()

    # -- construction --------------------------------------------------------

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        header = QHBoxLayout(bar)
        header.setContentsMargins(20, 0, 0, 0)
        header.setSpacing(16)

        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        self.title_label = ElidedLabel()
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        stack.addWidget(self.title_label)
        self.subtitle_label = elided_subtitle()
        stack.addWidget(self.subtitle_label)
        header.addLayout(stack, 1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)
        back = IconButton("back", "Back to the placement", size=BAR_HEIGHT)
        back.clicked.connect(self.backRequested.emit)
        actions.addWidget(back)
        self.primary = QPushButton("Build Profile")
        self.primary.setObjectName("PrimaryBar")
        self.primary.setFixedHeight(BAR_HEIGHT)
        self.primary.setCursor(Qt.CursorShape.PointingHandCursor)
        self.primary.clicked.connect(self._primary)
        actions.addWidget(self.primary)
        header.addLayout(actions)
        return bar

    @staticmethod
    def _field(label: str, control: QWidget) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        name = QLabel(label)
        name.setObjectName("Subtitle")
        column.addWidget(name)
        column.addWidget(control)
        return holder

    def _build_panel(self) -> None:
        p = self.panel_layout

        p.addWidget(heading("Detected"))
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.encoding_pill = Pill()
        self.count_pill = Pill()
        self.flagged_pill = Pill()
        for pill in (self.encoding_pill, self.count_pill, self.flagged_pill):
            chips.addWidget(pill)
        chips.addStretch(1)
        p.addLayout(chips)
        self.warnings_box = QVBoxLayout()
        self.warnings_box.setSpacing(6)
        p.addLayout(self.warnings_box)
        self.gamma_check = QCheckBox("Display Gamma")
        self.gamma_check.setToolTip("Draw the linear capture brightened. "
                                    "Sampling is unaffected.")
        self.gamma_check.toggled.connect(self._on_gamma_toggled)
        self.gamma_check.hide()
        p.addWidget(self.gamma_check)
        p.addWidget(divider())

        p.addWidget(heading("Profile"))
        self.type_combo = ComboBox()
        self.type_combo.currentIndexChanged.connect(self._on_options_changed)
        type_row = QWidget()
        row = QHBoxLayout(type_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.type_combo, 1)
        self.find_best_button = QPushButton("Auto Calculate")
        self.find_best_button.setToolTip("Auto calculate the best type.")
        self.find_best_button.clicked.connect(self._find_best)
        row.addWidget(self.find_best_button)
        p.addWidget(self._field("Type", type_row))

        self.white_combo = ComboBox()
        for mode, label in WHITE_LABELS.items():
            self.white_combo.addItem(label, mode)
        self.white_combo.currentIndexChanged.connect(self._on_options_changed)
        p.addWidget(self._field("White", self.white_combo))

        segments = QWidget()
        strip = QHBoxLayout(segments)
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(0)
        self.scale_buttons: dict[str, QPushButton] = {}
        for index, aim in enumerate(SCALE_FOR):
            button = QPushButton(SCALE_FOR_LABELS[aim])
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setStyleSheet(segment_style(first=index == 0,
                                               last=index == len(SCALE_FOR) - 1))
            button.toggled.connect(self._on_options_changed)
            strip.addWidget(button)
            self.scale_buttons[aim] = button
        strip.addStretch(1)
        self.scale_buttons["white"].setChecked(True)
        self.scale_for_field = self._field("Scale For", segments)
        p.addWidget(self.scale_for_field)

        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_options_changed)
        p.addWidget(self._field("Name", self.name_edit))
        p.addWidget(divider())

        p.addWidget(heading("Patches"))
        used = QHBoxLayout()
        self.used_label = QLabel()
        self.used_label.setFont(figures_font(13.0, self.used_label.font()))
        used.addWidget(self.used_label)
        used.addStretch(1)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Leave out only the flagged patches.")
        self.reset_button.clicked.connect(self._reset_patches)
        used.addWidget(self.reset_button)
        p.addLayout(used)
        self.merge_check = QCheckBox("Merge Repeated Patches")
        self.merge_check.toggled.connect(self._on_options_changed)
        self.merge_check.toggled.connect(self._refresh_patches)
        p.addWidget(self.merge_check)

        self.advanced_toggle = QPushButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setStyleSheet(
            "QPushButton { background: transparent; border: none; padding: 4px 0;"
            f" color: {TEXT_DIM}; text-align: left; font-weight: 600; }}"
            f"QPushButton:hover {{ color: {TEXT}; }}"
        )
        self.advanced_toggle.toggled.connect(self._show_advanced)
        p.addWidget(self.advanced_toggle)

        self.advanced = QWidget()
        column = QVBoxLayout(self.advanced)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)
        self.quality_combo = ComboBox()
        self.quality_combo.currentIndexChanged.connect(self._on_options_changed)
        column.addWidget(self._field("Quality", self.quality_combo))
        self.smoothing_edit = QLineEdit()
        self.smoothing_edit.setPlaceholderText("0.5")
        self.smoothing_edit.textEdited.connect(self._on_options_changed)
        self.smoothing_field = self._field("Smoothing (%)", self.smoothing_edit)
        column.addWidget(self.smoothing_field)
        self.scale_edit = QLineEdit()
        self.scale_edit.setPlaceholderText("Automatic")
        self.scale_edit.textEdited.connect(self._on_options_changed)
        column.addWidget(self._field("Scale", self.scale_edit))
        self.restrict_check = QCheckBox("Restrict Primaries")
        self.restrict_check.toggled.connect(self._on_options_changed)
        column.addWidget(self.restrict_check)
        p.addWidget(self.advanced)
        self._show_advanced(False)
        self._fill_menus()

        self.busy_strip = BusyStrip(width=PANEL_WIDTH - 56, font_px=12)
        self.busy_strip.hide()
        p.addWidget(self.busy_strip, 0, Qt.AlignmentFlag.AlignHCenter)

        self.result_box = QWidget()
        result = QVBoxLayout(self.result_box)
        result.setContentsMargins(0, 0, 0, 0)
        result.setSpacing(12)
        result.addWidget(divider())
        result.addWidget(heading("Result"))
        self.result_grid = QGridLayout()
        self.result_grid.setHorizontalSpacing(14)
        self.result_grid.setVerticalSpacing(6)
        result.addLayout(self.result_grid)
        self.result_rows = QVBoxLayout()
        self.result_rows.setSpacing(6)
        result.addLayout(self.result_rows)
        self.result_box.hide()
        p.addWidget(self.result_box)
        p.addStretch(1)

    def _fill_menus(self) -> None:
        """Type and quality, offering Automatic only where it is the default."""
        menus = (
            (self.type_combo, PROFILE_TYPE_LABELS, LARGE_CHART_DEFAULT[0]),
            (self.quality_combo, {q: q.title() for q in QUALITIES},
             LARGE_CHART_DEFAULT[1]),
        )
        for combo, labels, default in menus:
            combo.blockSignals(True)
            combo.clear()
            if not self._large:
                combo.addItem("Automatic", AUTOMATIC)
            for value, label in labels.items():
                combo.addItem(label, value)
            combo.setCurrentIndex(combo.findData(default) if self._large else 0)
            combo.blockSignals(False)
        self.find_best_button.setVisible(self._large)

    def _show_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self.advanced_toggle.setText(("▾  " if shown else "▸  ") + "Advanced")

    # -- population ---------------------------------------------------------

    def open(self, *, proxy: ImageProxy, image_path: Path, target, setup) -> None:
        """Show this capture's patches, sampled once, with nothing yet built."""
        self.setup = setup
        self.target = target
        self.image_path = Path(image_path)
        self.title_label.setText(self.image_path.name)
        self.subtitle_label.setText(
            f"Camera Profile  ·  {target.display_name}  ·  {setup.reference_name}"
        )
        encoding = setup.encoding
        self._proxies = {False: proxy}
        linear = encoding is not None and encoding.linear and proxy.image.profile is None
        if linear:
            # Linear values drawn as they are put a chart's midtones near black;
            # what is sampled is the pixels, never this.
            proxy = self._proxies[True] = ImageProxy(proxy.image, linear_display=True)
        self.gamma_check.blockSignals(True)
        self.gamma_check.setChecked(linear)
        self.gamma_check.blockSignals(False)
        self.gamma_check.setVisible(linear)
        self.canvas.set_proxy(proxy)

        data = setup.data
        self._flagged = {pid for pid, (patch, _) in data.excluded.items() if patch.flags}
        self._excluded = set(self._flagged)
        # Decided once, on what the chart opens with: a menu that changed its
        # entries as patches were clicked would be a control moving under the
        # pointer. Excluding below the line only stops Auto Calculate trying tables.
        self._large = len(data.all_patches) - len(self._excluded) >= TABLE_MIN_PATCHES
        self._fill_menus()

        self.encoding_pill.set_state(encoding.label if encoding else "", "unknown",
                                     detail=(f"Exponent {encoding.exponent:.2f} over "
                                             f"{encoding.steps} neutral steps."
                                             if encoding else ""))
        total = len(data.all_patches)
        self.count_pill.set_state(f"{total} Patches", "unknown")
        self.flagged_pill.set_state(
            f"{len(self._flagged)} Flagged" if self._flagged else "", "unknown"
        )
        while self.warnings_box.count():
            item = self.warnings_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for warning in setup.warnings:
            self.warnings_box.addWidget(caveat(warning))

        self.name_edit.setText(f"{self.image_path.stem} {date.today().isoformat()}")
        self.merge_check.setChecked(False)
        self.merge_check.setVisible(setup.merged_count is not None)
        if setup.merged_count is not None:
            self.merge_check.setToolTip(f"{setup.merged_count} readings when merged.")
        self._refresh_patches()
        self._invalidate()

    # -- patches ------------------------------------------------------------

    def _toggle_patch(self, patch_id: str) -> None:
        if self._building:
            return
        self._excluded ^= {patch_id}
        self._refresh_patches()
        self._invalidate()

    def _reset_patches(self) -> None:
        self._excluded = set(self._flagged)
        self._refresh_patches()
        self._invalidate()

    def _refresh_patches(self, *_args) -> None:
        if self.setup is None:
            return
        data = self.setup.data
        total = len(data.all_patches)
        self.used_label.setText(f"{total - len(self._excluded)} of {total}")
        self.reset_button.setEnabled(self._excluded != self._flagged)
        self.canvas.show_patches(data.regions, self._excluded, self._flagged,
                                 self._repeat_families())

    def _repeat_families(self) -> list[tuple[str, ...]]:
        """The families merging would pool now, by patch id; none while it is off."""
        from deltachrome.profiling import select_patches
        from deltachrome.profiling.patches import repeat_families

        if not self.merge_check.isChecked():
            return []
        try:
            used = select_patches(self.setup.data, self._excluded).used
        except ProfilingError:
            return []
        return [tuple(used[i].patch_id for i in family)
                for family in repeat_families(used)]

    def _on_gamma_toggled(self, on: bool) -> None:
        proxy = self._proxies.get(on)
        if proxy is not None:
            self.canvas.swap_proxy(proxy)

    # -- options ------------------------------------------------------------

    def _on_options_changed(self, *_args) -> None:
        if not self._constructed:
            return
        kind = self.type_combo.currentData()
        self.smoothing_field.setVisible(kind in TABLE_TYPES)
        chart_white = self.white_combo.currentData() == "chart"
        manual_scale = bool(self.scale_edit.text().strip())
        self.scale_for_field.setEnabled(not chart_white and not manual_scale)
        self._invalidate()

    def request(self) -> dict:
        """What Build Profile would run, or raise :class:`ProfilingError` saying why not."""
        from deltachrome.profiling import merge_repeats, select_patches

        data = select_patches(self.setup.data, self._excluded)
        if self.merge_check.isChecked():
            data = merge_repeats(data)
        kind = self.type_combo.currentData()
        quality = self.quality_combo.currentData()
        white = self.white_combo.currentData()
        scale = _number(self.scale_edit.text(), "Scale")
        smoothing = (_number(self.smoothing_edit.text(), "Smoothing")
                     if kind in TABLE_TYPES else None)
        scale_for = None
        if scale is None and white != "chart":
            scale_for = next(aim for aim, b in self.scale_buttons.items() if b.isChecked())
        options = ColprofOptions(
            # Placeholders where Automatic will choose; the options hold real values.
            profile_type="shaper" if kind == AUTOMATIC else kind,
            quality="medium" if quality == AUTOMATIC else quality,
            white=white,
            scale=scale,
            smoothing=smoothing,
            description=self.name_edit.text().strip() or self.image_path.stem,
            copyright=paths.remembered(paths.PROFILE_COPYRIGHT_KEY, "") or "",
            transmissive="transmissive" in (self.target.target_type or ""),
            restrict_primaries=self.restrict_check.isChecked(),
        )
        automatic = AUTOMATIC in (kind, quality)
        return {
            "data": data,
            "options": options,
            "keywords": dict(self.setup.keywords,
                             **({"merged_repeats": "yes"}
                                if self.merge_check.isChecked() else {})),
            "scale_for": scale_for,
            "automatic": automatic,
            "types": None if kind == AUTOMATIC else [kind],
            "qualities": None if quality == AUTOMATIC else [quality],
        }

    # -- building and saving ------------------------------------------------

    def _invalidate(self) -> None:
        """Whatever was built no longer describes the panel."""
        if not self._constructed:
            return
        self._built = None
        self.result_box.hide()
        self._update_primary()

    def _update_primary(self) -> None:
        if self._built is not None:
            self.primary.setText("Save Profile…")
            self.primary.setEnabled(True)
            self.primary.setToolTip("")
            self.find_best_button.setEnabled(True)
            return
        self.primary.setText("Build Profile")
        reason = ""
        if self.setup is None:
            reason = "No chart placed."
        elif paths.resolve_colprof() is None:
            reason = "colprof was not found. Set its location in Settings."
        else:
            total = len(self.setup.data.all_patches)
            if total - len(self._excluded) < MIN_PATCHES:
                reason = f"A profile needs at least {MIN_PATCHES} patches."
        self.primary.setEnabled(not reason and not self._building)
        self.primary.setToolTip(reason)
        self.find_best_button.setEnabled(not reason and not self._building)

    def _primary(self) -> None:
        if self._built is not None:
            self._save()
            return
        try:
            request = self.request()
        except ProfilingError as exc:
            QMessageBox.warning(self, "Cannot build", str(exc))
            return
        request["colprof"] = paths.resolve_colprof()
        self.set_building("Building the profile…")
        self.buildRequested.emit(request)

    def _find_best(self) -> None:
        """Automatic over every candidate, whatever the menus say."""
        try:
            request = self.request()
        except ProfilingError as exc:
            QMessageBox.warning(self, "Cannot build", str(exc))
            return
        request.update(automatic=True, types=None, qualities=None,
                       options=replace(request["options"], smoothing=None),
                       colprof=paths.resolve_colprof())
        self._finding = True
        self.set_building("Finding the best type…")
        self.buildRequested.emit(request)

    def set_building(self, message: str) -> None:
        self._building = True
        self.busy_strip.set_message(message)
        self.busy_strip.show()
        self.result_box.hide()
        for control in (self.type_combo, self.white_combo, self.name_edit,
                        self.merge_check, self.advanced, self.scale_for_field,
                        self.reset_button, self.find_best_button):
            control.setEnabled(False)
        self._update_primary()

    def _done_building(self) -> None:
        self._building = False
        self.busy_strip.hide()
        for control in (self.type_combo, self.white_combo, self.name_edit,
                        self.merge_check, self.advanced):
            control.setEnabled(True)
        self._on_options_changed()
        self._refresh_patches()

    def show_progress(self, message: str) -> None:
        self.busy_strip.set_message(message)

    def show_failed(self, failure) -> None:
        self._finding = False
        self._done_building()
        QMessageBox.warning(self, "Profile not built",
                            f"{failure.headline}\n\n{failure.detail}")

    def show_built(self, built) -> None:
        if self._finding:
            # The winner goes into the menus before anything can read them as a
            # change; it is what was built, so Save stays offered.
            for combo, value in ((self.type_combo, built.options.profile_type),
                                 (self.quality_combo, built.options.quality)):
                combo.blockSignals(True)
                combo.setCurrentIndex(combo.findData(value))
                combo.blockSignals(False)
            self._finding = False
        self._done_building()
        self._built = built
        self._fill_result(built)
        self._update_primary()

    def _fill_result(self, built) -> None:
        for layout in (self.result_grid, self.result_rows):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
        unit = QLabel("ΔE*2000")
        unit.setObjectName("Subtitle")
        self.result_grid.addWidget(unit, 0, 0)
        for column, name in enumerate(("Mean", "90th", "Max"), start=1):
            label = QLabel(name)
            label.setObjectName("Subtitle")
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.result_grid.addWidget(label, 0, column)
        rows = (("Built From", built.fit), ("Validation", built.held_out),
                ("Excluded", built.excluded))
        line = 1
        for name, summary in rows:
            if summary is None:
                continue
            self.result_grid.addWidget(QLabel(name), line, 0)
            for column, value in enumerate((summary.mean, summary.p90, summary.max),
                                           start=1):
                cell = QLabel(f"{value:.2f}")
                cell.setFont(figures_font(13.0, cell.font()))
                cell.setAlignment(Qt.AlignmentFlag.AlignRight)
                self.result_grid.addWidget(cell, line, column)
            line += 1
        options = built.options
        self.result_rows.addWidget(_value_row(
            "Type", f"{profile_type_label(options.profile_type)}, "
                    f"{options.quality.title()}"))
        if built.scale is not None:
            self.result_rows.addWidget(_value_row("Scale", f"{built.scale:.4f}"))
        for warning in built.run.warnings:
            self.result_rows.addWidget(caveat(warning))
        self.result_box.show()

    def _save(self) -> None:
        name = (self.name_edit.text().strip() or self.image_path.stem)
        folder = paths.remembered(paths.PROFILE_DIR_KEY) or str(Path.home())
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", str(Path(folder) / f"{name}.icc"),
            "ICC Profile (*.icc *.icm)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() not in (".icc", ".icm"):
            path = path.with_suffix(".icc")
        try:
            path.write_bytes(self._built.icc)
        except OSError as exc:
            QMessageBox.warning(self, "Not saved", str(exc))
            return
        paths.remember(paths.PROFILE_DIR_KEY, str(path.parent))
        self.primary.setText("Saved")
        QTimer.singleShot(1400, self._update_primary)


def _number(text: str, name: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ProfilingError(f"{name} must be a number.") from exc


def _value_row(key: str, value: str) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    name = QLabel(key)
    name.setObjectName("Subtitle")
    shown = QLabel(value)
    shown.setFont(figures_font(12.0, shown.font()))
    row.addWidget(name)
    row.addStretch(1)
    row.addWidget(shown)
    return holder
