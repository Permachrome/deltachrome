"""Small shared pieces.

The status badge is the only one worth arguing about. It takes a
:class:`~deltachrome_ui.status.Status` rather than a string, so there is no path
by which a caller can invent a label or pick its own color -- the demotion rule
in :mod:`deltachrome_ui.status` cannot be routed around by a widget that happens
to have a green stylesheet to hand.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui import mark
from deltachrome_ui.status import Status
from deltachrome_ui.status import TONE_QUALIFIED
from deltachrome_ui.theme import (
    ARROW_ZONE,
    BAR_HEIGHT,
    MARK_INK,
    RAMP,
    TEXT,
    TEXT_DIM,
    TEXT_SPENT,
    tone_colors,
)


class Ruled(QWidget):
    """A coloured rule, and words beside it. Every status in the application.

    There were four shapes doing this: a verdict badge, a state chip, the
    report's conformance caveat and a profile's warning, each a filled rectangle
    with a stroke and a radius, each styling itself. They are one shape now --
    a rule in the tone, and text in the tone, with nothing filled.

    **Why nothing filled.** A filled rounded rectangle is the shape of a button,
    and three of these four sat in rows that also held buttons. A verdict that
    looks pressable is not read as a statement, which is the one thing a verdict
    has to be. The rule carries the colour instead, and the colour is still the
    theme's -- :mod:`deltachrome_ui.status` decides which tone applies and
    nothing here can reach for a green the palette does not have.

    The rule **stretches to the text**, however many lines that is, so a one-word
    chip and a wrapped paragraph are the same widget at two sizes.
    """

    RULE_WIDTH = 3
    #: Never shorter than this, so a single short word still reads as a marked
    #: thing rather than as a dot beside some text.
    MIN_RULE_HEIGHT = 15

    def __init__(self, *, gap: int = 9, pad: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(gap)

        self.rule = QWidget()
        self.rule.setObjectName("ToneRule")
        self.rule.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.rule.setFixedWidth(self.RULE_WIDTH)
        self.rule.setMinimumHeight(self.MIN_RULE_HEIGHT)
        self.rule.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding
        )
        row.addWidget(self.rule)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, pad, 0, pad)
        self.body.setSpacing(0)
        row.addLayout(self.body, 1)

        # It never takes slack. The rule stretches to the widget, so a panel with
        # spare height below it turned a one-word chip into a three-hundred-pixel
        # stripe -- which is the same mistake the waiting bar made, in the same
        # direction, for the same reason.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def set_tone(self, tone: str) -> None:
        self._tone = tone
        color = tone_colors(tone)[0]
        self.rule.setStyleSheet(
            f"QWidget#ToneRule {{ background: {color}; border-radius: 2px; }}"
        )

    def tone_color(self) -> str:
        return tone_colors(getattr(self, "_tone", "unknown"))[0]


class StatusBadge(Ruled):
    """A verdict, shown in its tone, carrying its own explanation as a tooltip."""

    def __init__(self, status: Status | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.label = QLabel()
        self.label.setObjectName("ToneWord")
        self.body.addWidget(self.label)
        if status is not None:
            self.set_status(status)

    def set_status(self, status: Status) -> None:
        self.set_tone(status.tone)
        self.label.setText(status.label)
        self.label.setStyleSheet(f"color: {self.tone_color()};")
        # Every status in this vocabulary is one a user can reasonably not know,
        # so none of them ships without its sentence.
        self.setToolTip(status.hint)

    def text(self) -> str:
        """What it says. Kept because it was a ``QLabel`` and callers ask."""
        return self.label.text()


def figures_font(point_size: float, base: QFont | None = None) -> QFont:
    """The body face, with figures that keep their width, for a number that changes.

    One face for numbers everywhere: the charts draw theirs in the application's
    own font, and a readout beside them in Menlo read as a different quantity, the
    same argument that took mono out of the report's value column. What a mono face
    did buy -- digits that do not shuffle sideways while a corner is dragged -- is
    the font's own tabular figures, which keep the face.
    """
    font = QFont(base) if base is not None else QFont()
    font.setPointSizeF(point_size)
    font.setFeature(QFont.Tag("tnum"), 1)
    return font


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    return frame


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    return line


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionHeading")
    return label


def title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Title")
    return label


def elided_title(text: str = "") -> ElidedLabel:
    """A title in a fixed-width column, where the text is somebody's filename."""
    label = ElidedLabel(text)
    label.setObjectName("Title")
    return label


def elided_subtitle(text: str = "") -> ElidedLabel:
    label = ElidedLabel(text)
    label.setObjectName("Subtitle")
    return label


def subtitle(text: str, *, rich: bool = False) -> QLabel:
    """Secondary text. ``rich`` opts into HTML.

    Off by default on purpose: several subtitles carry filesystem paths, and a
    path containing ``&`` or ``<`` would be silently mangled by Qt's rich-text
    auto-detection.
    """
    label = QLabel(text)
    label.setObjectName("Subtitle")
    label.setWordWrap(True)
    label.setTextFormat(
        Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText
    )
    return label


class ElidedLabel(QLabel):
    """A label that shrinks its text to fit rather than its panel to fit the text.

    Word wrapping cannot help with the two strings this exists for. A capture's
    filename is one unbroken token -- ``qa62_LWF_BC_001_20211126_SH01_0411.jp2``
    has nowhere to break -- so a wrapping label reports a *minimum* width of the
    whole string, 398 px at the title size. Inside a 380 px panel that is not a
    long title: it is every control on the panel pushed off the right edge, with
    the horizontal scrollbar turned off so nothing can be scrolled back into
    view. An ICC profile's own description does the same thing more slowly, by
    wrapping to three lines to say that the file is tagged.

    So the text is elided to whatever width the label is actually given, at every
    width, and the whole of it is on hover. Which is also why this is a widget
    rather than a truncation at the call site: the panel is resizable in
    principle and a character count is a guess about a width.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None, *,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight) -> None:
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 -- Qt's own spelling
        self._full = text or ""
        # The untruncated string, always, so nothing is lost by eliding it.
        self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:
        """The full text, not what happens to be painted.

        Anything asking a label what it says wants what it was given -- a test
        checking a filename reached the screen, or a caller reading it back. The
        elision is a rendering, and rendering is not the label's answer.
        """
        return self._full

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(0, self.width())
        shown = (
            self.fontMetrics().elidedText(self._full, self._mode, width)
            if width else self._full
        )
        super().setText(shown)


def caveat(text: str) -> QLabel:
    """Small print that is load-bearing rather than decorative."""
    label = QLabel(text)
    label.setObjectName("Caveat")
    label.setWordWrap(True)
    return label


def field_row(name: str, value: str, *, tooltip: str = "") -> QWidget:
    """A dim label and a value, for the read-only property lists."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(10)
    key = QLabel(name)
    key.setObjectName("Subtitle")
    key.setFixedWidth(112)
    key.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    val = QLabel(value)
    val.setWordWrap(True)
    val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(key)
    layout.addWidget(val, 1)
    if tooltip:
        row.setToolTip(tooltip)
    return row


class ComboBox(QComboBox):
    """A combo box that draws its own drop-down chevron.

    Qt draws the drop-down as a square button with its own background and border,
    which lands on top of the 6px rounded corner and shows as two dark notches at
    the right edge. Making the subcontrol transparent in the style sheet removes
    the block -- and the arrow with it, because once a style sheet claims the
    subcontrol Qt only paints an arrow it was given as an image.

    Style sheets have no rotation, so the two-borders chevron renders as a right
    angle, and the remaining options are to ship a seven-pixel image file or to
    paint it. Painting it keeps the application a single installable package with
    no asset directory beside it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # An explicit list view, so the popup is the same on every platform and
        # so the style sheet reaches it at all. macOS gives a non-editable combo
        # box the *native menu* popup, which positions the current item over the
        # box and takes its metrics from the platform rather than from
        # ``QComboBox QAbstractItemView`` -- so the rule this application already
        # wrote for that popup was being ignored, and in a short combo the menu
        # came up clipped to a fragment of itself.
        self.setView(QListView())
        # Qt sizes a combo box to its widest *item*, which in a side panel is a
        # panel sized to "FADGI -- 3rd Edition, May 2023 . Photographic
        # Transparencies 4x5 and Larger". The list is what needs that width, not
        # the closed box, so the box takes the room it is given and the full
        # label is available where a long one always is: on hover.
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(12)
        self.currentIndexChanged.connect(self._describe)

    #: Matches ``QComboBox QAbstractItemView``'s own padding in the style sheet.
    #: Stated here because the popup's height has to be computed from the same
    #: number the rows are inset by, and the two being equal is what makes the
    #: last row clear the rounded corner.
    POPUP_PADDING = 4

    #: Above this many items a popup is a list to be scrolled; at or below it,
    #: it is a set of choices to be seen at once. Twelve is FADGI's four star
    #: levels and Metamorfoze's three with room over, and well under the fifteen
    #: material categories -- so every list in this application that a person
    #: reads *as a whole* fits, and the two that are genuinely long do not.
    FITS_WITHOUT_SCROLLING = 12

    def showPopup(self) -> None:  # noqa: N802 - Qt's name
        """Show every item where the list is short enough to show every item.

        The report was that a four-item popup felt uncontrollable on a Magic
        Mouse, which is the opposite of what a four-item popup should feel like:
        there is nothing below the fourth item, so there is nothing to scroll
        to. It could still be *scrolled* because pixel scrolling makes a range
        out of the few pixels of padding around the rows -- so a momentum
        gesture slid the list a little and sprang back, which reads as a control
        fighting the hand.

        So the two cases are separated rather than tuned. A short list is sized
        to hold all of it and scrolls **per item**, which makes its scroll range
        exactly zero -- it cannot move at all. A long one keeps pixel scrolling,
        because there a momentum gesture is the point and per-item jumps a whole
        row at a time.
        """
        fits = self.count() <= self.FITS_WITHOUT_SCROLLING
        self.setMaxVisibleItems(
            max(1, self.count()) if fits else self.FITS_WITHOUT_SCROLLING
        )
        self.view().setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerItem if fits
            else QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        super().showPopup()
        if fits:
            self._fit_popup_to_its_items()

    def _fit_popup_to_its_items(self) -> None:
        """Take the scroll arrows off a popup that has nothing to scroll to.

        ``setMaxVisibleItems`` is not enough, and the reason is circular. Qt's
        popup container lays out an up-arrow, the view, and a down-arrow, and it
        takes the arrows' height **out of the view** rather than adding it to the
        container -- so a two-item popup got a view one and a half rows tall,
        which made the view scrollable, which is the condition for showing the
        arrows. Measured on the chart selector: container 66 px, view 39, one
        visible 175x15 arrow strip, scroll range 1. On a list of two.

        So the arrows are hidden and the view is given the height its own rows
        ask for, which takes the scroll range to zero. Only where the list fits
        -- a long one genuinely needs them.

        Qt exposes none of this: the arrows are ``QComboBoxPrivateScroller`` and
        reach Python as bare ``QWidget`` children of the view's parent. They are
        identified as "every child widget of the container that is not the view",
        which is what they are, and the worst case if Qt ever adds another child
        is a popup that is the right size with something hidden in it.
        """
        view = self.view()
        container = view.parentWidget()
        if container is None:
            return
        for child in container.children():
            if isinstance(child, QWidget) and child is not view:
                child.hide()
        rows = sum(view.sizeHintForRow(row) for row in range(self.count()))
        frame = 2 * view.frameWidth() + self.POPUP_PADDING * 2
        view.setFixedHeight(rows + frame)
        container.setFixedHeight(view.height())

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt's name
        """Scroll the page, never the selection.

        Qt's default is that a wheel over a *closed* combo box changes what it
        holds -- so on a short list a single trackpad flick runs through every
        FADGI star level and stops on whichever one the gesture ended over, and
        the operator's guideline has silently changed while they were scrolling
        past it. It is the one control on these screens where an accidental
        gesture rewrites what a run is judged against.

        Ignoring it lets the event reach whatever scroll area the box sits in,
        which is what the gesture was for. The popup is unaffected -- it is a
        list view and scrolls the ordinary way -- and so is the keyboard, which
        is how a selection is meant to be walked.
        """
        event.ignore()

    def _describe(self) -> None:
        self.setToolTip(self.currentText())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        size = 8.0
        center_y = self.height() / 2 - size / 5
        right = self.width() - ARROW_ZONE / 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(TEXT if self.underMouse() else TEXT_DIM))
        pen.setWidthF(1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(right - size / 2, center_y)
        path.lineTo(right, center_y + size / 2)
        path.lineTo(right + size / 2, center_y)
        painter.drawPath(path)


class Pill(Ruled):
    """A compact state chip: a few words in a tone, with the detail on hover.

    Deliberately a different class from :class:`StatusBadge`, which takes a
    :class:`~deltachrome_ui.status.Status` so that no caller can invent a label
    or pick its own color. That rule exists to stop a qualified *verdict* being
    painted green, and it must not be routed around -- which is why they share a
    *shape* and not a constructor.

    Nothing here is a verdict. "Target Detected" and "Placed Automatically" are
    states of the screen before anything has been measured -- there is no
    conformance claim to demote and no reading to flatter -- so the tone is the
    caller's to choose. The vocabulary is still the theme's, so a pill cannot
    reach for a green the palette does not have.
    """

    def __init__(self, text: str = "", tone: str = "unknown",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.label = QLabel()
        self.label.setObjectName("ToneWord")
        self.body.addWidget(self.label)
        self.set_state(text, tone)

    def set_state(self, text: str, tone: str, *, detail: str = "") -> None:
        """Set what it says, how it reads, and what it says on hover.

        ``detail`` is where the diagnostic goes. The pill is the glance; the
        tooltip carries what a reader would want if the glance surprised them,
        which keeps a sentence available without spending a line on it.
        """
        self.set_tone(tone)
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {self.tone_color()};")
        self.setToolTip(detail)
        self.setVisible(bool(text))

    def text(self) -> str:
        return self.label.text()


class Callout(Ruled):
    """A paragraph in a tone: the report's conformance caveat, a profile's warning.

    The same rule as the chips, at paragraph size. It is the one place where
    dropping the fill costs something real -- the conformance caveat is the text
    this application least wants missed, and a filled amber panel is harder to
    read past than coloured text beside a line. So the rule runs the **full
    height of the paragraph** and the text stays in the tone rather than going
    dim: what is given up is the field behind it, not the colour.
    """

    #: Rule, gap and the right-hand margin -- what the paragraph does *not* get.
    OVERHEAD = Ruled.RULE_WIDTH + 13 + 4

    def __init__(self, text: str = "", tone: str = "qualified",
                 parent: QWidget | None = None) -> None:
        super().__init__(gap=13, pad=9, parent=parent)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setObjectName("ToneParagraph")
        self.body.addWidget(self.label)
        # A wrapping paragraph's height depends on the width it is given, and a
        # layout only asks when it is told to. Without this the callout reports
        # the height of one line and the panel below it is drawn over the rest.
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.set_state(text, tone)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt's name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt's name
        inner = max(1, width - self.OVERHEAD)
        return self.label.heightForWidth(inner) + 2 * 9

    def set_state(self, text: str, tone: str = "qualified") -> None:
        self.set_tone(tone)
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {self.tone_color()};")
        self.setVisible(bool(text))
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self.setMinimumHeight(self.heightForWidth(self.width()))

    def text(self) -> str:
        return self.label.text()


class BusyStrip(QWidget):
    """Something is happening, and it has not stopped.

    A message and an indeterminate bar, and that is all it is allowed to be. It
    used to be a paragraph on the loading screen explaining that JPEG 2000 is
    compressed and decodes more slowly than an uncompressed TIFF -- true, and an
    answer to a question nobody is asking while they wait.

    **The bar is indeterminate because the work behind it genuinely has no
    progress to report.** A JP2 decode returns when it returns; the finders run
    one after another. A bar that filled would be inventing a fraction.

    Shared rather than written twice, because the second place that needed one is
    the capture panel -- where its absence was not a cosmetic gap. Decoding this
    repository's ColorChecker SG master takes 24.6 seconds, and a button that
    stayed enabled and said the same thing throughout invited a second press,
    which started a second decode.
    """

    #: Wide enough to read a short message under, narrow enough for a 384 px
    #: panel. The bar is thin on purpose: it is a sign of life, not a gauge.
    WIDTH = 260
    HEIGHT = 4

    def __init__(self, message: str = "", *, width: int = WIDTH,
                 font_px: int = 13, parent: QWidget | None = None) -> None:
        from PySide6.QtWidgets import QProgressBar, QVBoxLayout

        super().__init__(parent)
        self._width = int(width)
        # It never takes slack. Given a panel's spare height it put the message
        # at the top and the bar five hundred pixels below it, which reads as two
        # unrelated things rather than as one wait.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setFixedWidth(self._width)
        # A QFont rather than a style sheet, and that is the load-bearing part.
        # A style sheet's font-size does not reach QWidget::font() until the
        # style is polished, so ``heightForWidth`` below measured the default
        # size and reported one line for a message that painted as two.
        font = self.label.font()
        font.setPixelSize(int(font_px))
        self.label.setFont(font)
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.bar = QProgressBar()
        # Range 0-0 is Qt's indeterminate bar: it sweeps rather than filling,
        # which is the honest rendering of work whose duration is not known.
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(self._width, self.HEIGHT)
        layout.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignHCenter)

        self.set_message(message)

    def set_message(self, text: str) -> None:
        """Set the message, and make room for however many lines it takes.

        The height is measured rather than left to the layout. A wrapping label
        reports the size hint of the width it would *like*, not of the width it
        has, so a message long enough to wrap reported the height of one line
        and was painted straight over the bar beneath it. Seen with "Sampling
        patches and measuring…", which is the longest thing either caller says.
        """
        self._message = text
        self.label.setText(text)
        self.label.setFixedHeight(max(1, self.label.heightForWidth(self._width)))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Measured again on the way onto the screen, because anything that
        # changes the font between construction and the first paint -- a style
        # sheet applied to an ancestor, most obviously -- changes how many lines
        # the message takes and nothing else would notice.
        super().showEvent(event)
        self.set_message(self._message)


def draw_glyph(painter: QPainter, box: QRectF, kind: str, color: str) -> None:
    """Draw one of the application's icons into ``box``.

    Painted rather than shipped. The same argument the combo box's chevron
    already rests on: an asset directory beside the executable is a second thing
    to install and a second thing to get out of step, and these are five shapes
    of a few strokes each.

    Each one is drawn in fractions of the box it is given, so the same code
    serves a 16 px toolbar icon and whatever comes later.
    """
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    x, y, w, h = box.x(), box.y(), box.width(), box.height()
    cx, cy = x + w / 2, y + h / 2

    if kind == "back":
        painter.drawLine(QPointF(x + w * 0.82, cy), QPointF(x + w * 0.18, cy))
        painter.drawLine(QPointF(x + w * 0.18, cy), QPointF(x + w * 0.44, cy - h * 0.26))
        painter.drawLine(QPointF(x + w * 0.18, cy), QPointF(x + w * 0.44, cy + h * 0.26))

    elif kind in ("prev", "next"):
        # A bare chevron, and deliberately not the "back" arrow drawn above it.
        # That one has a shaft and means *leave this screen*; it sits two cells
        # away on the same bar, and two arrows there meaning "out of the report"
        # and "the next chart of this capture" would be the worse of the two
        # possible confusions. A chevron points at a neighbour.
        sign = -1.0 if kind == "prev" else 1.0
        tip = cx + sign * w * 0.17
        tail = cx - sign * w * 0.13
        painter.drawLine(QPointF(tail, cy - h * 0.27), QPointF(tip, cy))
        painter.drawLine(QPointF(tip, cy), QPointF(tail, cy + h * 0.27))

    elif kind in ("placement", "rerun"):
        # A chart: nine patches in a grid. **Two actions, one picture**, because
        # they have one destination -- "back to the placement" on a fresh run and
        # "re-run the capture" on a saved one both open the screen where the
        # chart is placed, and an icon names where it goes. They stay two kinds
        # rather than one because they are still two actions, with two tooltips
        # and two signals, and the button's kind is how each is identified.
        #
        # It was four corner marks, which is what a *placement* is and not what
        # the screen is for; a reader who has not already learned the icon reads
        # crop marks. What the screen actually shows is a colour chart with its
        # regions on it, and that is a shape people recognise without being told.
        #
        # **Solid, and three by three.** Drawn as outlines the cells are 6 px
        # across at the header's 22 px glyph and the 1.6 px stroke closes them
        # up: at actual size a 3x3 outline is a grey smudge. Two by two survives
        # as an outline and reads as the generic "grid view" every application
        # has. Filled cells are unambiguous at both sizes, and a chart's patches
        # are filled.
        cells = 3
        gap = w * 0.09
        cell = (w - gap * (cells - 1)) / cells
        for row in range(cells):
            for col in range(cells):
                painter.fillRect(
                    QRectF(x + col * (cell + gap), y + row * (cell + gap), cell, cell),
                    QColor(color),
                )

    elif kind == "command":
        # A terminal prompt, because what it copies is a command line.
        painter.drawLine(QPointF(x + w * 0.20, y + h * 0.28), QPointF(x + w * 0.44, cy))
        painter.drawLine(QPointF(x + w * 0.44, cy), QPointF(x + w * 0.20, y + h * 0.72))
        painter.drawLine(
            QPointF(x + w * 0.54, y + h * 0.74), QPointF(x + w * 0.82, y + h * 0.74)
        )

    elif kind == "refresh":
        # An arc with a gap and a solid head at the end of it, along the tangent.
        # The head is computed from the arc's own endpoint rather than placed by
        # eye: two lines guessed at an angle render as a cross at 22 px.
        r = min(w, h) * 0.34
        start, span = 60.0, 280.0
        painter.drawArc(
            QRectF(cx - r, cy - r, r * 2, r * 2), int(start * 16), int(span * 16)
        )
        end = math.radians(start + span)
        ex, ey = cx + r * math.cos(end), cy - r * math.sin(end)
        dx, dy = -math.sin(end), -math.cos(end)
        nx, ny = -dy, dx
        head = QPainterPath(QPointF(ex + dx * r * 0.62, ey + dy * r * 0.62))
        head.lineTo(ex + nx * r * 0.38, ey + ny * r * 0.38)
        head.lineTo(ex - nx * r * 0.38, ey - ny * r * 0.38)
        head.closeSubpath()
        painter.fillPath(head, QColor(color))

    elif kind == "folder":
        path = QPainterPath(QPointF(x + w * 0.06, y + h * 0.82))
        path.lineTo(x + w * 0.06, y + h * 0.20)
        path.lineTo(x + w * 0.40, y + h * 0.20)
        path.lineTo(x + w * 0.50, y + h * 0.34)
        path.lineTo(x + w * 0.94, y + h * 0.34)
        path.lineTo(x + w * 0.94, y + h * 0.82)
        path.closeSubpath()
        painter.drawPath(path)

    elif kind == "settings":
        # Sliders rather than a gear. A gear is the more universal sign and it
        # does not survive this size: eight radial teeth on a 22 px box with a
        # 1.6 px stroke render as a sun, which is what "brightness" looks like.
        # Sliders stay legible. The cost is that the same picture sometimes means
        # "filter", and this screen does have a search field -- but that field is
        # a text box with its own placeholder, not an icon, and this sits in the
        # header's action group rather than beside it.
        for row, knob in ((0.24, 0.66), (0.50, 0.34), (0.76, 0.56)):
            yy = y + h * row
            painter.drawLine(QPointF(x + w * 0.08, yy), QPointF(x + w * 0.92, yy))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(x + w * knob, yy), w * 0.11, w * 0.11)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    elif kind == "check":
        painter.drawLine(QPointF(x + w * 0.16, y + h * 0.54),
                         QPointF(x + w * 0.40, y + h * 0.78))
        painter.drawLine(QPointF(x + w * 0.40, y + h * 0.78),
                         QPointF(x + w * 0.86, y + h * 0.22))

    elif kind == "save":
        painter.drawLine(QPointF(cx, y + h * 0.16), QPointF(cx, y + h * 0.60))
        painter.drawLine(QPointF(cx, y + h * 0.60), QPointF(cx - w * 0.18, y + h * 0.40))
        painter.drawLine(QPointF(cx, y + h * 0.60), QPointF(cx + w * 0.18, y + h * 0.40))
        path = QPainterPath(QPointF(x + w * 0.18, y + h * 0.66))
        path.lineTo(x + w * 0.18, y + h * 0.84)
        path.lineTo(x + w * 0.82, y + h * 0.84)
        path.lineTo(x + w * 0.82, y + h * 0.66)
        painter.drawPath(path)

    elif kind == "export":
        # The save glyph's arrow turned round: out of the tray rather than into
        # it. The two are the platform's own pair, and a report's file leaving
        # the application is exactly what the outward one means.
        painter.drawLine(QPointF(cx, y + h * 0.60), QPointF(cx, y + h * 0.16))
        painter.drawLine(QPointF(cx, y + h * 0.16), QPointF(cx - w * 0.18, y + h * 0.36))
        painter.drawLine(QPointF(cx, y + h * 0.16), QPointF(cx + w * 0.18, y + h * 0.36))
        path = QPainterPath(QPointF(x + w * 0.18, y + h * 0.66))
        path.lineTo(x + w * 0.18, y + h * 0.84)
        path.lineTo(x + w * 0.82, y + h * 0.84)
        path.lineTo(x + w * 0.82, y + h * 0.66)
        painter.drawPath(path)

    painter.restore()


class MarkBadge(QWidget):
    """The application's mark, at a fixed height, painted rather than loaded.

    A widget rather than a ``QLabel`` holding a pixmap, so it redraws at the
    right resolution when the window moves between a Retina display and an
    external one -- a pixmap baked at construction is the usual way that goes
    soft, and this sits in a header bar next to crisp text where the difference
    shows.

    It carries the application's name as its accessible name and its tooltip,
    because it is the one thing on the screen that says which program this is
    and a mark nobody can name is decoration.
    """

    def __init__(self, height: int, *, tooltip: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._height = int(height)
        self.setFixedSize(round(self._height * mark.ASPECT), self._height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)
        self.setAccessibleName("Deltachrome")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        mark.paint_mark(painter, QRectF(0, 0, self.width(), self.height()),
                        MARK_INK, RAMP)


def mark_cell(height: int = 26, *, tooltip: str = "Deltachrome") -> QWidget:
    """The mark in a cell the width of an action square, for a header bar.

    Both bars that carry the mark carry it this way, because the alternative is
    that they drift: the library set it against a 20 px left margin and Settings
    centred it in a ``BAR_HEIGHT`` cell, so the same logo sat eight pixels apart
    on two screens a person moves between with one click. A cell rather than a
    margin is the one of the two with an argument behind it -- every action on
    the right of every bar is a square of the bar's own height, flush to the
    frame, so the mark occupying the mirror of one is the bar's own geometry
    rather than a number somebody chose.
    """
    cell = QWidget()
    # Named so the header's transparency rule reaches it: a plain QWidget in
    # this bar paints the base background over the raised one, which put a dark
    # square behind the mark.
    cell.setObjectName("HeaderCell")
    cell.setFixedWidth(BAR_HEIGHT)
    layout = QHBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(MarkBadge(height, tooltip=tooltip), 0,
                     Qt.AlignmentFlag.AlignCenter)
    return cell


class IconButton(QPushButton):
    """A button whose label is a shape and whose name is on hover.

    The tooltip is not optional and is checked: an icon nobody can name is a
    control nobody can use, and this application's actions -- re-run the capture,
    copy the command that produced this run -- are not ones a shape can carry on
    its own.
    """

    SIZE = 32
    #: How much of the square the shape fills. A fraction rather than a margin,
    #: because the square is the height of whatever bar it sits in -- 52 px in
    #: the report's header -- and a fixed inset would draw an oversized icon.
    GLYPH_FRACTION = 0.42

    def __init__(self, kind: str, tooltip: str, *, size: int | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not tooltip:
            raise ValueError(f"an icon button ({kind}) must say what it does")
        self.setObjectName("IconButton")
        self.kind = kind
        self.setToolTip(tooltip)
        self.setFixedSize(size or self.SIZE, size or self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._flash: str | None = None
        self._flash_timer: QTimer | None = None

    def flash(self, kind: str = "check", ms: int = 1400) -> None:
        """Show a different glyph for a moment, then go back.

        For actions that leave nothing behind to look at. Copying a command puts
        eight hundred characters on the clipboard and changes nothing on screen,
        so without this the only way to find out whether the button worked is to
        paste somewhere and see -- and the button, being a shape, cannot do what
        the worded one did and say "Copied".

        Re-entrant on purpose: pressing it again during the flash restarts the
        interval rather than stacking timers, so a second copy is confirmed too.
        """
        self._flash = kind
        self.update()
        if self._flash_timer is None:
            self._flash_timer = QTimer(self)
            self._flash_timer.setSingleShot(True)
            self._flash_timer.timeout.connect(self._end_flash)
        self._flash_timer.start(ms)

    def _end_flash(self) -> None:
        self._flash = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # The base class first, so the hairline, the hover and the pressed state
        # are the ones the theme gives every button of this kind.
        super().paintEvent(event)
        painter = QPainter(self)
        span = min(self.width(), self.height()) * self.GLYPH_FRACTION
        # Three states, not two. A disabled icon used to draw in exactly the
        # resting colour, which is invisible on a button whose whole job is to
        # say whether there is another chart to step to: both arrows of a
        # stepper looked live on the first of five and on the last.
        if not self.isEnabled():
            color = TEXT_SPENT
        elif self._flash or self.underMouse():
            color = TEXT
        else:
            color = TEXT_DIM
        draw_glyph(
            painter,
            QRectF((self.width() - span) / 2, (self.height() - span) / 2, span, span),
            self._flash or self.kind,
            color,
        )


class Stepper(QWidget):
    """One of several, and the way to the one before it and the one after.

    **Where a frame holds several charts, this is how the reader moves between
    their reports.** It replaced a combo box, and the argument for the change is
    what the control is now *for*: the combo was a way of jumping about inside a
    capture the reader had already reached through a list of its charts, and the
    list has gone, so this is the only route between them. A drop-down asks for
    a press, a read and a second press to see the next of five; two arrows and a
    count ask for one press and say where you are without being opened.

    **The label is a fixed width**, computed from the longest position it can
    ever show rather than from the one it is showing. Everything to the right of
    it is the bar's actions, and a reader stepping through five charts must not
    have the buttons shift under the pointer between chart 9 and chart 10.

    The ends are dead rather than wrapping. Wrapping saves a press on the last
    chart and costs the one thing the control is for, which is knowing without
    reading whether there is another one.
    """

    #: The index wanted. Never the current one -- the ends are disabled.
    stepped = Signal(int)

    def __init__(self, index: int, count: int, *, noun: str = "Target",
                 height: int = 24, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._count = count

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        self.previous = IconButton(
            "prev", f"Previous {noun.lower()}", size=height)
        self.previous.clicked.connect(lambda: self.stepped.emit(self._index - 1))
        row.addWidget(self.previous)

        self.label = QLabel(f"{noun} {index + 1} of {count}")
        self.label.setObjectName("StepperLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedWidth(
            QFontMetrics(self.label.font()).horizontalAdvance(
                f"{noun} {count} of {count}"
            ) + 10
        )
        row.addWidget(self.label)

        self.following = IconButton(
            "next", f"Next {noun.lower()}", size=height)
        self.following.clicked.connect(lambda: self.stepped.emit(self._index + 1))
        # No hairline on this one. ``IconButton`` carries a left border so that
        # a row of bar actions reads as a strip of cells, and inside this widget
        # that put a divider between the label and the arrow after it while
        # there was none between the label and the arrow before it -- so the
        # group read ``| < Target 5 of 5 | >``, with one of its own parts fenced
        # off. The two arrows belong to the label, and the hairlines belong at
        # the group's edges: ``previous`` keeps its own, and whatever action
        # follows in the bar supplies the one on the right.
        self.following.setStyleSheet("QPushButton#IconButton { border-left: none; }")
        row.addWidget(self.following)

        self.previous.setEnabled(index > 0)
        self.following.setEnabled(index < count - 1)


class VerdictStripe(Ruled):
    """A run's verdict as a coloured rule and a word.

    It was a filled chip in a row of filled chips, and it read as a button --
    which is the one thing a verdict must not do, because a person who thinks a
    verdict is pressable does not trust it as a statement. Nothing here is
    filled, so there is nothing to press.

    The **qualification is beside the word rather than inside it.** "PASS
    (ADVISORY)" is fifteen characters of which the first four are the answer,
    and the parenthesis buried the part that a reader most needs to see at a
    glance. :class:`~deltachrome_ui.status.Status` supplies both halves, so this
    parses nothing.

    It sat *under* the word until the report's header was shortened, and the two
    arrangements say the same thing: the answer at full weight, its qualification
    smaller and dimmer next to it, neither hidden. Reading across is what makes
    the widget one line tall, and in a horizontal bar the verdict was the tallest
    thing in it -- so the stacked form was setting the height of a header whose
    other two lines had nothing to do with it.
    """

    #: How tall the rule stands for a run-level verdict. Taller than the text
    #: needs, because this one is the answer and the others are asides.
    RULE_HEIGHT = 26

    def __init__(self, status: Status | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(gap=11, parent=parent)
        self.rule.setMinimumHeight(self.RULE_HEIGHT)

        # One row inside the column. ``Ruled`` stacks whatever it is given, and
        # what this needs is the two halves on one line -- so they go in a row of
        # their own rather than the base class growing a mode.
        line = QWidget()
        line.setObjectName("HeaderText")
        line.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(line)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.headline = QLabel()
        self.headline.setObjectName("VerdictHeadline")
        row.addWidget(self.headline)
        self.qualifier = QLabel()
        self.qualifier.setObjectName("VerdictQualifier")
        row.addWidget(self.qualifier)
        row.addStretch(1)
        self.body.addWidget(line)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        if status is not None:
            self.set_status(status)

    def set_status(self, status: Status) -> None:
        self.set_tone(status.tone)
        fg = self.tone_color()
        self.headline.setText(status.headline)
        self.headline.setStyleSheet(
            f"color: {fg}; font-size: 15px; font-weight: 700; letter-spacing: 0.5px;"
        )
        self.qualifier.setText(status.qualifier.lower())
        self.qualifier.setVisible(bool(status.qualifier))
        # The qualification carries the **qualified** tone, whatever tone the
        # verdict itself carries. That is the only part of this that is not
        # simply "colour the words".
        #
        # Losing the conformance claim demotes a PASS from the pass tone to this
        # one, and demotes nothing else: a FAIL and a FAIL that cannot be quoted
        # are the same red, an INCOMPLETE and an advisory INCOMPLETE the same
        # blue. Toning the *verdict* differently was the obvious fix and is the
        # wrong one -- softening a FAIL to amber would make an unquotable failure
        # look milder than a plain one, which is the single direction this
        # application must never move a reader in, and it would collide with the
        # advisory PASS that is already amber.
        #
        # Marking the qualifier instead keeps every verdict at its own severity
        # and gives "this cannot be quoted" one signature everywhere it appears.
        self.qualifier.setStyleSheet(
            f"color: {tone_colors(TONE_QUALIFIED)[0]}; font-size: 11px;"
        )
        # The **full label** leads the tooltip, not only the explanation. It is
        # the same string the engine's own ``verdict_line`` prints, and
        # :mod:`deltachrome_ui.status` keeps it that way on purpose so a figure
        # read off the screen and one read out of the terminal are the same
        # figure. Splitting it across two lines to make the verdict scannable
        # must not take it off the screen altogether.
        self.setToolTip(f"{status.label}\n\n{status.hint}")
        for part in (self.headline, self.qualifier):
            part.setToolTip(self.toolTip())
