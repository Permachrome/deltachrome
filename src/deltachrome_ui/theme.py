"""Colors, and the mapping from a status tone to one.

:mod:`deltachrome_ui.status` decides which tone a verdict gets; this decides what
a tone looks like. Keeping them apart is what stops "make the passes green" from
quietly becoming "make the advisory passes green too".

The palette is dark because the work is judging images, and a bright chrome around
a photograph biases how it reads. The qualified tone is deliberately *amber and
not green* -- it has to be legible at a glance as "this is not a clean pass",
since the run it most often marks is an ISA target scored on nominal values, which
is the common case rather than an oddity.
"""

from __future__ import annotations

from deltachrome_ui.status import (
    TONE_ABSENT,
    TONE_FAIL,
    TONE_PASS,
    TONE_QUALIFIED,
    TONE_UNKNOWN,
)

BG = "#1a1a1d"
BG_RAISED = "#232327"
BG_INPUT = "#141416"
BORDER = "#33333a"
TEXT = "#e6e6ea"
TEXT_DIM = "#9a9aa4"
#: A control that is there and cannot be used. Between the dim text and the
#: raised background it sits on, so it reads as present and spent rather than
#: as one more thing to try -- the first arrow of a stepper on the first of
#: five charts, which must not look the same as the one beside it that works.
TEXT_SPENT = "#54545e"
ACCENT = "#ffd600"

#: Deltachrome's own ramp, top to bottom, from the mark's artwork.
#:
#: Two facts about it are worth stating where the constants live. The warm stop
#: is :data:`ACCENT` to within one unit of green -- the button that already
#: shipped *is* the bottom of this gradient, which is why nothing about the
#: accent had to move to adopt the brand. And the cool stop is a red close
#: enough to :data:`TONE_FAIL` that the ramp cannot be used as a fill on
#: anything a reader might take for a status: an element that begins at
#: ``#fb2929`` in an application which paints failure at ``#ff6050`` is saying
#: something it does not mean.
#:
#: So the rule this pair travels with: **the ramp belongs to the mark, and to
#: motion.** The mark, the application icon, and the indeterminate bar -- which
#: is neither a control nor a reading, sweeps rather than fills, and is the one
#: place in the interface where nothing is being judged. Not a button, not a
#: chart, not a chip, not a rule under a verdict.
RAMP = ("#fb2929", "#ffd601")

#: What the mark's delta is drawn in. A token rather than :data:`TEXT` directly,
#: because the delta follows the *surface* -- it is the ink on a dark panel and
#: would be the ground on a light one -- and only the chi is ever the ramp.
MARK_INK = TEXT

#: Width reserved at a combo box's right edge for the chevron :class:`~deltachrome_ui.widgets.ComboBox` paints there.
ARROW_ZONE = 26

#: The height of a screen's header bar, and therefore the size of every action
#: square in it: the squares are the bar, top to bottom, rather than buttons
#: sitting inside it. Shared, because the report and the library are the same
#: bar with different contents and a reader moving between them should not see
#: the chrome change height.
BAR_HEIGHT = 52

#: A floor under the verdict block, so the filename beside it starts in the same
#: place on every run.
#:
#: The block is sized to its own text, and the text is a verdict: ``FAIL`` is
#: 47 px and ``INCOMPLETE advisory`` is 168, so the whole middle of the header
#: shifted by up to 126 px according to what the answer turned out to be --
#: which is the one thing that should never move a layout, since a reader
#: comparing two reports is comparing exactly the runs whose verdicts differ.
#: Dropping the qualification was tried and only halves it: the verdicts are not
#: the same length as each other either, and ``INCOMPLETE`` against ``FAIL`` is
#: still 68 px.
#:
#: A floor rather than a fixed width. Nothing here should be able to clip a
#: verdict, so a longer one than this version knows about pushes as it used to
#: rather than being cut off.
VERDICT_WIDTH = 176

TONE_COLORS = {
    TONE_PASS: ("#7ee08a", "#1d3524"),
    TONE_FAIL: ("#ff6050", "#3a1f1c"),
    TONE_QUALIFIED: ("#ffc44d", "#3a2f18"),
    TONE_ABSENT: ("#8a8a94", "#26262b"),
    TONE_UNKNOWN: ("#8fb8ff", "#1e2738"),
}


def tone_colors(tone: str) -> tuple[str, str]:
    """Foreground and background for a status badge."""
    return TONE_COLORS.get(tone, TONE_COLORS[TONE_UNKNOWN])


STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}
/* Labels sit on cards and on the canvas, so they must not paint the base
   background over whatever is behind them. Badges and the placement warning set
   their own background directly on the widget, which wins over this. */
QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 19px; font-weight: 600; }}
QLabel#Subtitle {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#SectionHeading {{
    color: {TEXT_DIM}; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
}}
QLabel#Caveat {{ color: {TEXT_DIM}; font-size: 11px; }}
/* Which chart of a capture is on screen. Not dim: it is the position
   of what is being read, which is the one thing the two arrows beside
   it cannot say. */
QLabel#StepperLabel {{ color: {TEXT}; font-size: 12px; }}
QFrame#Card {{
    background: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#Divider {{ background: {BORDER}; max-height: 1px; border: none; }}
/* The waiting bar. Styled explicitly because the platform's own is a chunky
   raised widget with a groove, which reads as a form control on a screen whose
   only job is to say that something is still happening. */
QProgressBar {{
    background: {BORDER};
    border: none;
    border-radius: 2px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {RAMP[0]}, stop:1 {RAMP[1]});
    border-radius: 2px;
}}
QPushButton {{
    background: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover {{ border-color: #4a4a55; }}
QPushButton:pressed {{ background: #2b2b31; }}
QPushButton:disabled {{ color: #5a5a63; }}
/* A segmented group: buttons butted into one strip, sharing the hairline
   between them, with only the outer corners rounded. The same idea as the
   header's action squares -- several controls that belong to one question read
   as one object -- but these keep their words, because "Fit" and "1:1" are
   shorter as text than any icon is as a picture. */
QPushButton#SegmentLeft {{
    border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 7px 10px;
}}
QPushButton#SegmentMiddle {{
    border-radius: 0; border-left: none; padding: 7px 10px;
}}
QPushButton#SegmentRight {{
    border-top-left-radius: 0; border-bottom-left-radius: 0;
    border-left: none; padding: 7px 10px;
}}
QPushButton#Primary {{
    background: {ACCENT}; color: #1a1a1d; border: none; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #ffe14d; }}
QPushButton#Primary:disabled {{ background: #4a452a; color: #8a8578; }}
QLineEdit, QComboBox {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    selection-color: #1a1a1d;
}}
/* Unstyled, Qt draws a square native button hard against the right edge, whose
   corners then sit outside the 6px radius above as two dark notches. Styling it
   is what removes them; the inset is so the arrow is not touching the border. */
QComboBox {{ padding-right: {ARROW_ZONE}px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: {ARROW_ZONE}px;
    border: none;
    background: transparent;
}}
QComboBox QAbstractItemView {{
    background: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    outline: none;
    selection-background-color: {ACCENT};
    selection-color: #1a1a1d;
}}
/* The items in it. Without a height the row is sized from the font alone and
   clips a descender; without padding the text sits against the rounded edge. */
QComboBox QAbstractItemView::item {{ min-height: 22px; padding: 2px 8px; }}
/* The run library. Dense on purpose: a year of records is a thousand rows, and
   the row height is what decides how many of them a person can see at once. */
QTreeView#RunTable, QTableView#DataTable {{
    background: {BG};
    alternate-background-color: #1e1e22;
    gridline-color: #292930;
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
    selection-background-color: #2f2f38;
    selection-color: {TEXT};
}}
QTreeView#RunTable::item, QTableView#DataTable::item {{
    padding: 0px 8px; border: none;
}}
/* The library is a QTreeView, because a batch's records live under one row that
   opens -- and a tree draws no grid of its own, where a table does. The rules
   between columns are put back on the cell, and the row height with them: a
   tree has no vertical header to set a default section size on, and the row
   height is what decides how many of a year's records a person sees at once. */
QTreeView#RunTable::item {{
    border-right: 1px solid #292930;
    border-bottom: 1px solid #292930;
    height: 29px;
}}
QTreeView#RunTable::branch {{ background: transparent; border-image: none;
    image: none; }}
QTreeView#RunTable::item:hover, QTableView#DataTable::item:hover {{
    background: #26262c;
}}
QTreeView#RunTable::item:selected, QTableView#DataTable::item:selected {{
    background: #2f2f38; color: {TEXT};
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: {BG_RAISED};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid #292930;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QHeaderView::section:hover {{ color: {TEXT}; }}
/* Once a style sheet claims ``section``, Qt only paints a sort indicator it was
   given as an image -- the same trap as the combo box's chevron. The header
   paints its own; see :class:`~deltachrome_ui.run_table.SortHeader`. */
QHeaderView::up-arrow, QHeaderView::down-arrow {{ image: none; width: 0; height: 0; }}
QTableCornerButton::section {{ background: {BG_RAISED}; border: none; }}
QMenu {{
    background: {BG_RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: #33333d; }}
QMenu::item:disabled {{ color: #5a5a63; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: #3d3d46; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:horizontal {{
    background: #3d3d46; border-radius: 5px; min-width: 30px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: #4d4d58;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
/* The report. The left list chooses the metric; a tab chooses which reading of
   it. Both are selections and both read as one. */
/* The report's header. Raised, and separated from the report by a stroke --
   which this rule has always said and never drew: a plain QWidget ignores a
   style sheet's background and border unless it is given WA_StyledBackground,
   so the header rendered in the base colour with no edge. Set where the widget
   is built; the rule was never the missing part. */
QWidget#HeaderBar {{ background: {BG_RAISED}; border-bottom: 1px solid {BORDER}; }}
/* Its own columns. Transparent, because a plain QWidget otherwise paints the
   base background over the raised header -- a dark box around the filename, or
   around the mark, sized by whatever the layout gave it. Scoped by name so it
   does not cascade into the combo boxes inside it, which do want a background. */
QWidget#HeaderText, QWidget#HeaderCell {{ background: transparent; }}
/* The header's own combo boxes, and only those. A combo is 18px of text inside
   6px of padding and a 1px border, which is 32 -- and two of those stacked under
   a filename are most of the header's height. Two pixels of padding gives 24,
   which still clears the text, and the list the box opens is unaffected: it is
   the closed box that was setting the bar's height. Scoped by the bar rather
   than set on the widget, so the report's picker and the placement screen's copy
   of the same control stay the one class. */
QWidget#HeaderBar QComboBox {{ padding: 2px 9px; padding-right: {ARROW_ZONE}px; }}
/* A worded button on the header's second line sits where those combos sit, and
   at the base 7px of padding it was 32px against their 24 -- flush with the
   bar's bottom edge. Named rather than scoped by the bar, so the bar's own
   squares and its primary block are untouched. */
QPushButton#HeaderButton {{ padding: 2px 10px; }}
QLabel#VerdictHeadline {{ font-size: 15px; font-weight: 700; letter-spacing: 0.5px; }}
QLabel#VerdictQualifier {{ color: {TEXT_DIM}; font-size: 11px; }}
/* Every status in the application: a rule in the tone, words in the tone,
   nothing filled. The colour is set per widget because it comes from the tone;
   the type is set here because it does not. */
QLabel#ToneWord {{ font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }}
QLabel#ToneParagraph {{ font-size: 12px; }}
/* The report header's actions. Square, the full height of the bar, butted
   against each other with a hairline between and the last one flush against the
   window -- so the bar's own edge is the target rather than a rounded button
   floating twenty pixels inside it. */
QPushButton#IconButton {{
    background: transparent;
    border: none;
    border-left: 1px solid {BORDER};
    border-radius: 0;
    padding: 0;
}}
QPushButton#IconButton:hover {{ background: #30303a; }}
QPushButton#IconButton:pressed {{ background: #3a3a44; }}
/* The primary action, in the corner. The same accent as QPushButton#Primary
   and none of its geometry: an edge-to-edge block has no radius and no margin. */
QPushButton#PrimaryBar {{
    background: {ACCENT}; color: #1a1a1d; border: none; border-radius: 0;
    font-weight: 600; padding: 0 20px;
}}
QPushButton#PrimaryBar:hover {{ background: #ffe14d; }}
QPushButton#PrimaryBar:disabled {{ background: #4a452a; color: #8a8578; }}
/* A bar action that carries a NAME rather than a glyph, and is not the primary
   one. The icon squares beside it are for actions a shape can say -- refresh,
   settings, a chart's grid; "Batch Analysis" is not one of those, and it commits
   something, which is the test the accent block's own three actions pass. Same
   geometry as the block and none of its colour, so a bar has one accent in it. */
QPushButton#SecondaryBar {{
    background: transparent; color: {TEXT}; border: none; border-radius: 0;
    border-left: 1px solid {BORDER}; padding: 0 18px;
}}
QPushButton#SecondaryBar:hover {{ background: #30303a; }}
QPushButton#SecondaryBar:pressed {{ background: #3a3a44; }}
QPushButton#SecondaryBar:disabled {{ color: {TEXT_DIM}; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM}; padding: 6px 14px;
    border: none; border-bottom: 2px solid transparent; font-size: 12px;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QListWidget#Nav {{
    background: {BG_RAISED}; border-right: 1px solid {BORDER}; outline: none;
    font-size: 12px; padding: 6px 0px;
}}
QListWidget#Nav::item {{ padding: 2px 12px; border: none; }}
QListWidget#Nav::item:hover {{ background: #2a2a30; }}
QListWidget#Nav::item:selected {{
    background: {BG}; color: {TEXT}; border-left: 2px solid {ACCENT};
}}
/* The settings nav. The report's own Nav is a long tree of metric names where a
   tight row is what lets a family and its members read as one block; three
   sections is a short list of destinations, and the same 2 px of padding makes
   them look like an accident. */
/* The Guideline Builder's rows. Every control on them is 26 px so a table of
   twenty metrics fits on a screen, and the base padding of 6-7 px leaves 12 px
   of that for text -- which clips a descender off "Judged", the tail off every
   comma (so a list reads as though it were separated by full stops) and the
   underscore off "ECIRGB_v2". Measured on the render, not argued about. The
   header bar already does exactly this, for exactly this reason. */
QWidget#MetricRow QComboBox {{ padding: 2px 9px; padding-right: {ARROW_ZONE}px; }}
QWidget#MetricRow QLineEdit {{ padding: 2px 9px; }}
QWidget#MetricRow QPushButton {{ padding: 2px 10px; }}
QListWidget#SettingsNav {{
    background: {BG_RAISED}; border-right: 1px solid {BORDER}; outline: none;
    font-size: 13px; padding: 10px 0px;
}}
QListWidget#SettingsNav::item {{ padding: 8px 16px; border: none; }}
QListWidget#SettingsNav::item:hover {{ background: #2a2a30; }}
QListWidget#SettingsNav::item:selected {{
    background: {BG}; color: {TEXT}; border-left: 2px solid {ACCENT};
}}
QListWidget#BankList {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 4px;
    outline: none; font-size: 12px; padding: 4px 0px;
}}
QListWidget#BankList::item {{ padding: 4px 10px; border: none; }}
QListWidget#BankList::item:hover {{ background: #2a2a30; }}
QListWidget#BankList::item:selected {{
    background: {BG_RAISED}; color: {TEXT}; border-left: 2px solid {ACCENT};
}}
QPushButton#DockTab {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 4px;
    color: {TEXT_DIM}; font-size: 14px; padding: 0;
}}
QPushButton#DockTab:hover {{ color: {TEXT}; border-color: #4a4a55; }}
QToolTip {{
    background: #2b2b31; color: {TEXT};
    border: 1px solid {BORDER}; padding: 6px 8px; border-radius: 4px;
}}
"""



