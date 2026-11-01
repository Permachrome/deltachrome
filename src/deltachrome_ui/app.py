"""Application shell.

The shell is the library's, not a document window's. That is the point: batch
analysis and per-device time series are roadmap phase 3, and the difference
between adding them to this and adding them to a single-image window is a
rewrite.

**Setting up a run is one screen.** It used to be two -- choose the target, the
profile and the reference; then place the chart -- and the split was load-bearing
only while placement needed to be told which chart it was placing. Identification
places the chart on the way to naming it, so the corners exist before the first
screen would have been drawn, and the choices moved onto the canvas screen where
their consequences are visible.

What this shell still owns is everything the panel cannot: the decoded capture,
the worker threads, and therefore the two searches. A target chosen in the panel
comes back here as a request to find that chart, and the answer goes back to the
panel -- which is the same shape the flow always had, minus a screen.

**There is no explanation panel.** Every screen used to carry small ⓘ buttons
that opened a 330-pixel column of standing prose beside it. All of it was true
and none of it was needed twice, which was the argument for putting it behind an
icon -- but the argument for removing it outright is a different one and it is
about finding out what the application actually communicates. A control that
explains itself cannot be shown to be unclear: the operator opens the panel, the
confusion is resolved, and nothing is learned about the screen. So the standing
prose belongs in documentation kept apart from the application, and what the
application shows is what it can say for itself. The text is not lost -- it lives
in :mod:`deltachrome_ui.explanations`, which nothing imports, as the source for
that documentation.

What stayed on screen is unchanged and is the reason the distinction was drawn
in the first place: run-specific qualification. Why a metric could not be
finished, that a run was judged against nominal values, which patch was clipped.
That is part of a result rather than an explanation of one, and it was never
behind the icon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deltachrome import __version__
from deltachrome_ui.imageproxy import ImageProxy
from deltachrome_ui.library import RunSummary
from deltachrome_ui.library_view import LibraryView
from deltachrome_ui.paths import (
    batch_dir,
    ensure_run_dir,
    identification_candidates,
    restore_guideline_dir,
)
from deltachrome_ui.placement_view import PlacementView
from deltachrome_ui.profile_view import ProfileView
from deltachrome_ui.result_view import ResultView
from deltachrome_ui.batch_view import BatchView
from deltachrome_ui.session_view import SessionView
from deltachrome_ui.settings_view import SettingsView
from deltachrome_ui.mark import app_icon
from deltachrome_ui.theme import BG_INPUT, MARK_INK, RAMP, STYLESHEET
from deltachrome_ui.widgets import BusyStrip, MarkBadge
from deltachrome_ui.workers import (
    AnalysisRunner,
    BatchRunner,
    ChartsRunner,
    ImageLoader,
    LoadFailure,
    PlacementDetector,
    ProfileBuilder,
    ProfileGatherer,
    RecordEstablisher,
    Rescorer,
    TargetIdentifier,
    run_in_thread,
)

#: Formats the engine can read. JPEG 2000 is first-class here, not an afterthought:
#: it is a preservation master format across the sector.
IMAGE_FILTER = (
    "Master images (*.tif *.tiff *.jp2 *.jpf *.jpx *.j2k *.jpc);;All files (*)"
)

#: The export dialog's format menu, each entry with the extension it writes.
EXPORT_FILTERS = {
    "CSV (*.csv)": ".csv",
    "Text (*.txt)": ".txt",
    "XML (*.xml)": ".xml",
}


def export_path(path: str | Path, chosen_filter: str) -> Path:
    """The file an export writes, given the format chosen and the name returned.

    **The format is chosen before the dialog, never in it.** A format menu inside
    the macOS save panel does not change the type the panel itself is saving:
    with *Text* chosen it still hid a ``.csv`` extension and asked to replace
    ``name.csv``, before the application saw any name at all. So the dialog is
    offered the one format already chosen, and this only makes the name agree
    with it -- a suffix naming another export format is replaced, any other
    suffix is part of the name and the format's is appended.
    """
    from deltachrome.reporting.export import FORMATS

    path = Path(path)
    suffix = EXPORT_FILTERS.get(chosen_filter)
    if suffix is None:
        return path if path.suffix.lower() in FORMATS else path.with_name(path.name + ".csv")
    if path.suffix.lower() in FORMATS:
        return path.with_suffix(suffix)
    return path.with_name(path.name + suffix)


class LoadingView(QWidget):
    """Shown while a master decodes, or while the chart in it is worked out.

    Not a formality. The ColorChecker SG sample in this repository takes over
    twenty seconds to decode, because JP2 is compressed, so an unexplained frozen
    window would be the normal experience of opening one.

    It is a :class:`~deltachrome_ui.widgets.BusyStrip` centred in the window, and
    the argument for the strip is written there. What is left here is the
    centring and the fact that the message is *short*: it used to carry the
    capture's filename, which on a real digitization filename wrapped to two
    lines and overlapped what was under it.
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(22)
        # The one screen with nothing on it, and the longest anybody waits --
        # this repository's ColorChecker SG master takes over twenty seconds to
        # decode. A mark costs nothing here because there is nothing for it to
        # compete with, and it is the only moment the application gets to say
        # what it is without taking room from something a reader needs.
        layout.addWidget(MarkBadge(40), 0, Qt.AlignmentFlag.AlignHCenter)
        self.strip = BusyStrip("Loading\u2026", font_px=15)
        layout.addWidget(self.strip, 0, Qt.AlignmentFlag.AlignHCenter)

    def set_message(self, text: str) -> None:
        self.strip.set_message(text)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Deltachrome {__version__}")
        self.resize(1360, 880)
        self._pending_corners: list[tuple[float, float]] | None = None
        #: Every chart of a saved capture being re-run, as ``RunSummary.charts``.
        #: Restored into the frame's list of charts rather than onto the canvas
        #: alone, so a region search afterwards adds to the frame instead of
        #: replacing the one chart the canvas was holding.
        self._pending_charts: tuple = ()
        #: The folder and captures a batch is being set up for, held while the
        #: operator checks the placement on its first capture. ``None`` for every
        #: ordinary run, which is what the primary action branches on.
        self._pending_batch: tuple[Path, list[Path]] | None = None
        #: The worker running a batch, held so Stop can reach it.
        self._batch_worker = None
        self._proxy = None
        self._image_path: Path | None = None
        #: What an automatic placement did, ready to travel into the record. A
        #: placement nobody chose has to be visible in the archive, which is the
        #: only thing a reader a year later will have.
        self._placement_notes: list[str] = []
        #: What the capture was identified as, if it was. Held so the placement it
        #: already made can be reused, and so the record can say that nobody named
        #: the chart.
        self._identified = None
        #: The targets the last search chose among, or ``None`` for every one.
        self._searched: list[str] | None = None
        self._last_region = None
        #: The codes read off this capture, when the bank held anything to pair
        #: them with. Read once, in the identification pass, and handed both to
        #: the setup screen -- which says which file the run will use -- and to
        #: the run itself, so the two cannot be looking at different scans.
        self._scan = None
        #: Every placement the current capture supports, in reading order. Usually
        #: one; four when the frame is a board of QA-62 cards. Held here rather
        #: than on the placement screen because the shell owns the decoded capture
        #: and the search, and the screen shows one placement at a time.
        self._detections: list = []
        #: The target each of those placements is a placement OF, where the charts
        #: were identified individually. Empty when one target was named for the
        #: whole frame, which is the ordinary case.
        self._chart_targets: list = []
        #: Why a region that held a chart was not offered as one. Travels into the
        #: run's warnings, because a chart nobody measured has to be visible in the
        #: archive rather than only on a screen somebody has closed.
        self._unnamed: list[str] = []
        #: Why the chart at each offered position is uncertain, where it is. Parallel
        #: to ``_detections``; ``None`` where identification named the chart outright.
        self._chart_doubts: list = []
        #: Whether the search named the chart at each position, by identification
        #: or in a drawn region, rather than the operator choosing it. Parallel to
        #: ``_detections``. What the target chip says when that chart is shown.
        self._chart_named: list[bool] = []
        #: Charts left out of the run, by position. A region that is not a chart, or
        #: one the operator does not want measured this time. Ignored rather than
        #: deleted, so the list still shows what the search found and the undo is the
        #: same press again.
        #: How many targets the operator took out of this frame. Kept as a count
        #: rather than as the placements themselves, because what the record has
        #: to say is that the capture held more than was measured -- see
        #: :meth:`_drop_chart`.
        self._dropped_charts = 0
        #: Which chart the canvas is currently showing, so its corners can be taken
        #: back off the canvas before the screen moves on. Without this every drag on
        #: a multi-chart frame was discarded: the run reads the stored placements,
        #: and nothing wrote a correction back into them.
        self._shown_chart: int | None = None
        #: Charts whose handles have been moved since they were placed. Per chart,
        #: because the view's own flag is about whatever is on screen and a record
        #: must say, of ITS chart, whether a person put the corners there.
        self._chart_dragged: list[bool] = []
        #: Charts the operator placed rather than the search finding them. Kept
        #: because the record must say so, and because removing one is meaningful
        #: where removing a detected chart is not.
        self._chart_added: list[bool] = []
        #: Results from a board just measured, by run id. Held because the session
        #: they were combined into can be walked chart by chart before any of them
        #: has been saved, and the library cannot answer for a run it has not seen.
        self._fresh_results: dict = {}
        #: The decoded capture ``_fresh_results`` were read from, and whether it
        #: was placed on this session's placement screen. Usually that screen's
        #: own capture; after *Change Guidelines…* on a saved run it is not, and every
        #: redraw used to hand the report the placement screen's pixels anyway.
        self._live_proxy = None
        self._placed_here = True
        #: Charts of the capture on screen that could not be measured, as
        #: sentences. Kept because they are evidence about the frame: a run
        #: that quietly measured four of five charts would make the fifth
        #: read as absent rather than as undecided.
        self._chart_failures: list = []
        #: The library row the report on screen was opened from, so its chart
        #: selector knows which record to ask for another chart.
        self._open_record = None
        #: Where the last export went, so the next one opens there. A department
        #: exporting a day's captures puts them in one folder, and the home
        #: directory every time is a navigation per file.
        self._export_dir = Path.home()
        #: And in what format, so the menu opens on it.
        self._export_format = next(iter(EXPORT_FILTERS))
        #: The raw chart records "Change Guidelines…" is currently reading.
        #: Held between the decode and the analysis worker, which are two
        #: threads, so the second knows what the first was for.
        self._establishing_charts: list = []
        #: The proxy "Change Guidelines…" decoded, held only long enough to hand to
        #: the report once the reading it started has come back.
        self._establishing_proxy = None
        #: Whether the report on screen was opened from a session, so that going
        #: back returns there rather than to the library.
        self._came_from_session = False
        #: Which capture the frame map is currently waiting on a thumbnail for.
        #: Declared rather than left to ``getattr``: a reader can open one
        #: capture and another inside six tenths of a second, and the first
        #: read must not land on the second one's map.
        self._thumbnail_key: str | None = None
        #: Every chart identification the frame yielded, held between the search and
        #: the placement screen being drawn.
        self._pending_identifications: list = []
        self._pending_setup: tuple[str | None, str | None] = (None, None)
        #: The last capture decoded for a *report's* panel, and where it came
        #: from. One slot, separate from ``_proxy`` above, which is the capture
        #: being set up or just analyzed -- the two are different files whenever
        #: a saved run is opened while another capture is placed, and folding
        #: them together would let a report's decode become the image the next
        #: analysis measured.
        #:
        #: **The ceiling is therefore two decoded masters.** Measured on this
        #: repository's samples, one costs 175 MB (61 Mpx 8-bit) to 577 MB
        #: (101 Mpx 16-bit) plus 11 MB of overview, against 16.5 s and 25.1 s to
        #: decode the two JPEG 2000 files again. Keeping one is worth it many
        #: times over; keeping a history of them is how a desktop application
        #: ends up holding several gigabytes it will never look at again.
        self._capture_proxy = None
        self._capture_path: Path | None = None

        # The stack fills the window. There is no explanation panel beside it
        # any more -- see the module docstring.
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.library = LibraryView()
        self.library.settingsRequested.connect(self.show_settings)
        self.library.newAnalysisRequested.connect(self.open_image)
        self.library.batchRequested.connect(self._choose_batch)
        self.library.reuseRequested.connect(self._reuse_run)
        self.library.openRequested.connect(self._open_from_library)
        self.library.exportRequested.connect(self._export_from_library)

        self.loading = LoadingView()

        self.placement = PlacementView()
        self.placement.backRequested.connect(self.show_library)
        self.placement.continueRequested.connect(self._run_analysis)
        self.placement.targetChanged.connect(self._on_target_chosen)
        self.placement.chartSelected.connect(self._select_chart)
        self.placement.chartAdded.connect(self._add_chart)
        self.placement.chartDropped.connect(self._drop_chart)
        self.placement.regionSearched.connect(self._detect_in_region)
        self.placement.profileRequested.connect(self._open_profile)

        self.profile = ProfileView()
        self.profile.backRequested.connect(self.show_placement)
        self.profile.buildRequested.connect(self._build_profile)

        self.result = ResultView()
        self.result.backRequested.connect(self._leave_report)
        self.result.saveRequested.connect(self._save_result)
        self.result.rerunRequested.connect(self._rerun_saved)
        self.result.chartRequested.connect(self._show_chart)
        self.result.captureRequested.connect(self._load_capture_for_report)
        self.result.placementRequested.connect(self.show_placement)
        self.result.rescoreRequested.connect(self._rescore)
        self.result.establishRequested.connect(self._establish_measurement)
        self.result.sessionRequested.connect(self._show_combined)
        self.result.exportRequested.connect(
            lambda: self._export_capture(self.result.export_button))

        self.batch = BatchView()
        self.batch.backRequested.connect(self.show_library)
        self.batch.stopRequested.connect(self._stop_batch)
        self.batch.libraryRequested.connect(self._batch_finished)

        self.session = SessionView()
        self.session.backRequested.connect(self.show_library)
        self.session.saveCaptureRequested.connect(self._save_capture)
        self.session.captureRequested.connect(self._open_run_by_id)
        self.session.exportRequested.connect(
            lambda: self._export_capture(self.session.export_button))

        # Built once and kept, like every other screen here: the reference bank
        # indexes a folder to draw itself, and rebuilding the screen on each
        # visit would re-read every file in it to show a list that has not
        # changed.
        self.settings = SettingsView()
        self.settings.backRequested.connect(self.show_library)
        self.settings.runDirChanged.connect(self.library.reload)
        # The placement panel decides whether pairing is offerable by asking
        # whether the bank holds anything, so a file added here changes a screen
        # that is not on top.
        self.settings.referencesChanged.connect(self.placement.choices.refresh_bank)
        # And a guideline saved here has to reach the control that chooses one.
        # The placement panel's picker is built once and lives for the session,
        # so without this a guideline would not appear until a restart. The
        # report builds its own picker per report and needs no telling.
        self.settings.guidelinesChanged.connect(self.placement.choices.picker.reload)

        for widget in (self.library, self.loading, self.placement, self.profile,
                       self.result, self.session, self.settings, self.batch):
            self.stack.addWidget(widget)
        self.show_library()

    def closeEvent(self, event) -> None:
        """Let running work finish before the window and its threads are destroyed.

        A QThread parented to this window is deleted with it, and deleting a
        running QThread takes the process down. Closing mid-analysis is an
        entirely ordinary thing to do, so it must not be a crash.
        """
        for running in list(getattr(self, "_threads", {}).values()):
            running.thread.quit()
            running.thread.wait(5000)
        super().closeEvent(event)

    # -- navigation --------------------------------------------------------

    def show_settings(self) -> None:
        self.settings.refresh()
        self.stack.setCurrentWidget(self.settings)

    def show_library(self) -> None:
        # Leaving the placement screen abandons a batch that was being set up on
        # it. Held here rather than in the screen because the folder is the shell's
        # and a stale one would make the next ordinary run measure a hundred files.
        if self._batch_worker is None:
            self._pending_batch = None
            self.placement.set_batch(0)
        # Whatever session a report belonged to, it is behind us now. Left set, a
        # run opened later from the library would offer Back to a session of some
        # other capture -- which is a screen the reader never saw from here.
        self._came_from_session = False
        self.library.reload()
        self.stack.setCurrentWidget(self.library)

    def open_image(self, seed_corners=None) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a capture", str(Path.home()), IMAGE_FILTER
        )
        if not path:
            return
        self._pending_corners = seed_corners
        if seed_corners is None:
            # A fresh capture, not a saved run's: nothing of a previous re-run's
            # charts may carry into it.
            self._pending_charts = ()
        self.load_image(Path(path))

    def _open_run(
        self, run: RunSummary, chart: int = 0, *, focus: tuple | None = None,
        keep_view: bool = False,
    ) -> None:
        """Show a saved capture's report, one chart at a time.

        Without decoding the capture. The record holds every per-patch value and
        every edge, so the whole report renders from it -- what is missing is the
        pixels behind a mark, and the detail view says so rather than pretending
        the measurement is incomplete. Decoding a 300 MB master to read a number
        that is already on disk would make the common action the slow one.
        """
        from deltachrome_ui.library import load_result

        try:
            result = load_result(run.path, chart)
        except (OSError, ValueError) as exc:
            self._on_load_failed(LoadFailure(
                "That run could not be opened.",
                f"{run.path.name} could not be read: {exc}",
            ))
            return
        # Which record this came from, so the chart selector can ask it for another.
        # Held rather than passed through the signal because the signal says only
        # which chart is wanted; where to get it from is the screen's own state.
        self._open_record = run
        # Another chart of the capture on screen keeps the capture's pixels when
        # they are in memory -- matched on the fingerprint, as everywhere. Opening
        # a run from the library still decodes nothing.
        self.result.set_proxy(
            self._already_decoded(result.run.image or {}) if keep_view else None
        )
        self.result.set_result(
            result, saved=True, chart=(result.chart_index, result.chart_count),
            focus=focus, keep_view=keep_view,
        )
        self.stack.setCurrentWidget(self.result)

    def _open_from_library(self, run: RunSummary) -> None:
        """A row opened from the table, landing where the record's own shape says.

        A capture holding several targets lands on **All Targets**, which is
        where a board's question is actually asked -- did the system pass at
        every position -- and from which each target's own report is one click
        on its outline. A single-target record lands on its report exactly as
        before, there being nothing to combine.

        The chart's report is opened first regardless, because that is what sets
        ``_open_record``, and the combined screen reads the charts out of it.
        """
        self._open_run(run)
        if run.chart_count > 1:
            self._show_combined()

    def _open_run_by_id(self, run_id: str) -> None:
        """Open one capture's own report from inside a session.

        A session combines the runs without altering any of them, so each one's
        report is still the authority on what that chart measured.

        A board just measured is looked up here first: its runs are held in memory
        and have not been saved, so the library cannot answer for them yet.
        """
        fresh = list(self._fresh_results.values())
        ids = list(self._fresh_results)
        if run_id in self._fresh_results:
            # Not a saved record, so the chart selector must ask this list rather
            # than a file. Cleared explicitly: a stale row here after a new capture
            # was measured would send the selector to the previous one's file.
            self._open_record = None
            # Reached by clicking a capture inside the session, so Back belongs to
            # the session. A multi-chart run no longer lands there, so this is the
            # only way into a report that does -- and it has to be set here rather
            # than when the run finishes, or every report would offer a way back to
            # a screen the reader has not seen.
            self._came_from_session = True
            self.result.set_proxy(self._report_proxy())
            self.result.set_result(
                fresh[ids.index(run_id)], saved=False,
                chart=(ids.index(run_id), len(fresh)),
                placed_here=self._placed_here,
            )
            self.stack.setCurrentWidget(self.result)
            return
        # A chart of the capture already open. The library lists *captures*, so
        # only the first chart's id is ever in it -- a board of four cards has
        # three ids that exist, are asked for, and are nowhere in that list. The
        # record holding them is right here, and addressing a chart inside it is
        # by index, which is how the report's own stepper reaches them.
        index = self._chart_of_open_record(run_id)
        if index is not None:
            self._open_run(self._open_record, index)
            self._came_from_session = True
            return
        for run in self.library.runs:
            if run.run_id == run_id:
                self._open_run(run)
                self._came_from_session = True
                return
        QMessageBox.information(
            self, "Run not found",
            "That capture's record is no longer in the library.",
        )

    def _chart_of_open_record(self, run_id: str) -> int | None:
        """Which chart of the open capture record carries this run id."""
        from deltachrome_ui.library import read_record

        if self._open_record is None:
            return None
        try:
            charts = read_record(self._open_record.path).get("charts") or []
        except (OSError, ValueError):
            return None
        for index, chart in enumerate(charts):
            if str(chart.get("run_id") or "") == run_id:
                return index
        return None

    def _reuse_run(self, run: RunSummary) -> None:
        """Start a new placement from a saved run's corners.

        The seeded-state design paying off early: the canvas does not care that
        these four points came out of a JSON record rather than a default
        rectangle, and it will not care when they come out of a detector.
        """
        # A capture holding charts of two models states no one target at its top
        # level -- ``target_id`` is ``None`` -- and that read as "target unknown":
        # the capture was identified from scratch and its saved placements were
        # set aside. Its first chart's target is the one the screen opens on.
        self._pending_charts = tuple(run.charts)
        self._pending_setup = (
            run.target_id or (run.charts[0][0] if run.charts else None),
            run.profile_id,
        )
        if run.image_path and Path(run.image_path).exists():
            self._pending_corners = run.corners
            self.load_image(Path(run.image_path))
            return
        self._pending_corners = run.corners
        QMessageBox.information(
            self,
            "Choose the capture",
            f"The image this run recorded is not where it was:\n\n{run.image_path}"
            "\n\nChoose it and its corner placement will be restored.",
        )
        self.open_image(seed_corners=run.corners)

    # -- loading -----------------------------------------------------------

    def load_image(self, path: Path) -> None:
        self.loading.set_message("Loading File\u2026")
        self.stack.setCurrentWidget(self.loading)

        loader = ImageLoader(path)
        loader.progress.connect(self.loading.set_message)
        loader.loaded.connect(self._on_loaded)
        loader.failed.connect(self._on_load_failed)
        run_in_thread(loader, self)

    def _on_loaded(self, proxy: ImageProxy, path: Path) -> None:
        """Work out which chart this is, then show it placed.

        Identification first, because everything the panel offers is downstream
        of the target: which corners the handles are named for, whether a
        reference file is required, which run options a profile can use. Naming
        the chart before the screen is drawn means the panel opens already
        answering rather than asking.
        """
        self._proxy = proxy
        self._image_path = path
        self._identified = None
        # Cleared here rather than only in ``_identify``, which the reuse path
        # below skips: a scan left over from the previous capture would pair
        # this one against the last chart's code.
        self._scan = None
        # A BATCH IS NEVER IDENTIFIED. The engine has always declared the target
        # rather than naming it, and this is where that rule was being lost: the
        # folder's first capture opened down the ordinary path, the search ran, and
        # whatever it named became the target for all hundred. On a frame that is
        # mostly *subject* the search returns a confident wrong answer -- a
        # Smithsonian specimen sheet carrying an Object Level Target scores as an
        # Image Engineering UTT, because nothing better is available to rank -- and
        # unattended that is a hundred plausible wrong answers instead of one.
        #
        # Watched, it is only a bad suggestion, which is why the ordinary path still
        # identifies: the operator sees the placement and checks it. A batch has no
        # such moment for captures two to a hundred.
        if self._pending_batch is not None:
            self._open_placement()
            return
        # Reusing a saved run already knows what the chart is and where it was
        # placed, both checked by whoever made that run. Working it out again would
        # cost seconds to arrive at an answer already on disk.
        if not self._pending_setup[0]:
            self._identify()
            return
        self._open_placement()

    def _identify(self) -> None:
        """Which chart, and which physical unit, in one pass.

        The code scan rides along here for two reasons. It is the same question
        one level down -- identification says which *model* is in the frame and a
        printed code says which *unit*, and reference data is per unit -- and
        doing it now is what lets the setup screen say which file this capture
        will be measured against, instead of the run finding out after the decode
        and raising at somebody.

        Skipped when the bank is empty, there being nothing a code could pair
        with and no reason to spend even a tenth of a second on it.
        """
        from deltachrome_ui.reference_bank import bank_count

        self.loading.set_message("Detecting Target\u2026")
        self.stack.setCurrentWidget(self.loading)
        self._scan = None
        identifier = TargetIdentifier(
            image=self._proxy.image, scan_codes=bank_count() > 0,
            target_ids=self._narrow_search(),
        )
        identifier.progress.connect(self.loading.set_message)
        identifier.scanned.connect(self._on_scanned)
        identifier.finished.connect(self._on_identified)
        run_in_thread(identifier, self)

    def _on_scanned(self, scan) -> None:
        self._scan = scan

    def _code_notes(self) -> list[str]:
        """What the capture's printed code said, where it changes the reading.

        Only the case that is a finding: a chart identified itself and the bank
        holds no measurement of that unit, so the run is being judged against
        manufacturer values while a per-unit file could have existed. The screen
        says so before the run; this puts it in the record, which is the only
        thing a reader a year later will have. Silent when the run paired -- the
        record already names the file -- and silent when no code was read, that
        being the ordinary state of most charts.
        """
        from deltachrome_ui.choices import LIBRARY

        scan = self._scan
        choices = self.placement.choices
        if scan is None or not scan.texts or choices.mode == LIBRARY:
            return []
        resolution = choices.pairing()
        if resolution is not None and resolution.path is not None:
            return [
                f"This chart prints {', '.join(scan.texts)}, and the reference "
                f"bank holds {resolution.path.name} for that unit. The run was "
                "not paired with it."
            ]
        return [
            f"This chart prints {', '.join(scan.texts)} and no file in the "
            "reference bank answers to it, so no per-unit measurement of this "
            "specimen was available."
        ]

    def _placement_warnings(self) -> list[str]:
        """What the record must say about how this run's chart came to be chosen."""
        from deltachrome.imaging.detect import unidentified_regions_note

        note = unidentified_regions_note(len(self._unnamed))
        return [note] if note else []

    def _on_identified(self, results: list) -> None:
        """Every chart the frame was found to hold, of which one opens the screen.

        ``_identified`` stays the *first named* one, because that is what the setup
        panel's chip, its target choice and the run's own record are about: the chart
        being set up. The rest are held so the chart selector can offer them, and the
        ones that could not be named are held so the run can say they were skipped.
        """
        named = [r for r in results
                 if r.found and r.detection is not None and r.detection.found]
        self._identified = named[0] if named else (results[0] if results else None)
        # Every outcome, not only the named ones: a region two targets fitted holds a
        # chart just as surely as one region a single target fitted, and the operator
        # is here to settle which.
        self._pending_identifications = list(results)
        self._open_placement()

    def _open_placement(self) -> None:
        """Draw the one setup screen, with every choice already made.

        The choices are restored as a single choice rather than as a sequence.
        Each of them is also a user gesture, and a *target* gesture means "find
        this chart" -- so setting three of them in a row would ask for searches
        of charts about to be replaced by the next line. The panel holds that
        back and the placement is made once, below, when the whole choice stands.
        """
        proxy, path = self._proxy, self._image_path
        image = proxy.image
        self.placement.set_image(
            path,
            f"{image.width} × {image.height} px  ·  {image.bit_depth}-bit"
            + (f"  ·  {image.profile.description or 'no profile name'}"
               if image.profile is not None else "  ·  no ICC profile"),
        )
        self.placement.set_proxy(proxy, path)
        self.placement.set_untagged(image.profile is None)

        target_id, profile_id = self._pending_setup
        self._pending_setup = (None, None)
        self.placement.choices.restore(
            identification=self._identified,
            target_id=target_id,
            profile_id=profile_id,
        )
        # After the choice is restored, because the pairing line is redrawn by
        # the mode and the mode is part of what ``restore`` sets.
        self.placement.choices.set_scan(self._scan)
        # Placing for a folder rather than for this capture. The screen is the
        # same; only what the primary action starts differs, and the button says so.
        self.placement.set_batch(
            len(self._pending_batch[1]) if self._pending_batch else 0
        )
        if self._pending_batch is not None:
            self.placement.choices.ask_for_target(
                "A batch declares its target rather than identifying it. Which chart "
                "is in the frame is worked out from the capture, and on a frame that "
                "is mostly subject that search can name one confidently and wrongly "
                "-- which, unattended over a folder, is a wrong answer per capture "
                "rather than a suggestion somebody checks.\n\nChoose the chart "
                "these captures carry; the placement below follows it."
            )
        self.stack.setCurrentWidget(self.placement)
        # **This capture holds no charts until this capture is searched.** Nothing
        # cleared them at the way in: ``_apply_target`` did it on the way through,
        # and it now short-circuits when the frame already holds several -- so the
        # previous capture's charts survived into the next one and every visit to
        # this screen added to the count. Reported from the running application:
        # add a target, go home, open the same capture, and it says "Target 3 of 3".
        self._forget_charts()
        # Nothing is known about this capture yet -- identification has not landed,
        # or landed without naming anything -- so the canvas stays clean until
        # something says which chart this is. A restored setup is the exception:
        # those corners and that target were checked in another session.
        if target_id or self._identified is not None and self._identified.found:
            self.placement.settle_target()
        else:
            self.placement.unsettle_target()
        self._apply_target(self.placement.choices.target_id)

    def _on_target_chosen(self, target_id: str) -> None:
        """The operator named the chart, which is different from one being proposed.

        The combo always holds a value, so a capture nothing could be identified in
        opens showing whichever target is first in the list -- and drawing that
        chart's regions over the capture would be the screen asserting a chart it has
        no evidence for. A deliberate choice is evidence; the default is not. So the
        two paths into :meth:`_apply_target` are kept apart here rather than by a
        flag inside it.
        """
        from deltachrome_ui.choices import NO_TARGET

        # Decided from the id this was handed, not from the combo. They agree when
        # the signal came from the combo -- which is every press -- and reading the
        # widget instead made the two paths into :meth:`_apply_target` disagree
        # about their own argument, which is the kind of thing that works until
        # something calls it directly.
        if target_id and target_id != NO_TARGET:
            self.placement.settle_target()
        self._apply_target(target_id)

    def _apply_target(self, target_id: str) -> None:
        """Set the screen up for a chart, and find it.

        Called on opening and again whenever the operator chooses a different
        one, which is why the placement work lives here rather than in the open:
        changing the target changes where the chart is, and a panel still showing
        the previous chart's corners would be showing a placement of something
        else.
        """
        from deltachrome.imaging.detect import FINDERS, detect_placement
        from deltachrome.targets.registry import load_target
        from deltachrome_ui.choices import NO_TARGET

        # The blank entry. Nothing to set the screen up for and nothing to search
        # for, which is the whole of what it means -- the canvas is already clean
        # because ``unsettle_target`` put it that way.
        if not target_id or target_id == NO_TARGET:
            # Cleared, not merely left alone. This panel outlives the capture, so
            # the previous chart's target would otherwise still be what the screen
            # is set up for -- and *Add a Target* would build a chart of it.
            self.placement.set_target(None)
            self.placement.unsettle_target()
            self.placement.update_readiness()
            return

        target = load_target(target_id)

        # **A frame holding several charts retargets the one on screen**, rather
        # than throwing the frame away and searching it again for the new model.
        # It used to do the second, which made the multi-chart flow unusable: on a
        # capture whose two charts sit too close to separate, the operator adds the
        # second by hand and then says which model it is -- and that said "start
        # over", so the added chart AND the found one both vanished and the frame
        # came back holding one. The combo names the chart in front of you, and on
        # this screen "the chart in front of you" is the one selected.
        if self._retargets_one_chart():
            self._retarget_shown_chart(target)
            return

        self.placement.set_target(target)
        self._offer_charts([])
        # The chart was placed on the way to being identified -- the candidate
        # that won did so partly by its regions landing on patches -- so this is
        # a record of how the target was chosen, not a second search.
        self._placement_notes = []

        # Corners carried in from a saved run are a placement somebody already
        # made and checked, so they are not thrown away to search for another.
        restored = self._pending_charts
        if restored:
            self._pending_charts = ()
            self._pending_corners = None
            self._restore_charts(restored)
            return
        seeded = self._pending_corners
        if seeded:
            self._pending_corners = None
            self.placement.set_corners(seeded)
            return

        if len(self._pending_identifications) > 1:
            found = self._pending_identifications
            self._pending_identifications = []
            self._on_charts_identified(found)
            return
        carried = self._identification_placement(target_id)
        if carried is not None:
            self._on_detected([carried])
            return
        # Whether a finder exists is a property of the definition, so a chart
        # nothing can place answers immediately and on this thread rather than
        # showing a search that was never going to run.
        if target.fiducials.method not in FINDERS:
            self._on_detected([detect_placement(self._proxy.image, target)])
            return
        self._detect_placement(target)

    def _restore_charts(self, charts) -> None:
        """Put a saved capture's charts back in the frame, as the charts they were.

        **Every chart, and into the list rather than onto the canvas.** Re-run
        used to restore the record's first chart by setting the canvas's corners
        and nothing else, so a capture of five charts came back as one, and the
        frame's own list stayed empty -- a region search then made whatever it
        found the first and only chart, and the restored one vanished.

        Each placement is one somebody checked when the run was made, so it is
        neither searched for again nor claimed as found: the chip says it was
        restored, and the record keeps how it was placed originally.
        """
        from deltachrome.imaging.detect import Detection
        from deltachrome.targets.registry import load_target

        detections, targets = [], []
        for target_id, corners, placed in charts:
            try:
                target = load_target(target_id)
            except Exception:  # noqa: BLE001 -- a target since removed; skip it
                continue
            detections.append(Detection(
                corners=[tuple(map(float, point)) for point in corners],
                found=True,
                method=target.fiducials.method,
                reason="Restored from the saved run.",
                diagnostics={"restored": True, "placed_as": placed or "automatic"},
            ))
            targets.append(target)
        if not detections:
            return
        self.placement.settle_target()
        # Chosen when the run was made, not named by a search now. Passed in
        # rather than set after: offering the charts selects the first, and the
        # chip it set then would outlive a correction made afterwards.
        self._offer_charts(detections, targets, named=[False] * len(detections))
        self._select_chart(0)

    def _identification_placement(self, target_id):
        """The placement identification already made, if it is still the right one."""
        result = self._identified
        if result is None or not result.found or result.target_id != target_id:
            return None
        detection = result.detection
        return detection if detection is not None and detection.found else None

    def _detect_placement(self, target) -> None:
        """Find the chart, without leaving the screen it is being found on.

        This used to swap to the full-window wait, which severed a choice from
        its effect: the operator picked a chart from a list and the next thing
        they saw was a different screen. The canvas stays put and the panel goes
        inert with the wait on it, because every control on it would change the
        answer being computed.

        A refusal does not stop the run or open a dialog either -- it leaves the
        handles where they are with the reason on the panel, which is the only
        handover that leaves the operator better off than not having tried.
        """
        self.placement.set_busy("Detecting Target\u2026")
        detector = PlacementDetector(image=self._proxy.image, target=target)
        detector.finished.connect(self._on_detected)
        run_in_thread(detector, self)

    def _retargets_one_chart(self) -> bool:
        """Whether a target change belongs to the shown chart rather than the frame.

        Two cases, and both are "the operator is naming a chart, not correcting an
        identification of the whole capture": a frame holding more than one, and a
        chart added by hand, which has no identification to correct and a placement
        somebody made that a fresh search would discard.

        A frame holding one found chart keeps the old behaviour exactly -- there the
        combo really does mean "you named the capture wrongly, find this instead".
        """
        index = self._shown_chart
        if index is None or not (0 <= index < len(self._detections)):
            return False
        if len(self._detections) > 1:
            return True
        return index < len(self._chart_added) and self._chart_added[index]

    def _retarget_shown_chart(self, target) -> None:
        """Say which model the shown chart is, and leave the rest of the frame alone.

        A chart **added by hand** keeps its corners: there was no detection under it,
        the operator put it where it is, and re-searching would discard a placement
        they made deliberately. A chart that was **found** is looked for again as the
        new model, inside its own region rather than across the frame -- which is
        what ``detect_placement_near`` is for, and is the right scope because the
        operator is saying "that chart there is this model", not "there is one of
        these somewhere".
        """
        from deltachrome.imaging.detect import FINDERS, detect_placement_near

        index = self._shown_chart
        self._take_corners_off_the_canvas()
        self._chart_targets = self._targets_per_chart()
        self._chart_targets[index] = target
        self.placement.set_target(target)
        self.placement.settle_target()

        added = index < len(self._chart_added) and self._chart_added[index]
        corners = self._detections[index].corners
        if not added and corners and target.fiducials.method in FINDERS:
            found = detect_placement_near(self._proxy.image, target, corners)
            if found.found:
                self._detections[index] = found
                self._chart_dragged[index] = False
        self._refresh_charts(index)

    def _forget_charts(self) -> None:
        """Whatever the last capture held, this one does not hold it.

        Separate from :meth:`_offer_charts` because that one *offers* a list and
        this one says there is nothing to offer yet: it runs before the search that
        will fill it, and it has to leave ``_shown_chart`` unset so that nothing
        downstream reads a chart index into an empty frame.
        """
        self._detections = []
        self._chart_targets = []
        self._chart_doubts = []
        self._chart_named = []
        self._chart_added = []
        self._chart_dragged = []
        self._dropped_charts = 0
        self._shown_chart = None
        self._last_region = None
        self.placement.canvas.clear_other_placements()

    def _targets_per_chart(self) -> list:
        """One target per chart, materialised.

        The single-model path leaves this empty -- every chart is the same model and
        the panel's own target says which -- so anything that needs to change one
        chart's target has to fill it in first.
        """
        if len(self._chart_targets) == len(self._detections):
            return list(self._chart_targets)
        return [self.placement.target for _ in self._detections]

    def _detect_in_region(self, region) -> None:
        """Work out what chart is in the rectangle the operator drew, and place it.

        **It identifies rather than only placing**, which is the difference between
        a tool for a chart you have already named and a tool for a capture nothing
        could be found in. Identification has always taken the region to look in --
        that is how a frame holding several charts gets an answer per region -- so
        the rectangle feeds the same search, the same two gates and the same
        refusals, and the chart comes back named *and* placed from one pass.

        The case it is for: `chart_extent` finds the chart as the largest region
        unlike the frame's border, which is the whole frame whenever nothing
        separates them, and every finder's size gate is a fraction of that region.
        On the Smithsonian Odonata captures that puts the smallest admissible patch
        above the largest the Object Level Target prints and the search refuses.
        Told where to look, it names and places the chart in a fraction of a second.

        **What comes back is another chart in the frame, not a replacement for it.**
        Same rule whether the frame holds none or four: the operator pointed at a
        chart and said "this one too". On a capture nothing was identified in there
        is nothing to add to and it becomes the first; on one where two charts sit
        too close for the extent search to separate, it is how the second is reached
        without placing it by hand. *Remove This Target* is the undo either way.
        """
        if self._proxy is None:
            return
        self._last_region = tuple(region)
        self.placement.set_busy("Searching the Region\u2026")
        finder = TargetIdentifier(
            image=self._proxy.image, region=region,
            target_ids=self._narrow_search(),
        )
        finder.finished.connect(self._on_region_identified)
        run_in_thread(finder, self)

    def _on_region_identified(self, results: list) -> None:
        """Add the chart found in the drawn rectangle, or say what stopped it.

        A region that could not be named falls back to placing the chart the operator
        already chose, because "I know what it is, find it here" is the other half of
        why somebody draws a box -- and only then, so a confident identification is
        never overridden by a stale combo.
        """
        from deltachrome.targets.registry import load_target

        result = results[0] if results else None
        detection = result.detection if result is not None else None
        if result is None or detection is None or not detection.found:
            reason = (result.reason if result is not None else "") or (
                "Nothing could be placed in that region."
            )
            self._place_chosen_target_in_region(reason)
            return

        named = result.target_id if result.found else (
            next(iter(result.candidates), None).target_id
            if result.candidates else None
        )
        if named is None:
            self._place_chosen_target_in_region(result.reason)
            return
        doubt = None if result.found else result.reason
        self._append_chart(
            load_target(named), detection, doubt=doubt, added=False,
            named=bool(result.found),
        )

    def _place_chosen_target_in_region(self, reason: str) -> None:
        """Nothing was named there, so place what the operator already chose.

        Kept as a fallback rather than the first move: a chart identified in the
        region is a better answer than one asserted by a combo box that always holds
        a value, and this screen already has one way of asserting a chart nobody
        confirmed.
        """
        from deltachrome.imaging.detect import (
            Detection, FINDERS, detect_placement_near,
        )

        def refuse(why: str) -> None:
            self.placement.show_detection(Detection(reason=(
                f"{why} Draw a looser rectangle — one drawn inside the chart's own "
                "edges hides the marks the search needs — or choose the chart and "
                "place it by hand."
            )))

        target = self.placement.target
        region = self._last_region
        if target is None or region is None or self._proxy is None:
            refuse(reason)
            return
        if target.fiducials.method not in FINDERS:
            refuse(reason)
            return
        x0, y0, x1, y1 = region
        found = detect_placement_near(
            self._proxy.image, target,
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        )
        if not found.found:
            refuse(reason or found.reason)
            return
        self._append_chart(target, found, doubt=None, added=False)

    def _unplaced_chart(self) -> int | None:
        """An added chart nobody has moved, if the frame holds one.

        It is a **placeholder rather than a chart**: it sits on the canvas's default
        rectangle, it is not a placement of anything, and it is already the one
        state whose sampled regions are withheld for exactly that reason. So it
        carries no information, and there is never a reason to have two.
        """
        for index, added in enumerate(self._chart_added):
            moved = index < len(self._chart_dragged) and self._chart_dragged[index]
            if added and not moved:
                return index
        return None

    def _chart_already_here(self, detection) -> int | None:
        """The chart this placement is a placement *of*, if the frame holds it.

        **Two charts cannot occupy the same pixels**, so any real overlap between
        two placements means they are two placements of one chart -- which makes
        this a comparison with no threshold to tune, rather than a similarity score
        with a cut-off somebody has to defend. Two placements of the same chart
        agree to within a few pixels; two different charts share none at all.

        The case it is for: the operator draws a region, does not like what comes
        back -- a chart named that they did not mean, or a placement sitting badly
        -- and draws again. That is "try this again", and it was being counted as
        "and another chart here", so every press of *Search a Region* put one more
        target in the frame.

        Placeholders are excluded deliberately. An added chart nobody has moved
        sits on the canvas's default rectangle, which is a large box in the middle
        of the frame and would swallow the centre of almost any real chart -- and
        it is not a placement of anything, which is the same reason its regions are
        not drawn.
        """
        def box(corners):
            points = [(float(x), float(y)) for x, y in (corners or [])]
            if not points:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return min(xs), min(ys), max(xs), max(ys)

        def centre(bounds):
            x0, y0, x1, y1 = bounds
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

        def holds(bounds, point):
            x0, y0, x1, y1 = bounds
            return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

        fresh = box(detection.corners)
        if fresh is None:
            return None
        for index, held in enumerate(self._detections):
            if index < len(self._chart_added) and self._chart_added[index] and not (
                index < len(self._chart_dragged) and self._chart_dragged[index]
            ):
                continue
            theirs = box(held.corners)
            if theirs is None:
                continue
            if holds(theirs, centre(fresh)) or holds(fresh, centre(theirs)):
                return index
        return None

    def _append_chart(
        self, target, detection, *, doubt, added: bool, named: bool = False
    ) -> None:
        """Put one more chart in the frame, however it was arrived at.

        The bookkeeping is the same for a chart found in a drawn region and a chart
        added by hand, and keeping one copy of it is what stops the two drifting --
        five parallel lists, each of which has to grow by exactly one.

        **Unless the frame already holds a placeholder**, in which case this fills
        it. Reported from the running application: the operator presses *Add a
        Target*, thinks better of dragging it and reaches for *Search a Region*
        instead, and the frame comes back holding three charts where two were
        wanted -- the one that was found, the empty one they abandoned, and the one
        the region search named. Pressing *Add a Target* is a statement of intent
        to place a chart, and doing it another way is that same intent carried out,
        not a second chart.
        """
        self._take_corners_off_the_canvas()
        self._shown_chart = None
        # A placement landing on a chart the frame already holds is that chart
        # found again, not a second one -- see :meth:`_chart_already_here`. Then a
        # placeholder somebody abandoned. Only failing both is this another chart.
        spare = self._chart_already_here(detection)
        if spare is None:
            spare = self._unplaced_chart()
        if spare is None:
            self._chart_targets = self._targets_per_chart() + [target]
            self._detections = list(self._detections) + [detection]
            self._chart_doubts = list(self._chart_doubts) + [doubt]
            self._chart_named = self._named_per_chart()[:-1] + [named]
            self._chart_added = list(self._chart_added) + [added]
            self._chart_dragged = list(self._chart_dragged) + [False]
            spare = len(self._detections) - 1
        else:
            self._chart_targets = self._targets_per_chart()
            self._chart_targets[spare] = target
            self._detections[spare] = detection
            self._chart_doubts[spare] = doubt
            self._chart_named = self._named_per_chart()
            self._chart_named[spare] = named
            self._chart_added[spare] = added
            self._chart_dragged[spare] = False
        if target is not None:
            self.placement.settle_target()
        self._refresh_charts(spare)

    def _on_detected(self, detections: list) -> None:
        """Every placement the frame supports, of which one is shown.

        A frame usually holds one chart and then this is what it always was. Where
        it holds several -- the QA-62 is a small card shot four to a board -- they
        are offered one at a time rather than run together, because **a run is one
        chart**: each is analyzed, saved and re-scorable on its own, and the
        library already combines several saved runs into a verdict for the system.
        That is the workflow the session screen was built around, stated in its own
        words: analyze each chart as it comes, ask the combined question at the end.
        """
        self._offer_charts([d for d in detections if d.found] or list(detections))
        self._select_chart(0)

    def _on_charts_identified(self, results: list) -> None:
        """Every chart the frame holds, offered whether or not it could be named.

        **A shortlist is offered here and skipped on the command line**, and that
        asymmetry is the one this application already rests on: the desktop operator
        sees the chart, its target and its sampled regions and confirms them before
        anything is measured, where a scripted run has no such moment. Naming the
        wrong chart is worse than naming none *when nobody is looking*; when somebody
        is, a region that two targets fit is a region that certainly holds a chart,
        and dropping it makes that chart read as absent rather than as undecided.

        So a shortlisted region comes through with its leading candidate selected and
        the ambiguity stated, for the operator to confirm or change in the target
        combo beside it. A region that fitted nothing at all is still reported rather
        than offered: there is no candidate to lead with.
        """
        from deltachrome.targets.registry import load_target

        offered, unnamed = [], []
        for result in results:
            if result.found and result.detection is not None and result.detection.found:
                offered.append((load_target(result.target_id), result.detection, None))
                continue
            # A shortlist: two or more targets fitted and none led by enough. The
            # placement is the same either way -- they fitted the same four corners --
            # so the leading candidate's is the one to show.
            leading = next(iter(result.candidates), None)
            detection = result.detection
            if (leading is not None and len(result.candidates) > 1
                    and detection is not None and detection.found):
                offered.append((load_target(leading.target_id), detection, result.reason))
                continue
            unnamed.append(result.reason)

        self._unnamed = unnamed
        if not offered:
            # Nothing fitted. The combo still holds a value -- it always does -- so
            # the screen says outright that the choice is being handed over, and
            # draws nothing over the capture until it is made. Spraying the first
            # target in the list over a capture of something else is the one thing
            # this screen must not do, since checking the regions against the chart
            # is what it is for.
            self.placement.choices.clear_target()
            self._forget_charts()
            self.placement.unsettle_target()
            self.placement.choices.ask_for_target(
                (unnamed[0] if unnamed else
                 "No installed target could be found in this capture.")
                + "\n\nChoose the chart yourself, or drag a rectangle round it "
                  "with Search a Region and it will be looked for there."
            )
            return
        self.placement.settle_target()
        self._chart_doubts = [doubt for _t, _d, doubt in offered]
        self._offer_charts(
            [detection for _t, detection, _doubt in offered],
            [target for target, _d, _doubt in offered],
        )
        self._select_chart(0)

    def _add_chart(self) -> None:
        """Another chart in this frame, placed by hand.

        The frame that needs this is the one where the search found a single chart
        and the operator can see two: charts closer together than the extent search
        can separate come back as one region, and no amount of looking at that region
        finds the second. Rather than leave the second chart unmeasurable, it is
        added here and dragged onto -- which is what this screen has always been able
        to do for one chart, and is now able to do for the second.

        It starts on the canvas's own default rectangle, in the middle of the frame.
        Seeding it from a leftover candidate region was tried in principle and is not
        worth it: on the capture this exists for, the regions merged, so there is no
        leftover to seed from. The operator drags it.
        """
        from deltachrome.imaging.detect import Detection

        if self._proxy is None:
            return
        # One placeholder at a time. Pressing this twice without placing either
        # leaves two identical default rectangles, which is two things to step
        # through and nothing to look at; the second press shows the first.
        spare = self._unplaced_chart()
        if spare is not None:
            self._refresh_charts(spare)
            return
        image = self._proxy.image
        width, height = image.width, image.height
        inset_x, inset_y = width * 0.3, height * 0.3
        corners = [
            (inset_x, inset_y), (width - inset_x, inset_y),
            (width - inset_x, height - inset_y), (inset_x, height - inset_y),
        ]
        placed = Detection(
            corners=corners, found=True, method="",
            reason="Added by hand. Drag the handles onto the chart.",
        )
        # **It arrives with no target, and the canvas stays clean.** Carrying the
        # shown chart's target over was the obvious thing and it is the wrong
        # offer: a chart of the same model at a default rectangle, with reticles
        # up and its ROIs drawn, invites dragging -- which is the way of placing a
        # chart this screen wants reached for last. Blank says what is true, that
        # the frame holds another chart and nothing yet says which or where, and
        # leaves *Search a Region* as the thing to do about it.
        self._append_chart(None, placed, doubt=None, added=True)

    def _drop_chart(self, index: int) -> None:
        """Take a target out of the frame, whether it was found or added.

        **It used to be two actions and is now one**, which reverses a decision
        made when this was written. The argument for keeping a detected chart
        greyed rather than removing it was that a detection is *evidence about the
        capture* and should not vanish because somebody clicked. That is still
        true of the archive and it was the wrong thing to spend the screen on: the
        commonest reason to press this is that the search returned a region that
        is not a chart at all -- a QR code block on a ColorChecker SG is the case
        that prompted it -- and leaving a non-chart in the selector, with its own
        ROIs and its own place in `Target 2 of 2`, means the operator is still
        stepping through something they have already said is not there.

        So it goes, along with its placement, and the frame reverts to the charts
        that are left. **The evidence is kept where it belongs**, in the run's own
        warnings: the record still says how many further regions of this capture
        held something a chart could be placed in and were not measured. What is
        lost is the undo, and the cost of getting it wrong is one press of *Add a
        Target*.
        """
        if not (0 <= index < len(self._detections)):
            return
        self._take_corners_off_the_canvas()
        self._shown_chart = None
        self._chart_named = self._named_per_chart()
        for held in (self._detections, self._chart_targets, self._chart_doubts,
                     self._chart_named, self._chart_added, self._chart_dragged):
            if index < len(held):
                del held[index]
        self._dropped_charts += 1
        self._refresh_charts(min(index, len(self._detections) - 1))

    def _refresh_charts(self, index: int) -> None:
        """Redraw the strip and show one chart, after the list itself has changed."""
        from deltachrome_ui.placement_view import ChartOffer

        if not self._detections:
            self.placement.offer_charts([], 0, emptied=True)
            self.placement.canvas.clear_other_placements()
            return
        index = max(0, min(index, len(self._detections) - 1))
        self.placement.offer_charts(
            [ChartOffer(added=added) for added in self._chart_added],
            index,
        )
        self._select_chart(index)

    def _offer_charts(
        self, detections: list, targets: list | None = None, *, named: list | None = None
    ) -> None:
        """Hold this frame's placements, each with the target it is a placement of.

        Usually one target repeated -- a board of QA-62 cards is four of one model --
        but not necessarily, and the pairing is held rather than assumed so that a
        frame holding a device-level target beside a colour chart needs no second
        code path. Every chart is treated the same; same-model is the case where the
        targets happen to coincide.
        """
        from deltachrome_ui.placement_view import ChartOffer

        self._detections = list(detections)
        self._chart_targets = list(targets) if targets else []
        if len(self._chart_doubts) != len(self._detections):
            self._chart_doubts = [None] * len(self._detections)
        # Named by the search where it was offered with a target and no doubt --
        # which is what identification hands over -- and otherwise not: a
        # placement of a chart the operator chose names nothing.
        self._chart_named = list(named) if named is not None else [
            bool(targets) and doubt is None for doubt in self._chart_doubts
        ]
        # A fresh search replaces the operator's edits, because they were edits to a
        # different answer. Adding a chart and then changing the target would
        # otherwise carry a hand placement onto a frame it was never made for.
        self._chart_added = [False] * len(self._detections)
        self._chart_dragged = [False] * len(self._detections)
        self._dropped_charts = 0
        self._shown_chart = None
        self.placement.offer_charts(
            [ChartOffer() for _ in self._detections], 0
        )

    def _take_corners_off_the_canvas(self) -> None:
        """Whatever the canvas holds belongs to the chart it is showing.

        Read back rather than pushed by a signal, and at the two moments it matters
        -- before the screen shows another chart, and before anything runs. That
        catches a drag, a refinement and a Reset alike, because all three end on the
        canvas, where a ``cornersChanged`` subscription would have to be told about
        each new way of moving them.

        This was missing entirely, and on a multi-chart frame it made the placement
        screen's whole purpose a lie: the operator checked and corrected five
        placements and the run read the five the search had stored.
        """
        index = self._shown_chart
        if index is None or not (0 <= index < len(self._detections)):
            return
        corners = [(float(x), float(y)) for x, y in self.placement.canvas.corners()]
        stored = [(float(x), float(y)) for x, y in (self._detections[index].corners or [])]
        if corners and corners != stored:
            self._detections[index].corners = corners
            while len(self._chart_dragged) <= index:
                self._chart_dragged.append(False)
            self._chart_dragged[index] = True

    def _select_chart(self, index: int) -> None:
        """Show one of the placements found, and claim only what it did.

        Where the charts were identified separately this also switches the screen to
        *that* chart's target, so the sampled regions drawn over it, the reference
        requirement and the run that follows all belong to the chart on screen.
        """
        if not self._detections:
            return
        self._take_corners_off_the_canvas()
        index = max(0, min(index, len(self._detections) - 1))
        self._shown_chart = index
        doubt = self._chart_doubts[index] if index < len(self._chart_doubts) else None
        if index < len(self._chart_targets) and self._chart_targets[index] is None:
            # This chart is in the frame and nothing says which chart it is. The
            # combo goes blank and the canvas with it, so the screen asks rather
            # than assuming the chart shown before this one.
            self.placement.choices.clear_target()
            self.placement.set_target(None)
            self.placement.unsettle_target()
        elif index < len(self._chart_targets):
            chart_target = self._chart_targets[index]
            if chart_target.target_id != self.placement.choices.target_id:
                self.placement.set_target(chart_target)
                self.placement.choices.restore(target_id=chart_target.target_id)
            if doubt:
                self.placement.choices.show_shortlist(chart_target.target_id, doubt)
            else:
                # Every chart sets the chip for itself. It used to be set once for
                # the frame, so a chart found by *Search a Region* on a capture the
                # first pass could name nothing in kept that pass's "Choose the
                # Target" -- asking for a decision already made.
                named = self._named_per_chart()
                self.placement.choices.settle_chip(
                    chart_target.target_id, named=index < len(named) and named[index]
                )
        detection = self._detections[index]
        # The rest of the frame, behind this one. Drawn here rather than by the
        # placement view because the placements live in this shell: the view is
        # handed one chart at a time and has never known what else is in the
        # capture, which is the whole reason an ordinal could not be located.
        self.placement.canvas.show_other_placements([
            (n + 1, other.corners)
            for n, other in enumerate(self._detections)
            if n != index and other.corners
        ])
        self._placement_notes = self._placement_warnings()
        if doubt:
            # Stated where the target is chosen, because that is the control the
            # operator would use to settle it, and carried into the record because a
            # chart nobody could name and somebody confirmed has to say so.
            self._placement_notes.append(
                "The chart in this region could not be named outright and the leading "
                f"candidate was offered for confirmation. {doubt}"
            )
        added = index < len(self._chart_added) and self._chart_added[index]
        moved = index < len(self._chart_dragged) and self._chart_dragged[index]
        # **An added chart that nobody has moved has no placement yet.** It starts
        # on the canvas's default inset rectangle, in the middle of the frame and
        # over pixels that are almost certainly not a chart, so drawing its ROIs
        # there is a picture of something nobody has claimed -- the same fault as
        # drawing an unidentified chart's. The first drag settles it, which is the
        # moment the regions start being worth looking at.
        if added and not moved:
            self.placement.unsettle_placement()
        else:
            self.placement.settle_placement()
        if added:
            # set_corners before show_detection is what says "placed by hand": the
            # canvas reports a drag through cornersDragged, and a chart added here
            # was never detected, so the pill has nothing to claim.
            self.placement.set_corners(detection.corners)
            self.placement.clear_detection()
            return
        if detection.found:
            self.placement.set_corners(detection.corners)
            # How the chart was placed goes into the record's geometry, not here:
            # what is carried is only what the refinement had to say about it.
            self._placement_notes = self._placement_notes + list(detection.notes)
        self.placement.show_detection(detection)

    def _report_proxy(self):
        """The decoded capture the live results were read from.

        The placement screen's own where nothing else was recorded, which is the
        ordinary case: a run measured here was measured from that capture.
        """
        return self._live_proxy if self._live_proxy is not None else self._proxy

    def _named_per_chart(self) -> list[bool]:
        """``_chart_named``, the length of ``_detections`` whatever happened to it."""
        named = list(self._chart_named)[: len(self._detections)]
        return named + [False] * (len(self._detections) - len(named))

    @property
    def _came_from_session(self) -> bool:
        """Whether the report on screen was reached from All Targets.

        A property so the report's Back button says so: it is the one fact both
        where Back goes and what its tooltip names depend on, and two copies of it
        had drifted -- Back went to All Targets while saying "the library".
        """
        return self._back_to_targets

    @_came_from_session.setter
    def _came_from_session(self, value: bool) -> None:
        self._back_to_targets = bool(value)
        if hasattr(self, "result"):
            self.result.set_back_destination("All Targets" if value else None)

    def _was_identified(self) -> bool:
        """Whether the chart being run was worked out from the capture rather than
        named. Recorded as ``geometry.identified``; it used to be a warning."""
        return bool(self.placement.choices.identified and self._identified is not None)

    def _narrow_search(self) -> list[str] | None:
        """The targets Settings leaves on, held as the search that uses them starts.

        Held rather than re-read at run time, so the record names the set the chart
        was actually chosen from even if Settings changed in between.
        """
        self._searched = identification_candidates()
        return self._searched

    def _searched_targets(self) -> list[str] | None:
        """What the chart was named from, where Settings had narrowed it.

        ``None`` where the chart was not identified, since then the list decided
        nothing, and where nothing was turned off.
        """
        return getattr(self, "_searched", None) if self._was_identified() else None

    # -- analysis ----------------------------------------------------------


    # -- batches ---------------------------------------------------------------

    def _choose_batch(self) -> None:
        """Pick a folder, then set the run up on its first capture.

        **The batch starts on the placement screen, not on a dialog.** A folder
        cannot be run until somebody has said where the chart is in it, and the
        screen whose whole job is showing a placement for checking is the one that
        already exists. So the first capture opens exactly as an ordinary run does,
        with one difference: the primary action runs the folder.
        """
        from deltachrome.batch import captures_in

        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder of captures", str(Path.home())
        )
        if not folder:
            return
        try:
            captures = captures_in(folder)
        except DeltachromeError as exc:
            QMessageBox.warning(self, "Batch Analysis", str(exc))
            return
        if not captures:
            QMessageBox.information(
                self, "Batch Analysis",
                f"No captures in {Path(folder).name}.\n\nDeltachrome reads TIFF, "
                "JPEG 2000, PNG and JPEG. Anything else in a folder is passed over "
                "rather than counted as a capture that failed.",
            )
            return
        self._pending_batch = (Path(folder), captures)
        self._pending_corners = None
        self._pending_charts = ()
        self.load_image(captures[0])

    def _run_batch(self, corners) -> None:
        """Run the folder, from the placement the operator has just checked."""
        if self._pending_batch is None:
            return
        folder, captures = self._pending_batch
        choices = self.placement.choices
        target_id = choices.target_id
        profile_id = choices.profile_id or ""
        into = batch_dir(folder)

        # The guideline by the name a person chose it under. ``ProfilePicker``
        # is two combo boxes in a widget and has no ``currentText`` of its own,
        # so the ``hasattr`` this used to ask was always false and the header
        # read the profile id -- ``fadgi_1star_bound_volumes_general_2023``,
        # which is a variable and not a name.
        summary = choices.picker.summary
        self.batch.start(
            folder, len(captures),
            choices.target_combo.currentText() or target_id,
            summary.display_name if summary else (profile_id or ""),
            local_criteria=bool(summary and summary.local_criteria),
        )
        self.stack.setCurrentWidget(self.batch)

        worker = BatchRunner(
            captures=captures,
            target_id=target_id,
            profile_id=profile_id,
            into=into,
            corners=list(corners),
            # The reference the placement screen chose, read as a single run reads
            # it. Left out, every batch fell back to manufacturer values, and a
            # chart shipping none -- the LaserSoft, the DT charts -- had every
            # frame refused with "No reference file was supplied".
            reference_path=(choices.reference_choice
                            if choices.reference_mode == "file" else None),
            reference_library=(choices.reference_choice
                               if choices.reference_mode == "library" else None),
            options=choices.values(),
        )
        worker.progress.connect(self.batch.advance)
        worker.finished.connect(self._on_batch_done)
        worker.failed.connect(self._on_batch_failed)
        self._batch_worker = worker
        run_in_thread(worker, self)

    def _stop_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.stop()

    def _on_batch_done(self, result) -> None:
        self._batch_worker = None
        self._pending_batch = None
        self.placement.set_batch(0)
        self.batch.show_result(result)

    def _on_batch_failed(self, failure: LoadFailure) -> None:
        self._batch_worker = None
        self._pending_batch = None
        self.placement.set_batch(0)
        QMessageBox.critical(self, failure.headline, failure.detail)
        self.show_library()

    def _batch_finished(self, result) -> None:
        """Open the library with this batch's own row open.

        Not a screen of its own. Every record a batch makes is an ordinary one, so
        the place to read them is the table that reads every other record -- which
        also means sorting, searching and re-scoring work on a batch's output on
        the day it is written rather than being asked for later.

        The library gathers a batch's records under one row that opens, and shut
        is the right state for somebody who came looking for something else.
        Somebody who has just pressed "Open in Library" on the run that wrote
        them is not that person.
        """
        self.show_library()
        self.library.reveal_batch(Path(result.directory))

    def _run_analysis(self, corners) -> None:
        """Read the whole run off the one screen that set it up.

        Where the frame holds several charts this runs *all* of them and lands on
        the session, which is the question a board is asking. Going back to the
        placement screen between charts was the alternative and is too much to ask:
        five charts would be five round trips through a screen whose work -- finding
        and checking the placements -- was already done before the first run.
        """
        if self._proxy is None:
            return
        # The corners on screen belong to the chart on screen, and nothing else has
        # written them back. Taken before anything is decided, so a run reads what
        # the operator left rather than what the search stored.
        self._take_corners_off_the_canvas()
        if self._pending_batch is not None:
            # The placement on screen is this folder's seed. Everything else about
            # the screen means what it always meant; only what the button starts
            # differs, which is why this is a branch here and not a second screen.
            corners = self._detections[0].corners if self._detections else corners
            self._run_batch(corners)
            return
        if len(self._detections) > 1:
            self._run_every_chart()
            return
        if self._detections:
            # One chart, and the canvas is showing it -- but a chart *removed*
            # leaves the survivor in the list rather than on the canvas, so the
            # corners come from the list.
            corners = self._detections[0].corners or corners
        choices = self.placement.choices
        self.loading.set_message("Analyzing…")
        self.stack.setCurrentWidget(self.loading)

        runner = AnalysisRunner(
            image=self._proxy.image,
            target_id=choices.target_id,
            profile_id=choices.profile_id or "",
            corners=corners,
            reference_mode=choices.reference_mode,
            reference_choice_path=choices.reference_choice,
            options=choices.values(),
            scan=self._scan,
            placement_notes=(
                (self._placement_notes if self.placement.placed_automatically else [])
                + self._code_notes()
            ),
            placed="automatic" if self.placement.placed_automatically else "by_hand",
            identified=self._was_identified(),
            searched_targets=self._searched_targets(),
        )
        runner.progress.connect(self.loading.set_message)
        runner.finished.connect(self._on_analysed)
        runner.failed.connect(self._on_analysis_failed)
        run_in_thread(runner, self)

    def _open_profile(self, corners) -> None:
        """Sample the chart on screen and open the camera profile screen on it.

        The chart shown is the one profiled, on a frame of several as on a frame
        of one: a profile is built from one chart, and the operator has chosen
        which by having it on screen.
        """
        if self._proxy is None or self.placement.target is None:
            return
        self._take_corners_off_the_canvas()
        choices = self.placement.choices
        self.loading.set_message("Sampling patches…")
        self.stack.setCurrentWidget(self.loading)
        gatherer = ProfileGatherer(
            image=self._proxy.image,
            target_id=choices.target_id,
            corners=self.placement.canvas.corners(),
            reference_mode=choices.reference_mode,
            reference_choice_path=choices.reference_choice,
            scan=self._scan,
            image_path=self._image_path,
        )
        gatherer.progress.connect(self.loading.set_message)
        gatherer.finished.connect(self._on_profile_gathered)
        gatherer.failed.connect(self._on_profile_gather_failed)
        run_in_thread(gatherer, self)

    def _on_profile_gathered(self, setup) -> None:
        self.profile.open(proxy=self._proxy, image_path=self._image_path,
                          target=self.placement.target, setup=setup)
        self.stack.setCurrentWidget(self.profile)

    def _on_profile_gather_failed(self, failure: LoadFailure) -> None:
        self.stack.setCurrentWidget(self.placement)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Profile not started")
        box.setText(failure.headline)
        box.setInformativeText(failure.detail)
        box.exec()

    def _build_profile(self, request: dict) -> None:
        builder = ProfileBuilder(**request)
        builder.progress.connect(self.profile.show_progress)
        builder.finished.connect(self.profile.show_built)
        builder.failed.connect(self.profile.show_failed)
        run_in_thread(builder, self)

    def _run_every_chart(self) -> None:
        """Analyze every chart in the frame, sharing the one decode, then combine."""
        self._take_corners_off_the_canvas()
        choices = self.placement.choices
        charts = []
        ignored = self._dropped_charts
        for index, detection in enumerate(self._detections):
            target = (self._chart_targets[index] if index < len(self._chart_targets)
                      else self.placement.target)
            doubt = self._chart_doubts[index] if index < len(self._chart_doubts) else None
            added = index < len(self._chart_added) and self._chart_added[index]
            dragged = index < len(self._chart_dragged) and self._chart_dragged[index]
            # The record must not say a placement was automatic when half of it
            # was not -- the same rule the single-chart screen has always followed.
            # Said in the record's geometry rather than as a warning.
            if added:
                placed, notes = "by_hand", []
            elif dragged:
                placed, notes = (
                    "by_hand"
                    if detection.diagnostics.get("placed_as") == "by_hand"
                    else "automatic_then_moved"
                ), []
            elif detection.diagnostics.get("restored"):
                # Restored from a saved run: placed however it was placed then.
                placed, notes = detection.diagnostics.get("placed_as", "automatic"), []
            else:
                placed, notes = "automatic", list(detection.notes)
            if doubt:
                notes.append(
                    "The chart in this region could not be named outright and the "
                    f"leading candidate was confirmed on screen. {doubt}"
                )
            if ignored:
                notes.append(
                    f"{ignored} further chart(s) found in this capture were left out "
                    "of the run by the operator."
                )
            charts.append((target, detection.corners, notes, placed))

        self.loading.set_message(f"Analyzing {len(charts)} charts…")
        self.stack.setCurrentWidget(self.loading)
        runner = ChartsRunner(
            image=self._proxy.image,
            charts=charts,
            profile_id=choices.profile_id or "",
            reference_mode=choices.reference_mode,
            reference_choice_path=choices.reference_choice,
            options=choices.values(),
            scan=self._scan,
            notes=self._placement_warnings() + self._code_notes(),
            identified=self._was_identified(),
            searched_targets=self._searched_targets(),
        )
        runner.progress.connect(self.loading.set_message)
        runner.finished.connect(self._on_charts_analysed)
        runner.failed.connect(self._on_analysis_failed)
        run_in_thread(runner, self)

    def _on_charts_analysed(self, outcome) -> None:
        """Land on **All Targets** for a board, on its report for a lone chart.

        This has been round twice and the reason is worth keeping, because the
        screen changed underneath the argument. It first landed on the combined
        verdict; that was reversed because nothing on that screen was reached by
        *pressing* anything -- it listed the captures, and the report was what a
        run is made to produce, so every reader crossed a screen to get to one.

        The frame map is what makes it right again. The screen is now a picture
        of the capture with every target outlined where it was measured, and
        clicking one opens its report -- so it is no longer a list to be crossed
        but the place the targets are reached *from*, and it answers the question
        a board is laid out to ask before any one card does. The report is one
        click away, which is where the reversal put it, and the click now lands
        on the target the reader meant rather than on whichever came first.

        The runs are not saved yet, so the reports cannot be looked up in the
        library the way a saved capture's can. They are held here instead, and the
        stepper, All Targets and Save all read this one list.
        """
        results, failures = outcome
        self._fresh_results = {r.run.run_id: r for r in results}
        self._chart_failures = list(failures)
        if not results:
            # Every chart in the frame refused. There is no report to land on and
            # no session to combine, and the placement screen is where the corners
            # that produced this are still on screen.
            self.stack.setCurrentWidget(self.placement)
            QMessageBox.warning(
                self, "Nothing was measured",
                "No chart in this capture could be analyzed.\n\n"
                + "\n\n".join(str(f) for f in failures),
            )
            return
        # A report reached this way was not reached through the session, so Back
        # goes out to the library rather than to a screen nobody has seen.
        self._came_from_session = False
        self._open_record = None
        self._live_proxy, self._placed_here = self._proxy, True
        self.result.set_proxy(self._proxy)
        self.result.set_result(results[0], saved=False, chart=(0, len(results)))
        self.stack.setCurrentWidget(self.result)
        if len(results) > 1:
            # Built from the report rather than instead of it: the report is
            # what a chart's own readings live on, and the stepper, Save and the
            # frame map all read the list it was given. Landing on top of it
            # costs a widget switch.
            self._show_combined()

    def _show_combined(self) -> None:
        """Every chart of the capture on screen, read together.

        **Rebuilt on every press rather than kept**, because a chart can be judged
        again from the report between two presses of this, and a session held from
        before that would state a verdict against thresholds that are no longer the
        ones on screen.
        """
        from deltachrome.errors import ProfileError
        from deltachrome.session import session_from_records, session_from_results
        from deltachrome_ui.library import read_record

        # Record-shaped dicts for the frame map, built on whichever path ran.
        # A live result and a saved record hold the same two things it needs --
        # the capture's size and each chart's corners -- so the map is fed one
        # shape and never learns there were two ways here.
        records: list[dict] = []
        try:
            if self._open_record is None:
                results = list(self._fresh_results.values())
                if len(results) < 2:
                    return
                # "Target 1", not "chart 1 (Applied Image QA-62 Scanner SFR &
                # OEC #2)". These labels are what the metrics table's *Measured
                # By* column prints, once per capture per row, and on a board of
                # one model that is the same forty-character name repeated four
                # times to a cell -- elided to uselessness and saying nothing,
                # since every target in the frame is the same model. The ordinal
                # is what distinguishes them, and it is the ordinal the frame
                # map paints on the capture and the stepper counts through.
                session = session_from_results(
                    results,
                    labels=[f"Target {n}" for n in range(1, len(results) + 1)],
                )
                records = [
                    {"image": result.run.image, "geometry": result.run.geometry,
                     "target": result.run.target, "run_id": result.run.run_id}
                    for result in results
                ]
                for failure in self._chart_failures:
                    session.warnings = list(session.warnings) + [
                        f"[not measured] {failure}"
                    ]
            else:
                charts = read_record(self._open_record.path).get("charts") or []
                if len(charts) < 2:
                    return
                # Named the same way as a board just measured, and for a
                # sharper reason here: every chart of one capture carries the
                # same filename, so the default names them "QA62_2.jp2",
                # "QA62_2.jp2 (1)", "QA62_2.jp2 (2)" -- distinct and useless.
                session = session_from_records(
                    charts,
                    labels=[f"Target {n}" for n in range(1, len(charts) + 1)],
                )
                records = list(charts)
        except (OSError, ProfileError, ValueError) as exc:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("These charts cannot be combined")
            box.setText("A combined verdict needs charts judged against one set "
                        "of guidelines.")
            box.setInformativeText(str(exc))
            box.exec()
            return
        # A capture already in the library needs no save; one just measured is
        # saved as a **capture**, which is the same record and the same words a
        # chart's report writes. Only a session combining several captures
        # writes a session record.
        frame = self._frame_for(session, records)
        self.session.set_session(
            session, frame=frame,
            save_mode="none" if self._open_record is not None else "capture",
        )
        self.stack.setCurrentWidget(self.session)
        self._fill_frame_map(frame, records)

    def _frame_for(self, session, records: list[dict]):
        """The capture behind a combined verdict, when its charts share one.

        Built here rather than in the engine: a
        :class:`~deltachrome.session.CaptureOutcome` carries no placement on
        purpose -- combining readings needs none -- and widening the record so
        that one screen could draw a picture would put a presentation concern
        into the archive.

        The pixels are handed over only when this shell already has them
        decoded. A capture reopened from the library has none, and the map is
        built for that: the outlines come from each chart's own recorded
        corners, so where each chart sits is answered without a decode that
        costs five to sixteen seconds.
        """
        from deltachrome_ui.frame_map import capture_key, frame_from_records
        from deltachrome_ui.status import overall_status

        statuses = [
            overall_status(
                capture.overall, capture.conformance_claim,
                evidence_sound=capture.reference_basis == "per_unit_measurement",
            )
            for capture in session.captures
        ]
        notes = [
            "" if capture.reference_basis == "per_unit_measurement"
            else "nominal reference"
            for capture in session.captures
        ]
        if len(statuses) != len(records):
            return None
        frame = frame_from_records(records, statuses, notes)
        if frame is None:
            return None
        # Matched on the fingerprint rather than the path: the slot may be
        # holding a different capture entirely, and pixels from the wrong file
        # would be a picture of the wrong frame under the right outlines.
        held = self._report_proxy()
        wanted = capture_key((records[0].get("image") or {}).get("fingerprint"))
        if held is not None and wanted:
            from deltachrome.results.models import file_fingerprint

            try:
                if capture_key(file_fingerprint(held.image.path)) == wanted:
                    frame.image = held.overview
            except OSError:
                pass
        return frame

    def _fill_frame_map(self, frame, records: list[dict]) -> None:
        """Read a small picture of the capture, where the map has none.

        **Not the master.** Decoding ``QA62_2.jp2`` whole is 11.2 s and 303 MB
        resident, to fill a rectangle the map draws at five hundred pixels --
        and a JPEG 2000 carries its own reduced resolution levels, so the small
        version is already in the file and costs 0.61 s and 4.7 MB to read. An
        uncompressed TIFF needs no such trick and reads in 0.14 s whatever its
        size, being a disk read. Worst case across every master in this
        repository, including an 896 MB one: 0.61 s.

        That figure is why this needs no control. The capture panel asks before
        it decodes, shows a bar while it does and refuses a second press,
        because it costs seconds and hundreds of megabytes; none of that is
        worth building around six tenths of a second. It is still off the UI
        thread, and the map draws its outlines meanwhile -- which is exactly
        what it does when there are no pixels at all, so there is no waiting
        state to design.

        **Never for a list.** Six tenths of a second is fine for the one capture
        somebody opened and a minute for a batch of a hundred, so this runs on
        open and nowhere else.
        """
        from deltachrome_ui.frame_map import capture_key
        from deltachrome_ui.workers import ThumbnailLoader

        if frame is None or frame.image is not None or not records:
            return
        image = records[0].get("image") or {}
        key = capture_key(image.get("fingerprint"))
        path = image.get("path")
        if not key or not path or not Path(path).exists():
            # A master that has moved leaves the outlines, which is the whole
            # point of their coming from the record.
            return
        self._thumbnail_key = key
        loader = ThumbnailLoader(Path(path), key)
        loader.loaded.connect(self._on_thumbnail)
        run_in_thread(loader, self)

    def _on_thumbnail(self, image, key) -> None:
        """A thumbnail arrived. Draw it if it still belongs to what is on screen.

        A reader can open a capture, go back and open another inside six tenths
        of a second, and the first read would otherwise land on the second one's
        map -- a picture of the wrong capture under the right outlines, which is
        the failure the fingerprint match exists to prevent everywhere else.
        """
        if key != self._thumbnail_key:
            return
        if self.session.frame_block is not None:
            self.session.frame_block.set_image(image)

    def _on_analysed(self, result) -> None:
        # One chart, so no selector. Cleared for the same reason as above: this
        # report is not a view onto any saved record.
        self._open_record = None
        # And this run is now the capture's whole list. It was left alone, so a
        # board measured and not saved stayed in it: Save then wrote the board,
        # Export exported it, and judging this chart again re-judged the board's
        # charts and put one of them on screen.
        self._fresh_results = {result.run.run_id: result}
        self._live_proxy, self._placed_here = self._proxy, True
        # Before set_result: the report builds its sections from what it has.
        self.result.set_proxy(self._proxy)
        self.result.set_result(result, saved=False)
        self.stack.setCurrentWidget(self.result)

    def _leave_report(self) -> None:
        """Back to wherever the report was opened from.

        A report reached by clicking a capture inside a session belongs to that
        session, and sending it to the library instead would lose the combined
        verdict the operator was reading -- which on a board of charts is the answer
        they came for, the individual reports being the detail underneath it.
        """
        if self._came_from_session and self.session.session is not None:
            self._came_from_session = False
            # Rebuilt, not re-shown. The screen was built from whatever was on
            # file when it was last drawn, and a chart judged again since -- or a
            # whole capture read again by *Change Guidelines…* -- has new run ids and a
            # new verdict. Showing the old screen put stale ids under every
            # outline, and clicking one said the run was not in the library.
            # Where nothing can be rebuilt from, the screen as it was is still
            # where Back was going.
            self._show_combined()
            self.stack.setCurrentWidget(self.session)
            return
        self.show_library()

    def show_placement(self) -> None:
        """Back to the screen the run was set up on, with nothing to rebuild.

        The placement view is built once and lives for the session, so it is
        still holding this capture, this target, these four corners and whether
        anybody moved them. Going back is a change of page. It is offered only
        from a fresh run's report for exactly that reason: a run reopened from
        the library was placed in some other session, and the screen holds
        somebody else's chart or none at all -- which is what "Re-run Capture"
        is for.
        """
        if self._proxy is None or self.placement.proxy is None:
            return
        self.stack.setCurrentWidget(self.placement)

    def _show_chart(self, index: int) -> None:
        """Another chart of the capture already on screen.

        A read, not a run. Every chart's readings were measured when the capture was
        and are in the record beside the one being shown, so switching costs a parse
        and a redraw -- which is what makes a selector the right control for it, and
        what made returning to the placement screen between charts the wrong one.
        """
        # The same metric on the next chart, which is what stepping is for: a
        # reader on SFR50 of target 1 wants SFR50 of target 2, and was put back
        # on the overview every time. A chart that did not measure it opens on
        # the overview, as any missing page does.
        page = self.result.current_page
        # The capture panel too. Every target of the capture is in the same
        # frame, so the pixels already on screen are the next target's pixels,
        # and folding the panel -- or, on a saved capture, dropping the decode
        # and offering it again -- made comparing targets a reload per step.
        panel_open = not self.result.dock.collapsed
        if self._open_record is None:
            fresh = list(self._fresh_results.values())
            if 0 <= index < len(fresh):
                self.result.set_proxy(self._report_proxy())
                self.result.set_result(
                    fresh[index], saved=False, chart=(index, len(fresh)),
                    placed_here=self._placed_here, focus=page, keep_view=True,
                )
        else:
            self._open_run(self._open_record, index, focus=page, keep_view=True)
        if panel_open and self.result.proxy is not None:
            self.result.dock.set_collapsed(False)

    def _rescore(self, profile_id: str) -> None:
        """Judge the reading on screen against a different profile.

        The reason this is worth a control rather than a second analysis: the
        commonest thing wrong with a report is that the wrong level or the wrong
        material category was chosen, and the report is where that becomes
        visible. Re-reading the capture to answer it would cost a decode -- 16.4 s
        on the UTT's JPEG 2000 master -- to recompute numbers that have not
        changed and cannot.

        The screen stays where it is. A fifth of a second is not a wait to put a
        full-window progress bar in front of, and swapping screens and back would
        be more disruption than the work it is reporting.

        **Every chart of the capture is judged again, not the one on screen.** A
        capture record carries one combined verdict against one profile and
        ``capture_record`` requires it, so a board with one card at four star and
        the rest at three is not a state that can be saved or combined. The
        control asks the question of the capture; it merely stands on one chart's
        report, that being where a wrong level becomes visible.
        """
        result = self.result.result
        if result is None or not self.result.can_rescore:
            return
        self.result.set_rescoring(True)
        worker = Rescorer(
            results=list(self._fresh_results.values()) or [result],
            profile_id=profile_id,
        )
        worker.finished.connect(self._on_rescored)
        worker.failed.connect(self._on_rescore_failed)
        run_in_thread(worker, self)

    def _on_rescored(self, results: list) -> None:
        """The same capture, judged again -- and the reader put back where they were.

        ``set_proxy`` first for the same reason a fresh analysis does it: the
        report hands the panel whatever capture it has been given, and this one
        was decoded long ago.

        The capture panel is put back too. Rendering a result folds it away,
        which is right when a *different* run opens and wrong here: this is the
        run already on screen, answering a different question, and a reader who
        opened the panel to look at a patch has not asked for it to be shut.

        **The shell's list of charts is replaced, not just the one on screen.** A
        re-score is a new run with its own id, and that list is what the stepper
        steps through and what Save writes -- so without this, judging chart 2 of
        a board again and pressing Save wrote the judgement it replaced, with the
        header naming a profile the record did not hold.
        """
        if not results:
            return
        panel_open = not self.result.dock.collapsed
        index = self.result.chart_index if len(results) > 1 else 0
        # Whatever the count. A lone chart's re-judgement is a new run as much as
        # a board's is, and leaving the list holding the old one is the same bug
        # one chart smaller.
        self._fresh_results = {r.run.run_id: r for r in results}
        self.result.set_proxy(self._report_proxy())
        # The chart position is kept: this is the run already on screen answering a
        # different question, not a different run opening.
        self.result.set_result(
            results[min(index, len(results) - 1)], saved=False,
            placed_here=self._placed_here, keep_view=True,
            focus=self.result.current_page,
            chart=(index, len(results) if len(results) > 1
                   else self.result.chart_count),
        )
        self.result.dock.set_collapsed(not panel_open)

    def _on_rescore_failed(self, failure: LoadFailure) -> None:
        self.result.set_rescoring(False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("This run could not be judged again")
        box.setText(failure.headline)
        box.setInformativeText(failure.detail)
        box.exec()

    def _establish_measurement(self) -> None:
        """Read a saved run's capture again, from its own record.

        The report stays on screen and usable while this runs, the same way the
        capture panel's own load does: this is work alongside the report, not
        instead of it. What it asks the operator is nothing -- the target, the
        placement, the reference and the run options all travel with the record,
        which is the whole of what makes a control here different from the
        placement screen's "Re-run Capture": that one starts a new setup, and
        this reads the one already on file.
        """
        from deltachrome_ui.library import read_record

        result = self.result.result
        if result is None or self.result.can_rescore or self._open_record is None:
            return
        try:
            record = read_record(self._open_record.path)
        except (OSError, ValueError) as exc:
            self._on_establish_failed(LoadFailure(
                "This run's record could not be read.",
                f"{self._open_record.path.name}: {exc}",
            ))
            return
        charts = record.get("charts")
        self._establishing_charts = (
            list(charts) if isinstance(charts, list) and charts else [record]
        )

        self.result.set_establishing(True)
        held = self._already_decoded(result.run.image or {})
        if held is not None:
            # No thread for the decode -- there is nothing to wait for -- but
            # the analysis worker still runs, because every metric plugin runs
            # again exactly as it does for a fresh capture.
            self._establishing_proxy = held
            self._run_establisher(held.image)
            return

        path = Path((result.run.image or {}).get("path") or "")
        if not path.exists():
            self.result.set_establishing(False)
            QMessageBox.information(
                self,
                "Choose the capture",
                "The image this run recorded is not where it was:\n\n"
                f"{path}\n\nChoose it and this run's own placement will be read "
                "again.",
            )
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open the capture", str(Path.home()), IMAGE_FILTER
            )
            if not chosen:
                return
            path = Path(chosen)
            self.result.set_establishing(True)

        loader = ImageLoader(path)
        loader.loaded.connect(self._on_establish_image_loaded)
        loader.failed.connect(self._on_establish_failed)
        run_in_thread(loader, self)

    def _run_establisher(self, image) -> None:
        worker = RecordEstablisher(image=image, charts=self._establishing_charts)
        worker.finished.connect(self._on_established)
        worker.failed.connect(self._on_establish_failed)
        run_in_thread(worker, self)

    def _on_establish_image_loaded(self, proxy: ImageProxy, path: Path) -> None:
        # Cached the same way the capture panel's own load is -- a second look
        # at this capture, from the panel or from pressing this again, should
        # not cost a second decode.
        self._capture_proxy = proxy
        self._capture_path = path
        self._establishing_proxy = proxy
        self._run_establisher(proxy.image)

    def _on_established(self, outcome) -> None:
        """The capture read again, and the report handed a live measurement.

        Every chart of the record is read, not only the one on screen, for the
        reason :class:`~deltachrome_ui.workers.RecordEstablisher` gives: a
        capture record carries one combined verdict, and a stepper claiming
        every chart was re-established when only one was would be lying about
        the rest.

        ``saved`` becomes ``False``: this is a fresh measurement now, in the
        same shape a just-analyzed capture is in, and Save is what turns a
        fresh measurement into a library record whether it arrived this way or
        by placing a chart from scratch.
        """
        results, failures = outcome
        if not results:
            return
        self._fresh_results = {r.run.run_id: r for r in results}
        self._open_record = None
        # Live now, and not placed on this session's placement screen -- which
        # still holds whatever was placed last. So the report offers Re-run rather
        # than a way "back" to a screen showing another capture, and every later
        # redraw of these results is handed this capture's pixels, not that one's.
        self._live_proxy, self._placed_here = self._establishing_proxy, False
        index = self.result.chart_index if len(results) > 1 else 0
        self.result.set_proxy(self._establishing_proxy)
        self.result.set_result(
            results[min(index, len(results) - 1)], saved=False, placed_here=False,
            keep_view=True,
            focus=self.result.current_page,
            chart=(index, len(results) if len(results) > 1
                   else self.result.chart_count),
        )
        self.result.set_establishing(False)
        if failures:
            QMessageBox.information(
                self, "Some charts could not be read again",
                "\n\n".join(failures),
            )

    def _on_establish_failed(self, failure: LoadFailure) -> None:
        self.result.set_establishing(False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("This capture could not be read again")
        box.setText(failure.headline)
        box.setInformativeText(failure.detail)
        box.exec()

    def _on_analysis_failed(self, failure: LoadFailure) -> None:
        self.stack.setCurrentWidget(self.placement)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Analysis stopped")
        box.setText(failure.headline)
        box.setInformativeText(failure.detail)
        box.exec()

    def _on_capture_loaded(self, proxy: ImageProxy, path: Path) -> None:
        """Hand a decoded capture to the report that asked for it, and keep it.

        A method rather than a direct connection to ``result.set_proxy`` because
        the signal carries the path as well, and because a bound method of this
        window is what makes the delivery queued onto the GUI thread.

        Kept because the second look at the same capture is common -- open a run,
        close it, open it again -- and because what it saves is 16 to 25 seconds
        on a JPEG 2000 master against 175 to 577 MB of memory. One slot: this is
        a cache of the last one, not a history.
        """
        self._capture_proxy = proxy
        self._capture_path = path
        self.result.set_proxy(proxy)

    def _already_decoded(self, record: dict):
        """The capture this report wants, if it is already in memory.

        Both slots are asked: a run just analyzed is holding its own master, and
        a run whose capture was opened from a report is holding that one.

        The **fingerprint** is checked where the record carries one. Matching on
        the path alone would hand back pixels from before the file was replaced,
        and a report drawn over the wrong capture is exactly the class of silent
        wrongness this project refuses elsewhere. It is cheap -- blake2b over the
        size and both ends of the file, which is why the record stores it.
        """
        wanted = str(record.get("path") or "")
        if not wanted:
            return None
        candidates = (
            (self._image_path, self._proxy),
            (self._capture_path, self._capture_proxy),
        )
        for path, proxy in candidates:
            if proxy is None or path is None:
                continue
            try:
                if Path(path).resolve() != Path(wanted).resolve():
                    continue
            except OSError:
                continue
            recorded = record.get("fingerprint")
            if recorded:
                from deltachrome.results.models import file_fingerprint

                try:
                    if file_fingerprint(Path(path)) != recorded:
                        continue
                except OSError:
                    continue
            return proxy
        return None

    def _load_capture_for_report(self) -> None:
        """Decode the capture behind the report now on screen.

        On request rather than on open. Rendering a saved run costs no decode at
        all, and making every reader pay twenty seconds for a JPEG 2000 master to
        populate a panel they may not look at would take that back. The report
        stays on screen and usable while this runs, and the wait is shown *in the
        panel* rather than over the window: this is work alongside the report,
        not instead of it.
        """
        if self.result.result is None:
            return
        record = self.result.result.run.image or {}
        held = self._already_decoded(record)
        if held is not None:
            # No thread and no bar: there is nothing to wait for, and a bar that
            # appeared and vanished within a frame would read as a glitch.
            self.result.set_proxy(held)
            return

        path = Path(record.get("path") or "")
        if not path.exists():
            QMessageBox.information(
                self,
                "Choose the capture",
                "The image this run recorded is not where it was:\n\n"
                f"{path}\n\nChoose it to show the pixels behind each reading.",
            )
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open the capture", str(Path.home()), IMAGE_FILTER
            )
            if not chosen:
                # Nothing was started, so there is nothing to stop -- but the
                # panel is open and must not be left looking as though something
                # is happening.
                self.result.dock.set_loading(False)
                return
            path = Path(chosen)

        # Before the thread starts, so the panel is never in the state that
        # caused this: a live button over a decode nobody could see.
        self.result.dock.set_loading(True)
        loader = ImageLoader(path)
        loader.progress.connect(self.result.dock.busy.set_message)
        loader.loaded.connect(self._on_capture_loaded)
        loader.failed.connect(self._on_capture_load_failed)
        run_in_thread(loader, self)

    def _on_capture_load_failed(self, failure: LoadFailure) -> None:
        """The panel's own failure, which is not the window's.

        It used to share the opening handler, which returns to the library --
        so a master that had moved or would not decode took the reader off a
        perfectly good report to say so. The report is complete without the
        pixels; only the panel failed, and only the panel says so.
        """
        self.result.dock.set_loading(False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("The capture could not be opened")
        box.setText(failure.headline)
        box.setInformativeText(
            failure.detail
            + "\n\nThe report is unaffected: every reading in it was measured "
            "when the run was made. Only the pixels behind them are missing."
        )
        box.exec()

    def _rerun_saved(self, result) -> None:
        """Re-place a saved run's capture, every chart of it.

        From the open record where there is one, which holds them all; otherwise
        from the charts live on the report -- a capture read again by *Change
        Guidelines…* is held there and in no record. Built from the one chart on
        screen, as it was, a re-run of a board came back as one card.
        """
        from dataclasses import replace

        from deltachrome_ui.library import _charts_from, summarise

        if self._open_record is not None:
            self._reuse_run(self._open_record)
            return
        summary = summarise(result.run.to_dict(), Path(result.run.run_id))
        fresh = list(self._fresh_results.values())
        if len(fresh) > 1:
            summary = replace(summary, charts=tuple(
                chart for live in fresh for chart in _charts_from(live.run.to_dict())
            ))
        self._reuse_run(summary)

    def _save_capture(self) -> None:
        """Save the board from its own screen, as the capture that it is.

        The same record ``_save_result`` writes from a chart's report -- named
        after the image, holding every target -- rather than the session record
        this screen used to offer. A session record has no ``run_id``, no image
        and no charts, so the library summarised it as a row called
        ``session-a1b2c3-d4e5f6`` with no target and no capture name, and opening
        it built a report out of a record that is not one.

        That record is still right for a session combining several *captures*,
        which is a different object and keeps its own button.
        """
        fresh = list(self._fresh_results.values())
        if not fresh:
            return
        self._save_result(fresh[0])

    def _save_result(self, result) -> None:
        """One record per capture, holding every chart read from it.

        A board is one capture of one system at one moment, and five records in a
        library is five things to find again and keep together by memory. So the file
        holds all of them, with each chart's own record inside it whole and
        unaltered -- schema 7, re-scorable, and rendered by the same report code as
        ever. The container is additive.

        Written under the first chart's run id, so a second run of the same capture
        against a different profile sits beside the first rather than replacing it.
        """
        from deltachrome.session import capture_record

        directory = ensure_run_dir()
        charts = list(self._fresh_results.values()) or [result]
        # A report reached from a saved capture is already in the library; only a
        # fresh measurement gets here, and its siblings are the ones just measured.
        # The same names the two combining screens use, so a warning prefixed
        # in this record reads as the same thing the frame map paints on the
        # capture. They are labels and never identifiers -- nothing reads one
        # back -- so older records keeping their own wording costs nothing.
        record = capture_record(
            charts,
            labels=[f"Target {n}" for n in range(1, len(charts) + 1)],
        )
        path = record.save(directory / f"{record.capture_id}.json")
        held = ("Capture record written to" if len(charts) > 1
                else "Run record written to")
        QMessageBox.information(
            self, "Saved",
            f"{held}\n{path}"
            + (f"\n\n{len(charts)} charts, combined verdict {record.overall}."
               if len(charts) > 1 else ""),
        )
        self._fresh_results = {}
        self.show_library()

    def _capture_for_export(self):
        """The capture on screen, as the record Save writes or the library holds.

        A record already in the library is read as it is on disk -- what was
        saved is what is sent, not a rebuilding of it. Charts just measured are
        combined exactly as Save combines them, so exporting before saving and
        after cannot disagree.
        """
        from deltachrome.session import capture_record
        from deltachrome_ui.library import read_record

        if self._open_record is not None:
            return read_record(self._open_record.path)
        charts = list(self._fresh_results.values())
        if not charts and self.result.result is not None:
            charts = [self.result.result]
        if not charts:
            return None
        return capture_record(
            charts, labels=[f"Target {n}" for n in range(1, len(charts) + 1)],
        )

    def _export_capture(self, button=None) -> None:
        """Write the capture on screen as text, CSV or XML.

        Every chart of it, from whichever of its screens the press came from --
        the export is of the capture, as Save is.
        """
        from deltachrome.errors import DeltachromeError

        try:
            record = self._capture_for_export()
        except (OSError, ValueError, DeltachromeError) as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            return
        self._export_record(record, button)

    def _export_from_library(self, run) -> None:
        """A row's export: the record as it is on disk, as Open would read it."""
        from deltachrome.errors import DeltachromeError
        from deltachrome_ui.library import read_record

        try:
            record = read_record(run.path)
        except (OSError, ValueError, DeltachromeError) as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            return
        self._export_record(record)

    def _ask_export_format(self, button=None) -> str | None:
        """Which format, from a menu under the button pressed or at the pointer.

        Opens on the format chosen last, since a department exporting a day's
        captures exports them all the same way.
        """
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        actions = {}
        for key in EXPORT_FILTERS:
            actions[menu.addAction(key.replace("*", ""))] = key
        last = next((a for a, k in actions.items() if k == self._export_format), None)
        if last is not None:
            menu.setActiveAction(last)
        where = (button.mapToGlobal(button.rect().bottomLeft())
                 if button is not None else QCursor.pos())
        chosen = menu.exec(where)
        return actions.get(chosen)

    def _export_record(self, record, button=None) -> None:
        """Ask which format and where, then write the record."""
        from deltachrome.errors import DeltachromeError
        from deltachrome.reporting.export import export

        if record is None:
            return
        chosen = self._ask_export_format(button)
        if chosen is None:
            return
        data = record if isinstance(record, dict) else record.to_dict()
        stem = Path(str((data.get("image") or {}).get("filename") or "report")).stem
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Values",
            str(self._export_dir / f"{stem}{EXPORT_FILTERS[chosen]}"), chosen,
        )
        if not path:
            return
        self._export_format = chosen
        path = export_path(path, chosen)
        try:
            export(data, path)
        except (OSError, DeltachromeError) as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            return
        self._export_dir = path.parent
        if button is not None:
            button.flash()

    def _on_load_failed(self, failure: LoadFailure) -> None:
        self.show_library()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Could not open")
        box.setText(failure.headline)
        box.setInformativeText(failure.detail)
        box.exec()


def bundle_supplies_icon() -> bool:
    """Whether a packaged macOS application already carries its own icon.

    It does, rendered up to 1024 px into the bundle's ``.icns``, and the dock
    shows it crisply while the application is closed. Setting a window icon
    replaces it for as long as the application runs, and macOS drew that
    replacement visibly soft. A run from source has no
    bundle and still needs the call, or the dock shows Python's rocket; so
    does Windows, where the call also sets the title bar's icon.
    """
    return sys.platform == "darwin" and bool(getattr(sys, "frozen", False))


def main(argv: list[str] | None = None) -> int:
    ensure_run_dir()
    # Before anything reads the profile catalogue, so a guideline folder chosen in
    # a previous session is in force for the first list that is drawn from it.
    restore_guideline_dir()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Deltachrome")
    # Set before the first window exists, so the dock never shows the
    # interpreter's own icon even for the moment the library takes to build.
    # Without this the application has no icon at all and macOS falls back to
    # the Python rocket, which is the most visible unbranded surface it has.
    if not bundle_supplies_icon():
        app.setWindowIcon(app_icon(MARK_INK, RAMP, BG_INPUT))
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
