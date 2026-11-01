"""Take the user guide's screenshots, from the application itself.

The guide on https://permachro.me marks the places a screenshot belongs with
``[[figure: name.png | caption]]``. This renders each of them from the real main
window, driven the way an operator would drive it, under Qt's offscreen platform
at a scale factor of two -- so a 1440 x 900 window is a 2880 x 1800 PNG, which
the site displays at half size. The interface keeps changing and this re-shoots
every figure in one command.

**Only one capture of a target is shown: the LaserSoft 4x5**, in
``data/screenshot_target``. It is the one capture this project may publish; the
other captures belong to the organisations that published them. Measurements are another matter and
any of ours may be used, so the run is judged against the chart's own unit's
file, ``data/reference/T1910021.cxf`` -- the CxF beside the capture is a different
unit's and is in the bank only so that it lists more than one file.

**The application runs against a fresh home**, so no test guideline, private run
record or personal path appears. ``HOME`` is pointed at a scratch folder rather
than ``DELTACHROME_HOME``, because the latter pins the folders and Settings then
shows every *Change…* disabled; the scratch prefix is printed as ``~`` on the
screens that show a path.

Run from the repository root::

    python tools/capture_guide_screenshots.py
    python tools/capture_guide_screenshots.py --out some/folder

The default output is the site repository's ``content/guide/images``. Two figures
the guide marks are not taken, because both need a frame holding several targets
and the one publishable capture holds one: the numbered targets on the Placement
screen and the All Targets screen.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "data" / "screenshot_target" / "LaserSoft_4x5.tif"
#: The chart's own unit is T1910021, and its measurement is the run's reference.
#: The second file gives the bank more than one entry to list.
REFERENCES = (
    ROOT / "data" / "reference" / "T1910021.cxf",
    ROOT / "data" / "screenshot_target" / "T1710030.cxf",
)
DEFAULT_OUT = ROOT.parent / "permachro.me" / "content" / "guide" / "images"

#: Every full-window figure is this size in logical pixels, so the guide reads as
#: one product. Twice this in the PNG.
WINDOW = (1440, 900)
SCALE = 2

#: The guideline the analysis figures are judged against: the chart's own
#: material category, a 4x5 colour transparency.
PROFILE = "fadgi_4star_photographic_transparencies_4x5_and_larger_2023"
#: Further judgements of the same reading, saved so the library holds more than
#: one row. Each is a re-score, so nothing is measured twice.
OTHER_PROFILES = (
    "fadgi_3star_photographic_transparencies_4x5_and_larger_2023",
    "metamorfoze_full_2025",
)
#: The metric the Charts-tab figure opens on, and the patch it points at.
CHART_METRIC = "tone_response_oecf"
#: How many captures the batch folder holds, and after how many the progress
#: screen is taken. Copy-on-write clones of the one master.
BATCH_FOLDER = "Transparencies 2026-10-07"
BATCH_SIZE, BATCH_SHOT_AT = 6, 4
#: The guideline the builder figure makes, its parent, and the metric it sets to
#: Measured.
GUIDELINE_NAME = "Studio Transparencies"
GUIDELINE_PARENT = PROFILE
MEASURED_METRIC = "sampling_frequency"
#: Which of a library row's actions the row figure points at: Export Values.
EXPORT_ICON = 3
#: An explicit binding, so the Reference Files figure shows a decision beside the
#: codes the file answers to by itself. The chart's printed barcode.
BOUND_CODE = "1930021"


def _environment(home: Path) -> None:
    """Point the application at a clean home before anything of Qt is imported."""
    os.environ["HOME"] = str(home)
    for name in list(os.environ):
        if name.startswith("DELTACHROME_"):
            os.environ.pop(name)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = str(SCALE)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    sys.path.insert(0, str(ROOT / "src"))


def pump(seconds: float = 0.1) -> None:
    from PySide6.QtWidgets import QApplication

    end = time.monotonic() + seconds
    while True:
        QApplication.processEvents()
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def wait_for(condition, what: str, timeout: float = 300.0, state=None) -> None:
    from PySide6.QtWidgets import QApplication

    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise TimeoutError(f"Timed out waiting for {what}"
                               + (f": {state()}" if state else ""))
        QApplication.processEvents()
        time.sleep(0.02)


def _clone(source: Path, destination: Path) -> None:
    """A copy-on-write clone where the filesystem has them, a copy where not."""
    if sys.platform == "darwin":
        done = subprocess.run(["cp", "-c", str(source), str(destination)],
                              capture_output=True)
        if done.returncode == 0:
            return
    shutil.copyfile(source, destination)


def _mouse(widget, kind, pos, buttons) -> None:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    point = QPointF(*pos)
    button = (Qt.MouseButton.LeftButton if kind != QMouseEvent.Type.MouseMove
              else Qt.MouseButton.NoButton)
    event = QMouseEvent(kind, point, widget.mapToGlobal(point), button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


class Guide:
    def __init__(self, out: Path, home: Path) -> None:
        self.out = out
        self.home = home
        self.window = None
        out.mkdir(parents=True, exist_ok=True)

    # -- output -------------------------------------------------------------

    def save(self, widget, name: str, rect=None) -> None:
        pump(0.4)
        self._tidy_paths(self.window)
        pixmap = widget.grab(rect) if rect is not None else widget.grab()
        path = self.out / f"{name}.png"
        pixmap.save(str(path), "PNG")
        print(f"  {path.name}: {pixmap.width()} x {pixmap.height()}", flush=True)

    def _tidy_paths(self, root) -> None:
        """Print the scratch home as ``~`` wherever a label shows a path."""
        from PySide6.QtWidgets import QLabel

        prefixes = sorted({str(self.home.resolve()), str(self.home)}, key=len,
                          reverse=True)
        for label in root.findChildren(QLabel):
            text = label.text()
            tidied = text
            for prefix in prefixes:
                tidied = tidied.replace(prefix, "~")
            if tidied != text:
                label.setText(tidied)
                label.setToolTip(tidied)

    # -- the run ------------------------------------------------------------

    def run(self) -> None:
        from PySide6.QtWidgets import QApplication, QMessageBox

        from deltachrome.reference.library import ReferenceLibrary
        from deltachrome_ui.app import MainWindow
        from deltachrome_ui.paths import (
            ensure_reference_bank,
            ensure_run_dir,
            restore_guideline_dir,
        )
        from deltachrome_ui.theme import STYLESHEET

        ensure_run_dir()
        restore_guideline_dir()
        bank = ReferenceLibrary(ensure_reference_bank())
        for reference in REFERENCES:
            bank.add_file(reference)
        bank.bind(BOUND_CODE, REFERENCES[0].name)

        # Saving puts up a modal "Saved" box, which offscreen nobody can dismiss.
        QMessageBox.information = staticmethod(lambda *a, **k: None)

        app = QApplication.instance() or QApplication([])
        app.setApplicationName("Deltachrome")
        app.setStyleSheet(STYLESHEET)
        self.window = window = MainWindow()
        window.resize(*WINDOW)
        window.show()
        pump(0.5)

        result = self.placement_and_analysis()
        self.save_runs(result)
        self.batch()
        self.library()
        self.settings()
        window.close()
        pump(0.2)

    def placement_and_analysis(self):
        from deltachrome_ui.choices import LIBRARY

        window = self.window
        print("Placement", flush=True)
        window.load_image(CAPTURE)
        wait_for(lambda: window.stack.currentWidget() is window.placement,
                 "the placement screen")
        choices = window.placement.choices
        choices.mode_buttons[LIBRARY].setChecked(True)
        choices.select_profile(PROFILE)
        self._wait_for_sampling()
        self.save(window, "placement")

        self.search_a_region()

        print("Analysis", flush=True)
        window.placement.continueRequested.emit(list(window.placement.canvas.corners()))
        wait_for(lambda: window.stack.currentWidget() is window.result,
                 "the analysis screen")
        pump(1.0)
        self.save(window, "analysis-overview")

        view = window.result
        family = next(key[1] for key in view._page_index
                      if len(key) == 3 and key[2] == CHART_METRIC)
        view.show_metric(family, CHART_METRIC)
        pump(0.5)
        view.dock.tab.click()
        pump(1.5)
        metric = view.pages.currentWidget().current_metric()
        steps = [s for s in metric.get("per_patch") or metric.get("samples") or []
                 if s.get("patch_id")]
        if steps:
            view.selection.set_hovered(str(steps[len(steps) // 2]["patch_id"]))
        self.save(window, "metric-charts")
        return view.result

    def _wait_for_sampling(self) -> None:
        placement = self.window.placement
        wait_for(lambda: placement.canvas.showing_sample_regions,
                 "the sampled regions")
        pump(0.5)

    def search_a_region(self) -> None:
        """The rectangle mid-drag, round the whole card with a margin."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QMouseEvent

        placement = self.window.placement
        canvas = placement.canvas
        placement.region_button.click()
        pump(0.2)
        # The card's own corners, from the placement, pushed out by a margin.
        corners = canvas.corners()
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        # The card reaches well past its marks, most of all below them, where the
        # maker's name and barcode are printed.
        x0, y0 = min(xs) - 0.08 * width, min(ys) - 0.20 * height
        x1, y1 = max(xs) + 0.10 * width, max(ys) + 0.30 * height
        start = canvas.mapFromScene(x0, y0)
        end = canvas.mapFromScene(x1, y1)
        viewport = canvas.viewport()
        held = Qt.MouseButton.LeftButton
        _mouse(viewport, QMouseEvent.Type.MouseButtonPress, (start.x(), start.y()), held)
        steps = 12
        for i in range(1, steps + 1):
            x = start.x() + (end.x() - start.x()) * i / steps
            y = start.y() + (end.y() - start.y()) * i / steps
            event_pos = (x, y)
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QMouseEvent as _E
            from PySide6.QtWidgets import QApplication

            p = QPointF(*event_pos)
            QApplication.sendEvent(viewport, _E(
                QMouseEvent.Type.MouseMove, p, viewport.mapToGlobal(p),
                Qt.MouseButton.NoButton, held, Qt.KeyboardModifier.NoModifier))
            pump(0.02)
        self.save(self.window, "search-a-region")
        # Abandon the gesture rather than searching: the placement is already right.
        canvas.cancel_region_search()
        placement.region_button.setChecked(False)
        _mouse(viewport, QMouseEvent.Type.MouseButtonRelease, (end.x(), end.y()),
               Qt.MouseButton.NoButton)
        canvas.setDragMode(canvas.DragMode.ScrollHandDrag)
        pump(0.3)

    def save_runs(self, result) -> None:
        """The run on screen, and the same reading judged against two more."""
        from deltachrome.pipeline import rescore
        from deltachrome.session import capture_record
        from deltachrome_ui.paths import ensure_run_dir

        print("Saving runs", flush=True)
        directory = ensure_run_dir()
        for profile in OTHER_PROFILES:
            other = rescore(result, profile)
            record = capture_record([other], labels=["Target 1"])
            record.save(directory / f"{record.capture_id}.json")
            pump(1.1)  # so each row carries its own time
        self.window._save_result(result)
        pump(0.5)

    def batch(self) -> None:
        window = self.window
        print("Batch", flush=True)
        folder = self.home / "Pictures" / BATCH_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        captures = []
        for n in range(1, BATCH_SIZE + 1):
            copy = folder / f"Transparency_{n:03d}.tif"
            _clone(CAPTURE, copy)
            captures.append(copy)
        window._pending_batch = (folder, captures)
        window._pending_corners = None
        window._pending_charts = ()
        window.load_image(captures[0])
        wait_for(lambda: window.stack.currentWidget() is window.placement,
                 "the batch's placement")
        choices = window.placement.choices
        target = "lasersoft_iso12641_2_advanced_v1"
        choices.select_target(target)
        pump(0.3)
        if not window.placement.target_settled:
            # Already showing in the list, so choosing it fired no change.
            window._on_target_chosen(target)
        from deltachrome_ui.choices import LIBRARY

        choices.mode_buttons[LIBRARY].setChecked(True)
        choices.select_profile(PROFILE)
        wait_for(lambda: window.placement.placement_settled
                 and window.placement.canvas.showing_sample_regions,
                 "the batch placement", timeout=120,
                 state=lambda: (window.placement.placement_settled,
                                window.placement.canvas.showing_sample_regions,
                                choices.target_id))
        pump(0.5)
        window.placement.continueRequested.emit(list(window.placement.canvas.corners()))
        wait_for(lambda: window.batch.table.rowCount() >= BATCH_SHOT_AT,
                 "the batch to get partway")
        self.save(window, "batch-progress")
        wait_for(lambda: window._batch_worker is None, "the batch to finish")
        pump(0.5)

    def library(self) -> None:
        window = self.window
        print("Library", flush=True)
        window.show_library()
        pump(0.5)
        table = window.library.table
        model = table.model()
        # Open the batch, so the figure shows a batch row and what it holds.
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if model.hasChildren(index):
                table.expand(index)
        pump(0.3)
        self.save(window, "run-library")

        # A row under the pointer, its actions lit and the pointer on Export.
        # Set on the view rather than by a synthetic mouse move, which the
        # offscreen platform does not route as hover. The batch is shut again
        # first, so the figure is a few rows rather than the whole library.
        for r in range(model.rowCount()):
            if model.hasChildren(model.index(r, 0)):
                table.collapse(model.index(r, 0))
        pump(0.2)
        row = next(r for r in range(model.rowCount())
                   if not model.hasChildren(model.index(r, 0)))
        index = model.index(row, 0)
        table.hovered_row, table.hovered_icon = index, EXPORT_ICON
        table.viewport().update()
        pump(0.3)
        from PySide6.QtCore import QPoint, QRect

        rect = table.visualRect(index)
        header = table.header()
        top = header.mapTo(window, QPoint(0, 0)).y()
        row_bottom = table.viewport().mapTo(window, rect.bottomLeft()).y()
        left = table.mapTo(window, QPoint(0, 0)).x()
        crop = QRect(left - 12, top - 12, table.width() + 24,
                     row_bottom - top + 2 * rect.height() + 24)
        self.save(window, "run-table-row", crop)

    def settings(self) -> None:
        window = self.window
        print("Settings", flush=True)
        window.show_settings()
        settings = window.settings
        settings.nav.setCurrentRow(0)
        pump(0.3)
        self.save(window, "settings-general")

        settings.nav.setCurrentRow(2)
        bank = settings.bank
        bank.reload(select=REFERENCES[0].name)
        pump(0.5)
        self.save(window, "settings-reference-files")

        settings.nav.setCurrentRow(3)
        panel = settings.guidelines
        panel.add_button.click()
        pump(0.3)
        builder = panel.builder
        builder.picker.select_profile(GUIDELINE_PARENT)
        pump(0.3)
        builder.name.setText(GUIDELINE_NAME)
        from deltachrome_ui.guideline_builder import MEASURED

        for edit in builder._edits:
            if getattr(edit.spec, "metric_id", None) == MEASURED_METRIC:
                edit.state.setCurrentIndex(edit.state.findData(MEASURED))
        pump(0.5)
        self.save(window, "guideline-builder")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-home", action="store_true",
                        help="leave the scratch home in place afterwards")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="deltachrome-guide-"))
    _environment(home)
    try:
        Guide(args.out.resolve(), home).run()
    finally:
        if args.keep_home:
            print(f"Home left at {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
