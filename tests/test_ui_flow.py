"""The setup panel, the result screen, and the thread that joins them.

The setup tests are about one idea: **a control says it, or nothing does.** The
choices used to be a screen of their own carrying six paragraphs of prose -- what
the identifier keyed on, how many of a profile's metrics this chart could reach,
which capabilities it lacked, how automatic placement works. All of it true, and
all of it either shown by a control right beside it or answered better by the
report. What survives in the open is the one line that changes what the operator
has to do next: whether this chart needs a reference file.

The result tests are about the same idea as ``test_ui_status``: the report is the
last place a qualified verdict can be flattened into a green tick, so the
caveats travel with it.

The worker test guards a bug that was real, intermittent, and nasty -- see
:func:`test_a_running_worker_is_not_collected_when_the_caller_returns`.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.targets.registry import load_target  # noqa: E402
from deltachrome_ui.choices import (  # noqa: E402
    AUTOMATIC,
    LIBRARY,
    REFERENCE_REQUIRED,
    ChoicesPanel,
)
from deltachrome_ui.labels import condition_label  # noqa: E402
from deltachrome_ui.workers import run_in_thread  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def setup(qt_app):
    return ChoicesPanel()


@pytest.fixture
def empty_bank(tmp_path, monkeypatch):
    """Point the reference bank somewhere with nothing in it.

    Through the environment variable rather than by patching a function, because
    that is the resolution order the application itself documents and it is what
    a shared-drive deployment uses.
    """
    bank = tmp_path / "bank"
    bank.mkdir()
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(bank))
    return bank


@pytest.fixture
def stocked_bank(empty_bank):
    (empty_bank / "CCSG21.txt").write_text(
        "DESCRIPTOR \"ColorChecker SG\"\nTARGET_SERIAL CCSG21\n"
        "BEGIN_DATA_FORMAT\nSAMPLE_NAME LAB_L LAB_A LAB_B\nEND_DATA_FORMAT\n",
        encoding="utf-8",
    )
    return empty_bank


# isVisible() is False for every child of an unshown window, so these tests ask
# isVisibleTo(ancestor): would this be visible if the window were on screen.
def settle(ms: int = 60) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_until(predicate, *, seconds: float = 20.0) -> bool:
    """Pump the event loop until ``predicate`` holds, or the deadline passes."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        settle(25)
    return predicate()


# -- choosing ----------------------------------------------------------------


def test_the_level_list_only_holds_one_category(setup):
    """The reason for two combos: a single flat list collapses to a closed box
    reading '3-star', which says nothing about three-star what."""
    setup.select_profile("fadgi_3star_prints_photographs_2023")
    ids = {setup.profile_combo.itemData(i) for i in range(setup.profile_combo.count())}
    assert all("prints_photographs" in pid for pid in ids)


def test_a_profile_can_be_restored_from_its_id_alone(setup):
    """Needed by anything restoring a choice rather than making one."""
    assert setup.select_profile("metamorfoze_light_2025")
    assert setup.profile_id == "metamorfoze_light_2025"
    assert "Metamorfoze" in setup.category_combo.currentText()


def test_restoring_an_unknown_profile_reports_failure_rather_than_guessing(setup):
    assert not setup.select_profile("no_such_profile")


def test_changing_category_leaves_a_valid_level_selected(setup):
    for index in range(setup.category_combo.count()):
        setup.category_combo.setCurrentIndex(index)
        assert setup.profile_id, f"category {index} left nothing selected"


# -- what is said, and what is shown instead ---------------------------------


def test_the_profile_flag_survives_because_it_is_not_about_this_application(setup):
    """FADGI's transparency categories specify no color accuracy metric at all,
    so a report against one never mentions color and never says why. That is a
    property of the standard, and nothing on this panel shows it otherwise."""
    setup.select_target("colorchecker_sg_v1")
    setup.select_profile("fadgi_3star_photographic_transparencies_4x5_and_larger_2023")
    assert setup.profile_flags.isVisibleTo(setup)
    assert "no color accuracy" in setup.profile_flags.text()


def test_the_panel_states_no_coverage_forecast(setup):
    """It used to count a profile's metrics against a target's capabilities in
    prose, before anything had been measured. The report answers the same
    question from actual readings, which is both later and better."""
    setup.select_target("colorchecker_sg_v1")
    setup.select_profile("fadgi_3star_prints_photographs_2023")
    from PySide6.QtWidgets import QLabel

    text = " ".join(label.text() for label in setup.findChildren(QLabel))
    for gone in ("declares", "can measure", "Cannot measure", "not applicable"):
        assert gone not in text, gone


def test_the_identifiers_reasoning_is_a_tooltip_rather_than_a_paragraph(setup):
    """'aspect off by 1%, ramp +1.00, 0% flagged' is a diagnostic for whoever
    tunes the finder. What an operator needs is that a chart was named -- and
    which, which is the combo box directly below the pill."""
    from deltachrome.imaging.detect import Identification

    setup.show_identification(Identification(
        target_id="avian_ngt_v1", found=True,
        reason="Avian Rochester Next Generation Target, identified from the "
               "capture: aspect off by 1%, ramp +1.00.",
    ))
    assert setup.target_pill.text() == "Target Detected"
    assert "aspect off by" not in setup.target_pill.text()
    assert "aspect off by" in setup.target_pill.toolTip()


# -- the reference requirement -----------------------------------------------


def test_a_chart_with_no_values_of_its_own_says_a_file_is_required(setup):
    setup.select_target("colorchecker_sg_v1")
    assert setup.mode == AUTOMATIC
    assert setup.reference_pill.isVisibleTo(setup)
    assert setup.reference_pill.text() == REFERENCE_REQUIRED
    assert not setup.ready, "a run with nothing to measure against cannot start"


def test_a_chart_that_ships_nominal_values_asks_for_nothing(setup):
    """The ISA targets are seldom sold measured and fall back to the
    manufacturer's values, which costs the run its conformance claim -- said by
    the verdict, where it cannot be read past, rather than here."""
    setup.select_target("isa_golden_thread_dlt_v1")
    assert not setup.reference_pill.isVisibleTo(setup)
    assert setup.ready


def test_the_requirement_is_worded_the_same_for_every_chart_that_has_it(setup):
    """One bit, not a paragraph about print tolerance: a target either ships
    values to fall back on or it does not."""
    seen = set()
    for target_id in ("colorchecker_sg_v1", "xrite_it87_1_v1"):
        if setup.select_target(target_id) and setup.reference_pill.isVisibleTo(setup):
            seen.add(setup.reference_pill.text())
    assert seen == {REFERENCE_REQUIRED}


def test_pairing_with_the_bank_satisfies_the_requirement(setup, stocked_bank):
    """The chip means 'there is something left to do', so it goes when there is
    not -- otherwise its presence reads as 'this chart is fussy'."""
    setup.select_target("colorchecker_sg_v1")
    setup.refresh_bank()
    setup.mode_buttons[LIBRARY].setChecked(True)
    assert setup.reference_choice == stocked_bank
    assert not setup.reference_pill.isVisibleTo(setup)
    assert setup.ready


def test_pairing_opens_no_dialog_and_says_what_the_bank_holds(setup, stocked_bank):
    """The change this mode exists for. It used to ask for a folder on every run
    and remember it nowhere, so the one mode that can use a chart's printed code
    cost the most gestures of the three.

    What it reports is the bank's contents rather than its path: nobody using
    this mode has to think about the folder, and the whole point is that they do
    not have to."""
    setup.select_target("colorchecker_sg_v1")
    setup.refresh_bank()
    setup.mode_buttons[LIBRARY].setChecked(True)
    assert not setup.choose_button.isVisibleTo(setup)
    assert setup.path_label.text() == "1 reference file in the bank"
    assert str(stocked_bank) in setup.path_label.toolTip()


def test_an_empty_bank_cannot_be_paired_against(setup, empty_bank):
    """Disabled rather than hidden: the fix is one screen away, and a control that
    vanished would not say where."""
    setup.select_target("colorchecker_sg_v1")
    setup.refresh_bank()
    assert not setup.mode_buttons[LIBRARY].isEnabled()
    assert "Settings" in setup.mode_buttons[LIBRARY].toolTip()


def test_a_chart_that_compares_nothing_is_not_offered_a_reference_at_all(setup):
    """The QA-62 ships no aim values and reads both its capabilities off its
    slanted square, so a reference file chosen here would be read by nothing. A
    control that cannot affect the run invites an answer and records it beside a
    result it did not change -- so the whole group goes, and the run is ready."""
    setup.select_target("applied_image_qa62_sfr_v1")
    setup.select_profile("fadgi_3star_documents_unbound_general_2023")
    assert not setup.reference_heading.isVisibleTo(setup)
    assert not setup.mode_buttons[LIBRARY].isVisibleTo(setup)
    assert not setup.reference_pill.isVisibleTo(setup)
    assert setup.ready, "a chart needing no reference must not be blocked by one"


# -- restoring a whole choice ------------------------------------------------


def test_restoring_a_choice_asks_for_one_placement_rather_than_three(setup):
    """Every setter here is also a user gesture, and a target gesture means
    'find this chart'. Restoring a saved run touches three of them in a row, and
    each would ask for a search of a chart the next line replaces."""
    asked: list[str] = []
    setup.targetChanged.connect(asked.append)
    setup.restore(
        target_id="isa_golden_thread_dlt_v1",
        profile_id="fadgi_3star_prints_photographs_2023",
    )
    assert asked == [], "restoring asked for a placement mid-restore"
    assert setup.target_id == "isa_golden_thread_dlt_v1"
    assert setup.profile_id == "fadgi_3star_prints_photographs_2023"
    # And an ordinary choice afterwards still asks, once.
    setup.select_target("colorchecker_sg_v1")
    assert asked == ["colorchecker_sg_v1"]


# -- the worker's lifetime ---------------------------------------------------


class SlowWorker(QObject):
    """Long enough to still be running when the caller's reference goes away."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.ran = False

    def run(self) -> None:
        time.sleep(0.05)
        self.ran = True
        self.finished.emit("done")


def drain(owner, *, seconds: float = 20.0) -> bool:
    """Wait until no job is still registered on ``owner``.

    Deliberately not holding thread references and waiting on those: the records
    are the only owner of the QThread, so reaching into them to call ``wait``
    would defeat the lifetime rule the code under test relies on. Pumping the
    loop is what delivers the cleanup, so pumping is what this does.
    """
    return wait_until(lambda: not getattr(owner, "_threads", {}), seconds=seconds)


@pytest.fixture
def owner(qt_app):
    """An owner that outlives its threads.

    A QThread parented to a QObject is destroyed with it, and destroying a
    running QThread crashes the process -- so a test that lets its owner fall out
    of scope mid-run segfaults. The application's owner is the main window, which
    waits for its threads in ``closeEvent`` for exactly this reason.
    """
    holder = QObject()
    yield holder
    drain(holder)


def test_a_running_worker_is_not_collected_when_the_caller_returns(owner):
    """The bug this guards was real and intermittent.

    ``moveToThread`` requires the worker to have no parent, so after
    ``run_in_thread`` returns, nothing but the caller's local variable refers to
    it -- and Python is free to collect a worker that is still running. It
    survives a fast job and vanishes during a slow one, which is the worst
    possible way for a bug to behave.
    """
    received: list[object] = []

    def start() -> None:
        worker = SlowWorker()
        worker.finished.connect(received.append)
        run_in_thread(worker, owner)
        # `worker` goes out of scope here, exactly as it does in the app.

    start()
    gc.collect()  # the collection that used to take the worker with it

    assert wait_until(lambda: bool(received)), "the worker never reported back"
    assert received == ["done"], "the worker was collected before it finished"


def test_finished_workers_do_not_accumulate(owner):
    """The retention must be bounded, or a long session leaks a thread per run."""
    done: list[object] = []
    for _ in range(3):
        worker = SlowWorker()
        worker.finished.connect(done.append)
        run_in_thread(worker, owner)
    assert wait_until(lambda: len(done) == 3 and not owner._threads), (
        f"finished {len(done)}/3, {len(owner._threads)} thread(s) still retained"
    )
    assert not owner._threads, "threads were retained after finishing"


# -- the report --------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A real analysis, not a stand-in, so schema drift shows up here."""
    from fixtures import SyntheticChart

    tmp = tmp_path_factory.mktemp("report")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return tmp, reference, image, corners


def analyze(synthetic, profile_id):
    from deltachrome.pipeline import analyze

    _, reference, image, corners = synthetic
    return analyze(
        image, target="colorchecker_sg_v1", reference_path=reference,
        profile=profile_id, corners=corners,
    )


def render(qt_app, result):
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(result)
    return view


def all_text(widget) -> str:
    """Everything on screen: the labels, and the tables' contents.

    The report's lists became models rather than grids of labels, so what a
    reader sees is no longer all reachable through ``findChildren``.
    """
    from PySide6.QtWidgets import QLabel

    from deltachrome_ui.tables import DataTable

    lines = [c.text() for c in widget.findChildren(QLabel) if c.text()]
    for table in widget.findChildren(DataTable):
        model = table.model()
        for row in range(model.rowCount()):
            lines.append(" | ".join(
                str(model.index(row, column).data() or "")
                for column in range(model.columnCount())
            ))
    return "\n".join(lines)


def test_a_clean_conformance_run_reports_a_plain_pass(qt_app, synthetic):
    result = analyze(synthetic, "fadgi_3star_documents_unbound_general_2023")
    assert result.evaluation.conformance_claim
    text = all_text(render(qt_app, result))
    assert "PASS" in text
    assert "ADVISORY" not in text


def test_a_local_guideline_is_shown_as_indicative_with_its_reason(
    qt_app, synthetic, tmp_path, monkeypatch
):
    """A local guideline is not a published standard, whichever way a threshold
    moved, and the report has to say which guideline was actually applied -- by
    its own name, and without naming the standard it happened to start from."""
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path))
    from deltachrome.profiles.schema import load_profile

    base = load_profile("fadgi_3star_documents_unbound_general_2023")
    base.derive(
        "local_report_spec", thresholds={"white_balance": "value < 9"},
        display_name="Our House Standard",
    ).save(tmp_path)
    result = analyze(synthetic, str(tmp_path / "local_report_spec.json"))

    assert not result.evaluation.conformance_claim
    # ...and the run is not thereby advisory. The chart itself was measured, so the
    # reading stands; what it is not is a claim against a published standard, which
    # is a different sentence and is said as one. Demoting on this too meant a
    # department that wrote its own standard down could never meet it, which taught
    # the qualification to mean nothing.
    assert result.evaluation.evidence_is_sound
    view = render(qt_app, result)
    text = all_text(view)
    assert "PASS" in text
    assert "advisory" not in text.lower()
    # One line saying what kind of qualification this is...
    assert "Local guideline" in text
    # ...and the header naming the guideline that was actually applied, by the
    # name its author gave it and under a heading that says what it is.
    chosen = view.picker.category_combo.currentText()
    assert chosen == "LOCAL GUIDELINE — Our House Standard"
    # And nothing on the report reaches back to what it was derived from. The
    # record keeps that; a reader has no use for it and every reason to be
    # confused by it, the parent having possibly been chosen for convenience.
    header = text + chosen + view.picker.level_combo.currentText()
    assert "fadgi_3star_documents_unbound_general_2023" not in header
    assert "Thresholds derived from" not in header
    assert "Documents (Unbound)" not in header

    # And the exact string the CLI prints is still reachable, which is the
    # property deltachrome_ui.status exists to hold: a figure read off the screen
    # and one read out of the terminal are the same figure.
    from deltachrome_ui.widgets import VerdictStripe

    stripe = view.header.findChild(VerdictStripe)
    assert stripe.toolTip().startswith("PASS")
    assert result.evaluation.verdict_line == "PASS"


def test_every_metric_shows_its_requirement_beside_its_value(qt_app, synthetic):
    """A reading with no threshold next to it invites the reader to supply one
    from memory, and they differ by level, category and standard.

    It is the overview that carries them now rather than each metric's own
    page, where the limit is drawn on the chart as the band the reading sits
    in -- but it is still on screen, in words, next to the value."""
    result = analyze(synthetic, "fadgi_3star_documents_unbound_general_2023")
    text = all_text(render(qt_app, result))
    for metric in result.run.metrics:
        if metric.get("condition"):
            # As it is written on screen: ">= 294", not "value >= 294". The
            # criterion is the same string; the word that named its subject is
            # what the column heading and the row already say.
            assert condition_label(metric["condition"]) in text, metric["metric_id"]


def test_a_measurement_only_profile_never_reports_a_pass(qt_app, synthetic):
    """It produced numbers and judged nothing. Rendering that as success would
    invent a verdict the profile never asked for."""
    result = analyze(synthetic, "measurement_only_v1")
    text = all_text(render(qt_app, result))
    assert "MEASURED" in text
    assert "PASS" not in text


def test_warnings_are_carried_into_the_report(qt_app, synthetic):
    result = analyze(synthetic, "fadgi_3star_documents_unbound_general_2023")
    view = render(qt_app, result)
    if result.run.warnings:
        text = all_text(view)
        assert "Warning" in text
        assert result.run.warnings[0][:40] in text


def test_the_frontend_and_the_cli_agree_on_the_same_capture(synthetic, qt_app, capsys):
    """The point of a shared engine, asserted rather than assumed.

    The desktop runner and the command line reach the pipeline by different
    routes -- one hands it a decoded image and a resolved reference, the other a
    path and an argparse namespace -- so this checks that the routes converge on
    the same numbers rather than merely being expected to.
    """
    from deltachrome.cli import main
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.workers import AnalysisRunner

    tmp, reference, image, corners = synthetic
    profile = "fadgi_3star_documents_unbound_general_2023"

    runner = AnalysisRunner(
        image=load_image(image), target_id="colorchecker_sg_v1", profile_id=profile,
        corners=corners, reference_mode="file", reference_choice_path=reference,
    )
    produced: list[object] = []
    runner.finished.connect(produced.append)
    runner.failed.connect(produced.append)
    runner.run()  # synchronously; the threading is tested separately
    assert len(produced) == 1, produced
    gui_result = produced[0]

    saved = tmp / "cli_run.json"
    assert main([
        "analyze", str(image), "--target", "colorchecker_sg_v1",
        "--reference", str(reference), "--profile", profile,
        "--corners", " ".join(f"{x},{y}" for x, y in corners),
        "--save-run", str(saved),
    ]) == 0
    capsys.readouterr()

    import json

    cli_run = json.loads(saved.read_text(encoding="utf-8"))
    assert cli_run["overall"] == gui_result.run.overall
    assert cli_run["conformance_claim"] == gui_result.run.conformance_claim

    cli_metrics = {m["metric_id"]: m for m in cli_run["metrics"]}
    gui_metrics = {m["metric_id"]: m for m in gui_result.run.metrics}
    assert cli_metrics.keys() == gui_metrics.keys()
    for metric_id, cli_metric in cli_metrics.items():
        gui_metric = gui_metrics[metric_id]
        assert cli_metric["status"] == gui_metric["status"], metric_id
        if cli_metric["value"] is None:
            assert gui_metric["value"] is None, metric_id
        else:
            assert cli_metric["value"] == pytest.approx(gui_metric["value"]), metric_id


def test_reopening_a_saved_run_restores_what_it_was_judged_as_and_against(
    qt_app, synthetic, tmp_path
):
    """Reuse has to restore the whole choice. Restoring the corners but making the
    operator re-pick the target and profile from memory is how a capture gets
    re-scored against the wrong thing and looks fine doing it."""
    from deltachrome_ui.library import read_run

    result = analyze(synthetic, "fadgi_3star_documents_unbound_general_2023")
    saved = read_run(result.run.save(tmp_path / "run.json"))

    setup = ChoicesPanel()
    assert setup.select_target(saved.target_id)
    assert setup.select_profile(saved.profile_id)
    assert setup.target_id == "colorchecker_sg_v1"
    assert setup.profile_id == "fadgi_3star_documents_unbound_general_2023"
    assert saved.corners is not None and len(saved.corners) == 4


# -- automatic placement -------------------------------------------------------
#
# The frontend's half of automatic placement is small and its whole risk is in one
# place: a placement the operator did not make must be visible as such, on screen
# while it can still be corrected and in the record afterwards. These pin that.


def placement(qt_app, target_id="colorchecker_sg_v1"):
    from deltachrome_ui.placement_view import PlacementView

    view = PlacementView()
    view.set_target(load_target(target_id))
    return view


def _flat_proxy():
    """A canvas needs pixels before it will place a handle anywhere."""
    import numpy as np

    from deltachrome_ui.imageproxy import ImageProxy

    class Flat:
        width = height = 100
        scale = 1.0
        profile = None
        pixels = np.full((100, 100, 3), 0.5, dtype=np.float32)

    return ImageProxy(Flat())


def test_a_detected_placement_says_so_until_a_corner_is_moved(qt_app):
    """The record must not claim a placement was automatic when half of it was
    not, and the canvas is the only thing that knows a handle was dragged."""
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app)
    assert not view.placed_automatically

    view.show_detection(Detection(
        corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        found=True, method="printed_crosshairs",
        reason="Placed automatically from the chart's printed crosshairs.",
        notes=["Refined onto the chart's own printed features."],
    ))
    assert view.placed_automatically
    assert view.detection_pill.isVisibleTo(view)
    assert view.detection_pill.text() == "Placed Automatically"

    view.canvas.cornersDragged.emit()  # the signal a real drag emits
    assert not view.placed_automatically
    assert view.detection_pill.text() == "Placed by Hand"


def test_what_the_finder_keyed_on_is_kept_as_the_pills_own_detail(qt_app):
    """Two words and a color is what an operator needs from a placement they did
    not make: it was placed, so glance at it. The finder's account is still one
    hover away for the run where the glance surprises somebody."""
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app)
    view.show_detection(Detection(
        corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        found=True, method="printed_crosshairs",
        reason="Placed automatically from the chart's printed crosshairs.",
        notes=["Refined onto the chart's own printed features."],
    ))
    assert "printed crosshairs" in view.detection_pill.toolTip()
    assert "Refined" in view.detection_pill.toolTip()


def test_a_refusal_leaves_the_handles_alone_and_gives_the_reason(qt_app):
    """A refusal is not an error. It leaves the operator on the screen they would
    have used anyway with the reason on it -- the only handover that leaves them
    better off than not having tried."""
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app, "isa_fadgi_19264_v1")
    view.show_detection(Detection(
        method="corner_squares_and_dot",
        reason="ISA FADGI - ISO 19264 places on its corner squares and dot, "
               "which nothing here can find yet. Place the chart by hand.",
    ))
    assert not view.placed_automatically
    assert view.detection_pill.text() == "Place the Target by Hand"
    assert "corner squares and dot" in view.detection_pill.toolTip()


def test_the_panel_carries_no_placement_instruction_of_its_own(qt_app):
    """What a chart is placed on -- and the Golden Thread's corners are inside
    its corner modules, nowhere near the corners of the card -- is a fact about
    the target, not about the run. It belongs in the target documentation; on the panel it was a paragraph of blue text that sat there
    for the whole of a placement nobody was making by hand."""
    from PySide6.QtWidgets import QLabel

    view = placement(qt_app, "isa_golden_thread_dlt_v1")
    note = load_target("isa_golden_thread_dlt_v1").fiducials.notes or ""
    shown = " ".join(label.text() for label in view.findChildren(QLabel))
    assert note[:40] not in shown


def test_a_placement_arriving_is_not_a_person_placing_it(qt_app):
    """The corners moving and somebody moving them are different events, and the
    screen used to tell them apart by the order of two calls.

    A detection arrives *as* a corner change, so listening to the corners made
    "was this placed by hand" depend on ``set_corners`` being called before
    ``show_detection`` and never the other way round -- a rule nothing stated.
    It showed as the Golden Thread's hand-placement instruction appearing on a
    placement nobody had touched.
    """
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app, "isa_golden_thread_dlt_v1")
    proxy = _flat_proxy()
    view.set_proxy(proxy, Path("capture.tif"))
    # Deliberately the awkward order: the note is shown, then the corners land.
    view.show_detection(Detection(
        corners=[(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
        found=True, method="module_resolution_squares", reason="Placed.",
    ))
    view.set_corners([(11.0, 11.0), (89.0, 11.0), (89.0, 89.0), (11.0, 89.0)])
    assert view.placed_automatically
    assert view.detection_pill.text() == "Placed Automatically"


def test_choosing_a_target_clears_a_previous_detection(qt_app):
    """A note about a chart that is no longer selected is worse than none."""
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app)
    view.show_detection(Detection(
        corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        found=True, method="printed_crosshairs", reason="Placed automatically.",
    ))
    view.set_target(load_target("avian_ngt_v1"))
    assert not view.detection_pill.isVisibleTo(view)
    assert not view.placed_automatically


def test_a_search_makes_the_choices_inert_rather_than_leaving_the_screen(qt_app):
    """Changing the target used to swap to a full-window wait, which severed the
    choice from its effect. The canvas stays put; the panel goes inert, because
    every control on it would change the answer being computed."""
    from deltachrome.imaging.detect import Detection

    view = placement(qt_app)
    view.set_busy("Finding the ColorChecker Digital SG…")
    assert not view.choices.isEnabled()
    assert not view.continue_button.isEnabled()
    # The wait is the bar, and **only** the bar. It used to be said twice at
    # once -- a chip above a bar carrying the same sentence, the chip also
    # wearing the tone this screen uses for a placement needing attention.
    assert "Finding" in view.busy_strip.label.text()
    assert not view.detection_pill.isVisibleTo(view)

    view.show_detection(Detection(
        corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        found=True, method="printed_crosshairs", reason="Placed automatically.",
    ))
    assert view.choices.isEnabled()


def test_a_chart_with_nothing_to_measure_against_cannot_be_run(qt_app):
    """The pipeline says so by raising, which is a dialog arrived at after a
    wait and dismissable only. It is answered before the run instead."""
    view = placement(qt_app)
    view.choices.select_target("colorchecker_sg_v1")
    assert not view.continue_button.isEnabled()
    assert "reference file" in view.continue_button.toolTip()

    view.choices.select_target("isa_golden_thread_dlt_v1")
    assert view.continue_button.isEnabled()


def test_a_target_chosen_on_the_panel_asks_the_shell_to_place_it(qt_app):
    """The shell owns the decoded capture and the worker threads, so it does the
    finding. This is the whole of the panel's half of that."""
    view = placement(qt_app)
    asked: list[str] = []
    view.targetChanged.connect(asked.append)
    view.choices.select_target("avian_ngt_v1")
    assert asked == ["avian_ngt_v1"]


# -- identifying the chart -----------------------------------------------------


def test_an_identified_chart_is_selected_and_said(setup):
    """Selected rather than merely suggested: everything else on the panel is
    downstream of the target, so it cannot open already answering without one.
    What keeps that honest is that the choice stays a combo box."""
    from deltachrome.imaging.detect import Candidate, Identification

    setup.show_identification(Identification(
        target_id="avian_ngt_v1", found=True,
        candidates=[Candidate("avian_ngt_v1", "Avian Rochester Next Generation Target")],
        reason="Avian Rochester Next Generation Target, identified from the capture.",
    ))
    assert setup.target_id == "avian_ngt_v1"
    assert setup.identified
    assert setup.target_pill.isVisibleTo(setup)
    assert setup.target_pill.text() == "Target Detected"


def test_choosing_a_different_chart_stops_it_being_an_identification(setup):
    """The record must not say a chart was identified when the operator named it."""
    from deltachrome.imaging.detect import Identification

    setup.show_identification(Identification(
        target_id="avian_ngt_v1", found=True, reason="Identified.",
    ))
    assert setup.identified
    setup.select_target("colorchecker_sg_v1")
    assert not setup.identified
    assert not setup.target_pill.isVisibleTo(setup)


def test_two_charts_fitting_equally_names_neither(setup):
    """The honest answer where a layout is shared, which two shipped targets are.
    It hands the choice back, so it says so rather than leaving a combo box
    looking like a decision nobody made."""
    from deltachrome.imaging.detect import Identification

    setup.select_target("colorchecker_sg_v1")
    setup.show_identification(Identification(
        reason="This capture could be IT8.7/1 Transmissive Target or IT8.7/2 "
               "Reflective Target, and nothing measured here separates them. "
               "Choose which.",
    ))
    assert not setup.identified
    assert setup.target_pill.text() == "Choose the Target"
    assert "Choose which" in setup.target_pill.toolTip()
    # **And the selection is cleared**, which reverses what this asserted before.
    # The old rule was that naming nothing must not silently change what the panel
    # is set up for -- but this panel outlives the capture, so "unchanged" meant
    # the *previous* capture's chart stayed selected, and the screen then searched
    # for it and offered what it found as a chart of this one. A blank combo is
    # the honest handover, and a target explicitly restored still wins: see
    # ``test_a_restored_target_survives_an_identification_that_declined``.
    assert not setup.has_target


def test_a_restored_target_survives_an_identification_that_declined(setup):
    """A saved run carries a chart somebody already checked. Identification runs
    again on the way in and may decline -- which clears the combo -- so an explicit
    target has to be applied after it rather than instead of it. Ordering these the
    other way round lost the restored chart on any capture the search cannot name.
    """
    from deltachrome.imaging.detect import Identification

    setup.restore(
        identification=Identification(reason="Nothing here fitted any target."),
        target_id="colorchecker_sg_v1",
    )
    assert setup.target_id == "colorchecker_sg_v1"
    assert setup.has_target


# -- what thread a handler runs on ---------------------------------------------


def test_a_lambda_connection_runs_on_the_workers_thread(qt_app):
    """The measurement behind the rule below, made rather than assumed.

    PySide decides where to deliver a signal from the *receiver's* thread, and a
    lambda has no receiver: it is not a QObject, so there is nothing to ask. The
    connection is then direct and the handler runs wherever the signal was emitted
    -- which for a worker signal is the worker's own thread.
    """
    import threading

    class Worker(QObject):
        done = Signal(object)

        def run(self):
            self.done.emit(None)

    class Owner(QObject):
        def __init__(self):
            super().__init__()
            self.threads = {}

        def bound(self, _value):
            self.threads["bound method"] = threading.get_ident()

    owner, worker = Owner(), Worker()
    worker.done.connect(owner.bound)
    worker.done.connect(
        lambda _v: owner.threads.__setitem__("lambda", threading.get_ident())
    )
    thread = run_in_thread(worker, owner)
    for _ in range(50):
        settle(40)
        if len(owner.threads) == 2:
            break
    # Joined before the owner goes out of scope: a QThread destroyed while still
    # running takes the process with it, which is the very failure ``run_in_thread``
    # exists to prevent and would be a poor thing for its own test to cause.
    thread.quit()
    assert thread.wait(5000)

    here = threading.get_ident()
    assert owner.threads["bound method"] == here
    assert owner.threads["lambda"] != here


def test_no_worker_signal_is_connected_to_a_lambda():
    """A source scan, in the same spirit as ``test_ui_boundary``.

    The shell's load handler was connected as ``loaded.connect(lambda proxy: ...)``,
    so it ran on the decoding thread and called widget methods from there -- Qt's
    one hard rule, broken quietly for as long as it did no visible harm. It became
    audible only when that handler started doing something Qt checks for, parenting
    a new thread to the window: *"Cannot create children for a parent that is in a
    different thread."*

    The rule is therefore about the shape of the connection rather than about what
    the handler happens to do today: a worker signal is delivered to a bound method
    of a QObject, and anything the handler needs travels on the signal.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "deltachrome_ui"
              / "app.py").read_text(encoding="utf-8")
    offenders = re.findall(
        r"^\s*\w+\.(progress|loaded|finished|failed)\.connect\(\s*lambda.*$",
        source, re.MULTILINE,
    )
    assert not offenders, (
        "a worker signal is connected to a lambda, which PySide delivers directly "
        "on the worker's thread; connect a bound method and put what it needs on "
        "the signal"
    )
