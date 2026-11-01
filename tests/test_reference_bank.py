"""The bank of reference files: what it does to the folder, and what it shows.

Two halves, tested apart. The engine half moves files and rewrites a manifest, and
is the half where a mistake is destructive -- an overwrite loses a measurement that
a run has already been judged against. The frontend half is a list and a detail
panel, and the thing worth pinning there is that a code the file states for itself
is never offered as removable, because that is an offer this cannot honour.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.errors import MissingReferenceError  # noqa: E402
from deltachrome.reference.library import ReferenceLibrary  # noqa: E402


#: The local reference measurements this test reads. data/ is never committed,
#: so on a machine without it -- a build server -- the test skips rather than
#: fails, as every test needing local captures already does.
needs_local_references = pytest.mark.skipif(
    not Path("data/reference/ColorChecker_SG_M1.txt").is_file()
    or not Path("data/reference/T1910021.cxf").is_file(),
    reason="needs the local reference files in data/reference, which are not committed",
)

SG = "data/reference/ColorChecker_SG_M1.txt"
CXF = "data/reference/T1910021.cxf"


@pytest.fixture
def bank(tmp_path):
    directory = tmp_path / "references"
    directory.mkdir()
    return ReferenceLibrary(directory)


# -- the folder ----------------------------------------------------------


@needs_local_references
def test_a_file_is_copied_in_not_referenced(bank):
    """The whole reason for a bank. A path into somebody else's folder is not a
    promise, and the measurement a run was judged against has to still be there
    in a year."""
    entry = bank.add_file(SG)
    assert entry.path.parent == bank.directory
    assert entry.path.read_bytes() == open(SG, "rb").read()


@needs_local_references
def test_the_arrival_date_is_recorded_rather_than_read_off_the_file(bank):
    """A copy that preserves the timestamp reports when the *measurement* was
    made, and one that does not is changed by anything that touches the file.
    Neither answers 'when did this arrive here'."""
    entry = bank.add_file(SG)
    assert bank.added_on(entry.name)
    assert bank.added_on("never-added.txt") is None


@needs_local_references
def test_a_second_file_of_the_same_name_never_overwrites_the_first(bank):
    """An instrument that writes ``Measurement.txt`` every time is the normal
    case. Replacing the first would swap the reference under every run that had
    already paired with it."""
    first = bank.add_file(SG)
    second = bank.add_file(SG)
    assert first.name != second.name
    assert {e.name for e in bank.entries} == {first.name, second.name}


def test_a_file_of_a_kind_the_engine_does_not_index_is_refused(bank, tmp_path):
    stray = tmp_path / "notes.pdf"
    stray.write_text("not a measurement")
    with pytest.raises(MissingReferenceError):
        bank.add_file(stray)


@needs_local_references
def test_binding_a_code_makes_a_capture_pair_with_the_file(bank):
    """The case this exists for: i1Profiler writes no serial at all, so nothing
    inside the SG's own measurement names the unit it describes."""
    entry = bank.add_file(SG)
    assert entry.serial is None
    assert not bank.resolve(["CCSG21"]).path

    bank.bind("CCSG21", entry.name)
    assert bank.resolve(["CCSG21"]).path == entry.path

    bank.unbind("CCSG21")
    assert not bank.resolve(["CCSG21"]).path


@needs_local_references
def test_removing_a_file_takes_its_bindings_and_its_date_with_it(bank):
    """A binding pointing at a file that is gone is a manifest that lies, and the
    next file to arrive under that name would inherit the pairing."""
    entry = bank.add_file(SG)
    bank.bind("CCSG21", entry.name)
    bank.remove_file(entry.name)
    assert not entry.path.exists()
    assert not bank.resolve(["CCSG21"]).path
    assert bank.added_on(entry.name) is None


@needs_local_references
def test_a_manifest_written_by_hand_survives_an_edit(bank):
    """The application is not the only thing entitled to put something in it, and
    rewriting one binding must not delete keys this version has never heard of."""
    entry = bank.add_file(SG)
    bank.manifest_path.write_text(
        '{"bindings": {}, "notes": "left by a person"}', encoding="utf-8"
    )
    bank.reload()
    bank.bind("CCSG21", entry.name)
    assert bank.manifest()["notes"] == "left by a person"


@needs_local_references
def test_an_empty_manifest_is_removed_rather_than_left_saying_nothing(bank):
    """An empty manifest and no manifest are the same state to every reader, and
    a file saying ``{}`` makes an untouched bank look configured."""
    entry = bank.add_file(SG)
    bank.bind("CCSG21", entry.name)
    bank.remove_file(entry.name)
    assert not bank.manifest_path.exists()


# -- the panel -----------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def panel(qt_app, tmp_path, monkeypatch):
    directory = tmp_path / "references"
    directory.mkdir()
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(directory))
    from deltachrome_ui.reference_bank import ReferenceBankPanel

    return ReferenceBankPanel()


def _codes(panel) -> list[tuple[str, object]]:
    return [
        (panel.codes.item(i).text(), panel.codes.item(i).data(Qt.ItemDataRole.UserRole))
        for i in range(panel.codes.count())
    ]


def test_an_empty_bank_shows_nothing_about_a_file_that_is_not_there(panel):
    """Hiding only the parts that hold data left two section headings and a rule
    behind, laid out around a gap."""
    assert panel.list.count() == 0
    assert not panel.detail.isVisibleTo(panel)
    assert panel.empty_page.isVisibleTo(panel)
    assert not panel.remove_button.isEnabled()


def test_a_missing_bank_folder_reads_as_empty_and_not_as_an_error(qt_app, tmp_path,
                                                                 monkeypatch):
    """It ships empty and is created on first sight, so 'no such folder' is the
    ordinary state of a fresh install."""
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(tmp_path / "never-made"))
    from deltachrome_ui.reference_bank import ReferenceBankPanel

    panel = ReferenceBankPanel()
    assert panel.list.count() == 0
    assert panel.empty.text() == "No reference files added."


@needs_local_references
def test_the_two_kinds_of_code_are_marked_apart(panel):
    """One is a fact about the file and the other is a decision somebody made.
    Listing them together unmarked would offer to remove something this cannot."""
    entry = panel.library.add_file(SG)
    panel.library.bind("CCSG21", entry.name)
    panel.reload()

    shown = _codes(panel)
    bound = [text for text, code in shown if code]
    stated = [text for text, code in shown if not code]
    assert any(t.startswith("CCSG21") and "added here" in t for t in bound)
    assert any("from the file" in t for t in stated)


@needs_local_references
def test_a_code_the_file_states_is_never_offered_as_removable(panel):
    entry = panel.library.add_file(CXF)
    panel.reload()
    assert entry.serial == "T1910021"
    for row in range(panel.codes.count()):
        panel.codes.setCurrentRow(row)
        item = panel.codes.item(row)
        removable = bool(item.data(Qt.ItemDataRole.UserRole))
        assert panel.unbind_button.isEnabled() == removable


def test_a_file_answering_to_nothing_says_so(panel, tmp_path):
    """A file nothing can pair with is not an error and is not usable either, and
    the list of codes is the only place that can say which."""
    orphan = tmp_path / "x.txt"
    orphan.write_text("CGATS.17\nBEGIN_DATA_FORMAT\n", encoding="utf-8")
    panel.library.add_file(orphan)
    panel.reload()
    # "x" normalizes to a stem identifier, so blank the one thing it answers to.
    panel.library.entries[0].identifiers.clear()
    panel._show_selected()
    assert "answers to nothing" in panel.codes_note.text()


@needs_local_references
def test_binding_through_the_panel_reaches_the_folder(panel):
    entry = panel.library.add_file(SG)
    panel.reload()
    panel.code_input.setText("CCSG21")
    panel._bind()
    assert panel.library.resolve(["CCSG21"]).path == entry.path
    assert not panel.code_input.text()


@needs_local_references
def test_the_selected_file_survives_a_reload(panel):
    panel.library.add_file(SG)
    second = panel.library.add_file(CXF)
    panel.reload(select=second.name)
    assert panel.selected == second.name
    panel.reload()
    assert panel.selected == second.name


# -- pairing before the run, not during it ----------------------------------


class FakeScan:
    """What ``scan_codes`` returns, reduced to the fields pairing reads.

    ``labels`` derives from the texts exactly as the real class does rather than
    being set independently, so a stub can never claim a code is a label when
    the parser would disagree.
    """

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    @property
    def labels(self) -> list:
        from deltachrome.reference.openqualia import parse_label

        return [label for label in map(parse_label, self.texts) if label is not None]


@pytest.fixture
def filled_bank(tmp_path, monkeypatch):
    """A bank holding the SG measurement, bound to the code its card prints.

    Through the environment variable, which is the resolution order the
    application documents and what a shared-drive deployment uses.
    """
    if not Path(SG).is_file():
        pytest.skip("needs the local reference files in data/reference, which are not committed")
    import shutil

    directory = tmp_path / "references"
    directory.mkdir()
    shutil.copy(SG, directory / "ColorChecker_SG_M1.txt")
    (directory / "deltachrome-references.json").write_text(
        '{"bindings": {"CCSG21": "ColorChecker_SG_M1.txt"}}', encoding="utf-8"
    )
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(directory))
    return directory


def choices_paired_with(scan, qt_app, target="colorchecker_sg_v1"):
    """A setup panel in *Pair from library*, on a chart that compares something.

    The target matters: a chart declaring ``source_type: "none"`` compares none of
    its patches against anything, and the whole reference group is hidden for it.
    """
    from deltachrome_ui.choices import LIBRARY, ChoicesPanel

    panel = ChoicesPanel()
    panel.select_target(target)
    panel.mode_buttons[LIBRARY].setChecked(True)
    panel.set_scan(scan)
    return panel


def test_the_pairing_is_known_before_anything_is_measured(qt_app, filled_bank):
    """It used to be found out during the run, which made the one outcome worth
    knowing about -- a code matching nothing -- an exception after the decode."""
    from deltachrome_ui.choices import REFERENCE_PAIRED

    panel = choices_paired_with(FakeScan("CCSG21"), qt_app)
    resolution = panel.pairing()
    assert resolution.path.name == "ColorChecker_SG_M1.txt"
    assert "CCSG21" in panel.path_label.text()
    assert panel.reference_pill.text() == REFERENCE_PAIRED
    assert panel.ready


def test_a_code_matching_nothing_stops_the_run_on_the_setup_screen(qt_app, filled_bank):
    """And says so where the fix is, rather than raising after the decode.

    It stops the run whatever the target ships: falling back to nominal values
    for a chart the operator asked to pair *by unit* would answer a different
    question in the same report.
    """
    from deltachrome_ui.choices import REFERENCE_UNPAIRED

    panel = choices_paired_with(FakeScan("CCSG99"), qt_app)
    assert panel.pairing().path is None
    assert panel.reference_pill.text() == REFERENCE_UNPAIRED
    assert not panel.ready


def test_a_capture_carrying_no_code_says_so_in_every_mode(qt_app, filled_bank):
    """It is a fact about the capture rather than about the mode: this chart
    carries nothing machine-readable, so nothing here can pair it by unit. Said
    only under *Pair from library*, it read as a failure of that mode."""
    from deltachrome_ui.choices import AUTOMATIC, REFERENCE_NO_CODE

    panel = choices_paired_with(FakeScan(), qt_app)
    assert "prints no code" in panel.path_label.text()
    assert panel.reference_pill.text() == REFERENCE_NO_CODE
    assert not panel.ready

    panel.mode_buttons[AUTOMATIC].setChecked(True)
    assert panel.reference_pill.text() == REFERENCE_NO_CODE


def test_a_new_capture_opens_on_automatic(qt_app, filled_bank):
    """A capture opening on *Pair from library* is the application asserting
    something about the chart in front of the operator, and where that chart
    carries no code it asserts it and then fails."""
    from deltachrome_ui.choices import AUTOMATIC, LIBRARY

    panel = choices_paired_with(FakeScan("CCSG21"), qt_app)
    assert panel.mode == LIBRARY
    panel.restore(target_id="colorchecker_sg_v1")
    assert panel.mode == AUTOMATIC


def test_a_code_the_bank_pairs_selects_pair_from_library(qt_app, filled_bank):
    """The chart has said which unit it is and the bank holds it, so the capture
    opens paired rather than asking the operator to notice and switch."""
    from deltachrome_ui.choices import AUTOMATIC, LIBRARY, REFERENCE_PAIRED, ChoicesPanel

    panel = ChoicesPanel()
    panel.select_target("colorchecker_sg_v1")
    assert panel.mode == AUTOMATIC
    panel.set_scan(FakeScan("CCSG21"))
    assert panel.mode == LIBRARY
    assert panel.reference_pill.text() == REFERENCE_PAIRED

    # A code the bank does not know, or no code at all, asserts nothing.
    for scan in (FakeScan("CCSG99"), FakeScan()):
        panel.restore(target_id="colorchecker_sg_v1")
        panel.set_scan(scan)
        assert panel.mode == AUTOMATIC


def test_a_file_chosen_from_the_bank_by_hand_is_an_explicit_reference(
    qt_app, filled_bank
):
    """It was named, so the run resolves it as a named file -- code verification
    and all, since a disagreement then belongs in the record as a warning."""
    from deltachrome_ui.choices import AUTOMATIC, EXPLICIT_FILE, REFERENCE_CHOSEN

    panel = choices_paired_with(FakeScan("CCSG99"), qt_app)
    panel.bank_choice = filled_bank / "ColorChecker_SG_M1.txt"
    panel._on_mode_changed()
    assert panel.reference_mode == EXPLICIT_FILE
    assert panel.reference_choice == filled_bank / "ColorChecker_SG_M1.txt"
    assert panel.reference_pill.text() == REFERENCE_CHOSEN
    assert panel.ready
    # And it belongs to the mode it was made in: leaving it set would make
    # *Automatic* quietly use a file somebody chose two clicks ago.
    panel.mode_buttons[AUTOMATIC].setChecked(True)
    assert panel.bank_choice is None


@needs_local_references
def test_a_scan_already_read_is_not_read_again(tmp_path):
    """The frontend scans while it identifies the chart, so the pairing on screen
    and the pairing the run uses have to be one reading rather than two scans."""
    import shutil

    from deltachrome.reference.resolve import resolve_reference
    from deltachrome.targets.registry import load_target

    directory = tmp_path / "references"
    directory.mkdir()
    shutil.copy(SG, directory / "CCSG21.txt")
    choice = resolve_reference(
        target=load_target("colorchecker_sg_v1"),
        image=None,  # would raise if a scan were needed
        library_dir=directory,
        scan=FakeScan("CCSG21"),
    )
    assert choice.path.name == "CCSG21.txt"


def test_choosing_from_the_bank_can_bind_the_code_it_read(qt_app, filled_bank):
    """The one moment somebody has both facts in front of them: the code this
    capture carries, and the file it belongs to.

    Any code nothing else explains. The LaserSoft's 1930021 against T1910021 was
    the first, and the bank now pairs that one itself from LaserSoft's encoding;
    a binding is still what pairs a code to a file that states no serial, and
    doing it once means the chart pairs itself from then on.
    """
    from deltachrome_ui.reference_bank import BankPicker

    picker = BankPicker(["1930021"])
    assert picker.remember.isVisible() or picker.remember.isChecked()
    picker.list.setCurrentRow(0)
    picker.accept()

    assert picker.chosen is not None
    library = ReferenceLibrary(filled_bank)
    assert library.resolve(["1930021"]).path == picker.chosen


def test_a_bank_picker_with_no_code_offers_nothing_to_remember(qt_app, filled_bank):
    from deltachrome_ui.reference_bank import BankPicker

    picker = BankPicker([])
    assert not picker.remember.isChecked()
    picker.list.setCurrentRow(0)
    picker.accept()
    assert ReferenceLibrary(filled_bank).manifest().get("bindings", {}) == {
        "CCSG21": "ColorChecker_SG_M1.txt"
    }


# -- fetching a measurement the chart's own label points at ----------------

LABEL = (
    "https://targets.digitaltransitions.com/measurements"
    "?Manufacturer=DT&TargetType=DTNGT2&TargetID=DT-AR-2020041"
)


def _panel_scanning(text, monkeypatch, tmp_path):
    """A choices panel that has read one code off its capture."""
    from deltachrome.imaging.codes import CodeScanResult, DetectedCode
    from deltachrome_ui import choices as choices_module
    from deltachrome_ui.choices import ChoicesPanel

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(choices_module, "reference_bank_dir", lambda: tmp_path)

    panel = ChoicesPanel()
    panel.select_target("avian_dt_ngt_v2")
    panel.set_scan(
        CodeScanResult(
            codes=[DetectedCode(text=text, symbology="QRCode", decoder="opencv")]
        )
    )
    return panel


def test_a_label_offers_to_fetch_the_unit_it_names(monkeypatch, tmp_path):
    panel = _panel_scanning(LABEL, monkeypatch, tmp_path)
    label = panel.fetchable_label()
    assert label is not None
    assert label.target_id == "DT-AR-2020041"


def test_a_serial_belonging_to_another_model_is_still_offered(monkeypatch, tmp_path):
    """Superseded reasoning, kept because the change was deliberate.

    This originally asserted that a bare serial offers nothing, on the argument
    that only a label says where a measurement lives. That was true until the
    target definitions began declaring where their own units are published, and
    it left the ColorChecker SG -- whose QR *is* a bare serial -- unable to fetch
    anything at run time.

    What is left is a looser gate: any decoded code is offered as a lookup for
    whichever target is selected, so an SG serial against an NGT2 is offered and
    will 404. That is accepted rather than overlooked. Serial formats differ per
    maker with no general rule to check against, the cost of being wrong is one
    HTTP request and a message saying the unit is not published, and the
    alternative -- refusing to look up anything that does not match a guessed
    pattern -- would silently drop the case this exists for.
    """
    panel = _panel_scanning("CCSG21", monkeypatch, tmp_path)
    panel.select_target("avian_dt_ngt_v2")
    panel.set_scan(panel.scan)

    label = panel.fetchable_label()
    assert label is not None and label.target_id == "CCSG21"


@needs_local_references
def test_nothing_is_offered_when_the_bank_already_holds_the_unit(
    monkeypatch, tmp_path
):
    """Offering to download a measurement that is already filed invites a second
    copy of it under a second name."""
    ReferenceLibrary(tmp_path).add_measurement(
        open(SG, "rb").read(), "already-here.txt", bind="DT-AR-2020041"
    )
    panel = _panel_scanning(LABEL, monkeypatch, tmp_path)
    assert panel.fetchable_label() is None


def test_the_offer_is_a_button_and_never_the_identification_pass(monkeypatch,
                                                                 tmp_path):
    """A capture is not consent to make an outbound request. Reading the code
    must not reach the network -- only pressing the control may."""
    import deltachrome.reference.openqualia as openqualia

    def refuse(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the network was reached without a press")

    monkeypatch.setattr(openqualia, "fetch_measurement", refuse)
    panel = _panel_scanning(LABEL, monkeypatch, tmp_path)
    panel.show()
    assert panel.fetch_button.isVisible()


# -- filling the bank from a serial ---------------------------------------


def test_only_targets_whose_units_are_published_are_offered(monkeypatch, tmp_path):
    """Listing a target nobody publishes would offer a lookup that must 404."""
    from deltachrome.reference.openqualia import publishes_measurements
    from deltachrome.targets.registry import available_targets, load_target
    from deltachrome_ui.reference_bank import FetchBySerialDialog

    QApplication.instance() or QApplication([])
    dialog = FetchBySerialDialog()
    offered = {dialog.target_combo.itemData(i) for i in range(dialog.target_combo.count())}

    assert offered == {t for t in available_targets()
                       if publishes_measurements(load_target(t))}
    assert "isa_golden_thread_dlt_v1" in offered
    # The film targets, the UTT, the microform charts: nobody publishes them.
    assert "isa_transmissive_4x5_v1" not in offered


def test_the_operator_types_a_serial_and_never_a_target_type():
    """The host offers no listing to guess a target type from, matches it case-
    and whitespace-agnostically but not dash-agnostically, and its own web slug
    differs from its API spelling -- OLT-1x browses and 404s where OLT1x
    resolves. So the identity comes from the definition."""
    from deltachrome.reference.openqualia import label_for_serial
    from deltachrome.targets.registry import load_target

    label = label_for_serial(load_target("isa_object_level_1x_v1"), "M000559")
    assert label is not None
    assert "TargetType=OLT1x" in label.measurement_url
    assert "Manufacturer=ISA" in label.measurement_url
    assert "TargetID=M000559" in label.measurement_url


def test_a_target_with_no_declared_identity_yields_no_lookup():
    from deltachrome.reference.openqualia import label_for_serial
    from deltachrome.targets.registry import load_target

    assert label_for_serial(load_target("isa_transmissive_4x5_v1"), "X1") is None


@pytest.mark.parametrize("serial", ["", "   ", "has space", "semi;colon", "sl/ash"])
def test_a_serial_that_is_not_one_is_refused_before_any_request(serial):
    """The standard limits a TargetID to letters, digits and dashes, and this is
    the last point before a string becomes a URL."""
    from deltachrome.reference.openqualia import label_for_serial
    from deltachrome.targets.registry import load_target

    assert label_for_serial(load_target("isa_golden_thread_dlt_v1"), serial) is None


def test_the_dialog_will_not_fetch_until_the_serial_is_usable(monkeypatch, tmp_path):
    from deltachrome_ui.reference_bank import FetchBySerialDialog

    QApplication.instance() or QApplication([])
    dialog = FetchBySerialDialog()
    ids = [dialog.target_combo.itemData(i) for i in range(dialog.target_combo.count())]
    dialog.target_combo.setCurrentIndex(ids.index("isa_golden_thread_dlt_v1"))

    dialog.serial.setText("")
    assert not dialog.fetch_button.isEnabled()
    dialog.serial.setText("not a serial")
    assert not dialog.fetch_button.isEnabled()
    dialog.serial.setText("D00011")
    assert dialog.fetch_button.isEnabled()
    assert "D00011" in dialog.status.text()


def test_a_failed_fetch_is_reported_in_the_dialog_not_thrown():
    """LoadFailure passes only its headline to Exception, so it carries one
    element in `args` and exposes `.headline` and `.detail`. Both fetch failure
    handlers originally read `args[1]`, which raised IndexError *inside the
    handler* -- so a mistyped serial produced a traceback and no message. The
    unit tests missed it because they drive the engine's own fetch directly; only
    running the dialog against a real 404 reached this line.
    """
    from deltachrome_ui.reference_bank import FetchBySerialDialog
    from deltachrome_ui.workers import LoadFailure

    QApplication.instance() or QApplication([])
    dialog = FetchBySerialDialog()
    dialog._running = True
    dialog._on_failed(LoadFailure("Headline", "The detail an operator reads."))

    assert dialog.status.text() == "The detail an operator reads."
    assert dialog.fetch_button.isEnabled()
    assert dialog.fetched is None


def test_the_placement_panel_reports_a_failed_fetch_the_same_way(monkeypatch):
    """The other handler with the same bug."""
    from deltachrome_ui.choices import ChoicesPanel
    from deltachrome_ui.workers import LoadFailure

    QApplication.instance() or QApplication([])
    panel = ChoicesPanel()
    shown = {}
    monkeypatch.setattr(
        "deltachrome_ui.choices.QMessageBox.warning",
        lambda parent, title, text: shown.update(title=title, text=text),
    )
    panel._fetching = True
    panel._on_fetch_failed(LoadFailure("Headline", "The detail an operator reads."))

    assert shown == {"title": "Headline", "text": "The detail an operator reads."}
    assert panel.fetch_button.isEnabled()


def test_a_bare_serial_on_the_capture_also_offers_a_fetch(monkeypatch, tmp_path):
    """The run-time offer originally required an OpenQualia *label*, which left a
    gap: a chart printing a bare serial -- the ColorChecker SG's QR is exactly
    that -- was read, and the selected target knew where its units are published,
    and the two facts sat on the same screen without being put together. The
    operator would have had to retype the serial in Settings."""
    panel = _panel_scanning("D00011", monkeypatch, tmp_path)
    panel.select_target("isa_golden_thread_dlt_v1")
    panel.set_scan(panel.scan)

    label = panel.fetchable_label()
    assert label is not None
    assert label.target_id == "D00011"
    assert "TargetType=DLT" in label.measurement_url


def test_a_serial_and_a_label_are_not_the_same_claim(monkeypatch, tmp_path):
    """A label is the maker's own statement that this unit's measurement exists
    at that address. A serial is a number this application knows how to look up,
    and the lookup may come back empty. The two must not read alike."""
    from deltachrome.reference.openqualia import FROM_CHART, FROM_SERIAL

    panel = _panel_scanning(LABEL, monkeypatch, tmp_path)
    panel.select_target("isa_golden_thread_dlt_v1")
    panel.set_scan(panel.scan)
    panel.show()
    assert panel.fetchable_label().source == FROM_CHART
    from_label = panel.fetch_button.toolTip()

    panel = _panel_scanning("D00011", monkeypatch, tmp_path)
    panel.select_target("isa_golden_thread_dlt_v1")
    panel.set_scan(panel.scan)
    panel.show()
    assert panel.fetchable_label().source == FROM_SERIAL
    from_serial = panel.fetch_button.toolTip()

    # The label is the chart's own statement that its measurement exists; the
    # serial is a lookup that may come back empty, and only one of them may promise.
    assert "Download" in from_label and "Not every unit" not in from_label
    assert "Not every unit" in from_serial
    assert from_label != from_serial


def test_no_fetch_is_offered_for_a_target_nobody_publishes(monkeypatch, tmp_path):
    """Otherwise every film target would offer a lookup that must 404."""
    panel = _panel_scanning("D00011", monkeypatch, tmp_path)
    panel.select_target("isa_transmissive_4x5_v1")
    panel.set_scan(panel.scan)
    assert panel.fetchable_label() is None


def test_a_long_reference_filename_does_not_widen_the_panel(qt_app, filled_bank):
    """A filename is one unbroken token, so a wrapping label beside the buttons
    reported the whole name as its minimum width -- and inside the placement
    screen's fixed-width panel that pushed every control off the right edge,
    with no scrollbar to bring them back. Reported on 0.9.0b1."""
    from deltachrome_ui.placement_view import PANEL_WIDTH

    panel = choices_paired_with(FakeScan("CCSG21"), qt_app)
    name = "CCSG2016_i1p3_4xAVG_3M_2024-05-17_M1_D50_2deg_measurement_final.txt"
    panel.bank_choice = filled_bank / name
    panel._on_mode_changed()
    panel.show()
    qt_app.processEvents()
    assert name in panel.path_label.text()
    assert name in panel.path_label.toolTip()
    # The placement panel's own 20 px margins either side.
    assert panel.minimumSizeHint().width() <= PANEL_WIDTH - 40
