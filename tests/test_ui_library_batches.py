"""A batch's records, as one row of the library that opens.

A folder of a hundred captures run against one guideline writes a hundred
records at one instant. Listed flat and sorted by date, that is a hundred rows
between a person and everything else they have -- and several such folders is a
library nobody can see past. They already sit in a folder of their own on disk,
because ``batch_dir`` put them there so somebody looking at the filesystem could
tell one batch from a month of ordinary work; this is the same fact in the
table.

Three things have to stay true through the grouping, and each is a test here.

**A record inside a batch is an ordinary record.** It sorts, searches, opens and
re-scores exactly as one outside does, because it *is* one -- the grouping is a
way of showing them and nothing else.

**A batch is never given a verdict.** It asks whether *each* of a hundred
captures passed; pooling would let one bad frame fail the lot and lose which one,
and conformance in this field is stated per image. So the status column paints a
chevron on those rows and no dot at all.

**Nothing may be lost by it.** The count at the foot of the screen counts
records, a search reaches inside a folder that is shut, and deleting a batch
moves its folder rather than unlinking a hundred files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtCore import QModelIndex, QRect, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QStyleOptionViewItem,
)

from deltachrome_ui.library import (  # noqa: E402
    BatchGroup,
    TRASH_DIR_NAME,
    runs_in,
    scan_library,
    trash_batch,
)
from deltachrome_ui.run_table import (  # noqa: E402
    COLUMN_INDEX,
    RUN_ROLE,
    RunFilterProxy,
    RunTableModel,
    RunTableView,
    StatusDotDelegate,
)
from deltachrome_ui.theme import tone_colors  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def record(name: str, *, overall: str = "PASS", when: str = "2026-09-12T10:00:00+00:00",
           warnings: list[str] | None = None, target: str = "ISA Object Level Target",
           profile: str = "FADGI 3-Star") -> dict:
    return {
        "run_id": name, "created_utc": when, "overall": overall,
        "conformance_claim": False,
        "reference": {"basis": "manufacturer_nominal"},
        "image": {"filename": name},
        "target": {"target_id": "t", "display_name": target},
        "profile": {"profile_id": "p", "display_name": profile},
        "metrics": [], "warnings": warnings or [],
    }


def library(root: Path, *, captures: int = 4, failures: int = 1,
            index: dict | None = None) -> Path:
    """One loose run and one batch folder beside it."""
    (root / "sg.json").write_text(json.dumps(record(
        "sg_morning.tif", when="2026-09-14T09:00:00+00:00",
        target="ColorChecker Digital SG", profile="Metamorfoze Full",
        warnings=["patch K8 clipped"],
    )))
    folder = root / "batch-20260912-143200-Odonata-2026"
    folder.mkdir()
    for i in range(captures):
        folder.joinpath(f"c{i:03d}.json").write_text(json.dumps(record(
            f"odo_{i:04d}.tif",
            overall="FAIL" if i < failures else "PASS",
            when=f"2026-09-12T10:{i:02d}:00+00:00",
            warnings=[f"edge {i} short"] if i == 2 else [],
        )))
    folder.joinpath("batch.json").write_text(json.dumps(index if index is not None else {
        "created_utc": "2026-09-12T14:32:00+00:00",
        "target": {"target_id": "t", "display_name": "ISA Object Level Target 0.5X"},
        "profile": {"profile_id": "p", "display_name": "FADGI 3-Star"},
        "summary": f"{captures} of {captures} captures measured, {failures} failed.",
        "captures": [{"capture": "/masters/Odonata 2026/c000.tif"}],
    }))
    return folder


def table(qt_app, root: Path) -> RunTableView:
    model = RunTableModel()
    model.set_runs(scan_library(root))
    proxy = RunFilterProxy()
    proxy.setSourceModel(model)
    view = RunTableView()
    view.setModel(proxy)
    return view


def top_rows(view) -> list[str]:
    column = COLUMN_INDEX["file"]
    return [
        view.model().index(row, column).data()
        for row in range(view.model().rowCount())
    ]


def group_index(view) -> QModelIndex:
    for row in range(view.model().rowCount()):
        index = view.model().index(row, 0)
        if isinstance(index.data(RUN_ROLE), BatchGroup):
            return index
    raise AssertionError("no batch row in the table")


# -- the shape on disk becomes the shape on screen -----------------------------


def test_a_batch_is_one_row_that_opens(tmp_path):
    library(tmp_path, captures=20)
    rows = scan_library(tmp_path)

    assert len(rows) == 2, "one loose run and one batch"
    batch = next(row for row in rows if isinstance(row, BatchGroup))
    assert len(batch.runs) == 20
    assert "Odonata 2026" in batch.image_name
    assert "20 captures" in batch.image_name


def test_nothing_is_lost_by_the_grouping(tmp_path):
    """The count at the foot of the library counts records, not rows. A batch of
    a hundred reported as one run would make the library look empty."""
    library(tmp_path, captures=20)
    assert len(runs_in(scan_library(tmp_path))) == 21


def test_the_row_speaks_for_the_batch_rather_than_for_its_first_record(tmp_path):
    """Its own index says what was run and when, and the records inside were
    written at their own times over the hours a folder takes."""
    library(tmp_path, captures=4)
    batch = next(r for r in scan_library(tmp_path) if isinstance(r, BatchGroup))

    assert batch.target_name == "ISA Object Level Target 0.5X"
    assert batch.profile_name == "FADGI 3-Star"
    assert batch.created_utc == "2026-09-12T14:32:00+00:00"
    assert batch.summary.endswith("1 failed.")
    # Every warning every record raised, so the column counts what is in there.
    assert batch.warning_count == 1


def test_a_batch_whose_index_cannot_be_read_still_shows_its_records(tmp_path):
    """A record that vanished because a file beside it was truncated is the one
    outcome the library exists to prevent."""
    folder = library(tmp_path, captures=3)
    folder.joinpath("batch.json").write_text("{not json", encoding="utf-8")
    batch = next(r for r in scan_library(tmp_path) if isinstance(r, BatchGroup))

    assert len(batch.runs) == 3
    assert "Odonata-2026" in batch.image_name, "the folder's own name, as a fallback"


def test_the_records_are_children_of_their_batch_and_of_nothing_else(qt_app, tmp_path):
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    model = view.model()

    assert model.rowCount() == 2
    group = group_index(view)
    assert model.rowCount(group) == 4
    loose = next(
        model.index(row, 0) for row in range(model.rowCount())
        if not isinstance(model.index(row, 0).data(RUN_ROLE), BatchGroup)
    )
    assert model.rowCount(loose) == 0
    child = model.index(0, 0, group)
    assert child.parent() == group
    assert child.data(RUN_ROLE).image_name.startswith("odo_")


# -- a batch is never given a verdict ------------------------------------------


def test_a_batch_carries_a_tally_and_not_a_verdict(tmp_path):
    """Pooling a hundred captures into one word would let a single bad frame
    fail the lot and lose which one. Conformance here is stated per image."""
    library(tmp_path, captures=4, failures=1)
    batch = next(r for r in scan_library(tmp_path) if isinstance(r, BatchGroup))

    assert batch.status.label == "1 FAIL (ADVISORY), 3 PASS (ADVISORY)"
    assert "no combined verdict" in batch.status.hint


def test_the_status_column_paints_no_dot_on_a_batch_row(qt_app, tmp_path):
    """The rule above, where it would actually be broken. A dot is one word
    about a hundred captures, so there is none -- the cell is the chevron that
    opens the folder instead."""
    library(tmp_path, captures=4, failures=1)
    view = table(qt_app, tmp_path)
    delegate = StatusDotDelegate(view)

    def painted(index) -> set[int]:
        image = QImage(40, 30, QImage.Format.Format_ARGB32)
        image.fill(0)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 40, 30)
        painter = QPainter(image)
        delegate.paint(painter, option, index)
        painter.end()
        return {image.pixel(x, y) for x in range(40) for y in range(30)}

    group = group_index(view)
    run = view.model().index(0, 0, group)
    fail = int(tone_colors(run.data(RUN_ROLE).status.tone)[0].lstrip("#"), 16)

    assert any(pixel & 0xFFFFFF == fail for pixel in painted(run)), (
        "a record still shows its own verdict"
    )
    assert not any(pixel & 0xFFFFFF == fail for pixel in painted(group)), (
        "and the batch above it shows none"
    )


def test_a_batch_is_not_a_run_and_cannot_be_combined_into_a_session(qt_app, tmp_path):
    """A session pools captures to ask whether the system passed. A batch is a
    hundred independent verdicts, and pooling them is the one thing it must not
    become -- so selecting the folder's row selects no runs."""
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.setExpanded(group_index(view), True)
    view.selectAll()

    rows = view.selected_rows()
    runs = view.selected_runs()
    assert any(isinstance(row, BatchGroup) for row in rows)
    assert not any(isinstance(run, BatchGroup) for run in runs)
    assert len(runs) == 5, "the loose run and the four inside the batch, counted once"


# -- searching reaches inside a folder that is shut ----------------------------


def test_searching_the_batch_s_name_finds_it_and_what_is_in_it(qt_app, tmp_path):
    """The name is what a person calls the batch and it appears nowhere on the
    records inside, so without carrying it down the search returned a folder
    with nothing in it."""
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.model().set_query("odonata")

    assert top_rows(view) == [
        next(r.image_name for r in scan_library(tmp_path) if isinstance(r, BatchGroup))
    ]
    assert view.model().rowCount(view.model().index(0, 0)) == 4


def test_one_record_s_warning_does_not_drag_in_its_siblings(qt_app, tmp_path):
    """The batch's row carries the union of every warning its records raised, so
    that a shut folder can still be searched. Letting the records inherit that
    made searching for one warning over a folder of a hundred return the
    hundred."""
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.model().set_query("edge 2 short")

    assert len(top_rows(view)) == 1
    assert view.model().rowCount(view.model().index(0, 0)) == 1


def test_a_record_inside_a_batch_is_found_on_its_own_terms(qt_app, tmp_path):
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.model().set_query("odo_0002")

    assert view.model().rowCount(view.model().index(0, 0)) == 1


# -- deleting, and staying open ------------------------------------------------


def test_deleting_a_batch_moves_the_folder_and_its_index(tmp_path):
    """A folder of a thousand records cleared one confirmation at a time is not
    clearing it -- and a batch recovered from the trash has to come back as a
    batch rather than as a thousand loose files."""
    folder = library(tmp_path, captures=4)
    destination = trash_batch(folder)

    assert not folder.exists()
    assert destination.parent.name == TRASH_DIR_NAME
    assert (destination / "batch.json").is_file()
    assert len(list(destination.glob("c*.json"))) == 4
    assert len(runs_in(scan_library(tmp_path))) == 1, "the trash is not the library"


def test_the_model_still_answers_the_other_parent(qt_app, tmp_path):
    """``QAbstractItemModel.parent(index)`` and ``QObject.parent()`` are one name
    in Python, so declaring the tree's one hid the object's -- and every call
    Qt or this application made to ``model.parent()`` raised a TypeError from
    inside a line nobody wrote."""
    from PySide6.QtWidgets import QWidget

    owner = QWidget()
    owner.setObjectName("Owner")
    model = RunTableModel(owner)
    model.set_runs(scan_library(tmp_path))
    assert model.parent() is owner


def test_an_opened_batch_stays_open_across_a_reload(qt_app, tmp_path):
    """The library re-reads itself on every visit and after every delete, and a
    reader who opened a folder to look through it has not asked for it to be
    shut again each time."""
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.setExpanded(group_index(view), True)
    assert view.isExpanded(group_index(view))

    view.model().sourceModel().set_runs(scan_library(tmp_path))
    assert view.isExpanded(group_index(view)), "and it is still open"


def test_shutting_it_is_remembered_too(qt_app, tmp_path):
    library(tmp_path, captures=4)
    view = table(qt_app, tmp_path)
    view.setExpanded(group_index(view), True)
    view.setExpanded(group_index(view), False)

    view.model().sourceModel().set_runs(scan_library(tmp_path))
    assert not view.isExpanded(group_index(view))
