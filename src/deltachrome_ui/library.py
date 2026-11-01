"""The run library: saved analyzes, listed.

The app opens on this rather than on an empty document, and that is a structural
decision made early on purpose. Batch folders and per-device time series are
roadmap phase 3 and not built yet -- but :class:`~deltachrome.results.models.AnalysisRun`
already persists everything they need, per-patch values included, precisely so that
an old capture can be re-scored against a revised profile without re-shooting it.
A frontend built as a single-image window would have to be rebuilt around a library
later. This one starts as the library with one run in it.

Reading is deliberately forgiving. A record is JSON on disk that may have been
written by an older or newer version of the engine, so every field is fetched with
a default and a file that cannot be parsed becomes a visible bad row rather than a
crash on startup. What is *not* forgiving is the verdict: that goes through
:mod:`deltachrome_ui.status`, which refuses to show an unqualified pass for a run
that never earned one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deltachrome.profiles.evaluator import criteria_are_local
from deltachrome.reporting.labels import BASIS_LABELS
from deltachrome_ui.status import (
    Status,
    TONE_UNKNOWN,
    evidence_sound,
    status_for_record,
)


def _stamp(created_utc: str) -> datetime | None:
    try:
        return datetime.fromisoformat(created_utc)
    except (TypeError, ValueError):
        return None


def _stamp_display(created_utc: str) -> str:
    """ISO 8601, local time, to the minute.

    Ordered largest unit first so the column reads and sorts the same way, and
    carrying the time because two runs of the same chart on the same day is the
    ordinary case rather than the exception. Local rather than UTC: the question
    this answers is "was this the morning session or the afternoon one", which is
    a question about where the operator was.
    """
    stamp = _stamp(created_utc)
    if stamp is None:
        return created_utc or "unknown date"
    return stamp.astimezone().strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class RunSummary:
    """One row in the library. Enough to list without holding every record open."""

    path: Path
    run_id: str
    created_utc: str
    image_name: str
    image_path: str | None
    target_name: str
    profile_name: str
    #: The ids, kept beside the display names so a run can be *reused* and not
    #: merely read: reopening one restores what it was judged as and against,
    #: rather than making the operator remember and re-pick.
    target_id: str | None
    profile_id: str | None
    status: Status
    #: What the verdict was formed against, as a short phrase for the library's
    #: own column: ``Measured``, ``Nominal``, ``Derived``, or a combination.
    #: Read from the record rather than inferred from ``conformance_claim``,
    #: because the claim collapses two independent things -- which reference
    #: data was used, and whether the profile was the standard's own -- and a
    #: person filtering a year of runs wants to ask about them separately.
    basis: str
    #: Why the run cannot be quoted as conformance, verbatim from the record.
    #: The basis phrase is a label; these are the sentences behind it.
    caveats: tuple[str, ...]
    #: Every warning the run raised, not merely how many. The count alone is a
    #: dead end on a screen that cannot show what was warned about.
    warnings: tuple[str, ...]
    #: The placement this run used, in image pixels, when the record has one.
    #: This is what lets a new analysis of the same setup start from the last
    #: one's corners instead of from a default rectangle.
    corners: list[tuple[float, float]] | None
    #: How many charts this record holds. One for every record written before a
    #: capture could hold several, and for every capture that did hold one.
    chart_count: int = 1
    #: Every chart the record holds, as ``(target_id, corners, placed)``, so a
    #: re-run restores the frame rather than its first chart. ``placed`` is the
    #: record's own ``geometry.placed``, ``None`` on a record from before it was
    #: written.
    charts: tuple = ()
    #: Set when the file could not be read. Everything else is then a placeholder.
    error: str | None = None

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def created_display(self) -> str:
        return _stamp_display(self.created_utc)

    @property
    def created_at(self) -> datetime | None:
        return _stamp(self.created_utc)

    @property
    def is_readable(self) -> bool:
        return self.error is None


def _corners_from(record: dict[str, Any]) -> list[tuple[float, float]] | None:
    raw = (record.get("geometry") or {}).get("corners_px")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        return [(float(p[0]), float(p[1])) for p in raw]
    except (TypeError, ValueError, IndexError):
        return None


def establish_kwargs(record: dict[str, Any]) -> dict[str, Any] | None:
    """What it would take to read this chart's capture again, from its own record.

    The target, the profile, the corners, the reference and the run options it
    was judged under all travel with the record precisely so re-establishing the
    measurement is ``analyze`` with nothing asked of the operator.
    ``None`` where the record is missing something no amount of re-decoding
    supplies: a record with no placement, or one written before a target or
    profile id was carried at all.

    Read the same tolerant way :func:`deltachrome_ui.command.command_for_record`
    already does, and for the same reason: this is a record that may have been
    written by an older engine, not a shape this version controls.
    """
    target_id = (record.get("target") or {}).get("target_id")
    profile_id = (record.get("profile") or {}).get("profile_id")
    corners = _corners_from(record)
    if not target_id or not profile_id or corners is None:
        return None
    kwargs: dict[str, Any] = {
        "target": target_id,
        "profile": profile_id,
        "corners": corners,
        "reference_path": (record.get("reference") or {}).get("path"),
    }
    inset = (record.get("sampling") or {}).get("roi_inset")
    if inset is not None:
        kwargs["roi_inset"] = inset
    for key in ("original_size", "percentile_method", "frequency_units"):
        if record.get(key) is not None:
            kwargs[key] = record[key]
    # The corners are the record's own, so how they came to be is the record's
    # own too. Absent from records written before these were fields.
    geometry = record.get("geometry") or {}
    if geometry.get("placed") is not None:
        kwargs["placed"] = geometry["placed"]
    if geometry.get("identified") is not None:
        kwargs["identified"] = bool(geometry["identified"])
    return kwargs


def _basis_from(record: dict[str, Any]) -> str:
    """What this verdict was formed against, in a phrase.

    Two independent departures from a conformance test, and the record stores
    both separately even though ``conformance_claim`` folds them into one bit:
    the reference data may be the manufacturer's nominal values rather than a
    measurement of the chart, and the profile may be a local guideline rather
    than a published standard. A run can be
    either, both, or neither.
    """
    parts: list[str] = []
    basis = (record.get("reference") or {}).get("basis")
    if basis:
        parts.append(BASIS_LABELS.get(str(basis), str(basis)))
    if criteria_are_local(record.get("profile")):
        # "Local", not "Derived". What the column reports is what the criteria
        # *are* -- this site's own. There is no parent left to name.
        parts.append("Local")
    if parts:
        return " + ".join(parts)
    # A record from before the basis was written down. The claim is all there
    # is, and it must not be reported as "Measured" on that alone.
    return "Measured" if record.get("conformance_claim", True) else "Unstated"


def _strings(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(v) for v in values)


def summarise(record: dict[str, Any], path: Path) -> RunSummary:
    """One row, from either shape of record.

    A capture record carries ``run_id``, ``target`` and ``geometry`` at its top level
    precisely so that this needs no branch: the only thing it has to ask is how many
    charts are in there, because that is the one fact a row cannot derive.
    """
    image = record.get("image") or {}
    target = record.get("target") or {}
    profile = record.get("profile") or {}
    charts = record.get("charts")
    return RunSummary(
        chart_count=len(charts) if isinstance(charts, list) and charts else 1,
        path=path,
        run_id=str(record.get("run_id", "")),
        created_utc=str(record.get("created_utc", "")),
        image_name=str(image.get("filename") or path.stem),
        image_path=image.get("path"),
        target_name=str(target.get("display_name") or target.get("target_id") or "—"),
        profile_name=str(profile.get("display_name") or profile.get("profile_id") or "—"),
        target_id=target.get("target_id"),
        profile_id=profile.get("profile_id"),
        status=status_for_record(record),
        basis=_basis_from(record),
        caveats=_strings(record.get("conformance_caveats")),
        warnings=_strings(record.get("warnings")),
        corners=_corners_from(record),
        charts=_charts_from(record),
    )


def _charts_from(record: dict[str, Any]) -> tuple:
    """``RunSummary.charts``: each chart with a target and a placement, in order."""
    charts = record.get("charts")
    held = []
    for chart in (charts if isinstance(charts, list) and charts else [record]):
        if not isinstance(chart, dict):
            continue
        target_id = (chart.get("target") or {}).get("target_id")
        corners = _corners_from(chart)
        if target_id and corners:
            held.append((
                str(target_id), corners, (chart.get("geometry") or {}).get("placed"),
            ))
    return tuple(held)


def _unreadable(path: Path, reason: str) -> RunSummary:
    return RunSummary(
        path=path,
        run_id="",
        created_utc="",
        image_name=path.name,
        image_path=None,
        target_name="—",
        profile_name="—",
        target_id=None,
        profile_id=None,
        status=Status("UNREADABLE", TONE_UNKNOWN, reason),
        basis="—",
        caveats=(),
        warnings=(),
        corners=None,
        chart_count=0,
        error=reason,
    )


def read_run(path: Path) -> RunSummary:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _unreadable(path, f"Not valid JSON: {exc}")
    except OSError as exc:
        return _unreadable(path, f"Could not be read: {exc}")
    if not isinstance(record, dict):
        return _unreadable(path, "This file does not contain a run record.")
    return summarise(record, Path(path))


def read_record(path: Path) -> dict[str, Any]:
    """The full record, for the detail view. Raises rather than papering over."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


#: A batch writes this beside its records to say what the run was. It is not a
#: run record and must not be listed as an unreadable one.
BATCH_INDEX_NAME = "batch.json"


def scan_runs(directory: Path) -> list[RunSummary]:
    """Every run in a directory, newest first.

    Unreadable files are listed, not skipped. A record that silently vanished from
    the library because its JSON was truncated is worse than a visible bad row --
    the whole point of persisting raw measurements is that they can be relied on to
    still be there.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    # Through subfolders, because a batch writes its records into one of its own
    # -- a hundred rows arriving at one instant are hard to tell on disk from a
    # hundred that arrived over a month. The table should still see all of them:
    # it is the one place records are read, and a record the library cannot show
    # is a record nobody can sort, search or re-score.
    runs = [
        read_run(p) for p in sorted(directory.rglob("*.json"))
        if p.name != BATCH_INDEX_NAME
        and TRASH_DIR_NAME not in p.relative_to(directory).parts
    ]
    runs.sort(key=lambda r: (r.created_utc or "", r.path.name), reverse=True)
    return runs


@dataclass(frozen=True)
class BatchGroup:
    """One batch's records, as a row of the library that opens.

    **A batch is a hundred rows that arrived at one instant**, and in a table
    sorted by date that is a hundred rows between a person and everything else
    they have. On disk they are already a folder of their own -- ``batch_dir``
    put them there so that somebody looking at the filesystem could tell -- and
    this is the same fact in the table: one row that opens.

    Its fields are named after :class:`RunSummary`'s on purpose. Every column of
    the library reads its value off the row it is given, and for seven of the
    eight the two kinds of row mean the same thing by the same name: a target, a
    guideline, a count of warnings, a time. Only the two painted columns branch,
    and they branch in the delegate that paints them.

    **It is never a verdict.** A batch states conformance per image -- pooling
    would let one bad frame fail the lot and lose which one -- so ``status``
    carries a *tally* as its label and exists to sort and to be searched. The
    status column paints a disclosure chevron on these rows and no dot at all.
    """

    path: Path
    #: The records inside, already summarised. Held rather than re-read, since
    #: the table needs every one of them the moment the row is opened.
    runs: tuple[RunSummary, ...]
    image_name: str
    created_utc: str
    target_name: str
    profile_name: str
    basis: str
    status: Status
    #: Every warning every record in the batch raised, so that searching for one
    #: finds the batch that holds it even while the batch is closed.
    warnings: tuple[str, ...]
    #: What the batch's own index says became of it, in one sentence.
    summary: str
    caveats: tuple[str, ...] = ()
    target_id: str | None = None
    profile_id: str | None = None
    image_path: str | None = None
    corners: list[tuple[float, float]] | None = None
    run_id: str = ""
    error: str | None = None

    @property
    def chart_count(self) -> int:
        return len(self.runs)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def is_readable(self) -> bool:
        return self.error is None

    @property
    def created_display(self) -> str:
        return _stamp_display(self.created_utc)

    @property
    def created_at(self) -> datetime | None:
        return _stamp(self.created_utc)


def _tally(runs: tuple[RunSummary, ...]) -> Status:
    """How a batch's captures came out, as a count and never as a verdict.

    The tone is the worst any capture reached, which is a **sort key**: it is
    what puts the batches holding a failure at the top when the status column is
    sorted, and it is what "fail" matches when it is searched for. Nothing paints
    it -- see :class:`~deltachrome_ui.run_table.StatusDotDelegate`, which draws a
    chevron on these rows -- because a batch asks whether *each* of a hundred
    captures passed, and one dot cannot answer that.
    """
    if not runs:
        return Status("no records", TONE_UNKNOWN,
                      "This batch folder holds no run records.")
    counted: dict[str, tuple[Status, int]] = {}
    for run in runs:
        _, so_far = counted.get(run.status.label, (run.status, 0))
        counted[run.status.label] = (run.status, so_far + 1)
    ordered = sorted(counted.values(), key=lambda item: (item[0].rank, item[0].label))
    worst = ordered[0][0]
    label = ", ".join(f"{number} {status.label}" for status, number in ordered)
    return Status(
        label, worst.tone,
        f"{len(runs)} captures: {label}.  Every one of them is judged on its "
        "own — a batch states conformance per image, so there is no combined "
        "verdict here to read.",
    )


def _batch_group(folder: Path, runs: list[RunSummary]) -> BatchGroup:
    """One row for a batch folder, from its own index and the records in it.

    The index is what the batch wrote down about itself, and it is read
    forgivingly for the same reason a record is: it may have been written by
    another version, and a batch whose index is truncated must still show its
    records rather than disappearing with it.
    """
    try:
        index = json.loads((folder / BATCH_INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        index = {}
    if not isinstance(index, dict):
        index = {}
    ordered = tuple(sorted(runs, key=lambda r: (r.created_utc or "", r.path.name)))

    captures = index.get("captures")
    source = ""
    if isinstance(captures, list) and captures:
        first = (captures[0] or {}).get("capture")
        if first:
            # The folder the masters were in, which is what a person calls the
            # batch. The index's own ``directory`` is where the RECORDS went --
            # this folder -- so it cannot answer this.
            source = Path(str(first)).parent.name
    if not source:
        # ``batch-20260910-14322-Odonata-2026``: the stamp is fixed-width, so
        # what is left after it is the capture folder's name as ``batch_dir``
        # wrote it. A fallback, for an index that could not be read.
        parts = folder.name.split("-", 3)
        source = parts[3] if len(parts) == 4 else folder.name
    count = len(ordered)
    name = f"{source}   ·   {count} capture{'s' if count != 1 else ''}"

    target = index.get("target") or {}
    profile = index.get("profile") or {}
    bases = {run.basis for run in ordered}
    return BatchGroup(
        path=folder,
        runs=ordered,
        image_name=name,
        created_utc=str(index.get("created_utc") or
                        (ordered[0].created_utc if ordered else "")),
        target_name=str(target.get("display_name")
                        or (ordered[0].target_name if ordered else "—")),
        profile_name=str(profile.get("display_name")
                         or (ordered[0].profile_name if ordered else "—")),
        basis=bases.pop() if len(bases) == 1 else ("Mixed" if bases else "—"),
        status=_tally(ordered),
        warnings=tuple(w for run in ordered for w in run.warnings),
        summary=str(index.get("summary") or ""),
        target_id=target.get("target_id"),
        profile_id=profile.get("profile_id"),
    )


def scan_library(directory: Path) -> list[Any]:
    """The library's rows: loose runs, and one row per batch folder.

    Two levels and no more. A batch is the only thing that writes a folder of
    records, and a folder inside a folder is not a shape anything here produces
    -- so a record below a batch's own directory belongs to that batch, and
    everything else is a row in its own right.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    folders = sorted(
        p.parent for p in directory.rglob(BATCH_INDEX_NAME)
        if TRASH_DIR_NAME not in p.relative_to(directory).parts
    )
    inside: dict[Path, list[RunSummary]] = {folder: [] for folder in folders}
    loose: list[Any] = []
    for path in sorted(directory.rglob("*.json")):
        parts = path.relative_to(directory).parts
        if path.name == BATCH_INDEX_NAME or TRASH_DIR_NAME in parts:
            continue
        owner = next((f for f in folders if f in path.parents), None)
        run = read_run(path)
        if owner is None:
            loose.append(run)
        else:
            inside[owner].append(run)
    rows = loose + [_batch_group(folder, inside[folder]) for folder in folders]
    rows.sort(key=lambda row: (row.created_utc or "", row.path.name), reverse=True)
    return rows


def runs_in(rows: list[Any]) -> list[RunSummary]:
    """Every record the rows hold, batches opened out.

    What the library counts and what a scan is asserted against. A batch's rows
    are records like any other -- that is the whole of what makes them sortable,
    searchable and re-scorable -- and the grouping is a way of showing them.
    """
    out: list[RunSummary] = []
    for row in rows:
        out.extend(row.runs if isinstance(row, BatchGroup) else [row])
    return out


#: Where deleted records go. Inside the library folder rather than the operating
#: system's own trash: a library can be a shared lab drive or a project folder on
#: a network volume, where a per-user desktop trash either does not exist or is
#: somewhere nobody else can look.
TRASH_DIR_NAME = ".trash"


def trash_run(path: Path) -> Path:
    """Move a run record out of the library, without destroying it.

    Deleting is not ``unlink`` here, and that is a decision rather than caution.
    A record is the archive: it holds every per-patch value the run measured, and
    it exists so a capture can be re-scored against a revised profile years later
    without being re-shot. There is no way to regenerate one except by finding
    the master and repeating the placement by hand. A misclick in a list of two
    thousand rows must therefore be recoverable, and the recovery must not depend
    on the operator having a desktop at all.

    Returns where it went, so the caller can say.
    """
    path = Path(path)
    trash = path.parent / TRASH_DIR_NAME
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / path.name
    # Same capture, same profile, run twice and deleted twice: the second must
    # not overwrite the first on its way out.
    suffix = 1
    while destination.exists():
        destination = trash / f"{path.stem}.{suffix}{path.suffix}"
        suffix += 1
    path.replace(destination)
    return destination


def trash_batch(folder: Path) -> Path:
    """Move a whole batch's folder out of the library, without destroying it.

    The same promise :func:`trash_run` makes, at the size a batch actually
    reaches. A folder of a thousand records cleared one confirmation at a time is
    not clearing it -- and the index goes with them, which is what keeps a
    recovered folder a batch rather than a thousand loose records.
    """
    folder = Path(folder)
    trash = folder.parent / TRASH_DIR_NAME
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / folder.name
    suffix = 1
    while destination.exists():
        destination = trash / f"{folder.name}.{suffix}"
        suffix += 1
    folder.replace(destination)
    return destination


@dataclass(frozen=True)
class SavedEvaluation:
    """The evaluation half of a result, read back off disk.

    :class:`~deltachrome.profiles.evaluator.ProfileEvaluation` derives its
    caveats live from the profile object and the reference basis. A saved run has
    neither -- so the report cannot re-derive them and must be handed what was
    recorded. That is the whole reason ``conformance_caveats`` is now persisted:
    a reopened run that showed ``PASS (ADVISORY)`` and could not say why would
    be worse than one that did not open at all.
    """

    overall: str
    conformance_claim: bool
    conformance_caveats: list[str]
    #: The same caveats in one line, as the report's header shows them. Composed
    #: from the record's own two facts rather than by shortening the sentences,
    #: which is what lets a saved run and a fresh one produce the identical line.
    advisory_line: str | None = None
    #: Whether the numbers describe the chart that was photographed. ``None`` on a
    #: record that does not say, which is one written before the basis was stored;
    #: the tone then falls back to the claim, which is the old behaviour.
    #:
    #: **Appended rather than inserted.** This is built positionally in places, so
    #: a field added in the middle binds somebody else's argument -- which is
    #: exactly what happened, silently, and cost the report its caveat line.
    evidence_is_sound: bool | None = None


@dataclass(frozen=True)
class SavedResult:
    """What :class:`~deltachrome_ui.result_view.ResultView` needs, from a record.

    Shaped to match the live ``AnalysisResult`` exactly -- ``.run`` and
    ``.evaluation`` -- so the report renders a saved run through the same code
    that renders a fresh one. Two report renderers that could drift apart is the
    surest way to end up with a saved verdict that reads differently from the one
    that was saved.
    """

    run: Any
    evaluation: SavedEvaluation
    #: The chart this is, of how many in its capture. A record holding one leaves
    #: these at their defaults and every screen behaves as it always did.
    chart_index: int = 0
    chart_count: int = 1
    #: The other charts read from the same frame, unread until one is asked for.
    #: Kept as records rather than results so that opening a five-chart capture
    #: costs the same as opening a one-chart one.
    siblings: tuple = ()


#: Reasons a run cannot be quoted as conformance, for records written before the
#: caveats were persisted. Reconstructed from what schema 3 did store, and
#: deliberately vaguer than the originals: the profile's own display name was
#: never written down there, so an old local-guideline run can be told that it was
#: one but not which guideline.
def _recovered_caveats(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reference = record.get("reference") or {}
    if reference.get("basis") == "manufacturer_nominal":
        reasons.append(
            "judged against the manufacturer's nominal values for this model of "
            "chart, not against a measurement of the chart itself"
        )
    if criteria_are_local(record.get("profile")):
        reasons.append(
            "judged against a local guideline rather than a published standard"
        )
    elif not reasons:
        reasons.append(
            "this record predates the caveats being stored, and states that it is "
            "not a conformance result without recording which of the two reasons "
            "applied. Re-run the capture to record it."
        )
    return reasons


def load_result(path: Path, chart: int = 0) -> SavedResult:
    """One chart of a saved capture, in the shape the report renders.

    ``chart`` selects which. A record holding one -- every record written before a
    capture could hold several, and every capture that did hold one -- ignores it and
    behaves exactly as it always did.

    The capture's own combined verdict is deliberately NOT what comes back here. This
    is a chart's report, and it must show that chart's answer; the combined one is the
    session's to state, and showing it over a single chart's readings would attribute a
    board's failure to whichever card was on screen.
    """
    from deltachrome.profiles.evaluator import advisory_line

    record = read_record(Path(path))
    charts = record.get("charts")
    siblings: tuple = ()
    if isinstance(charts, list) and charts:
        siblings = tuple(charts)
        record = charts[max(0, min(chart, len(charts) - 1))]
    from deltachrome.results.models import AnalysisRun

    run = AnalysisRun.from_dict(record)
    claim = bool(record.get("conformance_claim", True))
    caveats = list(run.conformance_caveats)
    if not claim and not caveats:
        caveats = _recovered_caveats(record)
    line = advisory_line(
        reference_basis=(record.get("reference") or {}).get("basis"),
        local_criteria=criteria_are_local(record.get("profile")),
    )
    if not claim and not line:
        # A record from before either fact was stored. It knows it cannot be
        # quoted and not why, and saying only the first is the honest half.
        line = "Advisory: this run is not a conformance result."
    return SavedResult(
        run=run,
        evaluation=SavedEvaluation(
            overall=str(run.overall), conformance_claim=claim,
            conformance_caveats=caveats, advisory_line=line,
            evidence_is_sound=evidence_sound(record),
        ),
        chart_index=max(0, min(chart, len(siblings) - 1)) if siblings else 0,
        chart_count=len(siblings) or 1,
        siblings=siblings,
    )
