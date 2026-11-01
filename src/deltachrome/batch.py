"""A folder of captures, judged one at a time against one guideline.

**A batch is not a session, and the difference is the verdict.** A session pools
several charts to ask whether the *system* passed, because almost no chart measures
everything a standard asks for. A batch asks whether *each* of a hundred production
captures passed, and pooling them would let one bad frame fail the lot and lose
which one -- which is also the wrong shape for the domain, since conformance is
stated per image. So a batch is a list of independent verdicts with a summary over
them, and every record it writes is an ordinary one that opens, re-scores and
combines through the code that already exists.

Three rules follow from a batch being unattended.

**A failure never stops the run.** Every capture gets an outcome: measured, or
refused with the reason, recorded either way. A folder of a hundred where the
eleventh has no chart in it must produce ninety-nine results and one stated
absence, not eleven results and a traceback.

**Identification is off.** The target is declared. Open identification names the
chart from the capture, and on a frame that is mostly *subject* it can name one
confidently and wrongly -- a Smithsonian specimen sheet carrying an Object Level
Target scores as an Image Engineering UTT, on a ramp reading +1.00, because
nothing better is available to rank. With an operator watching that is a bad
suggestion; in an unattended run over a hundred frames it is a hundred plausible
wrong answers.

**Records are written as they are made.** That bounds memory to one decoded
capture, and it is what lets a run that died at frame ninety be resumed rather
than repeated. The index is rewritten after every capture for the same reason.

**What a batch can say that a hundred single runs cannot** is how its own captures
differ from each other. Across twenty captures of one chart on one rig, every
frame met the sharpness requirement and two sat four standard deviations below the
rest of the batch -- which no per-image verdict mentions, because none of them
failed. The batch's own distribution is the reference. See :func:`outliers`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from deltachrome.errors import DeltachromeError
from deltachrome.metrics.base import MetricResult
from deltachrome.profiles.schema import AnalysisProfile, load_profile
from deltachrome.results.models import environment_info, now_utc
from deltachrome.targets.schema import TargetDefinition

#: Suffixes a batch will attempt. Everything else in the folder is passed over
#: without comment -- a batch is pointed at a directory of captures and a directory
#: of captures routinely also holds a spreadsheet, a colour profile and a .DS_Store,
#: none of which is a failed capture and none of which should be reported as one.
IMAGE_SUFFIXES = frozenset({
    ".tif", ".tiff", ".jp2", ".jpf", ".jpx", ".j2k", ".png", ".jpg", ".jpeg",
})

#: The file a batch writes beside its records, naming what was run and what became
#: of every capture. Rewritten after each one, so a run that stops halfway leaves a
#: readable account of the half that finished.
INDEX_NAME = "batch.json"

#: Fewest captures a metric must have been measured on before this reports
#: outliers. Below about ten, the median absolute deviation is not a scale estimate
#: so much as a statement about which sample happened to land in the middle, and
#: flagging against it would invent findings on small batches.
MIN_FOR_OUTLIERS = 10

#: How many robust standard deviations from the batch's own median a reading must
#: sit before it is called out. Three, which on twenty captures of one chart flags
#: the two frames that measured softer than the rest and nothing else.
OUTLIER_SIGMAS = 3.0

#: Converts a median absolute deviation into a standard deviation for a normal
#: distribution. Textbook constant, named rather than inlined.
MAD_TO_SIGMA = 1.4826

#: How much of the profile's own criterion a reading must differ by, on top of
#: being statistically unusual, before it is worth an operator's attention.
#:
#: **Standard deviations measure unusualness and say nothing about importance**,
#: and on a batch that agrees with itself closely the two come apart completely.
#: Across twenty captures of one chart, incremental gain in the highlights read
#: 0.971 with a spread of 0.003, so a frame at 0.941 is *eleven* standard
#: deviations out and has moved six percent of the way across a criterion running
#: from 0.7 to 1.2. Reporting it buries the finding that matters -- two frames
#: measurably softer than the rest -- under ten that do not.
#:
#: The criterion is the only statement anywhere in the system of how large a
#: difference has to be before it means something, so it is what the deviation is
#: measured against: the width of a band, or the headroom to a one-sided
#: threshold. At fifteen percent the twenty Odonata captures report the soft frames
#: on both sharpness metrics and nothing else.
OUTLIER_SHARE = 0.15


@dataclass
class BatchCapture:
    """One capture's place in a batch: what happened, and where it was written.

    Deliberately narrow. It holds the readings a summary reduces and the path of
    the record, and never the run itself, because a batch of a few thousand
    captures would otherwise keep a few thousand runs in memory to answer questions
    the records on disk already answer.
    """

    path: Path
    outcome: str  # "measured" | "refused"
    reason: str | None = None
    record_path: Path | None = None
    overall: str | None = None
    conformance_claim: bool = False
    results: list[MetricResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: What the numbers were compared against. Separate from the claim above
    #: because they answer different questions: a run against a department's own
    #: written specification forfeits the claim and is not thereby doubtful, while
    #: a run against nominal values is only as good as the model it used.
    reference_basis: str = "per_unit_measurement"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def measured(self) -> bool:
        return self.outcome == "measured"

    def reading(self, metric_id: str) -> float | None:
        for result in self.results:
            if result.metric_id == metric_id and result.value is not None:
                value = float(result.value)
                return value if math.isfinite(value) else None
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": str(self.path),
            "outcome": self.outcome,
            "reason": self.reason,
            "record": self.record_path.name if self.record_path else None,
            "overall": self.overall,
            "conformance_claim": self.conformance_claim,
            "reference_basis": self.reference_basis,
            "warnings": list(self.warnings),
        }


@dataclass
class Outlier:
    """One capture sitting away from the rest of its own batch, on one metric."""

    metric_id: str
    capture: str
    value: float
    median: float
    sigmas: float
    units: str | None = None
    passed: bool | None = None

    @property
    def direction(self) -> str:
        """Which side of the batch this reading sits on."""
        return "above" if self.value > self.median else "below"

    def describe(self) -> str:
        verdict = ""
        if self.passed is True:
            verdict = " -- it met the requirement, and the rest of the batch did better"
        unit = f" {self.units}" if self.units else ""
        return (
            f"{self.capture}: {self.metric_id} reads {self.value:.3f}{unit} against a "
            f"batch median of {self.median:.3f}{unit}, {self.sigmas:.1f} standard "
            f"deviations out{verdict}."
        )


@dataclass
class BatchResult:
    """Every capture in one folder, judged against one guideline."""

    profile: AnalysisProfile
    target_id: str
    target_name: str
    directory: Path
    captures: list[BatchCapture]
    created_utc: str
    environment: dict[str, Any] = field(default_factory=dict)
    #: Set when the operator stopped the run rather than it finishing.
    stopped_early: bool = False
    #: Facts about the run as a whole rather than about any one capture. At
    #: present this is one: that the first capture held charts a batch does not
    #: measure. See :func:`_first_frame_notice`.
    notes: list[str] = field(default_factory=list)

    @property
    def measured(self) -> list[BatchCapture]:
        return [c for c in self.captures if c.measured]

    @property
    def refused(self) -> list[BatchCapture]:
        return [c for c in self.captures if not c.measured]

    @property
    def failed(self) -> list[BatchCapture]:
        return [c for c in self.measured if c.overall == "FAIL"]

    @property
    def passed(self) -> list[BatchCapture]:
        return [c for c in self.measured if c.overall in ("PASS", "MEASURED")]

    @property
    def conformance_claim(self) -> bool:
        """Whether every capture that was measured could be quoted as conformance."""
        return bool(self.measured) and all(c.conformance_claim for c in self.measured)

    @property
    def evidence_is_sound(self) -> bool:
        """Whether every measured capture's numbers describe its own chart."""
        return bool(self.measured) and all(
            c.reference_basis == "per_unit_measurement" for c in self.measured
        )

    def outliers(self) -> list[Outlier]:
        return outliers(self.measured)

    def unusual_captures(self) -> list[tuple[str, list["Outlier"]]]:
        """Captures unlike the rest of the batch, worst first, with what is odd.

        **Grouped by capture rather than listed per reading**, because the question
        a batch summary is asked is which frames to go and look at. Twenty captures
        of one chart on one rig agree with each other very closely, so the spread is
        tiny and a great many readings are several standard deviations from it: on a
        4-star profile the same twenty produced nineteen separate findings, which
        pushed the run's own list of captures off the screen and buried the two
        frames that mattered among seventeen that did not.

        Grouping also says the thing a per-reading list cannot. One frame soft on
        SFR50 *and* on sampling efficiency is one soft frame corroborated twice, and
        reading it as two findings makes it look like two problems.
        """
        grouped: dict[str, list[Outlier]] = {}
        for outlier in self.outliers():
            grouped.setdefault(outlier.capture, []).append(outlier)
        return sorted(
            grouped.items(), key=lambda item: -max(o.sigmas for o in item[1])
        )

    def summary_line(self) -> str:
        """What happened, in one sentence."""
        parts = [f"{len(self.measured)} of {len(self.captures)} captures measured"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.refused:
            parts.append(f"{len(self.refused)} could not be measured")
        if self.stopped_early:
            parts.append("the run was stopped before the end")
        return ", ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": "deltachrome/batch",
            "$schema_version": 1,
            "created_utc": self.created_utc,
            "directory": str(self.directory),
            "profile": {
                "profile_id": self.profile.profile_id,
                "display_name": self.profile.display_name,
                "local_criteria": self.profile.local_criteria,
            },
            "target": {"target_id": self.target_id, "display_name": self.target_name},
            "counts": {
                "captures": len(self.captures),
                "measured": len(self.measured),
                "refused": len(self.refused),
                "failed": len(self.failed),
            },
            "conformance_claim": self.conformance_claim,
            "stopped_early": self.stopped_early,
            "notes": list(self.notes),
            "summary": self.summary_line(),
            "outliers": [
                {
                    "metric_id": o.metric_id, "capture": o.capture, "value": o.value,
                    "median": o.median, "sigmas": round(o.sigmas, 2), "passed": o.passed,
                }
                for o in self.outliers()
            ],
            "captures": [c.to_dict() for c in self.captures],
            "environment": dict(self.environment),
        }


def outliers(
    captures: Sequence[BatchCapture],
    *,
    sigmas: float = OUTLIER_SIGMAS,
    share: float = OUTLIER_SHARE,
    minimum: int = MIN_FOR_OUTLIERS,
) -> list[Outlier]:
    """Captures that sit away from the rest of their own batch, per metric.

    **This is the one question a batch can answer that its captures cannot.** Every
    verdict in a batch may be a pass and two frames still be visibly unlike the
    other ninety-eight -- a focus that drifted, a light that moved, a chart that
    got fingerprinted -- and nothing in a per-image pass/fail says so, because
    nothing failed. Measured against twenty captures of one chart on one rig: every
    frame met the sharpness requirement and two read four standard deviations below
    the batch, which is the finding.

    Robust by construction. The centre is the median and the scale the median
    absolute deviation, because the thing being looked for is exactly the thing
    that would drag a mean and inflate a standard deviation -- with two outliers in
    twenty, a mean-and-sigma test hides them behind the width they themselves
    created.
    """
    found: list[Outlier] = []
    metric_ids: list[str] = []
    for capture in captures:
        for result in capture.results:
            if result.metric_id not in metric_ids:
                metric_ids.append(result.metric_id)

    for metric_id in metric_ids:
        readings = [(c, c.reading(metric_id)) for c in captures]
        readings = [(c, v) for c, v in readings if v is not None]
        if len(readings) < minimum:
            continue
        values = [v for _c, v in readings]
        middle = _median(values)
        deviation = _median([abs(v - middle) for v in values]) * MAD_TO_SIGMA
        if deviation <= 0:
            # Every capture agrees exactly. That is not a batch with no outliers so
            # much as a metric with no spread -- bit depth, the container format --
            # and dividing by it would call every reading infinitely unusual.
            continue
        example = next(
            r for c in captures for r in c.results if r.metric_id == metric_id
        )
        span = _criterion_span(example.condition, middle)
        for capture, value in readings:
            distance = abs(value - middle) / deviation
            if distance < sigmas:
                continue
            if span is not None and abs(value - middle) < share * span:
                # Unusual, and too small to act on. See OUTLIER_SHARE.
                continue
            result = next(r for r in capture.results if r.metric_id == metric_id)
            found.append(Outlier(
                metric_id=metric_id, capture=capture.name, value=value,
                median=middle, sigmas=distance, units=result.units,
                passed=result.passed,
            ))
    found.sort(key=lambda o: -o.sigmas)
    return found


#: The two shapes a numeric pass condition takes: a one-sided limit and a band.
#:
#: Matched strictly rather than by pulling every number out of the text, because
#: the conditions that are *not* numeric contain numbers too -- the master file
#: format reads "TIFF, JPEG 2000, PDF/A" and bit depth reads "8 or 16", and a loose
#: search makes the first of those a threshold of two thousand. What separates them
#: is that a numeric criterion compares against ``value`` and a list of permitted
#: values never mentions it.
_FLOAT = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_LIMIT = re.compile(rf"^\s*value\s*[<>]=?\s*({_FLOAT})\s*$")
_BAND = re.compile(rf"^\s*({_FLOAT})\s*<=?\s*value\s*<=?\s*({_FLOAT})\s*$")


def _criterion_span(condition: str | None, middle: float) -> float | None:
    """How large a difference the profile treats as the whole of the question.

    A band's width, where the criterion states two numbers; the headroom from the
    batch's own median to the limit, where it states one. ``None`` where the
    criterion names no number at all -- a list of permitted colour spaces, say --
    and the deviation is then judged on its statistics alone, there being nothing
    to weigh it against.
    """
    if not condition:
        return None
    band = _BAND.match(condition)
    if band:
        return abs(float(band.group(2)) - float(band.group(1))) or None
    limit = _LIMIT.match(condition)
    if limit:
        return abs(middle - float(limit.group(1))) or None
    return None


def captures_in(directory: str | Path, *, recursive: bool = False) -> list[Path]:
    """Every image in a folder, in a stable order.

    Sorted by name, because a batch has to be repeatable and a directory listing is
    not. Non-images are passed over silently: a folder of captures routinely holds
    a spreadsheet and a .DS_Store, and neither is a capture that failed.
    """
    root = Path(directory)
    if not root.is_dir():
        raise DeltachromeError(f"{root} is not a folder.")
    walk = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in walk
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _record_name(capture: Path, taken: set[str]) -> str:
    """A record's filename, from the capture's own, kept unique within the folder.

    Named after the capture rather than after the run id, because the first thing
    anyone does with a batch folder is look for the record belonging to a frame
    they are suspicious of, and a run id does not help them.
    """
    stem = capture.stem
    name = f"{stem}.json"
    index = 2
    while name in taken:
        name = f"{stem}_{index}.json"
        index += 1
    taken.add(name)
    return name


def already_done(into: str | Path) -> dict[str, str]:
    """Captures a previous run of this batch folder finished, by path.

    Read from the index rather than from the records present, because a record
    written for a capture that was later renamed on disk should not silently count
    as that capture's result.
    """
    index = Path(into) / INDEX_NAME
    if not index.is_file():
        return {}
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    done: dict[str, str] = {}
    for entry in data.get("captures") or []:
        capture, outcome = entry.get("capture"), entry.get("outcome")
        if capture and outcome == "measured":
            done[str(capture)] = str(entry.get("record") or "")
    return done


def run_batch(
    captures: Iterable[str | Path],
    *,
    target: str | Path | TargetDefinition,
    profile: str | Path | AnalysisProfile,
    into: str | Path,
    corners: Sequence[tuple[float, float]] | None = None,
    reference_path: str | Path | None = None,
    reference_library: str | Path | None = None,
    resume: bool = False,
    on_progress: Callable[[int, int, Path, BatchCapture | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    **run_options: Any,
) -> BatchResult:
    """Measure every capture in turn, writing each record as it is made.

    ``corners`` are a placement the operator has already checked, on this batch's
    first frame or any other of the same setup. They narrow where each capture is
    searched and never stand in for its own placement -- across twenty captures
    seeded from one, the corners actually found moved a median of 21 px and as much
    as 95 from the seed, so a chart that shifted is measured where it is. Without
    them each capture is searched whole, which works when the chart fills its frame
    and does not when it is a sixth of one.

    ``resume`` skips captures the index already records as measured, which is what
    makes a run that stopped at frame ninety cost ten frames rather than a hundred.

    A batch measures one chart per capture, and says once, from the first
    capture, when a frame held charts it will not measure -- see
    :func:`_first_frame_notice`. Asking every capture the same question was a
    per-capture option and is deferred past the beta.

    **``target`` is keyword-only and has no default**, and that is the contract
    rather than a style. It is where "identification is off" is *enforced*: there is
    no call to this function that leaves the chart to be worked out, so the hazard
    the module docstring describes cannot be reached by forgetting an argument. A
    default here -- even ``None``, meaning "identify it" -- would put a confident
    wrong answer behind an omission, which is exactly how it reached the desktop
    entry point once already. ``test_a_batch_cannot_be_asked_to_guess_its_target``
    pins it, so restoring a default fails a test rather than reopening this quietly.
    """
    from deltachrome.imaging.detect import detect_placement, detect_placement_near
    from deltachrome.imaging.loader import load_image
    from deltachrome.pipeline import analyze
    from deltachrome.session import capture_record
    from deltachrome.targets.registry import load_target

    target_def = target if isinstance(target, TargetDefinition) else load_target(target)
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)
    folder = Path(into)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in captures]
    done = already_done(folder) if resume else {}
    taken = {name for name in done.values() if name}

    result = BatchResult(
        profile=profile_def, target_id=target_def.target_id,
        target_name=target_def.display_name, directory=folder, captures=[],
        created_utc=now_utc(), environment=environment_info(),
    )

    for number, path in enumerate(paths, start=1):
        if should_stop is not None and should_stop():
            result.stopped_early = True
            break
        if str(path) in done:
            result.captures.append(_carried_over(
                path, folder / done[str(path)] if done[str(path)] else None
            ))
            _write_index(folder, result)
            if on_progress is not None:
                on_progress(number, len(paths), path, result.captures[-1])
            continue

        capture = _one(
            path, target_def=target_def, profile_def=profile_def, folder=folder,
            corners=corners, reference_path=reference_path,
            reference_library=reference_library, taken=taken,
            number=number, total=len(paths), run_options=run_options,
            batch_notes=result.notes,
            load_image=load_image, analyze=analyze, capture_record=capture_record,
            detect_placement=detect_placement, detect_placement_near=detect_placement_near,
        )
        result.captures.append(capture)
        # After every capture, not at the end. A run that dies at frame ninety has
        # to leave a readable account of the eighty-nine, and an index written once
        # at the end is the one thing a crash is guaranteed to take with it.
        _write_index(folder, result)
        if on_progress is not None:
            on_progress(number, len(paths), path, capture)

    _write_index(folder, result)
    return result


def _one(
    path: Path, *, target_def, profile_def, folder: Path, corners, reference_path,
    reference_library, taken: set[str], number: int, total: int,
    batch_notes: list[str] | None = None,
    run_options: dict[str, Any], load_image, analyze, capture_record,
    detect_placement, detect_placement_near,
) -> BatchCapture:
    """One capture, and never an exception.

    **Every failure is a value here, including the ones nobody predicted.** A batch
    is unattended and a folder of a hundred captures is a folder of a hundred
    chances to meet a truncated file, a colour space this build refuses, a chart
    that is not in the frame. Letting any of those end the run would throw away
    ninety-nine results to report one problem, so the catch is deliberately broad
    and the reason is recorded verbatim rather than summarised.
    """
    try:
        image = load_image(path)
        detection = (
            detect_placement_near(image, target_def, corners) if corners
            else detect_placement(image, target_def)
        )
        if not detection.found:
            reason = detection.reason
            if corners and detection.extent is not None:
                # "Only 0 colored patches found" reads as a fault in the
                # chart. In a batch it far more often means the chart is not where
                # the seed says -- moved, turned, or missing -- so say where the
                # search looked. Prefixed, so the finder's own sentence is kept.
                reason = f"Searched where the seed placement puts the chart. {reason}"
            return BatchCapture(path=path, outcome="refused", reason=reason)
        backwards = _ramp_reads_backwards(image, target_def, detection.corners)
        if backwards is not None:
            return BatchCapture(path=path, outcome="refused", reason=backwards)

        notes: list[str] = []
        chosen = reference_path
        if reference_library is not None and chosen is None:
            # Resolved per capture, not once for the batch: the code is printed on
            # the chart and read from the frame, so a folder holding two physical
            # units of one model pairs each with its own measurement. Where a batch
            # really is one chart photographed a hundred times this costs a scan a
            # frame and reaches the same file every time.
            from deltachrome.reference.resolve import resolve_reference

            choice = resolve_reference(
                target=target_def, image=image, library_dir=reference_library
            )
            chosen = choice.path
            notes.extend(choice.notes)
            notes.extend(choice.warnings)
        if number == 1 and batch_notes is not None:
            # Once for the whole run: a batch is a folder shot the same way
            # throughout, so a board in the first frame means boards in all of them.
            notice = _first_frame_notice(image, target_def, detection.corners)
            if notice:
                batch_notes.append(notice)
        run = analyze(
            image,
            target=target_def,
            profile=profile_def,
            corners=detection.corners,
            reference_path=chosen,
            extra_warnings=[*detection.notes, *notes],
            placed="automatic",
            **run_options,
        )
        record = capture_record([run])
        record_path = record.save(folder / _record_name(path, taken))
        return BatchCapture(
            path=path, outcome="measured", record_path=record_path,
            overall=str(run.overall), conformance_claim=bool(run.conformance_claim),
            results=list(run.evaluation.results), warnings=list(run.warnings),
            reference_basis=run.run.reference.get("basis", "per_unit_measurement"),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except DeltachromeError as failure:
        # Written for the operator already; the class name adds nothing to it.
        return BatchCapture(path=path, outcome="refused", reason=str(failure))
    except Exception as failure:  # noqa: BLE001 -- see the docstring
        return BatchCapture(
            path=path, outcome="refused",
            reason=f"{type(failure).__name__}: {failure}",
        )


def _ramp_reads_backwards(image, target_def, corners) -> str | None:
    """Why a found placement is not on the chart, if its own grey ramp says so.

    **A finder can return a confident placement of the wrong thing**, and in a
    batch nobody is looking. On a board of four QA-62 cards the crosshair finder
    returned a valid quadrilateral made of marks from different cards; measured,
    it gave SFR50 16% and 6.5 px of misregistration -- a FAIL describing nothing.
    The chart's own ladder catches it: read at that placement it ran backwards,
    -0.47 and -0.26, where every real placement checked -- twenty Odonata frames
    and four QA-62 cards -- reads +1.00.

    It is the gate identification already applies before naming a chart, at the
    same floor, applied here to a placement a batch is about to measure. It only
    ever refuses on evidence: a chart declaring no ramp, or a ramp that cannot be
    read, is measured exactly as before.
    """
    from deltachrome.imaging.detect import RAMP_AGREEMENT_FLOOR, ramp_agreement

    try:
        agreement = ramp_agreement(image, target_def, corners)
    except Exception:  # noqa: BLE001 -- no evidence is not evidence of a fault
        return None
    if agreement is None or agreement >= RAMP_AGREEMENT_FLOOR:
        return None
    return (
        f"The placement found is not on the chart: its grey ramp reads out of "
        f"order there ({agreement:+.2f})."
    )


def _carried_over(path: Path, record_path: Path | None) -> BatchCapture:
    """A capture a previous run of this batch already measured.

    **Its readings are read back rather than left blank.** A resumed run summarises
    the whole folder, not the part of it this invocation happened to measure, and a
    summary that reported twenty measured and none failed because nineteen of them
    came from disk would be wrong in the direction that matters. Everything needed
    is in the record: that is what the raw-data-first rule is for.
    """
    carried = BatchCapture(
        path=path, outcome="measured", record_path=record_path,
        reason="carried over from an earlier run of this batch",
    )
    if record_path is None or not record_path.is_file():
        return carried
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The index says this was measured and the record is gone or unreadable.
        # Worth carrying as measured -- re-measuring would contradict the index --
        # and not worth inventing readings for.
        return carried
    charts = record.get("charts")
    chart = charts[0] if isinstance(charts, list) and charts else record
    carried.overall = str(chart.get("overall") or record.get("overall") or "") or None
    carried.conformance_claim = bool(
        chart.get("conformance_claim", record.get("conformance_claim", False))
    )
    carried.warnings = [str(w) for w in (chart.get("warnings") or [])]
    carried.results = [
        MetricResult.from_record(entry) for entry in (chart.get("metrics") or [])
    ]
    carried.reference_basis = str(
        (chart.get("reference") or record.get("reference") or {}).get("basis")
        or "per_unit_measurement"
    )
    return carried


def _first_frame_notice(image, target_def, corners) -> str | None:
    r"""Whether the **first** capture held charts a batch will not measure.

    Without it a folder of boards is measured one card a frame in silence. The search runs **once**, on the first
    capture, and what it finds is reported for the batch rather than for that
    capture: one search against a whole run, where asking every capture costs a
    second search a frame -- measured at 7.9 s on the QA-62 board's JPEG 2000
    master, against a 10 s decode. That per-capture form is deferred.

    **It catches the common case exactly and says so.** A batch is a folder shot
    the same way throughout, so a board in the first frame means boards in all of
    them. A folder where only the forty-seventh capture holds a second chart is
    missed, and the notice says it looked at the first capture only rather than
    implying it has looked everywhere.

    Failures are swallowed: this is advisory, it runs beside a capture that has
    already been measured, and a run must not end because the notice could not be
    composed.
    """
    from deltachrome.imaging.detect import extra_charts as find_extra_charts

    try:
        missed = find_extra_charts(image, target_def, corners)
    except Exception:  # noqa: BLE001 -- advisory; never costs the run
        return None
    if not missed:
        return None
    return (
        f"The first capture held {len(missed) + 1} charts and a batch measures "
        "one in each -- the one its target and placement name. The others were "
        "not measured. Run the captures holding several charts on their own, "
        "where all of them are measured together. This looked at the first "
        "capture only."
    )


def _write_index(folder: Path, result: BatchResult) -> Path:
    path = folder / INDEX_NAME
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
