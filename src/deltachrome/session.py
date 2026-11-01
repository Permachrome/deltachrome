"""Several captures judged together against one profile.

Almost no chart measures everything a standard asks for. A ColorChecker SG has no
slanted edges and no printed scale; an ISA transmissive target has no color
patches at all. So sites routinely shoot two or three targets in one sitting and
read the results together -- the SG for color, something with edges for
resolution -- and the question they are actually asking is whether *the system*
meets the specification, not whether any one chart did.

A session is that question. It runs each capture exactly as a single analysis would
and then combines them, which means every rule below is about the combining and
nothing beneath it changes.

**Overlap.** Where two captures measure the same metric, the worst reading drives
the verdict and both are reported. Same reasoning as reducing a field of edges: a
mean can pass a system that one target says is failing, and the disagreement
between two charts is itself worth seeing -- it is the only signal available that
one of them is misplaced, mis-referenced, or not what it claims to be.

**Coverage.** A metric no capture could measure is named as a gap rather than
quietly dropped. This is the one thing a session can say that a single run cannot:
a lone chart's blind spots are inherent, while a session's are a choice about which
charts were shot, and could be closed by adding one.

**The verdict follows the single-run rule**, deliberately. Absence never fails --
not for a target, not for a session -- so a session passes when everything it
measured passed. Whether that pass covered the whole profile is a separate
question, answered separately, in ``covers_profile``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from deltachrome.errors import ProfileError
from deltachrome.metrics.base import MetricResult
from deltachrome.pipeline import AnalysisResult, analyze
from deltachrome.profiles.schema import AnalysisProfile, load_profile
from deltachrome.reference.cgats import PER_UNIT_MEASUREMENT
from deltachrome.results.models import environment_info, now_utc
from deltachrome.text import plural

#: Metrics that describe the *file* rather than the imaging system.
#:
#: The distinction matters only in a session. Two charts shot on one system should
#: agree about that system's color accuracy or its sharpness, and where they do not
#: something is wrong with one of them. They should *not* be expected to agree about
#: bit depth or sampling rate: those are properties of two different files, and two
#: files legitimately differ -- a reflective chart is shot at the object plane while a
#: transmissive one is shot at high magnification, so their sampling rates differ by
#: design and neither is in error.
#:
#: So these are judged per capture, each on its own, and never reported as a
#: disagreement. Any failure still carries into the session verdict: a file that
#: misses the specification misses it whichever chart it holds.
PER_CAPTURE_METRICS = frozenset(
    {
        "master_file_format",
        "bit_depth",
        "color_space",
        "sampling_frequency",
        "reproduction_scale_accuracy",
    }
)


@dataclass
class SessionCapture:
    """One image and everything needed to analyze it.

    Each carries its own target, corner placement and reference, because that is
    the point: the captures differ in what they can measure.

    ``image`` may be an already-decoded
    :class:`~deltachrome.imaging.loader.LoadedImage` as well as a path, and that is
    what lets **one frame holding several charts** be a session: each chart is an
    ordinary capture with its own placement, and they share the decode rather than
    paying for it four times. On the QA-62 board that is 10 seconds against 40.
    """

    image: str | Path | Any
    target: str | Path
    corners: Sequence[tuple[float, float]]
    reference: str | Path | None = None
    reference_table: int | str | None = None
    fiducial_extent_mm: tuple[float, float] | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        source = getattr(self.image, "path", self.image)
        return Path(source).name


@dataclass
class CaptureOutcome:
    """One capture's contribution to a session.

    A narrow view, and deliberately narrower than
    :class:`~deltachrome.pipeline.AnalysisResult`. Combining captures needs each
    one's metric results and a handful of labels; it does not need pixels, a
    placement, a sampler or a reference table. Saying so in the type is what lets a
    session be assembled two ways from one code path -- from runs just analyzed, and
    from **records read back off disk**, which is how a site actually works: shoot
    the charts, analyze each as it comes, then ask at the end whether the system
    passed. The alternative was re-running every capture to answer a question whose
    inputs are already saved, which on three masters is a minute of decoding for
    numbers that have not changed.
    """

    name: str
    results: list[MetricResult]
    conformance_claim: bool
    target_id: str
    target_name: str
    profile_id: str
    run_id: str
    overall: str
    reference_basis: str = "per_unit_measurement"
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_run(cls, name: str, run: Any) -> "CaptureOutcome":
        """From a live :class:`~deltachrome.pipeline.AnalysisResult`."""
        return cls(
            name=name,
            results=list(run.evaluation.results),
            conformance_claim=bool(run.conformance_claim),
            target_id=run.target.target_id,
            target_name=run.target.display_name,
            profile_id=run.run.profile.get("profile_id", ""),
            run_id=run.run.run_id,
            overall=run.overall,
            reference_basis=run.run.reference.get("basis", "per_unit_measurement"),
            warnings=list(run.warnings),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any], name: str | None = None) -> "CaptureOutcome":
        """From a saved run record.

        Everything read here was written by the pipeline and nothing is inferred.
        The record's own ``conformance_claim`` is honoured rather than re-derived:
        a saved run has neither its profile object nor its reference file to hand,
        and a session that quietly re-derived the claim as True would upgrade an
        advisory run by reading it back.
        """
        image = record.get("image") or {}
        target = record.get("target") or {}
        profile = record.get("profile") or {}
        return cls(
            name=name or str(image.get("filename") or record.get("run_id") or "capture"),
            results=[MetricResult.from_record(m) for m in (record.get("metrics") or [])],
            conformance_claim=bool(record.get("conformance_claim", True)),
            target_id=str(target.get("target_id") or ""),
            target_name=str(target.get("display_name") or target.get("target_id") or ""),
            profile_id=str(profile.get("profile_id") or ""),
            run_id=str(record.get("run_id") or ""),
            overall=str(record.get("overall") or "INCOMPLETE"),
            reference_basis=str(
                (record.get("reference") or {}).get("basis", "per_unit_measurement")
            ),
            warnings=list(record.get("warnings") or []),
        )


@dataclass
class MetricAcrossCaptures:
    """One metric's readings from every capture that could take it."""

    metric_id: str
    readings: list[tuple[str, MetricResult]] = field(default_factory=list)

    @property
    def measured(self) -> list[tuple[str, MetricResult]]:
        return [(name, r) for name, r in self.readings if r.outcome == "evaluated"]

    @property
    def covered(self) -> bool:
        return bool(self.measured)

    @property
    def failures(self) -> list[tuple[str, MetricResult]]:
        return [(name, r) for name, r in self.measured if r.passed is False]

    @property
    def governing(self) -> tuple[str, MetricResult] | None:
        """The reading the session's verdict rests on: any failure first, else the
        first pass. A metric two charts disagree about is failing on one of them,
        and the session should say so rather than average the disagreement away."""
        if self.failures:
            return self.failures[0]
        return self.measured[0] if self.measured else None

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.measured:
            return "PASS" if any(r.passed for _, r in self.measured) else "MEASURED"
        outcomes = {r.outcome for _, r in self.readings}
        if outcomes & {"incomplete", "not_implemented"}:
            return "INCOMPLETE"
        return "N/A"

    def summary(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "covered_by": [name for name, _ in self.measured],
            "not_measurable_by": [
                name for name, r in self.readings if r.outcome == "not_applicable"
            ],
            "values": {
                name: r.value for name, r in self.measured if r.value is not None
            },
            "disagreement": self.disagreement,
            "per_capture": self.per_capture,
        }

    @property
    def per_capture(self) -> bool:
        """Whether this metric describes each file rather than the system."""
        return self.metric_id in PER_CAPTURE_METRICS

    @property
    def disagreement(self) -> float | None:
        """Spread between captures that both measured this, or None.

        Reported because two charts measuring one system should agree, and when they
        do not the interesting question is which is wrong -- not what their average
        is. Withheld for per-capture metrics, where a difference between two files is
        expected and calling it a disagreement would bury the real ones in noise.
        """
        if self.per_capture:
            return None
        values = [r.value for _, r in self.measured if r.value is not None]
        return None if len(values) < 2 else float(max(values) - min(values))


@dataclass
class SessionResult:
    profile: AnalysisProfile
    captures: list[CaptureOutcome]
    created_utc: str
    warnings: list[str] = field(default_factory=list)

    @property
    def metrics(self) -> list[MetricAcrossCaptures]:
        combined: dict[str, MetricAcrossCaptures] = {
            spec.metric_id: MetricAcrossCaptures(spec.metric_id)
            for spec in self.profile.metrics
        }
        for capture in self.captures:
            for result in capture.results:
                combined.setdefault(
                    result.metric_id, MetricAcrossCaptures(result.metric_id)
                ).readings.append((capture.name, result))
        return list(combined.values())

    @property
    def failures(self) -> list[MetricAcrossCaptures]:
        return [m for m in self.metrics if m.failures]

    @property
    def uncovered(self) -> list[str]:
        """Metrics no capture in this session produced a value for."""
        return sorted(m.metric_id for m in self.metrics if not m.covered)

    @property
    def not_carried(self) -> list[str]:
        """Metrics no capture in this session *carries the feature for*.

        The gap a session can genuinely close and a single run cannot: shoot
        another chart. Split out of :attr:`uncovered`, which pools it with metrics a
        chart did carry and the measurement failed on -- and telling an operator to
        buy a target when the placement was wrong is the mistake that separation
        exists to prevent.
        """
        incomplete = set(self.incomplete)
        return [m for m in self.uncovered if m not in incomplete]

    @property
    def incomplete(self) -> list[str]:
        return sorted(m.metric_id for m in self.metrics if m.status == "INCOMPLETE")

    @property
    def covers_profile(self) -> bool:
        return not self.uncovered and not self.incomplete

    @property
    def conformance_claim(self) -> bool:
        return all(capture.conformance_claim for capture in self.captures)

    @property
    def evidence_is_sound(self) -> bool:
        """Whether every capture's numbers describe the chart that was photographed.

        Separate from the claim above for the reason stated in
        :attr:`~deltachrome.profiles.evaluator.ProfileEvaluation.evidence_is_sound`:
        a session judged against a department's own written specification met that
        specification, and only its reference can make the reading itself doubtful.
        """
        return all(
            capture.reference_basis == PER_UNIT_MEASUREMENT
            for capture in self.captures
        )

    @property
    def overall(self) -> str:
        """Follows the single-run rule: absence never fails.

        Whether the session reached the whole profile is a different question, and
        conflating them would penalise a site for the charts it happens to own.
        """
        if self.failures:
            return "FAIL"
        if self.incomplete:
            return "INCOMPLETE"
        measured = [m for m in self.metrics if m.covered]
        if not measured:
            return "INCOMPLETE"
        if not any(r.passed is not None for m in measured for _, r in m.measured):
            return "MEASURED"
        return "PASS"

    @property
    def coverage_line(self) -> str:
        metrics = self.metrics
        covered = sum(1 for m in metrics if m.covered)
        text = (f"{covered} of {plural(len(metrics), 'metric')} measured by "
                f"{plural(len(self.captures), 'capture')}")
        if self.uncovered:
            text += f"; {len(self.uncovered)} reached by none of them"
        return text

    @property
    def explanation(self) -> str:
        if self.failures:
            names = ", ".join(m.metric_id for m in self.failures)
            text = f"{plural(len(self.failures), 'metric')} failed: {names}."
        elif self.incomplete:
            text = (
                f"{plural(len(self.incomplete), 'metric')} had too little usable "
                "data: "
                f"{', '.join(self.incomplete)}."
            )
        else:
            text = "Every metric any capture measured met its threshold."
        if self.not_carried:
            text += (
                f" {plural(len(self.not_carried), 'metric')} "
                f"{'was' if len(self.not_carried) == 1 else 'were'} measured by no "
                f"capture in this session ({', '.join(self.not_carried)}); "
                "nothing failed on "
                "them, and adding a target that carries the feature would complete "
                "the profile."
            )
        if not self.conformance_claim:
            text = (
                "At least one capture was judged against manufacturer nominal values "
                "rather than a measurement of its own chart, so this is advisory "
                "and not a conformance result. " + text
            )
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_utc": self.created_utc,
            "profile": {
                "profile_id": self.profile.profile_id,
                "display_name": self.profile.display_name,
                "standard_reference": self.profile.standard_reference,
            },
            "captures": [
                {"name": c.name, "run_id": c.run_id,
                 "target": c.target_id, "overall": c.overall}
                for c in self.captures
            ],
            "overall": self.overall,
            "explanation": self.explanation,
            "conformance_claim": self.conformance_claim,
            "covers_profile": self.covers_profile,
            "coverage_line": self.coverage_line,
            "uncovered": self.uncovered,
            "metrics": [m.summary() for m in self.metrics],
            "warnings": self.warnings,
            "environment": environment_info(),
        }


def analyze_session(
    captures: Sequence[SessionCapture],
    *,
    profile: str | Path | AnalysisProfile,
    **shared: Any,
) -> SessionResult:
    """Run every capture against one profile and combine them."""
    if not captures:
        raise ProfileError("A session needs at least one capture.")
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)

    runs = [
        analyze(
            capture.image,
            target=capture.target,
            reference_path=capture.reference,
            reference_table=capture.reference_table,
            profile=profile_def,
            corners=capture.corners,
            fiducial_extent_mm=capture.fiducial_extent_mm,
            **shared,
        )
        for capture in captures
    ]
    return session_from_results(
        runs, labels=[capture.name for capture in captures], profile=profile_def
    )


def session_from_results(
    runs: Sequence[AnalysisResult],
    *,
    labels: Sequence[str] | None = None,
    profile: str | Path | AnalysisProfile | None = None,
) -> SessionResult:
    """Combine runs the caller has already made and still holds.

    The third way into a session, and the one a *single capture holding several
    charts* needs. :func:`analyze_session` runs the captures itself and
    :func:`session_from_records` reads them back off disk; neither suits a caller
    that wants to report each run in full as it is made and combine them afterwards,
    which is what the command line does with ``--detect-all``.
    """
    if not runs:
        raise ProfileError("A session needs at least one capture.")
    if profile is None:
        profile = runs[0].evaluation.profile
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)

    outcomes: list[CaptureOutcome] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for index, run in enumerate(runs):
        name = (labels[index] if labels else None) or Path(run.image.path).name
        if name in seen:
            name = f"{name} ({len(seen)})"
        seen.add(name)
        outcomes.append(CaptureOutcome.from_run(name, run))
        warnings.extend(f"[{name}] {w}" for w in run.warnings)

    return SessionResult(
        profile=profile_def, captures=outcomes, created_utc=now_utc(), warnings=warnings
    )


def session_from_records(
    records: Sequence[Mapping[str, Any]], *,
    profile: str | Path | AnalysisProfile | None = None,
    labels: Sequence[str] | None = None,
) -> SessionResult:
    """Combine runs that have already been analyzed and saved.

    The same combining as :func:`analyze_session`, without re-measuring anything --
    every number a session reduces was written into each record when the run was
    made. That is what the raw-data-first rule was for: a saved run can be asked a
    question it was not saved to answer.

    **The captures must have been judged against the same profile, and this
    refuses when they were not.** Combining metrics scored against different
    thresholds produces a verdict belonging to no standard: two captures both
    reading 3.2 delta-E, one judged against 3.5 and one against 2.0, would give a
    session that passes and a session that fails depending on which chart happened
    to be listed first. It is also the mistake that is easiest to make from a
    library, where a year of runs against a dozen profiles sit in one table.

    ``labels`` names the captures, exactly as it does for
    :func:`session_from_results`, and exists for the case the default cannot
    serve: the records of one *capture* holding several targets all carry the
    same filename, so the fallback names them ``board.tif``, ``board.tif (1)``,
    ``board.tif (2)`` -- which distinguishes them and identifies none of them.
    """
    if not records:
        raise ProfileError("A session needs at least one capture.")

    outcomes: list[CaptureOutcome] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        given = labels[index] if labels and index < len(labels) else None
        outcome = CaptureOutcome.from_record(record, name=given)
        if outcome.name in seen:
            outcome.name = f"{outcome.name} ({len(seen)})"
        seen.add(outcome.name)
        outcomes.append(outcome)

    profiles = {c.profile_id for c in outcomes if c.profile_id}
    if len(profiles) > 1:
        raise ProfileError(
            "These runs were judged against different profiles ("
            + ", ".join(sorted(profiles))
            + "), so they cannot be combined. Every metric a session reduces was "
            "scored against one profile's thresholds; pooling readings judged "
            "against different ones produces a verdict belonging to no standard. "
            "Re-run the captures against a single profile, or select only the runs "
            "that already share one."
        )

    if profile is None:
        if not profiles:
            raise ProfileError(
                "None of these records names the profile it was judged against, so "
                "there is nothing to combine them under. Pass one explicitly."
            )
        profile = next(iter(profiles))
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)

    warnings = [f"[{c.name}] {w}" for c in outcomes for w in c.warnings]
    return SessionResult(
        profile=profile_def, captures=outcomes, created_utc=now_utc(), warnings=warnings
    )


def capture_record(
    runs: Sequence[AnalysisResult],
    *,
    labels: Sequence[str] | None = None,
) -> "CaptureRecord":
    """Every chart read from one frame, as the one record that gets saved.

    The verdict, the caveats and the warnings come from combining the runs exactly as
    a session does -- one code path, so a board judged here and the same runs judged
    from the library cannot disagree. What is stored is the runs themselves, whole and
    unaltered, plus that combined answer.

    The runs must share a capture and a profile, which is not a restriction so much as
    a description: they were read from one frame in one pass, and pooling readings
    scored against different thresholds gives a verdict belonging to no standard.
    """
    from deltachrome.results.models import CaptureRecord

    if not runs:
        raise ProfileError("A capture record needs at least one chart.")
    combined = session_from_results(runs, labels=labels)
    first = runs[0].run
    return CaptureRecord(
        capture_id=first.run_id,
        created_utc=first.created_utc,
        image=dict(first.image),
        profile=dict(first.profile),
        charts=[result.run for result in runs],
        overall=combined.overall,
        overall_explanation=combined.explanation,
        conformance_claim=combined.conformance_claim,
        conformance_caveats=sorted({
            caveat for result in runs for caveat in result.run.conformance_caveats
        }),
        warnings=list(combined.warnings),
        environment=dict(first.environment),
    )


def load_session_file(path: str | Path) -> tuple[list[SessionCapture], str, dict[str, Any]]:
    """Read a session description: one profile, several captures.

    A file rather than a heap of repeated flags, because a session is the thing
    worth keeping. Which charts were shot, where their corners were placed and
    which reference each used is the record that makes a result reproducible, and a
    command line is a poor place to store it.

    Returns the captures, the profile id, and any session-wide settings -- which is
    where ``original_size`` belongs: it describes the work the system is being
    qualified for, so it is one answer for the sitting rather than one per capture.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "profile" not in data or "captures" not in data:
        raise ProfileError(
            f"{path.name}: a session file needs 'profile' and 'captures'. "
            'Example: {"profile": "fadgi_3star_prints_photographs_2023", "captures": '
            '[{"image": "sg.tif", "target": "colorchecker_sg_v1", '
            '"corners": "10,10 900,10 900,700 10,700", "reference": "sg.txt"}]}'
        )

    captures = []
    for i, entry in enumerate(data["captures"], start=1):
        missing = {"image", "target", "corners"} - set(entry)
        if missing:
            raise ProfileError(
                f"{path.name}: capture {i} is missing {sorted(missing)}."
            )
        captures.append(
            SessionCapture(
                image=_relative_to(path, entry["image"]),
                target=entry["target"],
                corners=_parse_corners(entry["corners"]),
                reference=(
                    _relative_to(path, entry["reference"]) if entry.get("reference") else None
                ),
                reference_table=entry.get("reference_table"),
                fiducial_extent_mm=(
                    tuple(entry["fiducial_extent_mm"])
                    if entry.get("fiducial_extent_mm") else None
                ),
                label=entry.get("label"),
            )
        )
    settings = {
        key: data[key] for key in ("original_size", "frequency_units",
                                   "percentile_method", "roi_inset")
        if data.get(key) is not None
    }
    return captures, data["profile"], settings


def _relative_to(session_path: Path, value: str) -> Path:
    """Paths in a session file resolve against the file, not the shell's cwd, so a
    session stays valid when it is re-run from somewhere else."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (session_path.parent / candidate)


def _parse_corners(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        points = [p for p in value.replace(";", " ").split() if p]
        parsed = []
        for point in points:
            x, _, y = point.partition(",")
            parsed.append((float(x), float(y)))
    else:
        parsed = [(float(x), float(y)) for x, y in value]
    if len(parsed) != 4:
        raise ProfileError(
            f"Expected four corners as x,y in TL TR BR BL order; got {len(parsed)}."
        )
    return parsed
