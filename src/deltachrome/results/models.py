"""Persisted result model.

Raw per-patch measurements are stored, not just aggregates. That is what makes it
possible to re-score an old capture against a revised profile, or to build a
time-series for a device, without going back and re-shooting the target.

The serialised form also records *how* the numbers were produced -- ICC profile,
white point, ROI inset, percentile method, reference file provenance -- because a
Delta E without those is not reproducible.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from deltachrome import __version__
from deltachrome.results.edges import EdgeRecord, RegistrationRecord

#: 4 added ``conformance_caveats`` and ``image.fingerprint``.
#: 5 spelled the record's keys in American English. Readers accept 3 and 4 --
#: see :data:`RENAMED_KEYS`.
#: 6 added ``image.color_records``. A run has no pixels once it is reopened, so
#: without it a report could show colour metrics as incomplete and be unable to say
#: that the capture was grayscale. Older records simply omit the key; nothing reads
#: it as a number of records they did not state.
#: 7 added ``metrics[].value_text`` -- the three file-level metrics read as a
#: word or a whole number rather than as three decimals, and the record now says
#: so rather than leaving each renderer to work it out from the metric's id.
#: 8 added ``original_size``, ``percentile_method`` and ``frequency_units`` --
#: the run options a profile's own algorithm can carry. They were read into
#: ``context.extras`` at analysis time and never written back out, so a
#: reopened run could not be judged again under the same conventions it was
#: judged under the first time -- and ``command_for_record`` had already been
#: reading them from the record's top level, so a saved run's Copy Command
#: silently dropped ``--original-size`` on every banded profile. Older records
#: simply omit the keys, which reads back as "no override was recorded".
#: 9 added ``metrics[].reading_units`` -- what one per-sample reading is in, where
#: that is not the aggregate's unit. Four metrics differ, and a readings table had
#: been heading their rows with the aggregate's unit. Older records are filled in
#: on read from :data:`LEGACY_READING_UNITS`.
SCHEMA_VERSION = 9


@dataclass
class PatchRecord:
    """Everything measured for a single patch, before any metric was applied."""

    patch_id: str
    row: int
    col: int
    role: str
    ramp_position: int | None
    rgb_mean: list[float]
    rgb_median: list[float]
    rgb_std: list[float]
    measured_lab: list[float] | None
    reference_lab: list[float] | None
    pixel_count: int
    clipped_high_fraction: float
    clipped_low_fraction: float
    flags: list[str] = field(default_factory=list)
    #: The four corners, in image pixels, of the region this patch was sampled
    #: from -- the inset quadrilateral, not the nominal patch. Under perspective
    #: it is a quadrilateral rather than a rectangle, and the inset is a
    #: parameter, so drawing the nominal patch instead would outline an area
    #: larger than the one that was measured.
    #:
    #: Kept here rather than on every ``PatchValue`` because the metrics repeat
    #: their per-patch lists: a hundred and forty patches across six metrics
    #: would store the same four corners six times. Metrics carry the patch id,
    #: which is the join.
    roi_px: list[list[float]] | None = None


@dataclass
class AnalysisRun:
    """One complete analysis, serialisable to JSON."""

    run_id: str
    created_utc: str
    image: dict[str, Any]
    target: dict[str, Any]
    reference: dict[str, Any]
    profile: dict[str, Any]
    geometry: dict[str, Any]
    sampling: dict[str, Any]
    color_management: dict[str, Any]
    patches: list[PatchRecord]
    #: Every slanted edge measured, whatever the verdict. Orientation and field
    #: position are recorded because the differences between edges, not the
    #: aggregate, are what point at astigmatism, field curvature or tilt.
    edges: list[Any]
    edge_field_summary: dict[str, Any]
    #: Where each color record put each edge, and the displacement between them.
    #: Kept for the same reason, and independently of whether the profile declares
    #: a registration criterion at all.
    registration: list[Any]
    registration_summary: dict[str, Any]
    #: The chart's own printed scales, read from this capture. The sampling rate the
    #: report quotes comes from here rather than from the file header, so the
    #: measurement that produced it travels with the result.
    rulers: dict[str, Any]
    metrics: list[dict[str, Any]]
    overall: str
    overall_explanation: str
    #: False when the run was scored against the manufacturer's nominal values rather
    #: than a measurement of the chart itself. The verdict is then advisory: the
    #: specimen's own print tolerance is inside every difference. Stored beside the
    #: verdict so a consumer reading ``overall`` alone cannot miss it.
    conformance_claim: bool = True
    #: Every reason this run cannot be quoted as conformance, in the words the
    #: evaluator formed them in.
    #:
    #: Persisted rather than re-derived, and that is not redundancy with
    #: ``conformance_claim``. One of the two reasons is that the profile was
    #: *derived* from a standard, which is a property of the profile object and
    #: was never written into the record -- so a reopened run could show a
    #: qualified verdict and be unable to say why. A caveat that survives the
    #: round trip is the whole point of qualifying a verdict at all.
    conformance_caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    #: Run options that belong to the run rather than to the profile -- what the
    #: operator (or ``analyze``'s own caller) overrode when this run was made.
    #: ``None`` means the profile's own default applied. Persisted so a run can
    #: be re-established from its own record under the conventions it was
    #: originally judged under, rather than silently reverting to defaults --
    #: see :func:`deltachrome.pipeline.analyze`'s parameters of the same names.
    original_size: str | None = None
    percentile_method: str | None = None
    frequency_units: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisRun":
        """Rebuild a run from its serialised form.

        Deliberately forgiving in one direction only. A field the file does not
        carry takes its default, and a field the file carries that this version
        does not know is dropped -- a record written by a newer or older engine
        opens rather than raising, which matters because these files are the
        point: they are what re-scoring an old capture against a revised profile
        works from, and a library that refuses to list its own history is worse
        than one that lists it with a gap.

        What is *not* forgiving is the verdict. ``overall`` and
        ``conformance_caveats`` are read straight through, never inferred, so
        nothing here can turn a qualified result into a clean one.
        """
        fields = {f.name for f in dataclass_fields(cls)}
        known = {k: v for k, v in dict(data).items() if k in fields}
        known["patches"] = [
            _record_from(PatchRecord, p) for p in (data.get("patches") or [])
        ]
        known["edges"] = [
            _record_from(EdgeRecord, e) for e in (data.get("edges") or [])
        ]
        known["registration"] = [
            _record_from(RegistrationRecord, r) for r in (data.get("registration") or [])
        ]
        for required in ("run_id", "created_utc"):
            known.setdefault(required, "")
        for required in ("image", "target", "reference", "profile", "geometry",
                         "sampling", "color_management", "edge_field_summary",
                         "registration_summary", "rulers"):
            known.setdefault(required, {})
        known.setdefault("metrics", [])
        known["metrics"] = [_with_reading_units(m) for m in known["metrics"]]
        known.setdefault("overall", "UNKNOWN")
        known.setdefault("overall_explanation", "")
        return cls(**known)

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisRun":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path


#: Schema of a capture record -- the file a saved analysis actually is now.
#:
#: Versioned apart from :data:`SCHEMA_VERSION` because the two change for different
#: reasons: a run's schema moves when what is measured about ONE CHART changes, and
#: this moves when what is held about a CAPTURE does. The run records nested inside
#: are ordinary schema-7 records, unaltered, which is the whole design -- see
#: :class:`CaptureRecord`.
CAPTURE_SCHEMA_VERSION = 1


@dataclass
class CaptureRecord:
    """One capture, and every chart read from it.

    **This reverses a decision, and the reversal is the operator's.** A run was one
    capture, one target, one placement, one verdict, and a frame holding five charts
    was five records -- which is the right shape for a *measurement* and the wrong
    shape for the *thing that was done*. What was done is one capture of one board on
    one system at one moment, and five rows in a library is five things to find again
    and keep together by memory. A department shooting a few boards a day accumulates
    thousands of rows a year, and nothing in a row says which four others belong with
    it.

    **What is not reversed is the run.** Every chart's record inside this is an
    ordinary :class:`AnalysisRun`, serialised exactly as it always was, schema 7 and
    unaltered -- so re-scoring takes one, the report renders one, and
    ``session_from_records`` combines them, all through the code that already exists.
    The container is additive: it holds them, it does not change them.

    Reading is forgiving in both directions. :meth:`from_dict` given a bare run record
    -- a file written before this existed -- returns a capture holding that one chart,
    because a year of saved runs must keep opening. Given a capture record it returns
    the charts. Nothing has to know which shape a file on disk is until it looks.
    """

    capture_id: str
    created_utc: str
    #: The file every chart was read from. One image, so one entry, rather than the
    #: same dictionary repeated in each chart's record.
    image: dict[str, Any]
    #: The profile every chart was judged against. One profile, for the same reason a
    #: session refuses to combine runs judged against several: pooling readings scored
    #: against different thresholds gives a verdict belonging to no standard.
    profile: dict[str, Any]
    charts: list[AnalysisRun]
    #: The combined verdict, formed the way a session forms one -- absence never
    #: fails, and the worst reading governs where two charts measure the same thing.
    overall: str
    overall_explanation: str = ""
    conformance_claim: bool = True
    conformance_caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CAPTURE_SCHEMA_VERSION

    @property
    def target_summary(self) -> str:
        """What to call this capture's contents in one cell of a table.

        The models it holds, not the count: "4 x Applied Image QA-62" says what was
        shot, where "5 charts" says only that something was.
        """
        names: dict[str, int] = {}
        for chart in self.charts:
            name = str((chart.target or {}).get("display_name")
                       or (chart.target or {}).get("target_id") or "unnamed")
            names[name] = names.get(name, 0) + 1
        if not names:
            return "—"
        return " + ".join(
            name if count == 1 else f"{count} x {name}" for name, count in names.items()
        )

    #: Reference bases, weakest first. A capture holding charts scored against
    #: different ones reports the weakest, because that is what governs it.
    _BASIS_ORDER = ("manufacturer_nominal", "population_mean", "per_unit_measurement")

    def _reference_summary(self) -> dict[str, Any]:
        """What the charts were scored against, as one run reader can read it."""
        if not self.charts:
            return {}
        if len(self.charts) == 1:
            return dict(self.charts[0].reference or {})
        bases = [(chart.reference or {}).get("basis") for chart in self.charts]
        known = [b for b in bases if b in self._BASIS_ORDER]
        if len(known) != len(bases):
            # One chart names a basis this version does not know. Saying nothing is
            # better than ranking it, and the claim below already withholds.
            return {}
        return {"basis": min(known, key=self._BASIS_ORDER.index)}

    def to_dict(self) -> dict[str, Any]:
        """The record, with the keys a run reader looks for carried at the top level.

        ``run_id``, ``target``, ``geometry`` and ``reference`` are emitted alongside
        the capture's own fields so that everything which already knows how to
        summarise a run keeps working on one of these unchanged -- the library's
        table especially. They are a view of the charts below, never a second copy
        of them.

        ``reference`` was missing and the library's Basis column read *Unstated* on
        every capture record because of it -- a run scored against nominal values
        looked like a run that declined to say what it was scored against. Where
        several charts disagree it carries the **weakest** of their bases, which is
        the same rule ``conformance_claim`` already applies one level up: a capture
        is quotable only if all of it is.
        """
        data = _jsonify({
            "schema_version": self.schema_version,
            "capture_id": self.capture_id,
            "run_id": self.capture_id,
            "created_utc": self.created_utc,
            "image": self.image,
            "profile": self.profile,
            "target": {"display_name": self.target_summary,
                       "target_id": (self.charts[0].target or {}).get("target_id")
                       if len(self.charts) == 1 else None},
            "geometry": self.charts[0].geometry if self.charts else {},
            "reference": self._reference_summary(),
            "overall": self.overall,
            "overall_explanation": self.overall_explanation,
            "conformance_claim": self.conformance_claim,
            "conformance_caveats": self.conformance_caveats,
            "warnings": self.warnings,
            "environment": self.environment,
        })
        data["charts"] = [chart.to_dict() for chart in self.charts]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaptureRecord":
        """A capture record, or a bare run record read as a capture holding one chart."""
        raw = dict(data)
        if "charts" not in raw:
            run = AnalysisRun.from_dict(raw)
            return cls(
                capture_id=run.run_id, created_utc=run.created_utc, image=run.image,
                profile=run.profile, charts=[run], overall=run.overall,
                overall_explanation=run.overall_explanation,
                conformance_claim=run.conformance_claim,
                conformance_caveats=list(run.conformance_caveats),
                warnings=list(run.warnings), environment=dict(run.environment),
            )
        charts = [AnalysisRun.from_dict(c) for c in (raw.get("charts") or [])]
        return cls(
            capture_id=str(raw.get("capture_id") or raw.get("run_id") or ""),
            created_utc=str(raw.get("created_utc") or ""),
            image=raw.get("image") or {},
            profile=raw.get("profile") or {},
            charts=charts,
            overall=str(raw.get("overall") or "UNKNOWN"),
            overall_explanation=str(raw.get("overall_explanation") or ""),
            conformance_claim=bool(raw.get("conformance_claim", True)),
            conformance_caveats=list(raw.get("conformance_caveats") or []),
            warnings=list(raw.get("warnings") or []),
            environment=raw.get("environment") or {},
            schema_version=int(raw.get("schema_version") or CAPTURE_SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CaptureRecord":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


#: Keys that changed spelling at schema 5, old name to new.
#:
#: A record is the archive, and the archive was written before the vocabulary was
#: settled. Renaming a field in the code cannot be allowed to make a year of
#: saved runs unreadable -- and the failure would not have been a graceful one:
#: the unknown key is dropped, the required argument is then missing, and the
#: record raises a TypeError naming a field the file does not contain.
RENAMED_KEYS = {
    "centre_px": "center_px",
    "centre_relative": "center_relative",
    "linearised": "linearized",
}


#: The per-reading unit of each metric whose readings are not in its aggregate's
#: unit, for records written before schema 9 stated it. Keyed on the metric id,
#: which is exactly what a record should never need -- and is acceptable here only
#: because it is a migration: it describes what four plugins wrote before the
#: record could say so, and it never grows. Every record since states its own.
LEGACY_READING_UNITS = {
    "lightness_uniformity": "dL* from family mean",
    "geometric_distortion": "ppi",
    "illumination_non_uniformity": "L*",
    "white_balance_non_uniformity": "L*",
}


def _with_reading_units(metric: Any) -> Any:
    """One metric entry, with ``reading_units`` filled in if it predates the key."""
    if not isinstance(metric, Mapping) or "reading_units" in metric:
        return metric
    entry = dict(metric)
    entry["reading_units"] = LEGACY_READING_UNITS.get(str(entry.get("metric_id", "")))
    return entry


def _record_from(cls: type, data: Any) -> Any:
    """One nested record, with unknown keys dropped and missing ones defaulted."""
    if not isinstance(data, Mapping):
        return data
    names = {f.name for f in dataclass_fields(cls)}
    values = {}
    for key, value in data.items():
        key = RENAMED_KEYS.get(key, key)
        if key in names:
            values[key] = value
    return cls(**values)


#: How much of a file's head and tail the fingerprint covers. A preservation
#: master runs to hundreds of megabytes and a library lists dozens of runs, so
#: hashing whole files to notice that one moved is not affordable. Size plus both
#: ends distinguishes captures from the same session -- which differ in their
#: pixel data throughout, not merely somewhere -- while reading two megabytes.
FINGERPRINT_CHUNK = 1 << 20


def file_fingerprint(path: str | Path) -> dict[str, Any] | None:
    """Enough to recognize a capture that has been moved or renamed.

    Not a checksum, and not offered as one: it would not detect a deliberate
    edit in the middle of a large file. It exists so a run whose image has moved
    can be offered a candidate rather than only an error, in the way a
    photographic catalogue reconnects missing originals.
    """
    try:
        path = Path(path)
        size = path.stat().st_size
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(size).encode("ascii"))
        with path.open("rb") as handle:
            digest.update(handle.read(FINGERPRINT_CHUNK))
            if size > FINGERPRINT_CHUNK * 2:
                handle.seek(-FINGERPRINT_CHUNK, 2)
                digest.update(handle.read(FINGERPRINT_CHUNK))
    except OSError:
        return None
    return {
        "algorithm": f"blake2b-128/size+{FINGERPRINT_CHUNK}B-ends",
        "size_bytes": size,
        "digest": digest.hexdigest(),
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_info() -> dict[str, Any]:
    return {
        "deltachrome_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars/arrays and dataclasses into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _finite(float(value))
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonify(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonify(asdict(value))
    return value


def _finite(x: float) -> float | None:
    """JSON has no NaN. A missing measurement serialises as null, not as 0.0."""
    return None if not np.isfinite(x) else x
