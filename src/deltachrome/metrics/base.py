"""Metric plugin infrastructure.

A metric plugin turns measured-vs-reference patch data into one aggregate number
plus a pass/fail. Standards become data-driven profiles on top of this, rather
than branches inside the engine.

The outcome vocabulary matters. A metric can end up in one of four states, and
collapsing any two of them would let a report overclaim:

``evaluated``       ran, produced a number, passed or failed
``not_applicable``  the target cannot measure this (an SG has no slanted edges)
``not_implemented`` the target could measure it, but Deltachrome has no plugin yet
``incomplete``      the plugin ran but had too little usable data to trust
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from deltachrome.errors import ProfileError
from deltachrome.profiles.bands import BandedCondition
from deltachrome.targets.schema import Patch, TargetDefinition

Outcome = Literal["evaluated", "not_applicable", "not_implemented", "incomplete"]

#: Percentile interpolation. FADGI's documentation does not specify one, so this is
#: an explicit, reported choice rather than a silent default. "linear" matches
#: numpy's and Excel's PERCENTILE; "nearest_rank" is the classical definition and
#: is never more lenient than an actual measured value.
PERCENTILE_METHODS = {
    "linear": "linear",
    "nearest_rank": "higher",
    "higher": "higher",
    "lower": "lower",
    "midpoint": "midpoint",
}
DEFAULT_PERCENTILE_METHOD = "linear"


#: Roles whose patches mean something without a reference value.
#:
#: Almost every metric here is a comparison against a measurement of the chart, and
#: a patch with no reference cannot take part in one. Lightness uniformity is the
#: exception: it is the spread of a set of nominally identical neutrals *about their
#: own mean*, so the set is its own reference and no external value is needed. That
#: matters for real charts -- the ISA Golden Thread prints twenty registration
#: squares spread over the whole frame, an areal-coverage uniformity sample better
#: than either other target offers, and prints no values for any of them.
#:
#: Kept deliberately narrow. Carrying reference-less patches into every metric would
#: also disarm a real safety check: a reference file for the wrong model of chart
#: currently leaves the patch data empty and the run refuses, and it can only do
#: that because a patch without a reference is normally dropped.
ROLES_WITHOUT_REFERENCE = frozenset({"achromatic_border"})


#: Capabilities that need more than one colour record **in the capture**, whatever
#: the target can do.
#:
#: A grayscale master is not a colour master with the chroma at zero; it is a file
#: carrying one record where these three metrics need three. The distinction is
#: load-bearing because the failure is silent and flattering: white balance is
#: Delta E(a*b*) against a neutral aim, and a grayscale capture returns exactly 0.00
#: on every patch -- a perfect score for a measurement that never happened. Colour
#: registration is the same shape of lie, reporting no displacement between records
#: that do not exist.
#:
#: So these are reported **incomplete**, not N/A. N/A means the target cannot carry
#: the measurement and is not a defect in the capture; here the target may carry it
#: perfectly well and the capture threw the information away. Incomplete keeps a
#: run off PASS, which is the only safe answer.
CAPTURE_COLOR_CAPABILITIES = frozenset(
    {"color_accuracy", "white_balance", "color_registration"}
)


@dataclass
class PatchDatum:
    """One patch's measured and reference color, ready for comparison.

    ``reference_lab`` is ``None`` for a patch carried without one -- see
    :data:`ROLES_WITHOUT_REFERENCE`. Metrics that compare against a reference must
    filter on :attr:`has_reference` rather than assume it is present.
    """

    patch: Patch
    measured_lab: np.ndarray
    reference_lab: np.ndarray | None
    flags: list[str] = field(default_factory=list)
    #: Center of the patch in target-relative coordinates. Resolved once by the
    #: pipeline rather than by each metric, so that every metric reports the same
    #: place for the same patch.
    position_relative: tuple[float, float] | None = None
    #: The reference file's own declared density for this patch, where it carries
    #: one -- a transmissive density ladder's ``D`` column, read generically off
    #: ``PatchMeasurement.raw`` rather than parsed specially. ``None`` for every
    #: reflective target and for a ramp measured against a file with no density
    #: column; :func:`deltachrome.metrics.tolerance.dynamic_range` is the one
    #: reader.
    reference_density: float | None = None

    @property
    def has_reference(self) -> bool:
        return self.reference_lab is not None

    @property
    def patch_id(self) -> str:
        return self.patch.patch_id

    @property
    def role(self) -> str:
        return self.patch.role


@dataclass
class MetricContext:
    """Everything a metric plugin is allowed to see."""

    target: TargetDefinition
    data: list[PatchDatum]
    #: Colour records the capture carries -- 3 for RGB, 1 for grayscale. A property
    #: of the file rather than of the target, and the reason it sits here rather
    #: than being fished out of ``extras`` is that the profile evaluator has to
    #: consult it before a metric runs: see :data:`CAPTURE_COLOR_CAPABILITIES`.
    capture_records: int = 3
    reference_provenance: dict[str, str] = field(default_factory=dict)
    #: ``per_unit_measurement`` or ``manufacturer_nominal``. Travels with the data
    #: rather than beside it, so a verdict cannot be formed without it being in scope.
    reference_basis: str = "per_unit_measurement"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchValue:
    """A per-sample metric value. Persisted raw, not just the aggregate.

    Named for the patch because that is what it began as, and most of what flows
    through it still is one. It also carries the samples that are not patches at
    all -- a flat-field spot, a printed rule, a position at which color records
    were differenced -- which is why ``measured_lab`` is meaningless for some of
    them and why the type does not insist on it.

    ``position_relative`` is where on the target the sample was taken, when the
    sample has a place. Recording it is what separates a metric that can be drawn
    as a field from one that can only be listed: a number's *position* is the
    difference between "the illumination is uneven" and "the left side is dark".
    It stays ``None`` where a value has no single place -- a gain-modulation
    reading spans a pair of patches, and putting it at either one would invent a
    location the measurement does not have.
    """

    patch_id: str
    role: str
    value: float
    measured_lab: tuple[float, float, float]
    reference_lab: tuple[float, float, float] | None
    flags: list[str] = field(default_factory=list)
    position_relative: tuple[float, float] | None = None
    #: The corners, in image pixels, of the region this reading was taken from.
    #:
    #: Populated only for samples that are **not** patches -- a flat-field spot,
    #: the band a printed rule was read from, the feature two color records were
    #: differenced across. A patch's region is on its :class:`PatchRecord`
    #: instead, stored once and joined by ``patch_id``, because the metrics
    #: repeat their per-patch lists and the same four corners would otherwise be
    #: written out once per metric. The samples here appear in a single metric
    #: each and number tens rather than hundreds, so there is nothing to save.
    roi_px: list[list[float]] | None = None


@dataclass
class VectorReading:
    """A reading whose *direction* is part of the measurement.

    A sibling of :class:`PatchValue` rather than a field on it, because the two
    are not the same granularity. Channel registration reports per *axis* by
    default -- what OpenDICE plots and what the thresholds were read against --
    while a displacement is a vector only where one position gave both axes. One
    region yields two per-axis readings and at most one vector, so folding the
    vector into the readings would either duplicate it or drop it.

    Direction is not decoration here. A magnification difference between the
    color records points *radially* and grows from the center of the frame; a
    plain translation is uniform in size and direction everywhere. Both produce
    the same magnitudes, and no arrangement of magnitudes tells them apart.
    """

    label: str
    #: Where the reading sits on the target, 0..1 in each axis.
    position_relative: tuple[float, float]
    #: The displacement itself, in image pixels.
    vector_px: tuple[float, float]
    #: Which comparison produced it -- for registration, the channel pair.
    source: str = ""


@dataclass
class MetricResult:
    """The outcome of running one metric."""

    metric_id: str
    outcome: Outcome
    units: str | None = None
    value: float | None = None
    #: The reading as it should be written, where the default three decimals is
    #: the wrong shape for it. Three metrics need this and they are the file-level
    #: ones: the master's container is ``JPEG 2000`` and not a number at all, its
    #: colour space is the name of an ICC profile, and its bit depth is ``8`` --
    #: a quantity that is counted rather than measured, and that ``8.000`` claims
    #: a precision for that does not exist. Every other metric leaves it ``None``
    #: and is formatted from :attr:`value`, so a reader can still tell a reading
    #: from a label: this is only ever set where the number is not the point.
    value_text: str | None = None
    #: What one per-sample reading is measured in, where that is not :attr:`units`.
    #: The aggregate and the readings it was reduced from are usually one quantity
    #: -- a mean of ΔE values is a ΔE -- and then this stays ``None``. Four metrics
    #: differ, and a readings table headed with the aggregate's unit mislabels every
    #: row: lightness uniformity is a coefficient of variation whose readings are
    #: signed L* differences from a family mean; geometric distortion is a percent
    #: whose readings are each printed rule's sampling rate; and the two flat-field
    #: metrics are spreads whose readings are each spot's L*. See
    #: :attr:`reading_unit` and :data:`LEGACY_READING_UNITS`.
    reading_units: str | None = None
    passed: bool | None = None
    condition: str | None = None
    aggregation: str | None = None
    per_patch: list[PatchValue] = field(default_factory=list)
    #: Readings that carry a direction as well as a magnitude. Empty for almost
    #: every metric; a report draws arrows when it finds them, which is the same
    #: read-the-shape-of-the-data rule that decides which metrics get a field map
    #: at all rather than a switch on the metric's name.
    vectors: list[VectorReading] = field(default_factory=list)
    reason: str | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the types the report will format, at the point they are set.

        Every one of these reaches a format string in
        :mod:`deltachrome.reporting.console` and in the frontend's tables, and a
        plugin that puts the wrong kind of thing in one produces a ``TypeError``
        *after* the whole analysis has run -- on a hundred-megapixel master that is
        twenty seconds of work thrown away, with a traceback pointing at the
        renderer rather than at the metric that did it.

        Not hypothetical: ``geometric_distortion`` shadowed its own ``units`` local
        with a dictionary, and every distortion result carried it until a capture
        with printed rules was rendered through the console. The test suite did not
        catch it because no test rendered that metric.
        """
        if self.units is not None and not isinstance(self.units, str):
            raise TypeError(
                f"{self.metric_id}: units must be a string or None, got "
                f"{type(self.units).__name__}. It is formatted into the report."
            )
        if self.value is not None and not isinstance(self.value, (int, float)):
            raise TypeError(
                f"{self.metric_id}: value must be a number or None, got "
                f"{type(self.value).__name__}."
            )
        if self.value_text is not None and not isinstance(self.value_text, str):
            raise TypeError(
                f"{self.metric_id}: value_text must be a string or None, got "
                f"{type(self.value_text).__name__}. It is shown in place of the "
                "number."
            )
        if self.reading_units is not None and not isinstance(self.reading_units, str):
            raise TypeError(
                f"{self.metric_id}: reading_units must be a string or None, got "
                f"{type(self.reading_units).__name__}. It heads a readings table."
            )
        if self.condition is not None and not isinstance(self.condition, str):
            raise TypeError(
                f"{self.metric_id}: condition must be a string or None, got "
                f"{type(self.condition).__name__}."
            )

    @property
    def reading(self) -> str:
        """The reading as a person should see it, whichever form it took."""
        if self.value_text is not None:
            return self.value_text
        return "-" if self.value is None else f"{self.value:.3f}"

    @property
    def reading_unit(self) -> str | None:
        """What each per-sample reading is in: its own unit, else the aggregate's."""
        return self.reading_units if self.reading_units is not None else self.units

    @property
    def counts_toward_overall(self) -> bool:
        return self.outcome == "evaluated"

    @property
    def status(self) -> str:
        if self.outcome == "evaluated":
            if self.passed is None:
                # Measured but unjudged. Rendering this as FAIL would invent a verdict
                # the profile never asked for.
                return "MEASURED"
            return "PASS" if self.passed else "FAIL"
        return {
            "not_applicable": "N/A",
            "not_implemented": "NOT IMPLEMENTED",
            "incomplete": "INCOMPLETE",
        }[self.outcome]

    @classmethod
    def from_record(cls, entry: Mapping[str, Any]) -> "MetricResult":
        """Rebuild one metric from its entry in a saved run record.

        The inverse of what :func:`deltachrome.pipeline._build_run` writes, and it
        exists for the same reason ``AnalysisRun.from_dict`` does: **the record is
        the archive**, and a format that can only be written is a format that makes
        a year of work unusable the moment the question changes. The question that
        changed here is combining several captures into a session -- a site shoots
        three charts, analyzes each, and then wants to know whether the *system*
        met the specification. Re-running all three to ask that would mean decoding
        three masters to recompute numbers already on disk.

        Per-sample readings come back too. They are what the aggregate was reduced
        from, and a session that could show only the aggregates would be unable to
        say *where* two charts disagree.
        """
        outcome = str(entry.get("outcome") or "incomplete")
        return cls(
            metric_id=str(entry["metric_id"]),
            outcome=outcome,  # type: ignore[arg-type]
            units=entry.get("units"),
            value=None if entry.get("value") is None else float(entry["value"]),
            value_text=entry.get("value_text"),
            reading_units=entry.get("reading_units"),
            passed=entry.get("passed"),
            condition=entry.get("condition"),
            aggregation=entry.get("aggregation"),
            per_patch=[
                PatchValue(
                    patch_id=str(item.get("patch_id", "")),
                    role=str(item.get("role", "")),
                    value=float(item.get("value", float("nan"))),
                    # Not persisted per reading -- a patch's colour lives on its own
                    # PatchRecord, joined by id, so the metrics do not repeat it.
                    measured_lab=(0.0, 0.0, 0.0),
                    reference_lab=None,
                    flags=list(item.get("flags") or []),
                    position_relative=(
                        tuple(item["position_relative"])  # type: ignore[arg-type]
                        if item.get("position_relative") else None
                    ),
                    roi_px=item.get("roi_px"),
                )
                for item in (entry.get("per_patch") or [])
            ],
            vectors=[
                VectorReading(
                    label=str(item.get("label", "")),
                    position_relative=tuple(item["position_relative"]),  # type: ignore[arg-type]
                    vector_px=tuple(item["vector_px"]),  # type: ignore[arg-type]
                    source=str(item.get("source", "")),
                )
                for item in (entry.get("vectors") or [])
                if item.get("position_relative") and item.get("vector_px")
            ],
            reason=entry.get("reason"),
            notes=list(entry.get("notes") or []),
            metadata=dict(entry.get("metadata") or {}),
        )


# -- plugin registry --------------------------------------------------------

MetricFn = Callable[[MetricContext, "MetricSpec"], MetricResult]
_REGISTRY: dict[str, MetricFn] = {}


def register_metric(metric_id: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if metric_id in _REGISTRY:
            raise ProfileError(f"Metric {metric_id!r} is already registered.")
        _REGISTRY[metric_id] = fn
        return fn

    return decorator


def get_metric(metric_id: str) -> MetricFn | None:
    return _REGISTRY.get(metric_id)


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


# -- profile-driven metric specification ------------------------------------


@dataclass
class MetricSpec:
    """One entry from a profile's ``metrics`` list."""

    metric_id: str
    requires_capability: str | None = None
    algorithm: dict[str, Any] = field(default_factory=dict)
    aggregation: str = "mean"
    pass_condition: str | None = None
    units: str | None = None
    notes: str | None = None
    #: A threshold that depends on a run input rather than on the capture -- see
    #: :mod:`deltachrome.profiles.bands`. When present it supersedes
    #: ``pass_condition`` for any run that declares the input.
    pass_condition_by: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricSpec":
        try:
            return cls(
                metric_id=data["metric_id"],
                requires_capability=data.get("requires_capability"),
                algorithm=data.get("algorithm", {}) or {},
                aggregation=data.get("aggregation", "mean"),
                pass_condition=data.get("pass_condition"),
                units=data.get("units"),
                notes=data.get("notes"),
                pass_condition_by=(
                    BandedCondition.from_dict(data["pass_condition_by"])
                    if data.get("pass_condition_by") else None
                ),
            )
        except KeyError as exc:
            raise ProfileError(f"Metric entry is missing required key {exc}") from exc

    @property
    def patch_selection(self) -> Any:
        return self.algorithm.get("patch_selection", "all")


def resolve_condition(
    spec: MetricSpec, ctx: MetricContext
) -> tuple[str | None, list[str], dict[str, Any]]:
    """The pass condition to judge against, given what this run declared.

    Returns ``(condition, notes, metadata)``. A ``condition`` of ``None`` where the
    spec declares bands means the run did not say which band applies, and the metric
    should report its value **unjudged** rather than guessing one.
    """
    banded = spec.pass_condition_by
    if banded is None:
        return spec.pass_condition, [], {}

    declared = (ctx.extras or {}).get(banded.parameter)
    options = ", ".join(f"{b.band} ({b.pass_condition})" for b in banded.bands)
    if declared in (None, ""):
        return None, [], {"banded_by": banded.parameter, "bands_available": options}

    band = banded.named(declared)
    notes = [
        f"Judged against the {band.describe()} band, from {banded.parameter}="
        f"{declared!r} declared for this run. The threshold depends on what the "
        "system is being qualified for, not on anything in the capture, so it is "
        "recorded here: a run under a different band is a different measurement."
    ]
    return band.pass_condition, notes, {
        "banded_by": banded.parameter,
        "band": band.band,
        "band_label": band.describe(),
        "declared": str(declared),
        "bands_available": options,
    }


# -- patch selection --------------------------------------------------------

_ROLE_EQ = re.compile(r"^\s*role\s*==\s*['\"](?P<role>[\w]+)['\"]\s*$")
_ROLE_IN = re.compile(r"^\s*role\s+in\s*\[(?P<roles>[^\]]*)\]\s*$")
_QUOTED = re.compile(r"['\"]([\w]+)['\"]")


def select_patches(data: Sequence[PatchDatum], selection: Any) -> list[PatchDatum]:
    """Filter patch data by a profile's ``patch_selection``.

    Accepts the literal strings used in the project's profile documents
    (``"all patches"``, ``"role == 'achromatic_ramp'"``) and a structured form
    (``{"roles": [...]}`` / ``{"patch_ids": [...]}``). Deliberately not ``eval`` --
    profiles are user-editable files and must never be executable.
    """
    if selection is None:
        return list(data)

    if isinstance(selection, dict):
        recognized = {"roles", "patch_ids", "all", "min_reference_L"}
        unknown = set(selection) - recognized
        if unknown:
            raise ProfileError(
                f"Unrecognised patch_selection key(s) {sorted(unknown)}. Supported: "
                f"{sorted(recognized)}."
            )
        if not selection:
            raise ProfileError(f"Unrecognised patch_selection object: {selection!r}")

        chosen = list(data)
        if "roles" in selection:
            roles = set(selection["roles"])
            chosen = [d for d in chosen if d.role in roles]
        if "patch_ids" in selection:
            ids = set(selection["patch_ids"])
            chosen = [d for d in chosen if d.patch_id in ids]

        # Metamorfoze scopes neutral assessment by lightness -- "maximum L* value
        # for assessing neutrals", L* 5 / 20 / 30 across its three levels -- meaning
        # the ladder is only judged down to that tone. The lighter levels do not
        # hold the deep shadows to the same standard, and judging patches they
        # exclude would fail captures those levels pass.
        floor = selection.get("min_reference_L")
        if floor is not None:
            chosen = [
                d for d in chosen
                if d.reference_lab is not None and float(d.reference_lab[0]) >= floor
            ]
        return chosen

    if isinstance(selection, (list, tuple)):
        ids = set(selection)
        return [d for d in data if d.patch_id in ids]

    if isinstance(selection, str):
        text = selection.strip()
        if text.lower() in {"all", "all patches", "*"}:
            return list(data)
        match = _ROLE_EQ.match(text)
        if match:
            role = match.group("role")
            return [d for d in data if d.role == role]
        match = _ROLE_IN.match(text)
        if match:
            roles = set(_QUOTED.findall(match.group("roles")))
            return [d for d in data if d.role in roles]
        raise ProfileError(
            f"Unsupported patch_selection {selection!r}. Supported forms: "
            "\"all patches\", \"role == 'name'\", \"role in ['a','b']\", "
            '{"roles": [...]}, {"patch_ids": [...]}, or a list of patch ids.'
        )

    raise ProfileError(f"Unsupported patch_selection type {type(selection).__name__}")


# -- aggregation ------------------------------------------------------------

_PERCENTILE_RE = re.compile(r"^percentile_(?P<p>\d{1,3}(?:\.\d+)?)$")


def aggregate(
    values: Sequence[float] | np.ndarray,
    method: str,
    *,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
) -> tuple[float, dict[str, Any]]:
    """Reduce per-patch values to one number; also return how it was derived."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ProfileError("Cannot aggregate an empty set of patch values.")

    meta: dict[str, Any] = {"n": int(arr.size)}
    name = method.strip().lower()

    if name == "mean":
        return float(arr.mean()), meta
    if name == "median":
        return float(np.median(arr)), meta
    if name in ("max", "max_per_patch", "worst_edge", "worst_case"):
        return float(arr.max()), meta
    if name == "min":
        return float(arr.min()), meta
    if name in ("std", "stdev"):
        return float(arr.std(ddof=1)) if arr.size > 1 else 0.0, meta
    if name == "range":
        return float(arr.max() - arr.min()), meta
    if name == "rms":
        return float(np.sqrt(np.mean(arr**2))), meta

    match = _PERCENTILE_RE.match(name)
    if match:
        p = float(match.group("p"))
        if percentile_method not in PERCENTILE_METHODS:
            raise ProfileError(
                f"Unknown percentile_method {percentile_method!r}; "
                f"expected one of {sorted(PERCENTILE_METHODS)}"
            )
        numpy_method = PERCENTILE_METHODS[percentile_method]
        value = float(np.percentile(arr, p, method=numpy_method))
        meta["percentile"] = p
        meta["percentile_method"] = percentile_method
        return value, meta

    raise ProfileError(
        f"Unknown aggregation {method!r}. Supported: mean, median, max "
        "(max_per_patch, worst_edge), min, std, range, rms, percentile_N."
    )


# -- pass/fail --------------------------------------------------------------

_CONDITION_RE = re.compile(
    r"^\s*value\s*(?P<op><=|>=|<|>|==|!=)\s*(?P<threshold>-?\d+(?:\.\d+)?)\s*$"
)
#: Two-sided form, e.g. "0.8 <= value <= 1.1". Needed because some criteria are
#: bands rather than ceilings -- gain modulation fails for being too flat *or* too
#: steep, and collapsing that to one bound would only ever catch half the fault.
_RANGE_RE = re.compile(
    r"^\s*(?P<low>-?\d+(?:\.\d+)?)\s*(?P<lop><=|<)\s*value\s*"
    r"(?P<hop><=|<)\s*(?P<high>-?\d+(?:\.\d+)?)\s*$"
)
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda v, t: v < t,
    "<=": lambda v, t: v <= t,
    ">": lambda v, t: v > t,
    ">=": lambda v, t: v >= t,
    "==": lambda v, t: v == t,
    "!=": lambda v, t: v != t,
}


def evaluate_condition(value: float, condition: str | None) -> tuple[bool | None, float | None]:
    """Test ``value`` against a profile pass condition.

    Accepts a one-sided comparison (``"value < 3.5"``) or a two-sided band
    (``"0.8 <= value <= 1.1"``).

    Returns ``(passed, threshold)``. A profile with no condition yields
    ``(None, None)`` -- a measured but unjudged metric, which is legitimate for
    informational metrics and must not be silently treated as a pass.
    """
    if condition is None:
        return None, None

    ranged = _RANGE_RE.match(condition)
    if ranged:
        low, high = float(ranged.group("low")), float(ranged.group("high"))
        lo_ok = value >= low if ranged.group("lop") == "<=" else value > low
        hi_ok = value <= high if ranged.group("hop") == "<=" else value < high
        # The reported threshold is the bound the value is closer to violating, so a
        # report shows what actually constrains it.
        nearer = low if abs(value - low) <= abs(value - high) else high
        return bool(lo_ok and hi_ok), nearer

    match = _CONDITION_RE.match(condition)
    if not match:
        raise ProfileError(
            f"Unsupported pass_condition {condition!r}. Expected the form "
            '"value < 3.5" using one of < <= > >= == !=, or a two-sided band such as '
            '"0.8 <= value <= 1.1".'
        )
    op = _OPS[match.group("op")]
    threshold = float(match.group("threshold"))
    return bool(op(value, threshold)), threshold


def _literal(value: float) -> str:
    """A threshold as a profile would have written it: no trailing zeros, no
    exponent for the magnitudes criteria actually use.

    Six significant figures, which rounds a converted threshold by at most 5e-7 of
    itself. That is deliberate. The criterion is one string, both displayed and
    evaluated, so a report saying ``value >= 1.72125`` is quoting the number the
    verdict was actually reached against -- keeping an exact threshold behind a
    rounded display is the same split that let a change of unit change a verdict.
    The rounding sits some five orders below the precision of any reading it
    judges.
    """
    text = f"{value:.6g}"
    return "0" if text in ("-0", "0.0", "-0.0") else text


def rescale_condition(condition: str | None, factor: float) -> str | None:
    """The same criterion, expressed in a unit ``factor`` times the original.

    Only sound where the value and its threshold are **the same physical
    quantity**, because it works by multiplying both sides of the comparison --
    which leaves the inequality untouched for any positive factor. That holds for a
    frequency reported in a different unit, which is the one caller here. It does
    not hold between quantities, and reaching for it to convert a threshold from,
    say, density to lightness would produce a confident and wrong criterion.

    The alternative -- scaling the measurement and leaving the threshold in the
    unit the profile declared -- silently changes the verdict, which is what this
    exists to prevent.
    """
    if condition is None:
        return None
    if factor <= 0:
        raise ProfileError(
            f"A unit conversion factor must be positive; got {factor!r}. A negative "
            "factor would reverse the comparison rather than re-express it."
        )
    if factor == 1.0:
        return condition

    ranged = _RANGE_RE.match(condition)
    if ranged:
        low = float(ranged.group("low")) * factor
        high = float(ranged.group("high")) * factor
        return (
            f"{_literal(low)} {ranged.group('lop')} value "
            f"{ranged.group('hop')} {_literal(high)}"
        )

    match = _CONDITION_RE.match(condition)
    if not match:
        raise ProfileError(
            f"Unsupported pass_condition {condition!r}. Expected the form "
            '"value < 3.5" using one of < <= > >= == !=, or a two-sided band such as '
            '"0.8 <= value <= 1.1".'
        )
    threshold = float(match.group("threshold")) * factor
    return f"value {match.group('op')} {_literal(threshold)}"


def not_applicable(spec: MetricSpec, target: TargetDefinition) -> MetricResult:
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="not_applicable",
        units=spec.units,
        condition=spec.pass_condition,
        reason=(
            f"{target.display_name} does not declare the capability "
            f"{spec.requires_capability!r}."
        ),
    )


def not_measurable_in_capture(spec: MetricSpec, records: int) -> MetricResult:
    """Reported when the *capture*, not the target, cannot carry the metric.

    Deliberately ``incomplete`` rather than ``not_applicable`` -- see
    :data:`CAPTURE_COLOR_CAPABILITIES` for why a pass here would be the worst
    available answer.
    """
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="incomplete",
        units=spec.units,
        condition=spec.pass_condition,
        reason=(
            f"{spec.metric_id!r} reads {spec.requires_capability!r}, which needs "
            f"three color records; this capture carries {records}. The target may "
            "well support it -- the information is absent from the file, not from "
            "the chart. Measuring anyway would report a perfect score for a "
            "comparison that was never made."
        ),
    )


def not_implemented(spec: MetricSpec) -> MetricResult:
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="not_implemented",
        units=spec.units,
        condition=spec.pass_condition,
        reason=(
            f"No plugin is registered for {spec.metric_id!r}. The target supports it "
            "and the profile requires it, so this result is incomplete -- it is not "
            "the same as the metric being inapplicable."
        ),
    )
