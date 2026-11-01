"""Run a profile's metrics against measured data and roll up an overall verdict."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from deltachrome.metrics.base import (
    CAPTURE_COLOR_CAPABILITIES,
    MetricContext,
    MetricResult,
    get_metric,
    not_applicable,
    not_implemented,
    not_measurable_in_capture,
)
from deltachrome.profiles.schema import AnalysisProfile
from deltachrome.reference.cgats import (
    MANUFACTURER_NOMINAL,
    PER_UNIT_MEASUREMENT,
    POPULATION_MEAN,
)
from deltachrome.targets.schema import TargetDefinition
from deltachrome.text import plural


def _were(count: int) -> str:
    """The verb agreeing with a count, for the sentences below."""
    return "was" if count == 1 else "were"


def _have(count: int) -> str:
    return "has" if count == 1 else "have"


#: The caveats again, in one line, for a screen.
#:
#: Not a second opinion: it is composed from the same two facts
#: :meth:`ProfileEvaluation.conformance_caveats` is composed from -- whether the
#: reference was the manufacturer's nominal data, and whether the profile was
#: derived from a standard rather than being one. The long forms stay exactly as
#: they are, because they are what the record archives and what the CLI prints;
#: this is what fits on a report header without pushing the table off the screen.
#:
#: Built from the facts rather than by shortening the prose so that a run
#: reopened from the library, which holds the sentences and not the objects that
#: produced them, can compose the identical line.
_ADVISORY_NOMINAL = (
    "No reference file provided. Metrics judged against manufacturer values."
)
#: Said instead of the above where the values are a population mean rather than a
#: printed aim. Both forfeit the conformance claim -- neither is a measurement of
#: the chart in front of the camera -- and they are not the same thing, so a reader
#: who could act on the difference is told it.
_ADVISORY_POPULATION = (
    "No reference file provided. Metrics judged against a mean of measured units "
    "of this model."
)
#: Says what the criteria *are*, not where they came from. It used to name the
#: parent profile -- "Thresholds derived from fadgi_4star_prints_photographs_2023"
#: -- which is a raw id on a report, and a connection to a table the operator may
#: have started from only because it was convenient. Once a guideline is saved it
#: is its own specification; the parent stays in the record and off the screen.
_ADVISORY_DERIVED = "Thresholds are this site's own, not a published standard's."


#: The two spellings a run record may carry for "these criteria are local". New
#: records state ``local_criteria``; ones written while the parent was recorded
#: state ``derived_from``, whose value is a parent profile id nothing reads any
#: more. Read in one place so a reader cannot be added that knows only one of
#: them -- which would silently restore the conformance claim to a local run, and
#: that is the one direction this must never fail in.
def criteria_are_local(profile_record: Mapping[str, Any] | None) -> bool:
    """Whether a saved run's profile block says its criteria were a site's own."""
    record = profile_record or {}
    return bool(record.get("local_criteria") or record.get("derived_from"))


def advisory_line(
    *, reference_basis: str | None, local_criteria: bool = False
) -> str | None:
    """One line saying why this run is not a conformance result, or ``None``.

    ``local_criteria`` is the one fact the parent profile id was ever carrying
    here. Its *value* was printed -- "Thresholds derived from
    fadgi_3star_documents_unbound_general_2023" -- which is a filename on a report.

    ``reference_basis`` is one of :data:`~deltachrome.reference.cgats.KNOWN_BASES`,
    or ``None`` on a record from before the basis was stored. Anything that is not
    a per-unit measurement is a caveat, which is the safe direction: a basis this
    version does not recognise says so rather than being quietly waved through.
    """
    parts = []
    if reference_basis == POPULATION_MEAN:
        parts.append(_ADVISORY_POPULATION)
    elif reference_basis is not None and reference_basis != PER_UNIT_MEASUREMENT:
        parts.append(_ADVISORY_NOMINAL)
    only_criteria = not parts
    if local_criteria:
        parts.append(_ADVISORY_DERIVED)
    if not parts:
        return None
    # "Advisory" is about the EVIDENCE. Where the only qualification is that the
    # criteria are the operator's own, the run is not advisory at all -- it met a
    # written specification, on a chart that was measured -- and saying it is
    # teaches the word to mean nothing on every screen it appears.
    if only_criteria:
        return "Local guideline: " + " ".join(parts)
    return "Advisory: " + " ".join(parts)


@dataclass
class ProfileEvaluation:
    profile: AnalysisProfile
    results: list[MetricResult]
    capability_report: dict[str, list[str]] = field(default_factory=dict)
    #: What the metrics were compared against. A run scored on the manufacturer's
    #: aim values for the model measures something real, but not the same thing as a
    #: run scored on a measurement of the chart in front of the camera.
    reference_basis: str = PER_UNIT_MEASUREMENT

    @property
    def evaluated(self) -> list[MetricResult]:
        return [r for r in self.results if r.outcome == "evaluated"]

    @property
    def failures(self) -> list[MetricResult]:
        return [r for r in self.evaluated if r.passed is False]

    @property
    def unjudged(self) -> list[MetricResult]:
        """Metrics that produced a number but had no pass condition to judge it."""
        return [r for r in self.evaluated if r.passed is None]

    @property
    def not_implemented(self) -> list[MetricResult]:
        return [r for r in self.results if r.outcome == "not_implemented"]

    @property
    def incomplete(self) -> list[MetricResult]:
        return [r for r in self.results if r.outcome == "incomplete"]

    @property
    def judged(self) -> list[MetricResult]:
        """Metrics that produced a number *and* had a threshold to judge it against."""
        return [r for r in self.evaluated if r.passed is not None]

    @property
    def overall(self) -> str:
        """``PASS``, ``FAIL``, ``INCOMPLETE`` or ``MEASURED``.

        A run that skipped a metric the profile requires and the target supports is
        never a PASS. Reporting one would be exactly the overclaiming this project
        set out not to do.

        ``MEASURED`` covers a profile that sets no thresholds at all -- a
        measurement-only profile. Calling that a PASS would imply something was
        tested and met, when nothing was judged.
        """
        if self.failures:
            return "FAIL"
        if self.not_implemented or self.incomplete:
            return "INCOMPLETE"
        if not self.evaluated:
            return "INCOMPLETE"
        if not self.judged:
            return "MEASURED"
        return "PASS"

    @property
    def not_applicable(self) -> list[MetricResult]:
        return [r for r in self.results if r.outcome == "not_applicable"]

    @property
    def coverage(self) -> dict[str, Any]:
        """How much of the profile this run actually tested.

        A pass says nothing about how much was examined, and on a partial target
        that gap is wide: a ColorChecker SG against FADGI's Prints and Photographs
        table passes twelve metrics and skips six, because it has no slanted edges
        and no scale. That is not a fault -- it is what the chart is -- and it is
        also not the whole standard. Sites routinely cover the rest with a second
        target in the same session, which only works if each report says plainly
        what it did and did not reach.
        """
        return {
            "declared": len(self.results),
            "judged": len(self.judged),
            "measured_unjudged": len(self.unjudged),
            "not_applicable": len(self.not_applicable),
            "incomplete": len(self.incomplete) + len(self.not_implemented),
            "not_measurable_by_this_target": sorted(
                r.metric_id for r in self.not_applicable
            ),
        }

    @property
    def coverage_line(self) -> str:
        c = self.coverage
        text = f"{c['judged']} of {plural(c['declared'], 'metric')} judged"
        if c["not_applicable"]:
            text += f", {c['not_applicable']} not measurable by this target"
        if c["measured_unjudged"]:
            text += f", {c['measured_unjudged']} measured without a threshold"
        if c["incomplete"]:
            text += f", {c['incomplete']} incomplete"
        return text

    @property
    def reference_caveat(self) -> str | None:
        """Why the EVIDENCE is not a measurement of the chart in front of the camera.

        A fact about the capture rather than a choice about it: the numbers being
        compared against describe the model, and the specimen's own print tolerance
        sits inside every delta-E. Nothing the operator can decide makes it untrue,
        which is what separates it from :attr:`criteria_caveat`.
        """
        reasons = self._reference_reasons()
        return reasons[0] if reasons else None

    @property
    def criteria_caveat(self) -> str | None:
        """Why the CRITERIA are not a published standard's own.

        A deliberate, recorded choice, and the whole point of the guideline
        builder: a department states the specification it works to. It holds
        whichever way a threshold moved -- a tightened profile is no more the
        standard than a relaxed one, and "stricter, so it still counts" is a rule
        that is easy to game and awkward to explain.

        **It is not a weakness in the run.** Folding it together with the reference
        caveat made it one: a department that wrote down its own standard could
        never see a verdict that was not qualified, however well the capture was
        measured, which taught the qualification to mean nothing. What it qualifies
        is the *claim to a standard*, and that is said in its own words.
        """
        if not self.profile.local_criteria:
            return None
        # The guideline's own name, and not the parent's id and departure list.
        # A departure sentence reads "with noise relaxed from 'value < 3' to
        # 'value < 9'", which is only meaningful beside the standard it departs
        # from -- so printing it here reinstated exactly the connection a saved
        # guideline is meant to have shed. The record keeps both.
        return (
            f"judged against {self.profile.display_name}, a local guideline "
            f"rather than a published standard"
        )

    @property
    def evidence_is_sound(self) -> bool:
        """Whether the numbers describe the chart that was photographed.

        The question a verdict's *tone* answers, and deliberately not the same one
        :attr:`conformance_claim` answers. A run may meet every threshold of a
        department's own written specification and be a perfectly good result; what
        it cannot be is better than the reference it was measured against.
        """
        return not self._reference_reasons()

    def _reference_reasons(self) -> list[str]:
        reasons = []
        # Granted only for a per-unit measurement, rather than withheld for a list
        # of known nominals. The list version means any basis this version does not
        # recognise -- a newer file, a hand-edited one -- is quietly promoted to a
        # conformance claim, which is the one direction this must never fail in.
        if self.reference_basis == POPULATION_MEAN:
            reasons.append(
                "judged against a mean of measured units of this model, not against "
                "a measurement of the chart itself"
            )
        elif self.reference_basis != PER_UNIT_MEASUREMENT:
            reasons.append(
                "judged against the manufacturer's nominal values for this model of "
                "chart, not against a measurement of the chart itself"
            )
        return reasons

    @property
    def conformance_caveats(self) -> list[str]:
        """Every reason this run cannot be quoted as conformance to a standard.

        Two independent ones and a run can carry both, but they are not the same
        kind of thing: see :attr:`reference_caveat` and :attr:`criteria_caveat`.
        The list and its wording are unchanged, because it is what the record
        stores and a year of saved runs has to keep reading.
        """
        return [c for c in (self.reference_caveat, self.criteria_caveat) if c]

    @property
    def advisory_line(self) -> str | None:
        """The caveats in one line, or ``None`` when there are none."""
        return advisory_line(
            reference_basis=self.reference_basis,
            local_criteria=self.profile.local_criteria,
        )

    @property
    def conformance_claim(self) -> bool:
        """Whether this run can be quoted as a conformance test."""
        return not self.conformance_caveats

    @property
    def verdict_line(self) -> str:
        """The verdict as it should be quoted -- never a bare PASS on nominal data.

        **Qualified by the evidence, not by the criteria.** A run against a
        department's own written specification, on a chart that was actually
        measured, met that specification: calling it advisory says the reading
        cannot be relied on, when what is true is only that it is not a claim
        against the standard the specification came from. That is said in the
        caveat, in the guideline's own name on the report, and in the record --
        three places, none of which is the word the verdict wears.
        """
        if self.evidence_is_sound:
            return self.overall
        return f"{self.overall} (ADVISORY)"

    @property
    def overall_explanation(self) -> str:
        text = self._verdict_explanation() + self._coverage_sentence()
        caveats = self.conformance_caveats
        if not caveats:
            return text
        # The lead says which kind of qualification this is, for the same reason
        # ``advisory_line`` does: a run that met a department's own specification on
        # a measured chart is not advisory, and the only thing it is not is a claim
        # against the standard that specification came from.
        lead = (
            "Not a conformance test against the standard: " if self.evidence_is_sound
            else "Advisory, not a conformance result: "
        )
        return lead + "; and ".join(caveats) + ". " + text

    def _verdict_explanation(self) -> str:
        if self.overall == "FAIL":
            names = ", ".join(r.metric_id for r in self.failures)
            return f"{plural(len(self.failures), 'metric')} failed: {names}."
        if self.overall == "INCOMPLETE":
            reasons = []
            if self.not_implemented:
                reasons.append(
                    f"{plural(len(self.not_implemented), 'metric')} "
                    f"{_have(len(self.not_implemented))} no plugin yet "
                    f"({', '.join(r.metric_id for r in self.not_implemented)})"
                )
            if self.incomplete:
                reasons.append(
                    f"{plural(len(self.incomplete), 'metric')} "
                    "lacked usable data "
                    f"({', '.join(r.metric_id for r in self.incomplete)})"
                )
            if not self.evaluated and not reasons:
                # Unreachable on a shipped standard -- each judges the file itself,
                # which any file that opens can answer -- so this is a guideline and
                # a chart with nothing in common, not a run where everything failed.
                return (
                    "Not a pass and not a failure: nothing this guideline judges can "
                    "be measured on this chart."
                )
            text = "Not a pass and not a failure: " + "; ".join(reasons) + "."
            if not self.evaluated:
                return text
            return (
                text + " Every metric that did run met its threshold, but the run "
                "does not demonstrate conformance."
            )
        if self.overall == "MEASURED":
            return (
                f"{plural(len(self.evaluated), 'metric')} "
                f"{_were(len(self.evaluated))} measured. This profile sets no "
                "thresholds, so nothing was judged and this is not a conformance claim."
            )
        unjudged = self.unjudged
        text = (f"All {plural(len(self.judged), 'judged metric')} met their "
                "thresholds.")
        if unjudged:
            text += (
                f" {plural(len(unjudged), 'metric')} "
                f"{_were(len(unjudged))} measured but carry no pass condition and "
                "did not affect this verdict."
            )
        return text

    def _coverage_sentence(self) -> str:
        """Appended to every verdict. A failure over part of a standard is as partial
        as a pass over part of one, and either can be quoted as the whole."""
        skipped = self.not_applicable
        if not skipped:
            return ""
        return (
            f" {plural(len(skipped), 'metric')} this target cannot measure "
            f"{_were(len(skipped))} not tested "
            f"({', '.join(r.metric_id for r in skipped)}), so this covers "
            f"{len(self.judged)} of the {len(self.results)} the profile declares."
        )


def evaluate_profile(
    profile: AnalysisProfile, target: TargetDefinition, ctx: MetricContext
) -> ProfileEvaluation:
    """Run every metric the profile declares, in profile order."""
    results: list[MetricResult] = []

    for spec in profile.metrics:
        capability = spec.requires_capability
        if capability and not target.supports(capability):
            results.append(not_applicable(spec, target))
            continue

        # Checked after the target, because "this chart has no slanted edges" is the
        # more specific truth than "this file has one colour record" when both hold.
        if capability in CAPTURE_COLOR_CAPABILITIES and ctx.capture_records < 3:
            results.append(not_measurable_in_capture(spec, ctx.capture_records))
            continue

        fn = get_metric(spec.metric_id)
        if fn is None:
            results.append(not_implemented(spec))
            continue

        results.append(fn(ctx, spec))

    return ProfileEvaluation(
        profile=profile,
        results=results,
        capability_report=profile.capability_report(target.capabilities),
        reference_basis=ctx.reference_basis,
    )


def required_capability_note(profile: AnalysisProfile, target: TargetDefinition) -> str | None:
    """Said out loud when a profile's *required* capabilities are absent.

    Absence never fails a run -- a metric the target cannot measure is not the
    capture's fault, and a site may well be covering it with another target. But a
    profile naming a capability as required is saying the standard leans on it, and
    a pass reached without it should not read the same as one reached with it.
    """
    missing = sorted(profile.required_capabilities - target.capabilities)
    if not missing:
        return None
    return (
        f"{target.display_name} cannot measure {', '.join(missing)}, which "
        f"{profile.display_name} lists as required for the category. Nothing here "
        "fails on that -- the chart is what it is -- but the metrics depending on it "
        "were not tested, so this run does not cover the category on its own. A "
        "second target that carries the feature would complete it."
    )
