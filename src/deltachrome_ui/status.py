"""How a verdict is allowed to look.

This is the smallest module in the frontend and the one most worth getting right.

The engine is careful never to state a number it cannot justify. A metric is not
merely pass or fail: it can be ``MEASURED`` (a real reading with no threshold to
judge it against), ``N/A`` (the target cannot measure it, which is not a fault),
``INCOMPLETE`` (something was attempted and could not be finished) or
``NOT IMPLEMENTED``. Orthogonal to all of those, a whole run can lose its
conformance claim -- because it was scored against a manufacturer's nominal values
instead of a measurement of the chart in front of the camera, or against a locally
derived specification rather than the standard itself.

A GUI is exactly where that collapses into a green tick. So the rule here is
stated once, in code, and used everywhere:

    **A run without a conformance claim never renders in the pass tone, whatever
    its verdict says.**

That is not cosmetic. Every ISA target Deltachrome ships runs on nominal values and
realistically always will, so this is the
common case rather than the corner case. A green PASS on one of those would be the
frontend telling a lie the CLI has never told.

Tones are semantic names, not colors. The stylesheet decides what they look like;
this module decides which one applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Semantic tones. A theme maps these to colors; nothing here knows about color.
TONE_PASS = "pass"
TONE_FAIL = "fail"
#: A real result that must not read as an unqualified success -- an advisory
#: verdict, or a measurement nothing judged.
TONE_QUALIFIED = "qualified"
#: Nothing was attempted, and that is not a fault.
TONE_ABSENT = "absent"
#: Something was attempted and could not be finished.
TONE_UNKNOWN = "unknown"


#: Worst first. What a status column sorts on when it is sorted ascending, which
#: is the only ordering a person clicking that column wants: the runs that need
#: attention gather at the top. Ranked by tone rather than by label so that a
#: verdict this version does not recognize still lands somewhere deliberate.
TONE_ORDER = {
    TONE_FAIL: 0,
    TONE_UNKNOWN: 1,
    TONE_QUALIFIED: 2,
    TONE_ABSENT: 3,
    TONE_PASS: 4,
}


@dataclass(frozen=True)
class Status:
    """How one verdict should be presented."""

    label: str
    tone: str
    #: One line saying what the label means, for a tooltip. Never empty: every
    #: status in this vocabulary is one a user can reasonably not know.
    hint: str
    #: The part of the label that *qualifies* the verdict rather than being it.
    #: Only ``ADVISORY`` so far, and it exists because "PASS (ADVISORY)" is
    #: seventeen characters of which the first four are the answer. A header can
    #: then set the answer large and the qualification small beneath it, without
    #: any presenter parsing a string it did not build.
    qualifier: str = ""

    @property
    def headline(self) -> str:
        """The verdict without its qualification.

        Derived from :attr:`label` rather than stored beside it, so the two
        cannot drift: ``label`` stays the authority and stays the same string the
        engine's own ``verdict_line`` prints.
        """
        suffix = f"({self.qualifier})"
        if self.qualifier and self.label.endswith(suffix):
            return self.label[: -len(suffix)].strip()
        return self.label

    @property
    def reads_as_success(self) -> bool:
        return self.tone == TONE_PASS

    @property
    def rank(self) -> int:
        """Where this sorts in a status column. See :data:`TONE_ORDER`."""
        return TONE_ORDER.get(self.tone, TONE_ORDER[TONE_UNKNOWN])


#: Per-metric statuses, keyed by :attr:`deltachrome.metrics.base.MetricResult.status`.
METRIC_STATUS = {
    "PASS": Status(
        "PASS", TONE_PASS,
        "Met the threshold."),
    "FAIL": Status(
        "FAIL", TONE_FAIL,
        "Did not meet the threshold."),
    "MEASURED": Status(
        "MEASURED", TONE_QUALIFIED,
        "Measured but not judged: these guidelines set no threshold for it."),
    "N/A": Status(
        "N/A", TONE_ABSENT,
        "This target can't measure it. Another target is needed."),
    "INCOMPLETE": Status(
        "INCOMPLETE", TONE_UNKNOWN,
        "Couldn't be completed."),
    "NOT IMPLEMENTED": Status(
        "NOT IMPLEMENTED", TONE_UNKNOWN,
        "Deltachrome can't measure this metric yet, so the run can't pass."),
}

_OVERALL = {
    "PASS": Status(
        "PASS", TONE_PASS,
        "Every judged metric met its threshold."),
    "FAIL": Status(
        "FAIL", TONE_FAIL,
        "At least one metric did not meet its threshold."),
    "MEASURED": Status(
        "MEASURED", TONE_QUALIFIED,
        "Measured but not judged: these guidelines set no thresholds."),
    "INCOMPLETE": Status(
        "INCOMPLETE", TONE_UNKNOWN,
        "A metric couldn't be completed, so the run can't pass."),
}

_UNRECOGNISED = Status(
    "UNKNOWN", TONE_UNKNOWN,
    "Verdict not recognized. The record may be from another version of "
    "Deltachrome.")


def metric_status(status_text: str) -> Status:
    return METRIC_STATUS.get(status_text, _UNRECOGNISED)


def overall_status(
    overall: str, conformance_claim: bool = True, *, evidence_sound: bool | None = None
) -> Status:
    """The run-level verdict, qualified by whether the EVIDENCE may be relied on.

    The label matches the engine's own ``verdict_line`` so that a figure read off
    the screen and a figure read out of the CLI are the same string. The tone is
    the part that is this module's decision, and the demotion below is the whole
    point of the module.

    **What demotes is the reference, not the criteria**, and that is a correction.
    A run loses its conformance claim two ways -- the numbers describe the model
    rather than this chart, or the thresholds are a local guideline rather than
    a standard's own -- and demoting on either meant a department that wrote its own
    standard down could never see an unqualified verdict, however well the capture
    was measured. Which taught the qualification to mean nothing: a tone that is
    always on is a tone nobody reads.

    They are not the same kind of thing. A nominal reference is a fact about the
    evidence and no decision makes it untrue. A derived guideline is a decision,
    recorded, visible in the guideline's own name on the report and in its list of
    departures -- and the run genuinely met it. So the verdict stands and says, in
    its tooltip and in the caveat above the table, what it is a verdict *about*.

    ``evidence_sound`` defaults to ``conformance_claim`` for a caller that cannot
    tell them apart -- a record written before the two were separable -- which is
    the old behaviour and the safe direction.
    """
    base = _OVERALL.get(overall, _UNRECOGNISED)
    if evidence_sound is None:
        evidence_sound = conformance_claim
    if evidence_sound:
        if conformance_claim:
            return base
        return Status(
            base.label,
            base.tone,
            base.hint + "  Judged against a local guideline, not a published "
            "standard.",
        )
    tone = base.tone
    if tone == TONE_PASS:
        # The demotion. A run judged against nominal data measured something real,
        # but it did not test the thing the label appears to claim.
        tone = TONE_QUALIFIED
    return Status(
        f"{base.label} (ADVISORY)",
        tone,
        base.hint + "  Not a conformance result: the reference isn't a "
        "measurement of this chart.",
        qualifier="ADVISORY",
    )


def status_for_record(record: dict[str, Any]) -> Status:
    """Verdict for a saved run, read from its JSON.

    Tolerant of a record written by another version: an absent
    ``conformance_claim`` predates the field, and the engine's own default when
    the attribute is missing is ``True``, so this matches the console renderer.
    """
    return overall_status(
        str(record.get("overall", "")),
        bool(record.get("conformance_claim", True)),
        evidence_sound=evidence_sound(record),
    )


def evidence_sound(record: dict[str, Any]) -> bool | None:
    """Whether a saved run's numbers describe the chart that was photographed.

    ``None`` where the record does not say, which is a record from before the
    basis was written down -- and the caller then falls back to the conformance
    claim, which is the old behaviour and withholds rather than grants.
    """
    basis = (record.get("reference") or {}).get("basis")
    return None if not basis else basis == "per_unit_measurement"
