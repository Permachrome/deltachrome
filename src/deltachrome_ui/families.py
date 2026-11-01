"""What a report is divided into.

A run measures one capture, but it measures several unrelated things about it,
and an operator arrives asking about one of them: is the color right, is it
sharp enough, is the lighting even. The old report answered by listing eighteen
metrics in one column and making the reader do the grouping.

**Families are read from the data, not listed here.** Each metric records the
capability it reads -- ``color_accuracy``, ``sfr``, ``flat_field`` -- and that is
already the engine's own vocabulary for "what kind of thing this measures". So
the report groups on it. A profile from a standard nobody has modeled yet
arrives with its metrics already sorted, and a new capability brings its own
compartment rather than falling into a default bucket.

What *is* listed here is presentation: an order to show them in, and a plain
name, because ``lightness_uniformity`` is a key and "Lightness Uniformity" is a
label. If a capability appears that this table does not know, it is shown under
its own name with the underscores taken out rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deltachrome_ui.status import (
    TONE_ABSENT,
    TONE_FAIL,
    TONE_PASS,
    TONE_QUALIFIED,
    TONE_UNKNOWN,
    Status,
    metric_status,
)

#: Capability -> (display name, order). Order runs from what a capture is judged
#: on most often to what it is judged on least, so the compartment someone opens
#: first is near the top.
FAMILY_NAMES: dict[str, tuple[str, int]] = {
    "color_accuracy": ("Color", 10),
    "white_balance": ("White Balance", 20),
    "tonal_response": ("Tone", 30),
    "sfr": ("Resolution", 40),
    "color_registration": ("Registration", 50),
    "lightness_uniformity": ("Uniformity", 60),
    "flat_field": ("Illumination", 65),
    "geometric_distortion": ("Geometry", 80),
    "reproduction_scale": ("Scale", 85),
}

#: Metrics that read no capability at all are properties of the file and of how
#: it was captured, rather than measurements of the image. They are genuinely a
#: group, and calling them "Other" would bury the fact.
CAPTURE_FAMILY = ("Capture & File", 90)

#: The worst status wins the compartment's badge, in this order. FAIL outranks
#: INCOMPLETE because a known failure is a firmer statement than an unfinished
#: measurement, and both outrank a reading nothing judged.
_SEVERITY = ["FAIL", "INCOMPLETE", "NOT IMPLEMENTED", "MEASURED", "PASS", "N/A"]

_FAMILY_TONE = {
    "FAIL": TONE_FAIL,
    "INCOMPLETE": TONE_UNKNOWN,
    "NOT IMPLEMENTED": TONE_UNKNOWN,
    "MEASURED": TONE_QUALIFIED,
    "PASS": TONE_PASS,
    "N/A": TONE_ABSENT,
}


@dataclass
class Family:
    """One compartment of a report."""

    key: str
    name: str
    order: int
    metrics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> Status:
        """The compartment's own badge: the worst verdict inside it.

        Never better than its worst member. A family showing PASS while one of
        its metrics failed would be the same flattening the whole frontend is
        built to avoid, one level up.
        """
        worst = "N/A"
        for metric in self.metrics:
            status = metric.get("status", "")
            if status in _SEVERITY and _SEVERITY.index(status) < _SEVERITY.index(worst):
                worst = status
        base = metric_status(worst)
        return Status(self.summary_label, _FAMILY_TONE.get(worst, TONE_UNKNOWN), base.hint)

    @property
    def summary_label(self) -> str:
        """How many of this family's metrics were judged, and how they went.

        A count rather than a bare verdict, because "3 of 5 judged" and "5 of 5
        judged" are different statements about the same PASS and the difference
        is coverage.
        """
        judged = [m for m in self.metrics if m.get("passed") is not None]
        if not judged:
            statuses = {m.get("status") for m in self.metrics}
            if statuses == {"N/A"}:
                return "N/A"
            return "not judged"
        failed = sum(1 for m in judged if not m["passed"])
        if failed:
            return f"{failed} of {len(judged)} failed"
        return f"{len(judged)} passed"


def _capabilities_from_profile(profile_id: str | None) -> dict[str, str | None]:
    """Metric id -> capability, asked of the installed profile.

    For records written before the capability was persisted. Rather than keep a
    private table of metric ids here -- which would be a second copy of the
    profiles, drifting -- the profile that produced the run is asked directly.
    A profile that is no longer installed simply yields nothing, and those
    metrics fall into the capture compartment.
    """
    if not profile_id:
        return {}
    try:
        from deltachrome.profiles.schema import load_profile

        profile = load_profile(profile_id)
    except Exception:
        return {}
    return {spec.metric_id: spec.requires_capability for spec in profile.metrics}


def families(
    metrics: list[dict[str, Any]], profile_id: str | None = None
) -> list[Family]:
    """Group a run's metrics into compartments, in presentation order."""
    legacy = _capabilities_from_profile(profile_id) if any(
        m.get("requires_capability") is None for m in metrics
    ) else {}
    grouped: dict[str, Family] = {}
    for metric in metrics:
        capability = metric.get("requires_capability") or legacy.get(
            metric.get("metric_id")
        )
        if capability:
            name, order = FAMILY_NAMES.get(
                capability, (capability.replace("_", " ").title(), 88)
            )
            key = capability
        else:
            name, order = CAPTURE_FAMILY
            key = "_capture"
        family = grouped.get(key)
        if family is None:
            family = grouped[key] = Family(key=key, name=name, order=order)
        family.metrics.append(metric)
    return sorted(grouped.values(), key=lambda f: (f.order, f.name))
