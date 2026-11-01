"""Describing the shipped profiles, for anything that has to present a choice.

Picking a profile is a three-way decision -- which body and edition, which
material category, which performance level -- and the material category is the
easy and consequential one to get wrong: FADGI's transparency tables specify no
color accuracy metric at all, while the reflective ones do. Someone who picks a
transparency profile for a print gets a report that never mentions color and
never says why.

So the flags below are not decoration. They are the things a person choosing from
a list needs told at the moment they choose, and they were previously computed
inside the CLI's print loop where no other front end could reach them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from deltachrome.metrics.base import MetricSpec
from deltachrome.profiles.schema import (
    DEFINITIONS_DIR,
    AnalysisProfile,
    available_profiles,
    is_local,
    load_profile,
    load_profile_file,
)

#: Group heading for a guideline written here rather than published elsewhere.
#: *Guideline* is the word this application uses for what a run is judged against,
#: and a heading reading "specifications" made this the one list that used another.
LOCAL_GROUP = "LOCAL GUIDELINE"

#: Metamorfoze's levels in increasing strictness, to match how star levels already
#: read (1-star to 4-star). Alphabetical would list them Extra Light, Full, Light,
#: which puts the strictest level in the middle and reads as meaningless.
_METAMORFOZE_STRICTNESS = {
    "Metamorfoze Extra Light": 1,
    "Metamorfoze Light": 2,
    "Metamorfoze Full": 3,
}


@dataclass(frozen=True)
class ProfileSummary:
    """One profile as it should appear in a list."""

    profile_id: str
    display_name: str
    body: str | None
    edition: str | None
    material: str | None
    star_level: int | None
    #: Metamorfoze bands by a named level rather than a star count, so a summary
    #: that only knew about stars labeled all three of its levels "custom".
    quality_level: str | None
    local_criteria: bool
    flags: list[str] = field(default_factory=list)

    @property
    def level_label(self) -> str:
        if self.star_level:
            return f"{self.star_level}-star"
        if self.quality_level:
            return self.quality_level
        return "custom"

    @property
    def rank(self) -> int:
        """Position within a group, ordered least to most demanding."""
        if self.star_level:
            return int(self.star_level)
        return _METAMORFOZE_STRICTNESS.get(self.quality_level or "", 0)

    @property
    def group(self) -> tuple[str, str, str]:
        """Body, edition and material, as a sort/heading key.

        A local guideline is filed under **its own name**, and under nothing else.
        It used to be filed under the category and the standard it was derived
        *from*, on the argument that this is what somebody scanning the list is
        looking for -- so a guideline called "Test3" appeared as "LOCAL
        SPECIFICATIONS -- Photographic Negatives: 35mm up to 4x5 in", which is
        three facts about a table the operator may have started from only because
        it was convenient, and not the one fact they gave it.

        Once saved, a guideline is its own specification. The parent is kept in the
        record, because a run's criteria are part of the archive; it is not shown,
        for the same reason the builder's parent combo disappears the moment the
        guideline is saved. A connection to a derivation nobody meant anything by
        can only mislead.
        """
        if self.local_criteria:
            return (LOCAL_GROUP, "", self.display_name)
        # Empty rather than a placeholder: Metamorfoze does not band by material
        # category at all, and a presenter should omit the line rather than print
        # a dash that looks like missing data.
        return (self.body or "", self.edition or "", self.material or "")


def summarise(profile: AnalysisProfile) -> ProfileSummary:
    ref = profile.standard_reference
    derived = profile.local_criteria
    #: Deliberately not the parent's. A derived guideline carried its parent's
    #: material category, which then appeared in the picker as though the
    #: guideline were about that material.
    material = None if derived else ref.get("material_category")

    flags: list[str] = []
    if is_local(profile.profile_id) and not derived:
        # Worth saying only where it is not already evident: a local *derived*
        # guideline files itself under LOCAL GUIDELINE and says so by where it is,
        # where a local file carrying a published id is shadowing that standard and
        # nothing else on screen can tell you.
        flags.append("local")
    if ref.get("provisional"):
        flags.append("provisional in the standard")
    if not any(
        m.metric_id == "average_color_accuracy" for m in profile.metrics
    ):
        # The trap this module exists for.
        # Neutral wording: for FADGI this is a property of the material
        # category, for Metamorfoze of the level. Both need saying.
        #
        # Said of a local guideline too, which it was not while a guideline was
        # presented as its parent's derivative: a guideline is now its own
        # specification, so the trap is its own as well -- and a guideline built
        # from a transparency table inherits it without anything having been
        # decided about colour at all.
        flags.append("specifies no color accuracy metric")

    return ProfileSummary(
        profile_id=profile.profile_id,
        display_name=profile.display_name,
        body=ref.get("body"),
        edition=ref.get("document_version"),
        material=material,
        star_level=ref.get("star_level"),
        quality_level=ref.get("quality_level"),
        local_criteria=derived,
        flags=flags,
    )


def catalogue(profile_ids: Iterable[str] | None = None) -> list[ProfileSummary]:
    """Every installed profile, summarised, ordered as a list should read."""
    ids = list(profile_ids) if profile_ids is not None else available_profiles()
    summaries = [summarise(load_profile(pid)) for pid in ids]
    summaries.sort(key=lambda s: (s.group, s.rank, s.profile_id))
    return summaries


@lru_cache(maxsize=8)
def metric_templates(body: str | None = None) -> dict[str, MetricSpec]:
    """Every metric the published tables declare, in the shape they declare it.

    The question this answers: *what would this metric look like if a profile that
    does not declare it were to?* A site building its own specification from FADGI
    X-Ray -- which states no colour accuracy metric at all -- should be able to add
    one, and a metric needs more than a threshold to run: a capability, an
    aggregation, a unit, and whatever its algorithm carries.

    None of that is invented here. Each is taken from the profiles that *do*
    declare the metric, by majority, and `body` is preferred over the rest --
    because the variation between bodies is not spelling. Metamorfoze's exposure
    is a plain `dL*` judged down to a lightness floor where FADGI's is `dL*2000`
    over the whole ramp, and the two are different numbers away from mid-grey.
    Taking one body's shape into the other's guideline would change the
    measurement while looking like a tidy-up.

    Only shipped definitions are read, so this is stable and safe to cache: a
    user's own guidelines are derived from these and cannot introduce a shape the
    tables do not have.
    """
    shapes: dict[str, list[tuple[bool, MetricSpec]]] = {}
    for path in sorted(DEFINITIONS_DIR.glob("*.json")):
        profile = load_profile_file(path)
        same = bool(body) and profile.body == body
        for spec in profile.metrics:
            shapes.setdefault(spec.metric_id, []).append((same, spec))

    templates: dict[str, MetricSpec] = {}
    for metric_id, found in shapes.items():
        preferred = [spec for same, spec in found if same] or [s for _, s in found]
        counted = Counter(
            (s.requires_capability, s.aggregation, s.units) for s in preferred
        )
        shape = counted.most_common(1)[0][0]
        templates[metric_id] = next(
            s for s in preferred
            if (s.requires_capability, s.aggregation, s.units) == shape
        )
    return templates


def vocabulary_of(profile: AnalysisProfile) -> str | None:
    """Which body's metric shapes this profile speaks, read from its own metrics.

    Needed because a metric needs a capability, an aggregation and a unit to run,
    and the bodies do not agree on them: Metamorfoze states exposure as a plain
    `dL*` judged down to a lightness floor where FADGI states `dL*2000` over the
    whole ramp. So when somebody *adds* a metric to a guideline, which shape they
    get changes the measurement.

    A guideline used to answer this with its parent's ``body``. It no longer has a
    parent, and the answer is in the guideline anyway: a profile that states
    thirteen metrics has stated thirteen votes for the vocabulary it is written
    in. Counting them is inference, but it is inference from the object's own
    contents rather than from a table it was once beside -- which is the whole
    distinction a severed guideline rests on.

    ``None`` when nothing is declared, which is what the majority template already
    means.
    """
    declared = {
        spec.metric_id: (spec.requires_capability, spec.aggregation, spec.units)
        for spec in profile.metrics
    }
    if not declared:
        return None
    bodies = {
        load_profile_file(path).body
        for path in sorted(DEFINITIONS_DIR.glob("*.json"))
    }
    best, score = None, -1
    for body in sorted(b for b in bodies if b):
        templates = metric_templates(body)
        agreed = sum(
            1 for metric_id, shape in declared.items()
            if metric_id in templates
            and (templates[metric_id].requires_capability,
                 templates[metric_id].aggregation,
                 templates[metric_id].units) == shape
        )
        if agreed > score:
            best, score = body, agreed
    return best


@lru_cache(maxsize=1)
def metric_order() -> tuple[str, ...]:
    """One order for every guideline, so two of them can be read side by side.

    Taken from the most complete published profile -- the order a real table puts
    its rows in -- with anything it does not declare appended. Not alphabetical:
    the tables group a file's properties, then its colour, then its resolution,
    and that grouping is what a person reads down.
    """
    profiles = [load_profile_file(p) for p in sorted(DEFINITIONS_DIR.glob("*.json"))]
    fullest = max(profiles, key=lambda p: len(p.metrics))
    order = [spec.metric_id for spec in fullest.metrics]
    order += sorted(set(metric_templates()) - set(order))
    return tuple(order)


def grouped(profile_ids: Iterable[str] | None = None):
    """The catalogue as ``[(group_key, [summary, ...]), ...]``."""
    groups: dict[tuple[str, str, str], list[ProfileSummary]] = {}
    for summary in catalogue(profile_ids):
        groups.setdefault(summary.group, []).append(summary)
    return sorted(groups.items(), key=lambda kv: kv[0])
