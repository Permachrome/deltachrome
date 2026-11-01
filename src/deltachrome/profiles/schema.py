"""Analysis profile schema and registry.

A profile bundles: which metrics run, which algorithm and parameters each uses,
and the pass/fail thresholds. Because it references published tables rather than
embedding logic, tracking a spec revision means editing data, not code.

``standard_reference`` is locked. A user editing thresholds on a profile called
"FADGI 3-Star" would otherwise produce a document that claims conformance to a
standard it no longer matches -- so editing a locked field forks the profile to a
new id instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dataclass_field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from deltachrome.errors import ProfileError
from deltachrome.metrics.base import _CONDITION_RE, _RANGE_RE, MetricSpec
from deltachrome.targets.schema import KNOWN_CAPABILITIES

DEFINITIONS_DIR = Path(__file__).parent / "definitions"
#: Where a user's own profiles live. Overridable, because a shared lab drive is a
#: perfectly reasonable place to keep the specification a whole department works to.
PROFILE_DIR_ENV = "DELTACHROME_PROFILE_DIR"

#: Where Deltachrome keeps its own things. The guideline folder is one of three
#: that default under it, and it is read here rather than only in the frontend so
#: that redirecting the home redirects *everything* -- a test suite, or a portable
#: install, must not be able to move two folders and leave the third writing into
#: a person's own. ``DELTACHROME_PROFILE_DIR`` still overrides it, being the
#: narrower instruction.
HOME_ENV = "DELTACHROME_HOME"


#: Set for this process by :func:`set_profile_dir`. Takes precedence over the
#: default and not over the environment variable, which is how a shared lab drive
#: is pinned regardless of what any window did.
_SESSION_PROFILE_DIR: Path | None = None


def user_profile_dir() -> Path:
    from_env = os.environ.get(PROFILE_DIR_ENV)
    if from_env:
        return Path(from_env)
    if _SESSION_PROFILE_DIR is not None:
        return _SESSION_PROFILE_DIR
    from deltachrome import home
    return home.default_folder(home.GUIDELINES)


def set_profile_dir(path: str | Path | None) -> Path:
    """Point the guideline folder somewhere else for this process.

    A department works to one specification, and the folder holding it is exactly
    the sort of thing that belongs on a shared drive rather than in one operator's
    home -- which the environment variable already allowed and nothing could ask
    for. Remembering the choice between sessions is the caller's business: this
    module has no settings file and should not acquire one.
    """
    global _SESSION_PROFILE_DIR
    _SESSION_PROFILE_DIR = Path(path) if path else None
    _index.cache_clear()
    return user_profile_dir()


def compare_conditions(before: str | None, after: str | None) -> str:
    """Which way a threshold moved: ``relaxed``, ``tightened``, or ``changed``.

    Direction is worth knowing and is not worth over-claiming. A ceiling raised is a
    relaxation and a floor raised is a tightening, which is mechanical; changing a
    one-sided condition into a band, or vice versa, is neither, and saying so is
    better than picking one.

    Note that the direction does **not** decide whether the result may still claim
    the standard. It may not, either way. A tightened profile is not the standard
    any more than a relaxed one is, and "stricter, so it still counts" is a rule
    that is easy to game and awkward to explain.
    """
    if before is None or after is None:
        return "changed"
    if before == after:
        return "unchanged"

    band_before, band_after = _RANGE_RE.match(before), _RANGE_RE.match(after)
    if band_before and band_after:
        width_before = float(band_before["high"]) - float(band_before["low"])
        width_after = float(band_after["high"]) - float(band_after["low"])
        if width_after > width_before:
            return "relaxed"
        return "tightened" if width_after < width_before else "changed"

    one_before, one_after = _CONDITION_RE.match(before), _CONDITION_RE.match(after)
    if one_before and one_after and one_before["op"][0] == one_after["op"][0]:
        moved = float(one_after["threshold"]) - float(one_before["threshold"])
        if moved == 0:
            return "changed"
        looser_upward = one_before["op"].startswith("<")
        return "relaxed" if (moved > 0) == looser_upward else "tightened"
    return "changed"


def _listed(values: Any) -> str:
    return ", ".join(str(v) for v in (values or [])) or "nothing"


def describe_departures(parent_name: str, departures: list[dict[str, Any]]) -> str:
    """One sentence naming every way this profile differs from its parent."""
    if not departures:
        # A guideline that states the standard's own numbers is still a *local
        # specification* and still forfeits the conformance claim, so it needs a
        # sentence rather than an empty one -- and the sentence has to say the
        # thing a reader would otherwise assume it was hiding.
        return f"{parent_name}, adopted unchanged"
    parts = []
    for entry in departures:
        if entry["change"] == "omitted":
            parts.append(f"{entry['metric_id']} omitted")
        elif entry["change"] == "unjudged":
            parts.append(f"{entry['metric_id']} measured but not judged")
        elif entry["change"] == "added":
            parts.append(
                f"{entry['metric_id']} added, {entry['to']}" if entry.get("to")
                else f"{entry['metric_id']} added, measured but not judged"
            )
        elif entry["change"] == "algorithm":
            # A changed list of permitted values is the common case and deserves
            # to be readable: "algorithm changed" says nothing a person can act
            # on, where "permits TIFF rather than TIFF, JPEG 2000" is the whole
            # of what a site decided.
            before, after = entry.get("from") or {}, entry.get("to") or {}
            if set(after) == {"allowed"}:
                parts.append(
                    f"{entry['metric_id']} permits {_listed(after.get('allowed'))} "
                    f"rather than {_listed(before.get('allowed'))}"
                )
            else:
                parts.append(f"{entry['metric_id']} algorithm changed")
        else:
            parts.append(
                f"{entry['metric_id']} {entry['direction']} "
                f"from '{entry['from']}' to '{entry['to']}'"
            )
    return f"{parent_name}, with " + "; ".join(parts)


@dataclass
class AnalysisProfile:
    profile_id: str
    display_name: str
    standard_reference: dict[str, Any]
    metrics: list[MetricSpec]
    required_capabilities: frozenset[str] = frozenset()
    optional_capabilities: frozenset[str] = frozenset()
    overall_pass_rule: str = "all applicable metrics must pass"
    editable: bool = True
    locked_fields: tuple[str, ...] = ("standard_reference",)
    notes: str | None = None
    source_path: Path | None = None
    #: What a derivation changed, and the sentence describing it. Held on the
    #: object and **never written to the file**: a departure reads "noise relaxed
    #: from 'value < 3' to 'value < 9'", which means nothing except beside the
    #: table it departs from, and a saved guideline no longer records that table.
    #: They exist for the moment of deriving -- the CLI prints them, the builder
    #: offers them on the save button -- and a guideline read back from disk has
    #: neither, which is the point rather than a loss. What survives the save is
    #: ``inherited``.
    departures_made: list[dict[str, Any]] = dataclass_field(default_factory=list)
    departure_summary: str | None = None

    @property
    def body(self) -> str | None:
        return self.standard_reference.get("body")

    def metric(self, metric_id: str) -> MetricSpec:
        for spec in self.metrics:
            if spec.metric_id == metric_id:
                return spec
        raise ProfileError(f"Profile {self.profile_id!r} has no metric {metric_id!r}")

    def fork(self, new_id: str, display_name: str | None = None) -> "AnalysisProfile":
        """Copy this profile under a new id, dropping the standard's identity.

        Used when a user edits a locked field: the numbers may be whatever they
        need, but the resulting document must not present itself as the standard.
        """
        if new_id == self.profile_id and not self.local_criteria:
            # Re-saving a local guideline under its own id is an amendment, which
            # is what re-editing one in the builder is. Re-saving a *published*
            # standard under its own id would overwrite that standard's identity
            # with a fork's, and must still refuse.
            raise ProfileError("A fork must use a different profile_id.")
        derived = {
            "local": True,
            "notice": (
                "Local guideline. Its criteria are this site's own and are not "
                "those of any published standard; results must not be described "
                "as conforming to one."
            ),
        }
        return replace(
            self,
            profile_id=new_id,
            display_name=display_name or f"{self.display_name} (custom)",
            standard_reference=derived,
            locked_fields=(),
            source_path=None,
        )

    def derive(
        self,
        new_id: str,
        *,
        thresholds: dict[str, str] | None = None,
        algorithms: dict[str, dict[str, Any]] | None = None,
        omit: Sequence[str] = (),
        unjudge: Sequence[str] = (),
        add: dict[str, MetricSpec] | None = None,
        display_name: str | None = None,
        notes: str | None = None,
        allow_identical: bool = False,
        inherited: Sequence[str] | None = None,
    ) -> "AnalysisProfile":
        """A local specification built from this one.

        The case this is for: projects are rarely uniformly one star level. "Four
        star for 35mm photographs, except sampling frequency, which we can set at
        2000 ppi" is a real and reasonable thing to want, and it is not FADGI
        4-star. So the result carries the parent's identity as provenance rather
        than as a claim, and records exactly where it departs.

        The departure list is the point. "Custom profile" tells a reader nothing;
        "4-star with sampling_frequency relaxed from 2940 to 2000 ppi" is the
        sentence that belongs in a project's documentation, and the one someone will
        want six months later when they ask what "our 4-star" meant.

        ``unjudge`` is the third thing a site does to a table, and it is not the
        same as omitting. A metric it names still **runs and reports its number**;
        it simply carries no criterion, so it reads MEASURED rather than PASS or
        FAIL -- which is what a department wants for a figure it tracks but does
        not gate on. Omitting removes the metric and the reading with it.

        Whatever shape the criterion had is what gets cleared: a threshold, a
        threshold banded by a run input, and the list of permitted values the
        file-level metrics carry in their algorithm. All three are judgements, and
        leaving one behind would produce a metric that says it is not judged and
        then judges.

        ``allow_identical`` permits a derivation that changes nothing. The refusal
        is right for a caller that meant to change something and did not -- which
        is what the command line and the API are -- and wrong for a site adopting a
        table as its own starting point: *"our standard is FADGI 3-star, for now"*
        is a real thing to record, and the guideline is where the site's later
        decisions will go. It costs the conformance claim either way, which is why
        it was refused and is also why refusing is not this module's call to make.

        ``add`` is the fourth, and it is the one that lets a site's specification
        be *more* than the table it came from. FADGI's X-Ray category states no
        colour accuracy metric at all; a department shooting radiographs on a rig
        it also uses for prints may reasonably want one measured anyway. The spec
        comes from the caller rather than being invented here --
        :func:`deltachrome.profiles.catalogue.metric_templates` takes it from the
        profiles that do declare it -- because a metric needs a capability, an
        aggregation and a unit to run, and guessing any of those would be a
        measurement this module made up.
        """
        thresholds = dict(thresholds or {})
        algorithms = dict(algorithms or {})
        omit = list(omit)
        unjudge = list(unjudge)
        add = dict(add or {})

        known = {spec.metric_id for spec in self.metrics}
        named = set(thresholds) | set(algorithms) | set(omit) | set(unjudge)
        unknown = sorted(named - known)
        if unknown:
            raise ProfileError(
                f"{self.profile_id!r} has no metric(s) {unknown}, so they cannot be "
                f"overridden, omitted or unjudged. It declares: {sorted(known)}."
            )
        already = sorted(set(add) & known)
        if already:
            raise ProfileError(
                f"{self.profile_id!r} already declares {already}, so they cannot be "
                "added. Give them a threshold instead."
            )
        mislabelled = sorted(k for k, v in add.items() if k != v.metric_id)
        if mislabelled:
            raise ProfileError(
                f"Added metric(s) {mislabelled} are filed under a different id than "
                "the spec they carry."
            )
        contradictory = sorted(
            set(unjudge) & (set(thresholds) | set(omit) | set(algorithms))
        )
        if contradictory:
            raise ProfileError(
                f"{contradictory} cannot be unjudged and also given a threshold, "
                "an algorithm or an omission. Unjudging removes the criterion; "
                "the two instructions disagree about what the metric should do."
            )

        metrics: list[MetricSpec] = []
        departures: list[dict[str, Any]] = []
        for spec in self.metrics:
            if spec.metric_id in omit:
                departures.append({
                    "metric_id": spec.metric_id,
                    "change": "omitted",
                    "from": spec.pass_condition,
                    "to": None,
                    "direction": "removed",
                })
                continue
            changed = replace(spec)
            if spec.metric_id in unjudge:
                departures.append({
                    "metric_id": spec.metric_id,
                    "change": "unjudged",
                    "from": spec.pass_condition or (
                        ", ".join(str(a) for a in spec.algorithm.get("allowed", []))
                        or None
                    ),
                    "to": None,
                    "direction": "no longer judged",
                })
                algorithm = dict(changed.algorithm)
                algorithm.pop("allowed", None)
                metrics.append(replace(
                    changed,
                    pass_condition=None,
                    pass_condition_by=None,
                    algorithm=algorithm,
                ))
                continue
            if spec.metric_id in thresholds:
                new_condition = thresholds[spec.metric_id]
                departures.append({
                    "metric_id": spec.metric_id,
                    "change": "threshold",
                    "from": spec.pass_condition,
                    "to": new_condition,
                    "direction": compare_conditions(spec.pass_condition, new_condition),
                })
                changed = replace(changed, pass_condition=new_condition)
            if spec.metric_id in algorithms:
                merged = {**changed.algorithm, **algorithms[spec.metric_id]}
                departures.append({
                    "metric_id": spec.metric_id,
                    "change": "algorithm",
                    "from": {k: changed.algorithm.get(k) for k in algorithms[spec.metric_id]},
                    "to": algorithms[spec.metric_id],
                    "direction": "changed",
                })
                changed = replace(changed, algorithm=merged)
            metrics.append(changed)

        for metric_id in sorted(add):
            spec = add[metric_id]
            departures.append({
                "metric_id": metric_id,
                "change": "added",
                "from": None,
                "to": spec.pass_condition,
                "direction": "added",
            })
            metrics.append(spec)

        if not departures and not allow_identical:
            raise ProfileError(
                f"Deriving {new_id!r} from {self.profile_id!r} would change nothing. "
                "Use the profile itself, which can still claim the standard."
            )

        # Every criterion the operator did not touch, which is nearly all of them:
        # moving one threshold on a FADGI profile carries eighteen metrics across
        # unexamined, algorithm parameters and all. ``inherited`` is what lets a
        # guideline tell a criterion it chose from one it was handed, once the
        # table it was handed them by is no longer recorded.
        #
        # ``inherited`` narrows and never grows. A caller passing the previous set
        # -- which is what amending a saved guideline does -- keeps only those
        # still untouched, because a row the operator has now edited is a decision
        # from here on and does not become inherited again by being left alone
        # next time.
        touched = {entry["metric_id"] for entry in departures}
        carried = {spec.metric_id for spec in metrics} - touched
        if inherited is not None:
            carried &= set(inherited)

        forked = self.fork(new_id, display_name=display_name)
        forked.standard_reference["inherited"] = sorted(carried)
        return replace(
            forked,
            # Stripped of the transcription notes, at both levels, and this is
            # accuracy before it is severance. A published profile's own note
            # says where its numbers were read from -- "Thresholds read from
            # FADGI 3rd Edition, May 2023, section 3.8 (Prints and
            # Photographs), pages 50-51" -- and carrying that into a guideline
            # that has moved a threshold makes the file assert something untrue
            # about itself. The same holds one level down: "FADGI requires gain
            # between 0.8 and 1.1" is false on a metric whose band the operator
            # has just changed.
            #
            # Dropped for the inherited metrics too, where it would still be
            # true. Nothing renders `MetricSpec.notes` -- the console report's
            # notes are `MetricResult.notes`, which plugins write at run time --
            # so the only reader is somebody opening the JSON, and they are
            # exactly the reader this is severed for.
            metrics=[replace(spec, notes=None) for spec in metrics],
            notes=notes,
            departures_made=departures,
            departure_summary=describe_departures(self.display_name, departures),
        )

    def to_dict(self) -> dict[str, Any]:
        """The document form, so a derived profile can be saved and re-used."""
        return {
            # 2 severed the parent: a local guideline records ``local`` and
            # ``inherited`` where it used to record ``derived_from``,
            # ``original_standard_reference``, ``departures`` and ``summary``.
            # Version 1 files still load -- ``local_criteria`` reads the old key.
            "$schema_version": 2,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "standard_reference": self.standard_reference,
            "applicable_target_capabilities_required": sorted(self.required_capabilities),
            "applicable_target_capabilities_optional": sorted(self.optional_capabilities),
            "metrics": [
                {
                    k: v for k, v in {
                        "metric_id": spec.metric_id,
                        "requires_capability": spec.requires_capability,
                        "algorithm": spec.algorithm or None,
                        "aggregation": spec.aggregation,
                        "pass_condition": spec.pass_condition,
                        "pass_condition_by": (
                            spec.pass_condition_by.to_dict()
                            if spec.pass_condition_by else None
                        ),
                        "units": spec.units,
                        "notes": spec.notes,
                    }.items() if v is not None
                }
                for spec in self.metrics
            ],
            "overall_pass_rule": self.overall_pass_rule,
            "profile_editable": self.editable,
            "profile_locked_fields": list(self.locked_fields),
            "notes": self.notes,
        }

    def save(self, directory: str | Path | None = None) -> Path:
        """Write to the user profile directory, or a directory of your choosing."""
        directory = Path(directory) if directory else user_profile_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.profile_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        _index.cache_clear()
        return path

    @property
    def local_criteria(self) -> bool:
        """Whether these criteria are a site's own rather than a published table.

        The one fact everything downstream actually needed. It was carried by
        ``derived_from``, whose *value* -- the parent's profile id -- nothing read
        and everything could leak. Guidelines written before that key was dropped
        still answer here, because a profile file is a user's file and a rename
        must not make one unreadable, exactly as a target definition still accepts
        the old ``centre`` spelling.
        """
        return bool(
            self.standard_reference.get("local")
            or self.standard_reference.get("derived_from")
        )

    @property
    def inherited(self) -> list[str]:
        """Metrics carried over from the table this was built from, untouched.

        What ``derived_from`` was really being kept for. Changing one threshold
        freezes every other criterion the parent declared -- eighteen of them on a
        FADGI profile, algorithm parameters included -- and nothing on the builder
        screen says so, because from the operator's side they changed one number.
        This is the list they did *not* decide, recorded in their own guideline's
        terms so that it survives the parent being severed.

        **Reconstructed for a guideline saved before it was recorded.** Those
        files state ``departures``, which names precisely what the operator did
        touch, so the complement is exact rather than a guess -- and getting it
        from them matters, because otherwise re-saving one would write an empty
        list, which does not mean "unknown". It means *every criterion here was
        chosen deliberately*, which is the flattering direction and the one this
        must never drift in.

        A local profile stating neither was written by hand, line by line, and an
        empty list is then the true answer.
        """
        stated = self.standard_reference.get("inherited")
        if stated is not None:
            return list(stated)
        departures = self.standard_reference.get("departures")
        if departures is None:
            return []
        touched = {entry["metric_id"] for entry in departures}
        return sorted({spec.metric_id for spec in self.metrics} - touched)

    @property
    def departures(self) -> list[dict[str, Any]]:
        # The field when this object came from a derivation; the old key when it
        # came off disk as a guideline saved before departures stopped being
        # written. Both are empty for a guideline saved since.
        return self.departures_made or self.standard_reference.get("departures") or []

    def capability_report(self, capabilities: frozenset[str]) -> dict[str, list[str]]:
        """Which required/optional capabilities a target does and does not have."""
        return {
            "required_present": sorted(self.required_capabilities & capabilities),
            "required_missing": sorted(self.required_capabilities - capabilities),
            "optional_present": sorted(self.optional_capabilities & capabilities),
            "optional_missing": sorted(self.optional_capabilities - capabilities),
        }


def profile_from_dict(data: dict[str, Any], source_path: Path | None = None) -> AnalysisProfile:
    try:
        metrics = [MetricSpec.from_dict(m) for m in data["metrics"]]
    except KeyError as exc:
        raise ProfileError(f"Profile definition is missing required key {exc}") from exc

    # The same closed vocabulary the targets are held to, checked from the other
    # side. A profile requiring a capability no target can ever declare would make
    # its metric permanently N/A, which reads as every target being unsuitable.
    named = {m.requires_capability for m in metrics if m.requires_capability}
    named |= set(data.get("applicable_target_capabilities_required", []))
    named |= set(data.get("applicable_target_capabilities_optional", []))
    unknown = sorted(named - KNOWN_CAPABILITIES)
    if unknown:
        raise ProfileError(
            f"Profile {data.get('profile_id')!r} names unknown capability/capabilities "
            f"{unknown}. Known: {sorted(KNOWN_CAPABILITIES)}."
        )

    seen = {m.metric_id for m in metrics}
    if len(seen) != len(metrics):
        raise ProfileError(f"Profile {data.get('profile_id')!r} lists a metric_id twice.")

    return AnalysisProfile(
        profile_id=data["profile_id"],
        display_name=data.get("display_name", data["profile_id"]),
        standard_reference=data.get("standard_reference", {}),
        metrics=metrics,
        required_capabilities=frozenset(data.get("applicable_target_capabilities_required", [])),
        optional_capabilities=frozenset(data.get("applicable_target_capabilities_optional", [])),
        overall_pass_rule=data.get("overall_pass_rule", "all applicable metrics must pass"),
        editable=data.get("profile_editable", True),
        locked_fields=tuple(data.get("profile_locked_fields", ("standard_reference",))),
        notes=data.get("notes"),
        source_path=source_path,
    )


def load_profile_file(path: str | Path) -> AnalysisProfile:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path}: invalid JSON -- {exc}") from exc
    return profile_from_dict(data, source_path=path)


@lru_cache(maxsize=1)
def _index() -> dict[str, Path]:
    """Shipped profiles, then the user's own -- which may shadow a shipped id.

    Shadowing is allowed on purpose: a site that has decided its own
    "fadgi_3star_..." means something particular should get what it decided, not a
    silent disagreement between what it named and what it got. ``is_local`` says
    which one answered.
    """
    found: dict[str, Path] = {}
    directories = [DEFINITIONS_DIR]
    user = user_profile_dir()
    if user.is_dir():
        directories.append(user)
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            try:
                found[json.loads(path.read_text(encoding="utf-8"))["profile_id"]] = path
            except (json.JSONDecodeError, KeyError) as exc:
                raise ProfileError(f"{path.name}: not a usable profile -- {exc}") from exc
    return found


def is_local(profile_id: str) -> bool:
    """Whether this id is answered by a user profile rather than a shipped one."""
    path = _index().get(profile_id)
    return path is not None and path.parent != DEFINITIONS_DIR


def available_profiles() -> list[str]:
    return sorted(_index())


def load_profile(profile_id_or_path: str | Path) -> AnalysisProfile:
    candidate = Path(profile_id_or_path)
    if candidate.suffix == ".json" or candidate.exists():
        return load_profile_file(candidate)
    index = _index()
    key = str(profile_id_or_path)
    if key not in index:
        raise ProfileError(
            f"Unknown profile {key!r}. Installed profiles: {available_profiles()}. "
            "Pass a path to a .json file to use a custom profile."
        )
    return load_profile_file(index[key])
