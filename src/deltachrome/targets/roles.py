"""Resolve patch roles from real measured reference data.

Which ColorChecker SG patches are the repeated near-white neutrals used for
lightness uniformity is not given by any published layout.

Rather than guess the remainder from a product photograph, Deltachrome derives
them from the per-unit CGATS file that must be imported anyway. This follows the
project's own working principle -- ground everything in real data -- and it
generalises: any chart with a repeated neutral scattered across its area gets the
same treatment without a hand-written patch list.

The rules:

* Roles written into the target definition are authoritative and are never
  overridden. For the SG that means the twelve ramp patches E5-J5 and E6-J6,
  which are corroborated by FADGI's own OpenDICE documentation.
* Remaining patches with chroma at or below ``achromatic_chroma_max`` are neutral.
* Among the neutrals, *every* group of mutually near-identical values with at
  least ``MIN_UNIFORMITY_CLUSTER`` members becomes ``achromatic_border`` -- the
  uniformity sample set. Everything else neutral is ``achromatic_other``;
  everything else is ``chromatic``.

  Taking only the single largest group, as an earlier version did, throws away real
  measurements: a ColorChecker SG's border is three interleaved neutral families
  (near-white, mid-gray, near-black), each repeated a dozen or more times around the
  chart. Keeping only the biggest discarded 28 of 45 usable sample points and
  measured uniformity at one lightness instead of three.

Every number that went into the decision is returned in :class:`RoleResolution`
and printed in the report, so the classification is inspectable rather than a
black box.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from deltachrome.errors import MissingReferenceError, TargetError
from deltachrome.imaging.color import CIE_D50, chroma_ab, delta_e, xyz_to_lab
from deltachrome.reference.cgats import MeasurementSet
from deltachrome.targets.schema import Patch, TargetDefinition
from deltachrome.text import plural

DEFAULT_ACHROMATIC_CHROMA_MAX = 4.0
DEFAULT_UNIFORMITY_TOLERANCE = 1.5
MIN_UNIFORMITY_CLUSTER = 4


@dataclass
class RoleResolution:
    """The outcome of role resolution, with the evidence that produced it."""

    target: TargetDefinition
    counts: dict[str, int]
    declared: list[str]
    uniformity_patches: list[str] = field(default_factory=list)
    uniformity_mean_lab: tuple[float, float, float] | None = None
    uniformity_spread_de: float | None = None
    uniformity_families: list[dict] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"Role resolution for {self.target.target_id}:"]
        for role, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {role:<20} {n:>4}")
        if self.declared:
            lines.append(
                f"  ({len(self.declared)} role(s) taken from the target definition, not inferred)"
            )
        if self.uniformity_patches:
            lines.append(
                f"  uniformity set: {len(self.uniformity_patches)} patches across "
                f"{len(self.uniformity_families)} neutral family/families"
            )
            for family in self.uniformity_families:
                L, a, b = family["mean_lab"]
                lines.append(
                    f"    L*{L:6.2f} a{a:6.2f} b{b:6.2f}  n={family['count']:>2}  "
                    f"reference spread {family['spread_de']:.2f} dE2000"
                )
        if self.unmeasured:
            lines.append(
                f"  {len(self.unmeasured)} patch(es) absent from the reference file: "
                + ", ".join(self.unmeasured[:12])
                + (" ..." if len(self.unmeasured) > 12 else "")
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _lab_of(measurement, white_point):
    """One measurement's Lab, from whichever representation the file carries.

    **XYZ is not a lesser form of Lab and has to be read.** Reading only ``lab``
    meant a file carrying XYZ -- or spectra, which are integrated to XYZ when the
    file is read -- had every patch counted as unmeasured, and the run stopped with
    a message about patch ids that all matched perfectly.

    No chromatic adaptation is applied here, deliberately. Roles are a coarse
    classification: whether a patch is neutral, and which neutrals repeat. The
    pipeline's own reference table adapts, reports that it did, and is what every
    reported number is measured against; doing it twice in two places is how the
    two come to disagree.
    """
    if measurement is None:
        return None
    if measurement.lab is not None:
        return np.asarray(measurement.lab, dtype=np.float64)
    if measurement.xyz is not None:
        xyz = np.asarray(measurement.xyz, dtype=np.float64)
        # CGATS XYZ is conventionally scaled 0-100.
        if float(np.max(xyz)) > 3.0:
            xyz = xyz / 100.0
        return xyz_to_lab(xyz, white_point=white_point)
    return None


def resolve_roles(
    target: TargetDefinition,
    reference: MeasurementSet,
    *,
    achromatic_chroma_max: float = DEFAULT_ACHROMATIC_CHROMA_MAX,
    uniformity_tolerance: float = DEFAULT_UNIFORMITY_TOLERANCE,
    white_point: tuple[float, float, float] = CIE_D50,
) -> RoleResolution:
    """Assign roles to every ``unassigned`` patch using the reference measurements."""
    labs: dict[str, np.ndarray] = {}
    unmeasured: list[str] = []
    named = 0
    for patch in target.patches:
        measurement = reference.get(patch.patch_id)
        if measurement is not None:
            named += 1
        lab = _lab_of(measurement, white_point)
        if lab is None:
            unmeasured.append(patch.patch_id)
            continue
        labs[patch.patch_id] = lab

    if not target.patches:
        # A target may carry no patches at all -- a dedicated resolution chart is
        # slanted edges and a scale and nothing else. It needs no reference data
        # because it makes no colorimetric claim, and refusing to resolve roles for
        # it would make the whole class of target unusable.
        return RoleResolution(target=target, counts={}, declared=[], notes=[])

    if target.reference_data.get("source_type") == "none":
        # The target carries patches and says outright that none of them is
        # compared against anything -- the QA-62 prints a twenty-step ring and is
        # judged entirely on its slanted square. There is nothing to resolve roles
        # *from*, and nothing to resolve: a definition making that claim has to
        # declare every role itself, which is checked here rather than left to
        # produce an unassigned patch nothing ever selects.
        unassigned = [p.patch_id for p in target.patches if p.role == "unassigned"]
        if unassigned:
            raise TargetError(
                f"{target.target_id} declares no reference source, so nothing can "
                f"resolve a role from measured color -- but {len(unassigned)} of its "
                f"patches are left unassigned: {', '.join(unassigned[:8])}"
                + (" ..." if len(unassigned) > 8 else "")
            )
        return RoleResolution(
            target=target,
            counts=dict(Counter(p.role for p in target.patches)),
            declared=[p.patch_id for p in target.patches],
            notes=[],
        )

    if not labs:
        # Two different faults, and telling them apart is the difference between
        # "you have the wrong file" and "this file needs something Deltachrome
        # cannot do". Reporting the first for the second sent a reader looking for
        # a chart mismatch that was not there, on a file naming all 864 patches
        # correctly and carrying spectra its reader had not been taught to read.
        if not named:
            raise MissingReferenceError(
                f"Reference file covers none of {target.target_id}'s "
                f"{len(target.patches)} patch ids. Check that the file is for this "
                "model of chart, and that its SAMPLE_NAME values look like "
                f"{target.patch_ids[:3]}."
            )
        raise MissingReferenceError(
            f"Reference file names {named} of {target.target_id}'s "
            f"{len(target.patches)} patches, so it is the right chart -- but none of "
            "them carries anything Deltachrome can read a color from. It needs Lab, "
            "XYZ, or spectra it can integrate."
        )

    declared = [p.patch_id for p in target.patches if p.role != "unassigned"]
    notes: list[str] = []

    # Split unassigned patches into neutral and chromatic by measured chroma.
    neutral_ids: list[str] = []
    resolved: dict[str, str] = {}
    for patch in target.patches:
        if patch.role != "unassigned":
            continue
        lab = labs.get(patch.patch_id)
        if lab is None:
            continue
        if float(chroma_ab(lab)) <= achromatic_chroma_max:
            neutral_ids.append(patch.patch_id)
        else:
            resolved[patch.patch_id] = "chromatic"

    families = _repeated_clusters(neutral_ids, labs, tolerance=uniformity_tolerance)

    declared_uniformity = [
        p.patch_id for p in target.patches if p.role == "achromatic_border"
    ]
    if not families and declared_uniformity:
        # The target names its own uniformity set. Nothing has to be inferred from
        # reference data, and on a chart that prints no values for those patches
        # nothing could be -- so this is not the "none found" case.
        uniformity = sorted(declared_uniformity)
        spread = mean_lab = None
        family_summary: list[dict[str, object]] = []
    elif not families:
        uniformity = []
        spread = mean_lab = None
        family_summary = []
    else:
        uniformity = sorted(pid for family in families for pid in family["members"])
        spread = max(float(f["spread"]) for f in families)
        mean_lab = families[0]["mean"]
        family_summary = [
            {
                "mean_lab": f["mean"],
                "count": len(f["members"]),
                "spread_de": round(float(f["spread"]), 3),
                "members": sorted(f["members"]),
            }
            for f in families
        ]

    for pid in neutral_ids:
        resolved[pid] = "achromatic_border" if pid in uniformity else "achromatic_other"

    patches: list[Patch] = []
    for patch in target.patches:
        if patch.role != "unassigned":
            patches.append(patch)
        elif patch.patch_id in resolved:
            patches.append(patch.with_role(resolved[patch.patch_id]))
        else:
            patches.append(patch)

    patches, ramp_note = _order_ramp(patches, labs)
    if ramp_note:
        notes.append(ramp_note)

    new_target = target.with_patches(patches)
    counts: dict[str, int] = {}
    for patch in patches:
        counts[patch.role] = counts.get(patch.role, 0) + 1

    stranded = [pid for pid in unmeasured if new_target[pid].role == "unassigned"]
    if stranded:
        notes.append(
            f"{plural(len(stranded), 'patch', 'patches')} missing from the reference "
            "file and skipped."
        )

    return RoleResolution(
        target=new_target,
        counts=counts,
        declared=declared,
        uniformity_patches=sorted(uniformity),
        uniformity_mean_lab=mean_lab,
        uniformity_spread_de=spread,
        uniformity_families=family_summary,
        unmeasured=unmeasured,
        notes=notes,
    )


def _repeated_clusters(
    ids: list[str], labs: dict[str, np.ndarray], tolerance: float
) -> list[dict]:
    """Find every group of mutually near-identical Lab values, largest first.

    Greedy seed-and-grow: take the biggest group any seed can gather, remove its
    members, repeat. With at most a few hundred patches this is trivially cheap and
    avoids the arbitrary parameters of a general clustering algorithm.

    Only groups of at least ``MIN_UNIFORMITY_CLUSTER`` patches are returned -- a
    neutral that appears once is a tonal step, not a repeated uniformity reference.
    """
    remaining = list(ids)
    families: list[dict] = []

    while remaining:
        best: list[str] = []
        for seed in remaining:
            members = [
                pid for pid in remaining if float(delta_e(labs[pid], labs[seed])) <= tolerance
            ]
            if len(members) > len(best):
                best = members
        if len(best) < MIN_UNIFORMITY_CLUSTER:
            break

        stack = np.stack([labs[pid] for pid in best])
        mean = stack.mean(axis=0)
        families.append(
            {
                "members": best,
                "mean": (float(mean[0]), float(mean[1]), float(mean[2])),
                "spread": float(np.max(delta_e(stack, np.broadcast_to(mean, stack.shape)))),
            }
        )
        remaining = [pid for pid in remaining if pid not in set(best)]

    families.sort(key=lambda f: -f["mean"][0])
    return families


def _order_ramp(
    patches: list[Patch], labs: dict[str, np.ndarray]
) -> tuple[list[Patch], str | None]:
    """Re-derive the order of the ramp's steps from measured L*, lightest first.

    Ramp *membership* is documented (FADGI OpenDICE names E5-J6 for the SG), but the
    reading order within those patches is an assumption. Deriving the order from the
    measured reference L* makes it a fact, and disagreeing with the declared order is
    worth saying out loud rather than silently accepting either one -- a ramp read
    backwards would still pass a max-per-patch threshold while making the plotted
    OECF curve meaningless.

    What is re-derived is the order of the **steps**, not a ranking of the patches.
    A chart may print one ladder at several field positions -- the UTT prints its
    twenty steps along four arms of a star -- and those four patches are one
    density read four times, not four densities. So the declared position groups
    the patches and the measurement orders the groups.
    """
    ramp = [p for p in patches if p.role == "achromatic_ramp" and p.patch_id in labs]
    if len(ramp) < 2:
        return patches, None

    # Group before ordering. ``ramp_position`` states the ladder's *structure* --
    # how many steps the chart prints -- and that is a fact about the printed
    # object, where the reading order is the assumption being checked. The UTT
    # prints one twenty-step ladder along four arms of a star and declares
    # position 1-20 four times over; ranking its patches individually turned that
    # into an eighty-step ladder whose order among four identical steps was
    # whatever the sort happened to yield, and made the OECF curve eighty points
    # wide over twenty printed densities.
    steps: dict[object, list[Patch]] = {}
    for p in ramp:
        # A definition is a file a user writes, so an unstated position is its own
        # step rather than an error -- which is what ranking every patch did.
        key = p.ramp_position if p.ramp_position is not None else ("unstated", p.patch_id)
        steps.setdefault(key, []).append(p)

    if len(steps) < 2:
        return patches, None

    def _level(members: list[Patch]) -> float:
        """The step's lightness. A mean, because members of one step are repeat
        readings of one printed density; how far they differ is uniformity's
        question and not this one's."""
        return float(np.mean([float(labs[m.patch_id][0]) for m in members]))

    declared_order = sorted(steps, key=lambda k: (isinstance(k, tuple), k))
    measured_order = sorted(steps, key=lambda k: -_level(steps[k]))
    derived = {key: i + 1 for i, key in enumerate(measured_order)}

    note = None
    if declared_order != measured_order:
        was = {key: i + 1 for i, key in enumerate(declared_order)}
        moved = sorted(
            (key for key in steps if was[key] != derived[key]),
            key=lambda key: derived[key],
        )
        note = (
            "Tonal ramp order re-derived from measured reference L*: "
            + ", ".join(
                f"step {was[key]} -> {derived[key]} "
                f"({'/'.join(m.patch_id for m in steps[key])}, "
                f"L* {_level(steps[key]):.2f})"
                for key in moved
            )
            + ". Using the measured order."
        )

    derived = {
        m.patch_id: position
        for key, position in derived.items()
        for m in steps[key]
    }

    out: list[Patch] = []
    for p in patches:
        if p.patch_id in derived:
            out.append(
                Patch(
                    patch_id=p.patch_id,
                    row=p.row,
                    col=p.col,
                    role=p.role,
                    ramp_position=derived[p.patch_id],
                    notes=p.notes,
                    # Region is not decoration: row and col only mean anything
                    # within one lattice, so dropping it here silently moved every
                    # re-ordered patch onto a region the target does not declare.
                    region=p.region,
                )
            )
        else:
            out.append(p)
    return out, note
