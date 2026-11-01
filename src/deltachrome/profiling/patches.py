"""The patches a camera profile is built from: device RGB beside reference XYZ.

**Device values, not colour-managed ones.** Everything else in Deltachrome reads
a capture through its embedded ICC profile, because what is judged there is
colour. Here the profile is what is being *made*, so the numbers wanted are the
ones a profile will later stand in front of -- ``rgb_mean`` straight off the
sampler -- and the capture usually carries no profile to read through anyway.

**A patch is used, excluded, or not data.** Used patches go into the ``.ti3``.
Excluded patches have a reference value and are left out, either because the
sampler flagged them or because the operator chose to; they are still scored
through the finished profile, which is how a deselection says what it cost. A
patch with no reference value is neither, and is simply not part of the set.

**Repeats can be merged** (:func:`merge_repeats`), and are not by default, which
is what ``colprof`` does with a ``.ti3`` it is handed. A colour the chart prints
many times is then one reading -- the mean of its device values against the mean
of its references -- rather than a dozen readings outweighing every other colour
in the fit. Which patches are repeats is decided by the reference alone, and by
**separation rather than distance**: a family is merged only where its spread is
at most :data:`REPEAT_SEPARATION` of the distance to the nearest colour outside
it. The SG's three border families spread 0.16 to 0.49 ΔE2000 with the next
colour 5 to 7 away; the LaserSoft's fine ramps put neighbouring steps 0.4 to 1.0
apart with no gap at all, and a distance that merged the first would average the
second's steps together.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from deltachrome.errors import ProfilingError
from deltachrome.imaging.color import delta_e, lab_to_xyz, xyz_to_lab
from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.icc import CIE_D50, ICC_PCS_D50
from deltachrome.imaging.loader import LoadedImage
from deltachrome.imaging.sampling import DEFAULT_ROI_INSET, sample_patches
from deltachrome.reference import MeasurementSet
from deltachrome.reference.sample_ids import reconcile_sample_ids
from deltachrome.targets.roles import resolve_roles
from deltachrome.targets.schema import TargetDefinition

#: Why a patch was left out when nobody chose to leave it out. The sampler's own
#: flags, which already mean "these pixels are not this patch's colour": clipped
#: at either end, a median disagreeing with the mean, a region off the frame.
EXCLUDED_FLAGGED = "flagged"
#: Why a patch was left out when somebody did.
EXCLUDED_CHOSEN = "chosen"

#: The largest share of a patch's pixels that may touch the encoding floor, with
#: nothing else wrong, and the patch still be used for a profile. QC flags any
#: patch over 0.1%, which is right for a judgement and too strict for a fit: on
#: the LaserSoft's eciRGB v2 master seven dark patches flag at 0.16-0.48%, one
#: channel grazing zero with nothing bright beside it, and their means agree with
#: their medians to 0.0003 -- the floor pixels move nothing. Twice the worst of
#: them. A patch with a channel *pinned* while another is bright is out of gamut
#: or at the sensor's floor, flagged as such, and still left out.
MARGINAL_LOW_FRACTION = 0.01

#: The fewest used patches a profile is attempted from. A 3x3 matrix is nine
#: unknowns and each patch gives three equations, but the curves need the
#: patches spread along the tone scale too; twelve is the SG's own ramp.
MIN_PATCHES = 12

#: How tight a family must be, against the nearest colour outside it, to be the
#: same colour printed again: its largest internal ΔE2000 at most this fraction
#: of the smallest ΔE2000 to anything else. The SG's families read 0.03 to 0.07.
REPEAT_SEPARATION = 0.2

#: How near two references must be (ΔE2000) to be one family: every pair in it,
#: not just neighbours. Separation then decides. The held-out folds group on the
#: same distance.
REPEAT_LINK_DE = 1.0


@dataclass(frozen=True)
class ProfilePatch:
    """One patch: what the camera recorded and what the chart actually is."""

    patch_id: str
    #: Device values in [0, 1], the sampler's per-channel mean.
    rgb: np.ndarray
    #: The reference, as XYZ relative to the ICC's D50 with Y in [0, 1] -- the
    #: white ``colprof`` itself converts with, so the file carries no mismatch.
    xyz: np.ndarray
    #: The same reference as Lab against CIE D50, which is what every ΔE in
    #: Deltachrome is computed against.
    lab: np.ndarray
    #: What the sampler flagged that leaves the patch out, empty for a usable
    #: one -- a marginal floor (:data:`MARGINAL_LOW_FRACTION`) is not among them.
    flags: tuple[str, ...] = ()
    #: The patches a merged reading stands for, this one first; empty for a
    #: patch read on its own.
    members: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return self.members or (self.patch_id,)


@dataclass
class ProfilingSet:
    """Every patch with a reference value, split into used and excluded."""

    target_id: str
    used: list[ProfilePatch]
    #: patch id -> (the patch, why it was left out)
    excluded: dict[str, tuple[ProfilePatch, str]]
    warnings: list[str] = field(default_factory=list)
    #: The neutral ramp's patch ids, declared or resolved from the reference --
    #: what a computed white scale aims at.
    ramp: list[str] = field(default_factory=list)

    #: Each sampled patch's region in image pixels, for drawing and for picking
    #: a patch out of the capture by clicking it.
    regions: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def all_patches(self) -> list[ProfilePatch]:
        return self.used + [patch for patch, _ in self.excluded.values()]


def _marginal_floor(sample) -> bool:
    """Whether the only thing wrong is a few pixels grazing zero. See
    :data:`MARGINAL_LOW_FRACTION`."""
    return (
        bool(sample.flags)
        and all(flag.startswith("clipped_low(") for flag in sample.flags)
        and sample.clipped_low_fraction <= MARGINAL_LOW_FRACTION
    )


def profilable(target: TargetDefinition) -> tuple[bool, str]:
    """Whether a camera profile can be built from this chart at all, and why not.

    A property of the definition, so it is knowable before a pixel is read and a
    control can be withheld rather than offered and refused.
    """
    if not target.grid_regions or not target.patches:
        return False, f"{target.display_name} has no colour patches to profile from."
    if target.reference_data.get("source_type") == "none":
        return False, (
            f"{target.display_name} ships no reference values, so there is nothing "
            "to profile against."
        )
    return True, ""


def select_patches(data: ProfilingSet, excluded: Iterable[str]) -> ProfilingSet:
    """The same patches, with exactly ``excluded`` left out.

    For the screen, which samples once and then lets patches be clicked in and
    out: what is left out is the operator's whole answer, flagged patches
    included -- one they click back in is used, clipped or not. Each left-out
    patch says why: flagged if the sampler flagged it, chosen otherwise.
    """
    out = set(excluded)
    used: list[ProfilePatch] = []
    left: dict[str, tuple[ProfilePatch, str]] = {}
    for patch in data.all_patches:
        if patch.patch_id in out:
            left[patch.patch_id] = (
                patch, EXCLUDED_FLAGGED if patch.flags else EXCLUDED_CHOSEN
            )
        else:
            used.append(patch)
    if len(used) < MIN_PATCHES:
        raise ProfilingError(
            f"Only {len(used)} patches are used; a profile needs at least {MIN_PATCHES}."
        )
    return replace(data, used=used, excluded=left)


def gather_patches(
    image: LoadedImage,
    target: TargetDefinition,
    reference: MeasurementSet,
    corners: Sequence[tuple[float, float]],
    *,
    exclude: Iterable[str] = (),
    roi_inset: float = DEFAULT_ROI_INSET,
) -> ProfilingSet:
    """Sample every patch and pair it with its reference, or refuse.

    ``exclude`` names patches the operator has taken out. Flagged patches are
    taken out without being named; naming a flagged one changes nothing.
    """
    # Imported here: the pipeline imports every metric, and this module is
    # otherwise light enough for a frontend to import at startup.
    from deltachrome.pipeline import _reference_lab

    ok, why = profilable(target)
    if not ok:
        raise ProfilingError(why)
    if image.channels != 3:
        raise ProfilingError(
            f"{image.path.name} has {image.channels} colour record(s). A camera "
            "profile here is built from RGB."
        )

    warnings: list[str] = []
    reference, alias_note = reconcile_sample_ids(reference, target)
    if alias_note:
        warnings.append(alias_note)
    # The roles QC resolves, so the ramp a scale aims at is the ramp tone response
    # is judged on.
    ramp = [
        patch.patch_id for patch in resolve_roles(target, reference).target.patches
        if patch.role == "achromatic_ramp"
    ]
    # The same adaptation QC applies, so a profile and the run that later checks
    # it read one reference identically.
    labs, lab_warnings = _reference_lab(reference, CIE_D50)
    warnings.extend(lab_warnings)

    placement = place_chart(
        corners,
        image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )
    # Noise needs Lab, and Lab needs the profile this capture does not have yet.
    sampling = sample_patches(
        image, placement, target, inset=roi_inset, measure_noise=False
    )
    warnings.extend(sampling.warnings)

    chosen = set(exclude)
    unknown = sorted(chosen - {patch.patch_id for patch in target.patches})
    if unknown:
        raise ProfilingError(
            f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not "
            f"patches of {target.display_name}."
        )

    used: list[ProfilePatch] = []
    excluded: dict[str, tuple[ProfilePatch, str]] = {}
    regions: dict[str, np.ndarray] = {}
    for patch in target.patches:
        lab = labs.get(patch.patch_id)
        sample = sampling.samples.get(patch.patch_id)
        if lab is None or sample is None:
            continue
        flags = tuple(sample.flags)
        if _marginal_floor(sample):
            flags = ()
        entry = ProfilePatch(
            patch_id=patch.patch_id,
            rgb=np.clip(np.asarray(sample.rgb_mean, dtype=np.float64), 0.0, 1.0),
            xyz=lab_to_xyz(lab, white_point=ICC_PCS_D50),
            lab=np.asarray(lab, dtype=np.float64),
            flags=flags,
        )
        regions[patch.patch_id] = np.asarray(sample.polygon_px, dtype=np.float64)
        if flags or not sample.usable:
            excluded[patch.patch_id] = (entry, EXCLUDED_FLAGGED)
        elif patch.patch_id in chosen:
            excluded[patch.patch_id] = (entry, EXCLUDED_CHOSEN)
        else:
            used.append(entry)

    if not used and not excluded:
        raise ProfilingError(
            "No patch could be both sampled and matched to a reference value. Check "
            "that the reference file is for this chart and that the placement is right."
        )
    # A matrix alone has nine numbers to fit. Fewer patches than that is not a
    # profile but an interpolation through every point.
    if len(used) < MIN_PATCHES:
        raise ProfilingError(
            f"Only {len(used)} usable patches; a profile needs at least {MIN_PATCHES}."
        )
    return ProfilingSet(
        target_id=target.target_id, used=used, excluded=excluded, warnings=warnings,
        ramp=ramp, regions=regions,
    )


def repeat_families(patches: Sequence[ProfilePatch]) -> list[list[int]]:
    """Indices of ``patches`` printing one colour more than once, family by family.

    Families are linked at :data:`REPEAT_LINK_DE`, and kept only where every
    pair inside them is within that distance too and they stand apart from every
    other colour by :data:`REPEAT_SEPARATION`.
    """
    if len(patches) < 2:
        return []
    labs = np.stack([patch.lab for patch in patches])
    count = len(patches)
    distances = np.stack([
        delta_e(labs, np.broadcast_to(labs[i], labs.shape)) for i in range(count)
    ])
    # Connected components of "within the link distance", by flood fill.
    label = np.full(count, -1)
    for start in range(count):
        if label[start] >= 0:
            continue
        label[start] = start
        stack = [start]
        while stack:
            here = stack.pop()
            for there in np.flatnonzero((distances[here] < REPEAT_LINK_DE) & (label < 0)):
                label[there] = start
                stack.append(int(there))
    families = []
    for root in np.unique(label):
        members = np.flatnonzero(label == root)
        outside = np.flatnonzero(label != root)
        if len(members) < 2 or not len(outside):
            continue
        spread = distances[np.ix_(members, members)].max()
        nearest = distances[np.ix_(members, outside)].min()
        # Every pair within the link distance, not only a chain of neighbours:
        # a fine ramp standing alone is isolated and still several colours.
        if spread < REPEAT_LINK_DE and spread <= REPEAT_SEPARATION * nearest:
            families.append([int(i) for i in members])
    return families


def merge_repeats(data: ProfilingSet) -> ProfilingSet:
    """The same set with each family of repeated used patches as one reading.

    The reading is the family's mean device value against the mean of its
    references in XYZ, and carries the first patch's id and every member's.
    Excluded patches are not merged: each was left out for a reason of its own.
    """
    families = repeat_families(data.used)
    if not families:
        return data
    merged_into = {i: family for family in families for i in family}
    used: list[ProfilePatch] = []
    for index, patch in enumerate(data.used):
        family = merged_into.get(index)
        if family is None:
            used.append(patch)
        elif index == family[0]:
            group = [data.used[i] for i in family]
            xyz = np.mean([p.xyz for p in group], axis=0)
            used.append(replace(
                patch,
                rgb=np.mean([p.rgb for p in group], axis=0),
                xyz=xyz,
                lab=xyz_to_lab(xyz, white_point=ICC_PCS_D50),
                members=tuple(p.patch_id for p in group),
            ))
    if len(used) < MIN_PATCHES:
        raise ProfilingError(
            f"Merged, only {len(used)} different colours are left; a profile needs "
            f"at least {MIN_PATCHES}."
        )
    return replace(data, used=used)
