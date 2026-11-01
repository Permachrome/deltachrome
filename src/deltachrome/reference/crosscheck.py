"""Does this reference file describe the chart this definition describes?

The existing guards answer a coarser question. A file for the wrong *model* is
caught because its patch ids do not cover the target's. A file for the wrong
*unit* is caught, where the chart carries a code, by reading the code. Neither
catches a file that names every patch correctly and states different colours for
some of them -- and that is a real case rather than a hypothetical one.

**How it was found.** Image Science Associates revised the FADGI - ISO 19264
chart at some point after the unit OpenDICE published, changing a contiguous
block of patches below its central modules. The per-unit measurements Digital
Transitions publish are of the revised chart; this repository's nominal values
come from the earlier one. The two agree on 135 of 168 patches to a median of
0.29 dE2000 and disagree on the other 33 by a **mean of 37.8**, and the sample
capture settles which is which: measured through this engine it sits 1.49 dE2000
from the earlier values and 37.8 from the published ones. So both sets of numbers
are right, about different objects.

Pairing such a file by serial is correct and produces a good measurement. Pairing
it against the *other* version's geometry, or reading its numbers as though they
described the chart the definition was built from, is a run whose colour metrics
are wrong on a fifth of the chart and which looks entirely ordinary.

**What separates the cases is scale, and there are three of them rather than two.**
Two units of one model *from one production batch* differ by 0.2 to 0.9 dE2000
across their patches -- measured over 50 Golden Thread Device Level units, 40
Object Level and 17 DT NGT2, all of which turn out to be single populations.
Two units of one model from **different batches** differ by far more, and that is
normal rather than faulty: the ISA FADGI 19264 has two populations 7.1 dE2000
apart, the DT NTPT two 13.7 apart, the DT NTVT two around 8. A different *version*
of the chart -- patch colours revised without the product being renamed -- differs
by tens on the patches that changed and not at all on the rest.

So :data:`DISAGREEMENT_DE` at 5.0 sits above within-batch noise and below
everything else, and it deliberately fires on a batch difference as well as on a
wrong chart. A nominal 7 dE2000 from the chart in front of you is worth being told
about, because the answer is the same in both cases: fetch the unit's own file.

This warns and never refuses, and names no culprit, because it genuinely cannot
tell which of three things happened -- another chart's file, another batch of the
same chart, or a revision the definition predates. The operator may also have the
newer chart and the newer file, in which case they are right and the definition's
nominals are the stale half.
"""

from __future__ import annotations

import numpy as np

from deltachrome.imaging.color import delta_e
from deltachrome.reference.cgats import MeasurementSet
from deltachrome.targets.schema import TargetDefinition

#: Per-patch difference above which a reference and a definition's own nominals
#: are describing different objects rather than two units of one model. An order
#: of magnitude above real unit-to-unit variation.
DISAGREEMENT_DE = 5.0

#: How many patches must disagree before it is reported. A single patch that far
#: out is a bad measurement of one patch; a population of them is a different
#: chart. Both a floor and a fraction, so the rule reads the same on a 30-patch
#: card and a 168-patch one.
MIN_DISAGREEING = 3
MIN_FRACTION = 0.05


def crosscheck_against_nominal(
    reference: MeasurementSet, target: TargetDefinition
) -> str | None:
    """Compare a supplied reference against the target's own nominal values.

    Returns a warning, or ``None`` when they agree or there is nothing to
    compare. Call only for a reference that is *not* the nominal file itself --
    comparing that with itself answers nothing.
    """
    nominal_path = target.nominal_reference_path
    if nominal_path is None or not reference.samples:
        return None

    from deltachrome.reference import read_reference
    from deltachrome.reference.sample_ids import reconcile_sample_ids

    try:
        nominal, _ = reconcile_sample_ids(read_reference(nominal_path), target)
    except Exception:  # noqa: BLE001 -- a nominal we cannot read is not this check's business
        return None

    shared = [
        patch_id
        for patch_id in (p.patch_id for p in target.patches)
        if (a := reference.samples.get(patch_id)) is not None
        and (b := nominal.samples.get(patch_id)) is not None
        and a.lab is not None
        and b.lab is not None
    ]
    if len(shared) < MIN_DISAGREEING:
        return None

    supplied = np.array([reference.samples[p].lab for p in shared])
    declared = np.array([nominal.samples[p].lab for p in shared])
    differences = delta_e(supplied, declared, "delta_e_2000")
    disagreeing = differences > DISAGREEMENT_DE
    count = int(disagreeing.sum())
    if count < MIN_DISAGREEING or count < MIN_FRACTION * len(shared):
        return None

    worst = [shared[i] for i in np.argsort(differences)[::-1][:6]]
    return (
        f"This reference file disagrees with {target.display_name}'s own "
        f"manufacturer values on {count} of {len(shared)} patches, by a mean of "
        f"{differences[disagreeing].mean():.1f} dE2000 and up to "
        f"{differences.max():.1f} -- worst at {', '.join(worst)}. Two units from one "
        "production batch differ by well under 1, so this is not ordinary "
        "manufacturing spread. Three things produce it and this cannot tell them "
        "apart: another chart's measurement; another BATCH of the same chart, "
        "which is normal and routinely runs to 7-14 dE2000 on these models; or a "
        "revision the definition predates, manufacturers being free to change "
        "patch colors without renaming the product. In the last two cases the "
        "file is right and these manufacturer values are the stale half. The "
        "measurement uses the file you supplied either way, which is the correct "
        "choice if it is genuinely a measurement of the chart in the frame -- and "
        "a per-unit file always beats these values. Check the geometry still "
        "matches before trusting a result from a chart that may have been revised."
    )
