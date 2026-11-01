"""Reconciling a reference file's patch names with a target's own.

One chart, two vocabularies. Deltachrome names the Golden Thread Device Level
Target's patches ``1`` to ``30``, because that is what the chart prints beside
them -- and the OpenQualia standard agrees, naming those very targets as its
example of the bare-number case: "If the physical target has patch labels with
numbers only, the field SAMPLE_ID or SAMPLE_NAME shall contain the patch number
as labeled on the target." The measurement files published for them name the
same patches ``A1`` to ``A30``.

So the files do not follow the standard they are published under, and the
target definitions here do. That is worth stating in that order, because it
decides what this module is allowed to be: **a reconciliation, reported**, and
not a normalisation. Deltachrome does not get to quietly rename another
vendor's data to match its own and leave the operator with a run that looks
ordinary.

Three properties make that safe:

* **A mapping is declared by the target, never guessed.** A rule like "strip a
  leading letter" applied to any file would eventually fold ``A1`` and ``B1``
  of a real two-column chart onto one patch, and the chart it silently
  corrupted would be one nobody was testing.
* **It only ever applies when it beats reading the file as it stands.** A file
  that already joins is used exactly as written.
* **It says so.** The note travels into the run's warnings, so a report can be
  read back in a year and still show that the file named its patches
  differently from the chart.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from deltachrome.errors import DuplicateSampleError, ReferenceError
from deltachrome.reference.cgats import MeasurementSet
from deltachrome.targets.schema import TargetDefinition

#: Where a target declares the spellings a reference file might use for its
#: patches. A list of templates containing ``{id}``, tried in order.
ALIASES_KEY = "sample_id_aliases"


def _fold(name: str) -> str:
    """A patch name with its punctuation removed, for the last-resort match."""
    return re.sub(r"[^A-Za-z0-9]+", "", name).upper()


def reconcile_sample_ids(
    reference: MeasurementSet, target: TargetDefinition
) -> tuple[MeasurementSet, str | None]:
    """Join a reference file to a target's patch ids, renaming only if it helps.

    Returns the measurement set to use and a note to carry into the run's
    warnings, or ``None`` when the file was used exactly as written -- which is
    the ordinary case and produces no note at all.
    """
    patch_ids = [patch.patch_id for patch in target.patches]
    if not patch_ids or not reference.samples:
        return reference, None

    by_row = _declared_row_order(reference, target)
    if by_row is not None:
        return by_row

    wanted = set(patch_ids)
    direct = len(wanted & set(reference.samples))
    if direct == len(wanted):
        return reference, None

    # A per-patch declared label comes first, because it is a *statement about the
    # chart* rather than a pattern over ids. The ISA FADGI - ISO 19264 needs one:
    # it numbers its patches 1-168 in reading order and prints a ragged A-N by
    # 1-16 grid around its central modules, so no template generates the mapping.
    printed = {
        patch.printed_id: patch.patch_id
        for patch in target.patches
        if patch.printed_id and patch.printed_id in reference.samples
    }
    best: tuple[int, str, dict[str, str]] | None = None
    if printed:
        best = (len(printed), "the labels the chart prints", printed)

    templates = target.reference_data.get(ALIASES_KEY) or []
    for template in templates:
        if not isinstance(template, str) or "{id}" not in template:
            continue
        # Built from our ids outward, so two target patches can never collide
        # onto one reference sample: the mapping is a function of patch_id.
        mapping = {
            template.format(id=patch_id): patch_id
            for patch_id in patch_ids
            if template.format(id=patch_id) in reference.samples
        }
        if best is None or len(mapping) > best[0]:
            best = (len(mapping), template, mapping)

    # Last, and only if nothing declared does better: a match that ignores
    # punctuation. Two published measurements of one DT Nexus Validation Target
    # name the same patch ``C-7`` and ``C7``, so a vendor can differ from itself
    # between production runs by a hyphen. This is safe where a template is not,
    # because it is checked for collisions first: it is applied only when every
    # target patch folds to a distinct key, so it can never merge two patches.
    if best is None or best[0] <= direct:
        folded = {_fold(patch_id): patch_id for patch_id in patch_ids}
        if len(folded) == len(patch_ids):
            mapping = {
                name: folded[_fold(name)]
                for name in reference.samples
                if _fold(name) in folded and name not in wanted
            }
            if len(mapping) > direct:
                best = (len(mapping), "ignoring punctuation", mapping)

    if best is None or best[0] <= direct:
        # Only when nothing joins as written: a chart that prints numbers -- the
        # Golden Thread's 1 to 30 -- is named by the file's numbers already.
        layout = _layout_order(reference, target) if direct == 0 else None
        if layout is None:
            return reference, None
        mapping, note = layout
        renamed = {
            mapping[name]: replace(measurement, sample_name=mapping[name])
            for name, measurement in reference.samples.items()
        }
        return replace(reference, samples=renamed), note

    count, template, mapping = best
    renamed = {
        mapping.get(name, name): (
            replace(measurement, sample_name=mapping[name])
            if name in mapping
            else measurement
        )
        for name, measurement in reference.samples.items()
    }
    example = (
        template.format(id=patch_ids[0]) if "{id}" in template else next(iter(mapping))
    )
    how = f"the pattern {template!r}" if "{id}" in template else template
    note = (
        f"The reference file names its patches like {example!r} where "
        f"{target.display_name} labels them {patch_ids[0]!r}. {count} of "
        f"{len(wanted)} patches were matched by {how}, which the target definition "
        "declares. The chart's own patch ids are what Deltachrome reports."
    )
    return replace(reference, samples=renamed), note


# -- OpenDICE profiles, matched row by row -------------------------------------------

#: Where a target declares how OpenDICE's profile for it lists its patches: a list of
#: ``[name in the profile, patch id]`` pairs in the profile's row order. Needed where
#: the profile's names cannot be joined on their own -- the film ladders number their
#: two blocks 1-24 straight through, and the UTT's profile names a colour patch and a
#: grey step both ``B1``.
OPENDICE_KEY = "opendice_names"


def _declared_row_order(
    reference: MeasurementSet, target: TargetDefinition
) -> tuple[MeasurementSet, str | None] | None:
    """Rename an OpenDICE profile's rows by the order the target declares.

    Matched on the names as well as the positions: the declaration is used only when
    the file lists exactly the names it says, in exactly that order, so a profile for
    a different chart, or one edited by hand, is never renamed by position alone.
    A file that repeats a name and is not settled this way is refused.
    """
    if not reference.row_names:
        return None
    declared = target.reference_data.get(OPENDICE_KEY) or []
    if declared and [name for name, _ in declared] == reference.row_names:
        renamed = {
            patch_id: replace(measurement, sample_name=patch_id)
            for (_, patch_id), measurement in zip(declared, reference.samples.values())
        }
        note = (
            f"The reference file is an OpenDICE profile, matched to "
            f"{target.display_name} by the order OpenDICE lists its patches in, which "
            "the target definition declares."
        )
        return replace(reference, samples=renamed, row_names=[]), note

    repeated = sorted({n for n in reference.row_names if reference.row_names.count(n) > 1})
    if repeated:
        raise DuplicateSampleError(
            f"The OpenDICE profile names {', '.join(repeated)} more than once, and "
            f"{target.display_name} declares no row order that says which patch each "
            "one is."
        )
    return None


# -- Files that number their patches and name none ---------------------------------

#: CGATS.17's keyword for how many samples one row of the *measured* layout holds.
#: A file numbering its samples 1..N is counting through that layout row by row,
#: and this is the only thing in it saying how long a row was.
LAYOUT_ROW_LENGTH_KEY = "LGOROWLENGTH"

#: Chroma below which a step of a declared ramp reads as neutral, when checking a
#: layout order. Looser than role resolution's 4.0 because what is being told apart
#: is a neutral column from a chromatic one, and the nearest chromatic column of an
#: IT8 sits in the twenties.
LAYOUT_RAMP_CHROMA = 6.0

#: How many declared ramp steps the check needs before it means anything.
LAYOUT_RAMP_MIN_STEPS = 4


def _layout_order(
    reference: MeasurementSet, target: TargetDefinition
) -> tuple[dict[str, str], str] | None:
    """Patch ids for a file that numbers its samples ``1``..``N`` and names none.

    BabelColor PatchTool, among others, writes a measurement as ``SAMPLE_ID`` 1 to
    N in the order the instrument read it. That order is a fact about how somebody
    measured, not about the chart -- the Wolf Faust IT8.7/2 file this was written
    for was read down the columns, A1 to L1 and then A2, because a strip reader
    takes a column at a pass. What the file does say is ``LGOROWLENGTH``: how many
    samples a row of the measured layout held. On a chart that is not square that
    settles which way the rows ran -- 12 on a 12 x 22 array can only mean the
    columns were read as rows -- and nothing else. It does not say which corner the
    count began at, and a count begun at L1 is as consistent with it as one begun
    at A1.

    So the order is taken from A1, which is where every layout convention begins,
    and is then **checked against the definition before it is used**: the target's
    declared neutral ramp must, under that order, read as a ramp -- every step
    neutral and lightness falling in the declared direction. A count begun at any
    other corner, or a file numbered row-wise when it claims columns, puts the ramp
    on a chromatic column or reverses it, and fails. No capture is consulted, so a
    bad capture cannot choose the order.

    Returns ``None`` when the file does not number its samples this way, so the
    ordinary refusal for an unjoined file stands. Raises when it does and the order
    cannot be established, saying what would fix it.
    """
    names = list(reference.samples)
    if not names or not all(name.isdigit() for name in names):
        return None
    count = len(names)
    if sorted(int(name) for name in names) != list(range(1, count + 1)):
        return None

    def refuse(why: str) -> ReferenceError:
        return ReferenceError(
            f"The reference file numbers its patches 1 to {count} and names none of "
            f"them, so which patch of {target.display_name} each number means depends "
            f"on how it was measured. {why} Export it again with patch names "
            "(SAMPLE_NAME A1, A2, ...)."
        )

    grids = [r for r in target.grid_regions if r.grid.rows * r.grid.columns == count]
    if len(grids) != 1:
        raise refuse(f"No single patch array on the chart holds {count} patches.")
    region = grids[0]
    rows, columns = region.grid.rows, region.grid.columns
    raw = reference.metadata.get(LAYOUT_ROW_LENGTH_KEY, "").strip().strip('"')
    if not raw.isdigit():
        raise refuse(f"It does not state {LAYOUT_ROW_LENGTH_KEY}, the row length of "
                      "the layout it was measured in.")
    length = int(raw)
    if rows == columns or length not in (rows, columns):
        raise refuse(f"Its {LAYOUT_ROW_LENGTH_KEY} of {length} does not say which way "
                      f"a {rows} x {columns} array was read.")
    by_cell = {(p.row, p.col): p for p in target.patches if p.region == region.region_id}
    down_columns = length == rows and rows != columns
    mapping = {}
    for name in names:
        k = int(name) - 1
        cell = (k % rows, k // rows) if down_columns else (k // columns, k % columns)
        if cell not in by_cell:
            raise refuse("Its numbering runs past the chart's own cells.")
        mapping[name] = by_cell[cell].patch_id

    ramp = sorted(
        (p for p in by_cell.values() if p.role == "achromatic_ramp" and p.ramp_position),
        key=lambda p: p.ramp_position,
    )
    if len(ramp) < LAYOUT_RAMP_MIN_STEPS:
        raise refuse("The chart declares no neutral ramp to check a reading order "
                      "against.")
    inverse = {patch_id: name for name, patch_id in mapping.items()}
    labs = [reference.samples[inverse[p.patch_id]].lab for p in ramp]
    if any(lab is None for lab in labs):
        raise refuse("Its ramp patches carry no Lab to check a reading order against.")
    lightness = np.array([lab[0] for lab in labs], float)
    chroma = np.array([np.hypot(lab[1], lab[2]) for lab in labs], float)
    if chroma.max() > LAYOUT_RAMP_CHROMA or not np.all(np.diff(lightness) < 0):
        raise refuse(
            f"Counted {'down the columns' if down_columns else 'along the rows'} from "
            f"{by_cell[(0, 0)].patch_id}, as its {LAYOUT_ROW_LENGTH_KEY} implies, the "
            "chart's neutral ramp does not come out neutral and descending, so it was "
            "counted some other way."
        )
    first = by_cell[(0, 0)].patch_id
    direction = "down each column" if down_columns else "along each row"
    note = (
        f"The reference file numbers its patches 1 to {count} without naming them. "
        f"They were read as counted {direction} from {first}, which its "
        f"{LAYOUT_ROW_LENGTH_KEY} of {length} states, and the order was checked "
        f"against the chart's neutral ramp ({ramp[0].patch_id} to "
        f"{ramp[-1].patch_id}). A file exported with patch names needs no such step."
    )
    return mapping, note
