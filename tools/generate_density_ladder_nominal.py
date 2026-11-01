"""Build the nominal reference files for targets whose aim data is a density ladder.

Four of the film targets modelled here carry **no Lab at all**. What the Library of
Congress publishes with OpenDICE for each is a column of optical densities and
nothing else, because the chart they describe has no colour patches. Deltachrome's
metrics work in Lab, so the ladder has to be turned into aim L*a*b* -- and that
conversion is a choice, not an arithmetic identity, which is why it lives in a
generator script with its reasoning attached rather than in a hand-written data file.

The two choices, made once here and applied to every ladder:

``anchor``  A density ladder says how much light each step passes relative to the
            others. It does not say what lightness the lightest step should come
            out at, and for a transmissive original there is no single right
            answer -- that mapping is the system's tone response, the thing being
            measured. FADGI's transparency tables give a highlight aim of
            96 +/- 2 L*, so every ladder is anchored there: the standard's choice
            rather than ours.

``chroma``  a* = b* = 0 for every step. These are neutral scales, and white balance
            is Delta E(a*b*) with L* excluded, so this aim measures exactly what
            that metric is for -- whether the system renders a neutral as a
            neutral. It is a claim about the target restated as an aim, not a
            measurement of any specimen.

The densities themselves are carried through into each file beside the derived Lab,
because the density is what was actually published and the Lab is what was inferred
from it.

**Patch ids are assigned here, and they are the join to the target definition.**
OpenDICE names every one of these steps ``PatchN``, numbered straight through both
copies of the ladder, which says nothing about which copy a step belongs to. Each
definition addresses its two blocks separately -- there is no single lattice -- so
the ids are rewritten as ``<block><step>``, and ``BLOCKS`` below records which
prefix goes with which region. Getting that mapping wrong would still sample every
patch and still produce a plausible ramp, which is why it is stated in one place.

Run from the repository root:

    python tools/generate_density_ladder_nominal.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deltachrome.imaging.color import (
    FADGI_TRANSPARENCY_HIGHLIGHT_L,
    density_to_lightness,
)

DEFINITIONS = Path(__file__).resolve().parents[1] / "src/deltachrome/targets/definitions"

SHARED_CAVEAT = (
    "Manufacturer nominal aim values for the model, not a measurement of any "
    "individual chart. Two further steps sit between this file and a measurement, "
    "and both are choices rather than arithmetic: the published data is DENSITY "
    f"ONLY, anchored here so the lightest step reads L* {FADGI_TRANSPARENCY_HIGHLIGHT_L:.0f} "
    "-- FADGI's own transparency highlight aim -- because a density ladder carries "
    "no lightness of its own; and a* = b* = 0 restates the manufacturer's claim "
    "that the scales are spectrally neutral as an aim. Density describes the film; "
    "L* describes the appearance of a reproduction, and for a transmissive original "
    "the mapping between them IS the tone response being measured. Treat tone "
    "response against this file as advisory of a rendering intent, not as "
    "conformance."
)


@dataclass(frozen=True)
class Ladder:
    """One published density ladder, and everything the output file must declare."""

    output: str
    descriptor: str
    manufacturer: str
    originator: str
    #: Densities in the order the published profile lists them. Not sorted here:
    #: the profile's own order is what its ``PatchN`` numbering means, and the
    #: definitions place their patches by that numbering.
    densities: tuple[float, ...]
    #: Prefix per copy of the ladder, in the order the published file repeats it.
    blocks: tuple[str, ...]
    source: str
    notes: tuple[str, ...] = field(default_factory=tuple)


#: The wedge ISA's transmissive film targets print, as OpenDICE publishes it for
#: the 4x5 (Profile_Negative4x5.txt). Shared by the 35mm and the 120: every
#: capture of either fits it best, and the ladders published
#: for them fit neither their own samples nor each other's.
ISA_FILM_WEDGE = (0.05, 0.12, 0.19, 0.33, 0.48, 0.72, 0.88, 1.15, 1.43, 1.71, 2.05, 2.25)


LADDERS = (
    Ladder(
        output="isa_transmissive_4x5_v1_nominal.txt",
        descriptor="ISA transmissive film target 4x5 -- manufacturer nominal aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_Negative4x5.txt (Library of Congress)",
        densities=ISA_FILM_WEDGE,
        blocks=("L", "R"),
        source=(
            "Density ladder from OpenDICE's Profile_Negative4x5.txt, published by the "
            "Library of Congress at digitizationguidelines.gov. Confirmed to be this "
            "target by its accompanying sample image 'Negative 4x5.tif', which is the "
            "same chart -- ten quarter-disc features at five pair positions, two 2x6 "
            "wedges, the diagonal, the ISA logo and the Don Williams signature -- as a "
            "negative."
        ),
    ),
    Ladder(
        output="isa_transmissive_120_v1_nominal.txt",
        descriptor="ISA transmissive film target 120 -- manufacturer nominal aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_Negative4x5.txt (Library of Congress)",
        densities=ISA_FILM_WEDGE,
        blocks=("T", "B"),
        source=(
            "The ISA film wedge, D 0.05 to 2.25, from OpenDICE's "
            "Profile_Negative4x5.txt -- the ladder published for this chart's 4x5 "
            "sibling. NOT the 120's own published file: OpenDICE's "
            "Profile_Negative120.txt (D 0.06 to 1.83) does not fit its own sample "
            "capture '120_target.tif', leaving 9.2 L* rms and needing a negative "
            "flare term, where the 4x5's ladder fits it at 2.2 L* rms with flare "
            "near zero. Every ISA 35mm capture tried, OpenDICE's own included, also "
            "fits the 4x5's ladder best, so one wedge serves all three ISA film "
            "targets. The two "
            "wedges measure the same step for step, so which block is which does "
            "not matter."
        ),
        notes=(
            "CHOSEN PARTLY BY FIT, which is said rather than hidden: four captures "
            "of two formats from two sources agree on this ladder and disagree with "
            "the one published for the 120, but only ISA can confirm what the chart "
            "prints.",
        ),
    ),
    Ladder(
        output="isa_transmissive_35mm_v1_nominal.txt",
        descriptor="ISA transmissive film target 35mm -- manufacturer nominal aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_Negative4x5.txt (Library of Congress)",
        densities=ISA_FILM_WEDGE,
        blocks=("L", "R"),
        source=(
            "The ISA film wedge, D 0.05 to 2.25, from OpenDICE's "
            "Profile_Negative4x5.txt -- the ladder published for this chart's 4x5 "
            "sibling. It replaces twelve recommended L* values of untraced origin "
            "(93, 76, 70, 62, 53, 38, 28, 20, 17, 13, 9, 7). Fitted with free gain and "
            "flare, three ISA 35mm captures -- a 35mm, a slide and OpenDICE's 'Negative "
            "35mm (2)' -- read 3.2, 3.3 and 2.2 L* rms against this ladder, 4.9, 6.5 "
            "and 8.2 against the ladder OpenDICE publishes for that sample "
            "(Profile_35mm_Negative2.txt, D 0.06 to 1.83), and 7.2, 8.3 and 9.8 "
            "against the values it replaces. One wedge serves all three ISA film "
            "targets."
        ),
        notes=(
            "CHOSEN PARTLY BY FIT, which is said rather than hidden: four captures "
            "of two formats from two sources agree on this ladder, but only ISA can "
            "confirm what the chart prints.",
        ),
    ),
    Ladder(
        output="microform_35mm_positive_v1_nominal.txt",
        descriptor="35mm microform resolution target, positive -- nominal aim values",
        manufacturer="Image Science Associates (Microfilm Preservation Target, discontinued by 2021)",
        originator="OpenDICE Profile_35mmPositive.txt (Library of Congress)",
        densities=(1.34, 1.16, 0.99, 0.86, 0.73, 0.58, 0.45, 0.35, 0.20, 0.04),
        blocks=("UL", "LR"),
        source=(
            "Density ladder from OpenDICE's Profile_35mmPositive.txt, published with its "
            "sample capture 'Positive 35mm.tif'. Ten steps, repeated as Patch1-10 and "
            "Patch11-20 for the chart's two L-shaped ladders. Confirmed against the "
            "capture: each L holds exactly ten steps, monotone along the L, and the two "
            "agree with each other to 0.01 in normalized code value at every step."
        ),
        notes=(
            "The densities DESCEND with patch number on this positive, which is the "
            "published order and is kept. Patch1 is the densest step and sits at the far "
            "end of each L's short arm; Patch10 is the clear step at the far end of the "
            "long arm. The negative of the same chart numbers the same physical steps in "
            "the same direction and its densities ascend, which is what says the "
            "numbering follows the print rather than the polarity.",
            "The two files are NOT interchangeable. A positive and a negative of one "
            "chart are different specimens with different aim data, and running either "
            "against the other's file inverts the ramp.",
        ),
    ),
    Ladder(
        output="microform_35mm_negative_v1_nominal.txt",
        descriptor="35mm microform resolution target, negative -- nominal aim values",
        manufacturer="Image Science Associates (Microfilm Preservation Target, discontinued by 2021)",
        originator="OpenDICE Profile_35mm_Negative.txt (Library of Congress)",
        densities=(0.06, 0.25, 0.40, 0.50, 0.65, 0.83, 0.93, 1.05, 1.21, 1.39),
        blocks=("UL", "LR"),
        source=(
            "Density ladder from OpenDICE's Profile_35mm_Negative.txt, published with its "
            "sample capture 'Negative_35mm_ICC (1).tif'. Ten steps, repeated as Patch1-10 "
            "and Patch11-20. Confirmed against the capture, which is the same chart as "
            "the positive -- same four corner crosses one inch apart, same tilted "
            "rectangle, same two L-shaped ladders at the same relative positions, sampled "
            "with the positive's geometry unchanged."
        ),
        notes=(
            "These ten densities are the first ten of the twelve OpenDICE publishes for "
            "the ISA 120 target, to the digit. That is worth knowing and is not evidence "
            "the two charts are related: it is a reason to check both files before "
            "trusting either.",
            "The two files are NOT interchangeable. A positive and a negative of one "
            "chart are different specimens with different aim data, and running either "
            "against the other's file inverts the ramp.",
        ),
    ),
)


def build(ladder: Ladder) -> str:
    lightness = density_to_lightness(
        np.asarray(ladder.densities), anchor_L=FADGI_TRANSPARENCY_HIGHLIGHT_L
    )
    rows = [
        f"{block}{step}\t{L:.3f}\t0.000\t0.000\t{density:.2f}"
        for block in ladder.blocks
        for step, (density, L) in enumerate(zip(ladder.densities, lightness), start=1)
    ]
    caveat = " ".join((SHARED_CAVEAT, *ladder.notes))
    lines = [
        "CGATS.17",
        "",
        f'ORIGINATOR\t"Derived from {ladder.originator}"',
        f'DESCRIPTOR\t"{ladder.descriptor}"',
        f'MANUFACTURER\t"{ladder.manufacturer}"',
        'ILLUMINATION_NAME\t"D50"',
        'OBSERVER_ANGLE\t"2"',
        'KEYWORD\t"DELTACHROME_REFERENCE_BASIS"',
        'DELTACHROME_REFERENCE_BASIS\t"manufacturer_nominal"',
        'KEYWORD\t"DELTACHROME_NOMINAL_SOURCE"',
        f'DELTACHROME_NOMINAL_SOURCE\t"{ladder.source}"',
        'KEYWORD\t"DELTACHROME_NOMINAL_CAVEAT"',
        f'DELTACHROME_NOMINAL_CAVEAT\t"{caveat}"',
        "",
        "NUMBER_OF_FIELDS\t5",
        "BEGIN_DATA_FORMAT",
        "SAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B\tD",
        "END_DATA_FORMAT",
        "",
        f"NUMBER_OF_SETS\t{len(rows)}",
        "BEGIN_DATA",
        *rows,
        "END_DATA",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    for ladder in LADDERS:
        (DEFINITIONS / ladder.output).write_text(build(ladder), encoding="utf-8")
        lightness = density_to_lightness(
            np.asarray(ladder.densities), anchor_L=FADGI_TRANSPARENCY_HIGHLIGHT_L
        )
        print(
            f"wrote {ladder.output}  "
            f"({len(ladder.densities)} steps x {len(ladder.blocks)} blocks "
            f"{'/'.join(ladder.blocks)})"
        )
        for density, L in zip(ladder.densities, lightness):
            print(f"    D {density:5.2f} -> L* {L:7.3f}")


if __name__ == "__main__":
    main()
