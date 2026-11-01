"""Turn an OpenDICE reference profile into the CGATS a target definition reads.

OpenDICE -- the Library of Congress's open implementation of the FADGI device
level analysis -- ships one ``Profile_*.txt`` per target it recognizes. The
format is a tab-separated table with one header line:

    <tab>L<tab>a <tab>b<tab>D
    A0<tab>92.628<tab>-0.652<tab>2.51
    D6<tab>92.33<tab>-0.658<tab>2.322<tab>0.09

The first column is the patch id as printed on the chart. ``D`` is filled in
**only on the neutral ramp steps**, which is what makes the ramp identifiable
without inspecting chroma, and is how each target's ``ramp_position`` ordering
below was derived rather than guessed.

**What basis these values are.** They are not design aims. A design aim is a
round number -- the UTT's nominal file is whole L\\* units throughout, which is
what gives it away. These are not: 388 of NGT1's 390 values are multiples of
0.002 while only 90 are multiples of 0.01, which is the arithmetic fingerprint of
a **mean of five readings each recorded to two decimals**. The IT8.7/2 profile
carries the same signature. ISA state the convention outright on the Golden
Thread Object Level Target, which prints "All values are batch averages" beside
its own table.

So these are measurements -- of a batch, not of the chart in front of you. That
is worse than an aim in one specific way and better in another. Better, because a
batch mean is closer to any given unit than a design aim is. Worse, because it
*looks* like a per-unit measurement and is not, and a run against it cannot be a
conformance claim for the specimen. So they ship on the ``manufacturer_nominal``
basis, the run reports ADVISORY, and the caveat in each file says which of the
two kinds of "nominal" it is.

Run from the repository root:

    python tools/generate_opendice_nominal.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "data/additional_targets"
DEFINITIONS = ROOT / "src/deltachrome/targets/definitions"

BATCH_AVERAGE_TELL = (
    "The values are batch averages, not design aims and not a measurement of the "
    "chart in front of you. Their own arithmetic says so: they quantize to 0.002 "
    "rather than to 0.01, which is what a mean of five readings each taken to two "
    "decimals does, and ISA print the same convention outright on the Golden "
    "Thread Object Level Target -- 'All values are batch averages'. A batch mean "
    "sits closer to any individual unit than a design aim would, which makes it "
    "better data; it also looks exactly like a per-unit measurement and is not, "
    "which is why a run against it reports ADVISORY and makes no conformance "
    "claim for the specimen."
)


@dataclass(frozen=True)
class Source:
    """One profile to convert, and everything the output file has to declare."""

    profile: str
    output: str
    descriptor: str
    manufacturer: str
    originator: str
    source: str
    caveat: str
    #: Extra lines appended to the caveat. Kept separate so the shared text above
    #: stays one string and each target says only what is true of it.
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Patches the source profile does not carry at all, appended after its own
    #: rows verbatim -- for the IT8 grayscale strip, which OpenDICE never measured
    #: and which comes from the chart's own vendor batch file instead. Each row is
    #: (patch_id, L*, a*, b*); these never carry a density value.
    extra_rows: tuple[tuple[str, float, float, float], ...] = field(default_factory=tuple)
    #: Patches the source profile carries and the output must not -- values true of
    #: the measured specimen's maker and of nobody else's. See ``IT8_VENDOR_COLUMNS``.
    omit: frozenset[str] = field(default_factory=frozenset)


# The IT8 grayscale strip, GS0-GS23, in neither OpenDICE profile -- fetched
# directly from LaserSoft's own currently published batch-average files
# (silverfast.com/it8calibration, fetched 2026). See each IT8 Source's own
# `source`/`caveat` text for what these batch labels do and do not establish.
#: IT8.7/1 and IT8.7/2 fix columns 1-19 of the colour array and leave A20-L22 to
#: the manufacturer, so a value there is true of one vendor's chart and of no
#: other's. Measured: against a Wolf Faust IT8.7/2's own measurement, the LaserSoft
#: values this file ships sit 3.8 dE2000 off across columns 1-19 -- batch and print
#: medium -- and 31.3 across 20-22, where the two vendors printed different colours.
#: A fallback that judged those would fail every other vendor's chart on patches
#: that were never meant to match, so they are left out: a chart's own batch file
#: carries them and they are judged then, and not otherwise.
IT8_VENDOR_COLUMNS = frozenset(f"{row}{col}" for row in "ABCDEFGHIJKL" for col in (20, 21, 22))
IT8_VENDOR_NOTE = (
    "Columns 20-22 (A20-L22) are not in this file. The IT8.7 standard leaves them to "
    "each manufacturer, so values for them describe one vendor's chart and no other; "
    "they are judged only against a reference file for the chart in front of the "
    "camera, which carries them."
)

GS_STEPS_E180106 = (
    ("GS0", 87.06, -0.89, -1.62), ("GS1", 84.19, -0.55, -2.01),
    ("GS2", 81.09, -0.70, -1.89), ("GS3", 76.96, -1.28, -1.18),
    ("GS4", 73.95, -1.37, -0.99), ("GS5", 70.33, -1.35, -1.03),
    ("GS6", 67.05, -1.38, -0.85), ("GS7", 63.18, -1.33, -0.80),
    ("GS8", 59.09, -1.19, -0.94), ("GS9", 54.73, -1.50, -0.83),
    ("GS10", 51.01, -2.83, -1.28), ("GS11", 47.47, -2.56, -1.58),
    ("GS12", 42.85, -2.32, -0.92), ("GS13", 39.40, -2.69, -1.18),
    ("GS14", 35.72, -2.73, -1.19), ("GS15", 31.17, -2.68, -1.46),
    ("GS16", 27.09, -2.63, -1.88), ("GS17", 23.71, -2.90, -1.48),
    ("GS18", 18.96, -2.10, -1.47), ("GS19", 14.67, -2.13, -1.86),
    ("GS20", 11.05, -1.64, -1.53), ("GS21", 6.15, -1.40, -1.49),
    ("GS22", 2.83, -0.93, -1.12), ("GS23", 1.46, -0.14, -0.38),
)
GS_STEPS_R100213 = (
    ("GS0", 89.90, -1.32, 1.75), ("GS1", 86.97, -1.71, 2.63),
    ("GS2", 82.70, -2.17, 2.83), ("GS3", 78.52, -2.27, 3.13),
    ("GS4", 73.79, -1.35, 3.59), ("GS5", 70.23, -1.33, 3.84),
    ("GS6", 66.27, -0.41, 3.68), ("GS7", 62.48, -0.61, 3.40),
    ("GS8", 58.81, 0.04, 3.50), ("GS9", 55.82, -0.04, 3.25),
    ("GS10", 52.86, -0.15, 3.21), ("GS11", 49.79, 0.06, 2.89),
    ("GS12", 46.81, -0.27, 2.65), ("GS13", 43.30, -0.25, 2.21),
    ("GS14", 39.87, 0.44, 2.28), ("GS15", 37.03, 0.26, 2.64),
    ("GS16", 33.16, 0.30, 2.77), ("GS17", 29.27, 1.55, 3.18),
    ("GS18", 24.54, 1.87, 3.90), ("GS19", 20.61, 2.69, 3.93),
    ("GS20", 16.66, 3.33, 4.10), ("GS21", 13.00, 4.30, 3.45),
    ("GS22", 9.04, 3.74, 2.71), ("GS23", 6.26, 0.20, 2.48),
)

SOURCES = (
    Source(
        profile="OpenDICE_Profile_NGT1.txt",
        output="avian_ngt_v1_nominal.txt",
        descriptor="Avian Rochester Next Generation Target -- batch average aim values",
        manufacturer="Avian Rochester",
        originator="OpenDICE Profile_NGT1.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_NGT1.txt, published by the Library of Congress with "
            "its sample capture of unit AR-NGT-2016023. Confirmed to be this target "
            "by the layout: 130 patch ids A0-M9 against the chart's own printed "
            "column letters A-M and row digits 0-9, and by the twelve density-bearing "
            "steps D2-I2 and D6-I6 landing exactly on the two rows of neutrals the "
            "chart prints."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "Avian Rochester measure and ship data with each chart; that file "
            "supersedes this one and should be used whenever it is to hand.",
        ),
    ),
    Source(
        profile="OpenDICE_Profile_DT_NGT.txt",
        output="avian_dt_ngt_v2_nominal.txt",
        descriptor="DT Next Generation Target V2 -- batch average aim values",
        manufacturer="Avian Rochester",
        originator="OpenDICE Profile_DT_NGT.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_DT_NGT.txt, published with its sample capture of unit "
            "DT-AR-2019056. The V2 shares the earlier NGT's lattice and patch ids "
            "exactly -- same 13 columns, same 10 rows, same twelve neutral steps at "
            "D2-I2 and D6-I6 -- and is a different chart for all that: its neutrals "
            "are printed lighter, at a mean L* 96.1 against the NGT's 92.4. The two "
            "reference files are NOT interchangeable and this is the check that says "
            "so."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "Avian Rochester measure and ship data with each chart; that file "
            "supersedes this one and should be used whenever it is to hand.",
        ),
    ),
    Source(
        profile="OpenDICE_Profile_ObjectDICE.txt",
        output="isa_object_level_1x_v1_nominal.txt",
        descriptor="ISA Golden Thread Object Level Target 1X -- batch average aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_ObjectDICE.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_ObjectDICE.txt. Confirmed against the chart itself: "
            "this target PRINTS its own reference table beside the patches, and the "
            "table on OpenDICE's ObjectLevel_2 sample -- serial M000390, captioned "
            "'All values are batch averages' -- matches this file to the last decimal "
            "on every value transcribed from it."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "THE TABLE PRINTED ON YOUR OWN CARD IS THE ONE TO USE, and this target "
            "makes that unusually easy: the values are printed beside the patches. "
            "OpenDICE ships two sample captures of this target and they carry "
            "DIFFERENT tables -- the other one, ObjectDICE.tif, prints the same table "
            "as Golden Thread Device Level Target unit D00453, which this repository "
            "already ships as isa_golden_thread_dlt_v1_nominal.txt. The two differ by "
            "0.99 Delta E2000 on average and 2.46 at worst, which is batch drift "
            "rather than a redesign; the same batch table appears on both the Device "
            "Level and Object Level products, which is what says it identifies a print "
            "run and not a product.",
        ),
    ),
    Source(
        profile="OpenDICE_Profile_FADGI_19264.txt",
        output="isa_fadgi_19264_v1_nominal.txt",
        descriptor="ISA FADGI - ISO 19264, early version -- batch average aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_FADGI_19264.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_FADGI_19264.txt, published with its sample capture of "
            "the chart in its early form -- the one that prints its patch numbers 1 to "
            "168 beside the patches, which is what makes these sample names checkable "
            "against the chart by eye. Confirmed to be this target three ways from the "
            "capture: 168 patches fit one 14 x 21 lattice to 0.11% of the fiducial "
            "quad; the sixteen density-bearing entries land exactly on the sixteen "
            "neutral circles, whose measured RGB range is 4.4 against 36.4 for the "
            "array as a whole; and measured luminance correlates with the stated L* at "
            "0.973, against 0.006 for the same values shuffled."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "THIS FILE AND THE COMMERCIAL VERSION'S ARE NOT INTERCHANGEABLE. The two "
            "charts share a layout exactly -- same lattice, same 168 positions, same "
            "sixteen-step neutral ramp at the same densities -- and their colour "
            "patches were re-chosen between versions: only 30 of 168 have any close "
            "match in the other file, and the worst pair differs by 74 Delta E2000. "
            "Sharing a layout is what makes the confusion possible and what makes it "
            "invisible, since every patch still samples correctly.",
        ),
    ),
    Source(
        profile="OpenDICE_Profile_FADGI_19264_Commercial.txt",
        output="isa_fadgi_19264_commercial_v1_nominal.txt",
        descriptor="ISA FADGI - ISO 19264, commercial release -- batch average aim values",
        manufacturer="Image Science Associates",
        originator="OpenDICE Profile_FADGI_19264_Commercial.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_FADGI_19264_Commercial.txt, published with its sample "
            "capture of unit S/N F00025. The sample names are 1 to 168 in reading "
            "order over the occupied cells; this version of the chart prints grid "
            "labels A-N by 1-16 instead of those numbers, and the definition records "
            "the printed label beside every patch so the two can be reconciled."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "THIS FILE AND THE EARLY VERSION'S ARE NOT INTERCHANGEABLE -- see the "
            "early version's file for the evidence. The neutral ramp is the one part "
            "the two do share: all sixteen densities are identical to the digit.",
        ),
    ),
    Source(
        profile="OpenDICE_Profile_IT8-7.1.txt",
        output="it8_7_1_transmissive_v1_nominal.txt",
        descriptor="IT8.7/1 transmissive target -- batch aim values",
        manufacturer="(various; this batch LaserSoft Imaging, on Fuji Provia)",
        originator="OpenDICE Profile_IT8-7.1.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_IT8-7.1.txt, published with its sample capture of a "
            "LaserSoft Imaging IT8.7/1 target printed on Fuji Provia Professional "
            "Film, batch E180106. The 264 patch ids A1-L22 match the chart's own "
            "printed row letters and column numbers, and the twelve density-bearing "
            "steps are column 16 in row order A to L, which is the neutral column of "
            "the IT8 layout. GS0-GS23, the 24-step grayscale strip below the array, "
            "were added directly from LaserSoft's own currently published "
            "batch-average file for E180106 (silverfast.com/it8calibration, fetched "
            "2026), rather than through OpenDICE, whose profile carries no values for "
            "the strip at all. That file's own A1-L22 values agree with OpenDICE's to "
            "0.000 dE(ab) across all 264 patches, exactly -- the strongest cross-check "
            "this project has for any reference file -- so the GS values it also "
            "carries are trusted on the same batch identity."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "IT8 reference data is per BATCH by design, and the batch is printed on "
            "the chart -- E180106 on the specimen these values came from, beside a "
            "barcode carrying the same string. Every IT8 vendor publishes the data "
            "for every batch they sell. Using this file against a chart from another "
            "batch is the exact error the per-unit rule exists to prevent, and it is "
            "avoidable here in a way it is not for most targets: read the chart's own "
            "batch number and fetch that batch's file.",
            "These values describe a photographic emulsion on a film base. Lightness "
            "is what the reference states; density is stated only for the neutral "
            "column.",
        ),
        extra_rows=GS_STEPS_E180106,
        omit=IT8_VENDOR_COLUMNS,
    ),
    Source(
        profile="OpenDICE_Profile_IT8.7.2.txt",
        output="it8_7_2_reflective_v1_nominal.txt",
        descriptor="IT8.7/2 reflective target -- batch aim values",
        manufacturer="(various; LaserSoft Imaging, on Kodak Endura -- batch not confirmed)",
        originator="OpenDICE Profile_IT8.7.2.txt (Library of Congress)",
        source=(
            "OpenDICE's Profile_IT8.7.2.txt, published with its sample capture of a "
            "LaserSoft Imaging IT8.7/2 target printed on Kodak Professional Endura. "
            "The 264 patch ids A1-L22 match the chart's own printed row letters and "
            "column numbers, and the twelve density-bearing steps are column 16 in "
            "row order A to L, which is the neutral column of the IT8 layout. THE "
            "BATCH IS NOT R100213, AND THAT WAS CHECKED RATHER THAN ASSUMED: "
            "OpenDICE's own file carries no CGATS preamble at all -- no ORIGINATOR, "
            "SERIAL or batch field of any kind, only a bare patch/L*/a*/b* table, per "
            "digitizationguidelines.gov's own note that its reference files were "
            "'measured by Konica Minolta FD7' -- so these 264 values are an "
            "independent spectrophotometer measurement of one physical chart, not a "
            "republished manufacturer average. Checked against LaserSoft's own "
            "currently published R100213 batch-average file "
            "(silverfast.com/it8calibration, fetched 2026), the two disagree by a "
            "mean of 1.6 dE(ab) and up to 8.6 on one patch (K2) -- more than the "
            "0.2-0.9 dE2000 this project has measured as ordinary within-batch spread "
            "elsewhere, so R100213 is probably the wrong batch, or FD7 and the mean's "
            "own instrument disagree by more than either alone, or both. GS0-GS23, "
            "the 24-step grayscale strip below the array, could not be measured the "
            "same way -- FD7 was pointed at the colour array, not the strip -- so "
            "they are taken from R100213 anyway, flagged: the best data available for "
            "a feature this file otherwise says nothing about, from a batch this "
            "file's other 264 patches do not confirm."
        ),
        caveat=BATCH_AVERAGE_TELL,
        notes=(
            "IT8 reference data is per BATCH by design, and the batch is printed on "
            "the chart, beside a barcode carrying the same string. Every IT8 vendor "
            "publishes the data for every batch they sell. Using this file against a "
            "chart from another batch is the exact error the per-unit rule exists to "
            "prevent, and it is avoidable here in a way it is not for most targets: "
            "read the chart's own batch number and fetch that batch's file. For the "
            "264-patch colour array specifically, that advice is stronger than usual "
            "-- these are not a batch file at all, but one physical chart's own FD7 "
            "measurement, so they already carry the specimen's own accident with no "
            "batch mean to average it away.",
        ),
        extra_rows=GS_STEPS_R100213,
        omit=IT8_VENDOR_COLUMNS,
    ),
)


def read_profile(path: Path) -> list[tuple[str, float, float, float, str]]:
    """Patch id, L*, a*, b* and density-as-written for every row of a profile."""
    rows: list[tuple[str, float, float, float, str]] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise SystemExit(f"{path.name} is empty.")
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) < 4 or not cells[1].strip():
            raise SystemExit(f"{path.name}: cannot read row {line!r}")
        density = cells[4].strip() if len(cells) > 4 else ""
        rows.append(
            (cells[0].strip(), float(cells[1]), float(cells[2]), float(cells[3]), density)
        )
    return rows


def build(source: Source, rows: list[tuple[str, float, float, float, str]]) -> str:
    has_density = any(density for *_, density in rows)
    fields = ["SAMPLE_NAME", "LAB_L", "LAB_A", "LAB_B"] + (["D"] if has_density else [])

    data = []
    rows = [row for row in rows if row[0] not in source.omit]
    for patch_id, lightness, a, b, density in rows:
        cells = [patch_id, f"{lightness:.3f}", f"{a:.3f}", f"{b:.3f}"]
        if has_density:
            # Written only where the profile wrote one. A blank is not a zero:
            # zero density is a perfectly clear step, and every non-ramp patch
            # would then claim to be one.
            cells.append(density if density else "-")
        data.append("\t".join(cells))

    for patch_id, lightness, a, b in source.extra_rows:
        cells = [patch_id, f"{lightness:.3f}", f"{a:.3f}", f"{b:.3f}"]
        if has_density:
            cells.append("-")  # never measured for the extra rows.
        data.append("\t".join(cells))

    caveat = " ".join((source.caveat, *source.notes,
                       *((IT8_VENDOR_NOTE,) if source.omit == IT8_VENDOR_COLUMNS else ())))
    lines = [
        "CGATS.17",
        "",
        f'ORIGINATOR\t"Derived from {source.originator}"',
        f'DESCRIPTOR\t"{source.descriptor}"',
        f'MANUFACTURER\t"{source.manufacturer}"',
        'ILLUMINATION_NAME\t"D50"',
        'OBSERVER_ANGLE\t"2"',
        'KEYWORD\t"DELTACHROME_REFERENCE_BASIS"',
        'DELTACHROME_REFERENCE_BASIS\t"manufacturer_nominal"',
        'KEYWORD\t"DELTACHROME_NOMINAL_SOURCE"',
        f'DELTACHROME_NOMINAL_SOURCE\t"{source.source}"',
        'KEYWORD\t"DELTACHROME_NOMINAL_CAVEAT"',
        f'DELTACHROME_NOMINAL_CAVEAT\t"{caveat}"',
        "",
        f"NUMBER_OF_FIELDS\t{len(fields)}",
        "BEGIN_DATA_FORMAT",
        "\t".join(fields),
        "END_DATA_FORMAT",
        "",
        f"NUMBER_OF_SETS\t{len(data)}",
        "BEGIN_DATA",
        *data,
        "END_DATA",
    ]
    return "\n".join(lines) + "\n"


def ramp_order(rows: list[tuple[str, float, float, float, str]]) -> list[tuple[str, float]]:
    """The ramp's patch ids in lightest-first order, with their densities.

    Ordered by density rather than by position, because on the NGT the twelve
    steps are printed as two rows running in opposite directions and reading them
    left to right interleaves the ladder. Density is the quantity the ladder is
    built on, so it is what orders it.
    """
    steps = [(patch, float(density)) for patch, *_, density in rows if density]
    return sorted(steps, key=lambda step: step[1])


def main() -> None:
    for source in SOURCES:
        existing = DEFINITIONS / source.output
        # A file another tool now writes -- the batch means derived from published
        # per-unit measurements replaced two of these -- is not this tool's to
        # overwrite, and running it used to put the older values back silently.
        if existing.exists() and "population_mean" in existing.read_text(encoding="utf-8"):
            print(f"skipped {source.output}  (now written by derive_nominal_from_openqualia.py)")
            continue
        rows = read_profile(PROFILES / source.profile)
        (DEFINITIONS / source.output).write_text(build(source, rows), encoding="utf-8")
        steps = ramp_order(rows)
        print(f"wrote {source.output}  ({len(rows)} patches, {len(steps)} ramp steps)")
        print(
            "   ramp, lightest first: "
            + ", ".join(f"{patch}({density:g})" for patch, density in steps)
        )


if __name__ == "__main__":
    main()
