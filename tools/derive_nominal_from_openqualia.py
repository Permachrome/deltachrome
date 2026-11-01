"""Build a target's nominal reference from published per-unit measurements.

The two Golden Thread nominals this repository shipped were each **one print
run's printed table** -- the Device Level's transcribed from unit D00453's card,
the Object Level's from OpenDICE's profile, which matches unit M000390's. That is
a perfectly reasonable thing to ship when it is the only data there is, and it
turns out to be measurably the wrong thing once a population is available: a
single batch's table sits wherever that batch sat, and the population sits
somewhere else.

DT Nexus publishes per-unit measurements for these targets under the OpenQualia
label standard -- 1,454 Device Level units and 1,266 Object Level 1x, public.
This averages a stratified sample of them into a nominal that sits at the centre
of the population instead of at one batch's corner.

**The test that decides whether that is worth doing** is not the distance from a
nominal to some unit, which has no scale. It is that distance against **how far
two real units sit from each other**. A nominal closer than the unit-to-unit
spread is centred in the population; one further away is offset from it, and is
then a worse predictor of the chart in front of you than a randomly chosen other
chart of the same model would be. Both shipped Golden Thread nominals were
outside their spread. Run with ``--report`` this prints that comparison, scored
**leave-one-out** so a candidate is never tested against a unit it was built
from.

Three choices in here were measured rather than argued, and each could sensibly
have gone the other way:

* **Pooled across measurement conditions.** The published units are a mix of M0,
  M1 and M2, confounded with instrument era, and the group means differ by
  0.51-0.67 dE2000 -- the same order as the within-group unit spread. Restricting
  to one condition gives a cleaner definition and a *worse* nominal: on the
  Device Level, an M1-only mean scores 0.483 against M1 units and 0.833 against
  M2 units, where the pooled mean scores 0.601 overall and beats every
  single-condition mean. The reason is what a nominal is *for*: it is used
  exactly when nobody has a measurement of this unit, so there is no condition to
  match, and what matters is the expected distance to an arbitrary chart.
* **The stated Lab is averaged, not the spectra.** Both are present and the two
  differ by 0.05 dE2000, which is simply this engine's own integration agreement
  rather than a difference between methods. The project's rule decides it: a
  file's own stated Lab is preferred to integrating its spectra, because a reader
  that silently disagrees with a number the file asserts is worse than one that
  cannot read spectra.
* **A stratified sample, not the whole catalogue.** Held-out error converges by
  about ten units to the floor implied by the unit-to-unit spread itself
  (sigma/sqrt(2) = 0.587 on the Device Level), so fifty is already at it and
  1,454 would buy nothing. The sample is drawn evenly across years because
  *drift* is the thing a small sample could get wrong -- and it was checked: every
  year's own mean sits 0.30-0.59 dE from the whole population with no trend.

The serial lists are committed below rather than re-sampled on each run, because
the point of a derivation tool is that it produces the same file twice. Downloads
are cached, so a re-run costs nothing and does not hammer somebody else's server.

Usage::

    python tools/derive_nominal_from_openqualia.py --report
    python tools/derive_nominal_from_openqualia.py --write
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltachrome.imaging.color import delta_e  # noqa: E402
from deltachrome.reference import read_reference  # noqa: E402
from deltachrome.reference.openqualia import fetch_measurement, parse_label  # noqa: E402
from deltachrome.reference.sample_ids import reconcile_sample_ids  # noqa: E402
from deltachrome.targets.registry import DEFINITIONS_DIR, load_target  # noqa: E402

HOST = "https://targets.digitaltransitions.com/measurements"
CACHE = Path(__file__).resolve().parents[1] / "out" / "openqualia-units"


@dataclass(frozen=True)
class Source:
    """One target's derivation: where the units come from and what to write."""

    target_id: str
    manufacturer: str
    target_type: str
    output: str
    descriptor: str
    #: Sampled evenly across the years the host publishes, so that drift -- the
    #: one thing a small sample could get wrong -- is inside the sample rather
    #: than outside it.
    serials: tuple[str, ...]
    replaces: str


SOURCES = (
    Source(
        target_id="isa_golden_thread_dlt_v1",
        manufacturer="ISA",
        target_type="DLT",
        output="isa_golden_thread_dlt_v1_nominal.txt",
        descriptor="Golden Thread Device Level Target -- batch mean of published per-unit measurements",
        replaces=(
            "the table printed on unit D00453's own card, transcribed from a capture of it: "
            "one print run's batch average, which sits 1.086 dE2000 from an average published "
            "unit against a unit-to-unit spread of 0.830"
        ),
        serials=(
            "D00001", "D00031", "D00054", "D00077", "D00100", "D00102",
            "D00111", "D00126", "D00134", "D00142", "D00143", "D00162",
            "D00178", "D00194", "D00210", "D00211", "D00224", "D00233",
            "D00243", "D00252", "D00259", "D00282", "D00306", "D00334",
            "D00358", "D00359", "D00377", "D00396", "D00413", "D00432",
            "D00055", "D00462", "D00497", "D00531", "D00564", "D00472",
            "D00625", "D00688", "D00750", "D00812", "D00813", "D00934",
            "D01053", "D01174", "D01294", "D01139", "D01344", "D01396",
            "D01447", "D01499",
        ),
    ),
    Source(
        target_id="isa_object_level_1x_v1",
        manufacturer="ISA",
        target_type="OLT1x",
        output="isa_object_level_1x_v1_nominal.txt",
        descriptor="ISA Golden Thread Object Level Target 1X -- batch mean of published per-unit measurements",
        replaces=(
            "OpenDICE's Profile_ObjectDICE.txt, the table printed on unit M000390's own card: "
            "one print run's batch average, which sits 0.853 dE2000 from an average published "
            "unit against a unit-to-unit spread of 0.610"
        ),
        serials=(
            "M000559", "M000576", "M000593", "M000610", "M000628", "M000629",
            "M000635", "M000641", "M000647", "M000653", "M000668", "M000768",
            "M000788", "M000808", "M000828", "M000829", "M000885", "M000941",
            "M000997", "M001053", "M001054", "M001122", "M001191", "M001260",
            "M001329", "M001330", "M001380", "M001430", "M001479", "M001529",
            "M001530", "M001568", "M001607", "M001647", "M001686", "M001687",
            "M001722", "M001758", "M001794", "M001830",
        ),
    ),
)

CAVEAT = (
    "A batch mean of {n} individually measured units, not a measurement of the chart in "
    "front of you and not a design aim. It is the best available answer to 'what does a "
    "chart of this model look like' and it is still not an answer to 'what does THIS chart "
    "look like', which is why a run against it reports ADVISORY and makes no conformance "
    "claim for the specimen. Two things follow. First, if your own unit's serial is "
    "published -- and most are -- fetch it: Deltachrome reads the chart's OpenQualia label "
    "and will pair the capture with that unit's own file, which is a conformance test where "
    "this is not. Second, the values printed on your own card are a batch average too, of "
    "your batch rather than of the population; they are not more authoritative than these "
    "and are drawn from a smaller sample. The residual error this leaves is genuine "
    "unit-to-unit variation and cannot be reduced by averaging more units: held-out error "
    "converges to the floor set by the population's own scatter by about ten units."
)

SOURCE_NOTE = (
    "Mean of the L*a*b* values stated by {n} per-unit measurement files published by "
    "Digital Transitions under the OpenQualia label standard, fetched from "
    "{host}?Manufacturer={mfr}&TargetType={ttype}&TargetID=<serial>. Serials, sampled "
    "evenly across the years the host publishes: {serials}. "
    "Replaces {replaces}. "
    "Scored leave-one-out against the same units, this file sits {after:.3f} dE2000 from an "
    "average chart against the previous file's {before:.3f}, with the unit-to-unit spread at "
    "{spread:.3f}. Pooled across the M0/M1/M2 measurement conditions the units are a mix of, "
    "which was measured rather than assumed: a single-condition mean is better within its own "
    "condition and worse overall, and a nominal is used precisely when there is no per-unit "
    "measurement whose condition it could be matched to. The files' own stated Lab is averaged "
    "rather than their spectra; the two differ by 0.05 dE2000, which is this engine's "
    "integration agreement rather than a difference between the methods. "
    "Regenerate with tools/derive_nominal_from_openqualia.py."
)


def unit_path(source: Source, serial: str) -> Path:
    return CACHE / source.target_type / f"{serial}.oqm.txt"


def obtain(source: Source, serial: str) -> Path:
    """Fetch one unit's measurement, or reuse the cached copy."""
    path = unit_path(source, serial)
    if path.exists() and path.stat().st_size:
        return path
    label = parse_label(
        f"{HOST}?Manufacturer={source.manufacturer}"
        f"&TargetType={source.target_type}&TargetID={serial}"
    )
    fetched = fetch_measurement(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fetched.content)
    return path


def load_units(source: Source) -> tuple[list[str], np.ndarray, list[str]]:
    """Every unit's Lab table, on this target's own patch ids."""
    target = load_target(source.target_id)
    ids = [str(n) for n in range(1, 31)]
    serials, rows, conditions = [], [], []
    for serial in source.serials:
        try:
            raw = read_reference(obtain(source, serial))
        except Exception as exc:  # noqa: BLE001 -- a unit that will not load is skipped, loudly
            print(f"    ! {serial}: {exc}", file=sys.stderr)
            continue
        condition = raw.metadata.get("MEASUREMENT_CONDITION") or raw.measurement_mode or "?"
        measurement, _ = reconcile_sample_ids(raw, target)
        try:
            lab = np.array([measurement.samples[i].lab for i in ids], dtype=float)
        except (KeyError, TypeError):
            print(f"    ! {serial}: does not cover all {len(ids)} patches", file=sys.stderr)
            continue
        if lab.shape != (len(ids), 3):
            continue
        serials.append(serial)
        rows.append(lab)
        conditions.append(condition)
    return serials, np.stack(rows), conditions


def leave_one_out(units: np.ndarray) -> float:
    """Held-out error of a mean built from every other unit."""
    return float(np.mean([
        delta_e(np.delete(units, i, axis=0).mean(axis=0), units[i], "delta_e_2000").mean()
        for i in range(len(units))
    ]))


def spread(units: np.ndarray) -> float:
    return float(np.mean([
        delta_e(units[i], units[j], "delta_e_2000").mean()
        for i in range(len(units)) for j in range(i + 1, len(units))
    ]))


def previous_error(source: Source, units: np.ndarray) -> float | None:
    """How far the file being replaced sits from an average unit."""
    path = DEFINITIONS_DIR / source.output
    if not path.exists():
        return None
    old = read_reference(path)
    ids = [str(n) for n in range(1, 31)]
    try:
        table = np.array([old.samples[i].lab for i in ids], dtype=float)
    except (KeyError, TypeError):
        return None
    return float(np.mean([delta_e(table, u, "delta_e_2000").mean() for u in units]))


def render(source: Source, serials, units, conditions, before, after, sprd) -> str:
    mean = units.mean(axis=0)
    ids = [str(n) for n in range(1, 31)]
    note = SOURCE_NOTE.format(
        n=len(serials), host=HOST, mfr=source.manufacturer, ttype=source.target_type,
        serials=" ".join(serials), replaces=source.replaces,
        after=after, before=before if before is not None else float("nan"), spread=sprd,
    )
    lines = [
        "CGATS.17",
        "",
        'ORIGINATOR\t"Derived by Deltachrome from per-unit measurements published '
        'by Digital Transitions under the OpenQualia label standard"',
        f'DESCRIPTOR\t"{source.descriptor}"',
        'MANUFACTURER\t"Image Science Associates"',
        'ILLUMINATION_NAME\t"D50"',
        'OBSERVER_ANGLE\t"2"',
        'KEYWORD\t"DELTACHROME_REFERENCE_BASIS"',
        # The same test the parser applies, made here so the file and the reader
        # cannot disagree: a mean whose held-out error is smaller than the spread of
        # the units it was built from is centred in the population, and a mean that
        # is not is no better than the printed table it replaces. Neither earns a
        # conformance claim -- neither is a measurement of the chart in front of the
        # camera -- and the difference is worth telling a reader who could act on it.
        f'DELTACHROME_REFERENCE_BASIS\t"'
        f'{"population_mean" if after < sprd else "manufacturer_nominal"}"',
        'KEYWORD\t"DELTACHROME_NOMINAL_HELD_OUT_DE"',
        f"DELTACHROME_NOMINAL_HELD_OUT_DE\t{after:.3f}",
        'KEYWORD\t"DELTACHROME_NOMINAL_POPULATION_SPREAD_DE"',
        f"DELTACHROME_NOMINAL_POPULATION_SPREAD_DE\t{sprd:.3f}",
        'KEYWORD\t"DELTACHROME_NOMINAL_POPULATION_UNITS"',
        f"DELTACHROME_NOMINAL_POPULATION_UNITS\t{len(serials)}",
        'KEYWORD\t"DELTACHROME_NOMINAL_SOURCE"',
        f'DELTACHROME_NOMINAL_SOURCE\t"{note}"',
        'KEYWORD\t"DELTACHROME_NOMINAL_CAVEAT"',
        f'DELTACHROME_NOMINAL_CAVEAT\t"{CAVEAT.format(n=len(serials))}"',
        "",
        "NUMBER_OF_FIELDS\t4",
        "BEGIN_DATA_FORMAT",
        "SAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B",
        "END_DATA_FORMAT",
        "",
        f"NUMBER_OF_SETS\t{len(ids)}",
        "BEGIN_DATA",
    ]
    for i, patch in enumerate(ids):
        lines.append(f"{patch}\t{mean[i,0]:.3f}\t{mean[i,1]:.3f}\t{mean[i,2]:.3f}")
    lines += ["END_DATA", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="overwrite the shipped nominal files")
    parser.add_argument("--report", action="store_true",
                        help="show the comparison without writing anything")
    args = parser.parse_args()
    if not (args.write or args.report):
        args.report = True

    for source in SOURCES:
        target = load_target(source.target_id)
        print(f"\n=== {target.display_name}")
        serials, units, conditions = load_units(source)
        mix = {c: conditions.count(c) for c in sorted(set(conditions))}
        print(f"    {len(serials)} units, conditions {mix}")

        sprd = spread(units)
        after = leave_one_out(units)
        before = previous_error(source, units)
        print(f"    unit vs unit               {sprd:.3f} dE2000")
        if before is not None:
            verdict = "OUTSIDE" if before > sprd else "inside"
            print(f"    shipped nominal vs a unit  {before:.3f}  ({verdict} the spread)")
        print(f"    this derivation (LOO)      {after:.3f}")

        text = render(source, serials, units, conditions, before, after, sprd)
        if args.write:
            path = DEFINITIONS_DIR / source.output
            path.write_text(text, encoding="utf-8")
            print(f"    written: {path}")
        else:
            print("    (--write to replace the shipped file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
