# Adding a target

A chart arrives as a capture, usually a reference file, and a name. What has to
come out the other end is a chart that places itself, names itself without
stealing anybody else's captures, joins its reference, and appears everywhere the
application lists what it supports. Most of the ways this goes wrong are silent —
a lattice a third of a patch out returns 864 plausible ΔE values — so every step
below ends in a check with a number, not in a look.

The steps are in the order they are worth doing. **Step 1 decides whether there
is a new target at all**, and is the one most easily skipped: the first two charts
taken through this process both turned out to be served by definitions already
shipped, and what each needed was somewhere else entirely.

---

## 0. Intake

Put what arrived in `data/new_targets/` — it is gitignored, because a capture is
the operator's own and a reference file carries an instrument serial. Record,
before anything is measured:

- **What the card prints about itself**: maker, model, edition or date, print
  medium, batch or serial, and what kind of code (barcode, QR, Data Matrix, text
  only). Read it off the capture, not from memory of the product line — the
  LaserSoft 120 turned out to print *ISO 12641-2 Candidate 2017:10*, which is not
  the *Advanced* chart its 4x5 sibling is.
- **What the capture is**: `deltachrome inspect-icc <capture>` — depth, profile,
  channels. A capture in a narrow space against a wide-gamut chart will clip, and
  that has to be known before any number is believed (step 5).
- **What the reference file is**: `deltachrome inspect-reference <file>` — sample
  count, how patches are named, illuminant and observer, spectra or not. A file
  that names patches `1`–`264` names nothing on the chart, and its order is a
  fact about how somebody measured it (step 4).

## 1. Is it a new target?

Run the current build on the capture before writing anything:

```bash
deltachrome identify <capture>
python tools/check_detection.py --capture <capture> [--region x0,y0,x1,y1] [--target ID ...]
```

The second gives identification, automatic placement, and — given `--target` — an
overlay in `data/overlays` and the inset sensitivity of an existing definition at
the placement detection found. `--region` is *Search a Region*: a rectangle round
the whole card, for a chart that is a small part of a busy frame.

**A new definition is warranted when the chart's relative geometry differs, or
when something the definition states about the chart would be false for it.**
Not when the product name differs. Concretely, a new definition when any of:

- the overlay of the nearest existing definition misses patches, or its worst
  inset sensitivity exceeds **0.02**;
- the fiducial method or the marks it places on differ;
- capabilities differ — it carries edges, rules or a uniformity set the other
  does not, or lacks ones the other declares;
- it would need different **shipped nominal values** to be judged without a
  reference file;
- its `display_name` or `manufacturer` would be untrue of it.

Otherwise it is **the same target**, and the work is a committed placement (step
3), the reference (step 4), and a sentence in the definition's notes saying the
specimen was checked. Format, film stock and print date do not make a new target
by themselves: the reference file is per unit and carries all of that.

*Two layouts sharing one definition is not the same as two vendors sharing one
layout.* IT8.7/2 fixes columns 1–19 and leaves 20–22 to the vendor, so a generic
definition's geometry fits every vendor's chart while its shipped nominal fits
one: on the Wolf Faust chart, the shipped LaserSoft batch values sit 3.8 ΔE2000
off in columns 1–19 and **31.3** off in 20–22. Check the nominal against the new
specimen's own measurement wherever both exist, by column group.

## 2. Writing the definition

Start from the nearest existing definition and change what step 1 said differs.
Every field is load-bearing; the ones most often got wrong:

| Field | What to check |
|---|---|
| `target_id` | `maker_model_variant_v1`. Never reused, never renamed — it is in every saved record. |
| `target_type` | Ends `reflective_…` or `transmissive_…`; Settings ▸ Targets groups on it. |
| `fiducials.method` | Must be a key of `detect.FINDERS`, or listed in `tests/test_detect.py::NOT_YET_FINDABLE` with the reason. Prefer a mark a finder can *locate* over one a hand can see: printed discs, pluses and keylines. |
| geometry | Measured by a tool in `tools/locate_*`, never typed from a spec sheet. Relative to the fiducial quad. |
| `capabilities` / `not_capable_of` | Every capability is one or the other. A declared lack is a statement the report repeats. |
| `dimensional_reference` | `available: false` makes sampling frequency N/A; omitted makes it INCOMPLETE. They mean different things. |
| `reference_data.source_type` | `per_unit_file`, a shipped nominal (with `DELTACHROME_REFERENCE_BASIS` and a caveat in the file), or `none`. |
| `sample_id_aliases` / `printed_id` | Only when a *vendor's published files* name patches differently from the chart. Declared, never guessed. Files that number patches and name none are handled generically (step 4), provided the definition declares its ramp. |
| `opendice_names` | Only when OpenDICE's profile for the chart cannot be joined by name: a name used twice, or blocks numbered straight through. Pairs of `[name in the profile, patch id]` in the profile's row order; the file must match it exactly. |
| `metric_defaults.color_accuracy.uses_regions` | Only when the chart prints its neutral ramp apart from its color patches, as the UTT does. Names the regions that count as color patches under "all patches". |
| `openqualia` | With a `host`, only if the maker publishes per-unit measurements there. Verify with one real fetch. |
| roles | Ramp and uniformity sets declared where the chart prints them; `resolve_roles` infers the rest from the reference. |

## 3. Committed placement and overlay

Add a `Placement` to `tools/render_target_overlays.PLACEMENTS`: capture path,
corners, and the tool that measured them. Then:

```bash
python tools/render_target_overlays.py --target <id> --labels
```

**Pass**: every ROI inside its patch on the picture, every edge ROI across one
clean edge, every declared rule read, inset sensitivity worst **< 0.02**. Look at
the worst patch it names, at full resolution.

## 4. The reference

```bash
deltachrome inspect-reference <file>
deltachrome analyze <capture> --target <id> --reference <file> --profile <p> --corners "…"
```

**Pass**: every patch the file covers joins (the header's sample count against
the target's), and mean ΔE2000 against a *per-unit* file lands where a good
capture of that kind of chart lands — about 1–2 on a reflective chart. A join
that "works" with a wrong patch order gives 30–60; test the order rather than
assuming it. A file that numbers its samples 1–N and names none is read in the
order its `LGOROWLENGTH` implies, counted from the first patch, and only if the
chart's declared neutral ramp then reads neutral and descending
(`reference/sample_ids.py`); otherwise it is refused and wants exporting with
patch names. The Wolf Faust file was read down the columns: 1.86 ΔE2000 that way,
39.0 across the rows. A target with no declared ramp cannot use that route.

**If the target ships fallback values, check them against the new specimen's own
file, by region.** A layout standard may leave part of the chart to each vendor —
IT8.7 does, columns 20–22 — and values there must not ship. The nominal files are
written by `tools/generate_opendice_nominal.py` and
`tools/derive_nominal_from_openqualia.py`; edit the tool, not the file, and diff
every file it rewrites.

Then the code: `deltachrome identify <capture>` should read what the card prints.
Record what it actually reads, and if it disagrees with the card, measure the
bars before calling it a misread. LaserSoft's barcode reads `1730030` off a card
printed `T1710030`, and that is an encoding of the serial, the same on every
LaserSoft chart here — now applied by the bank (`lasersoft_barcode`). A maker's
encoding goes in `reference/library.py` beside it, run forward from the bank's
serials, never backward from a capture.

## 5. Running it

Analyze against every profile the chart is meant for. Look at the flag breakdown
before the verdict: a capture in sRGB of a Provia chart flags 243 of 864 patches
`gamut_clipped`, which is the capture's fault and must not be read as the
definition's.

## 6. Automatic placement and identification

```bash
python tools/check_detection.py            # every committed placement
python tools/check_detection.py --target <id>
```

**Pass**, on the new capture:

- `detect_placement` reproduces the committed corners to **under 1% of the
  chart's shorter side**, refined to about a pixel;
- `identify_target` names it, or shortlists it with the siblings that share its
  layout — and the shortlist says which;
- if the chart sits small in a busy frame, *Search a Region* round the card names
  and places it. `analyze --detect` searches only the largest region and may
  refuse where the application succeeds.

**And on every other committed capture**: nothing that was named before is named
differently, and nothing that was placed before moves. A new target is a new
candidate for every capture anybody has, so this is where adding one breaks
something else. If the new chart cannot be placed, either the finder is fixed —
re-running the whole table — or the definition moves onto a mark a finder can
find. Refusing is acceptable for a release; naming the wrong chart is not.

## 7. Tests

- A test file for the target or its family (`tests/test_<family>_target.py`):
  the definition loads, its geometry matches what was measured, its capabilities
  are what the report will claim, its reference joins.
- Real captures are not committed, so anything needing one skips when it is
  absent; the measured numbers go in the definition's own notes instead.
- Run the whole suite. `tests/test_detect.py` fails on an unfindable method.

## 8. Everywhere it is listed

- The user guide's target pages, which are maintained outside this repository:
  a row in the targets table and the *Reference values* table, and a placement
  picture from `python tools/render_target_illustrations.py --target <id>`.
  Charts with features the definition does not describe need an entry in that
  tool's per-target table.
- Keep the capture under `data/`, and reference files in `data/reference/` — with a binding in `deltachrome-references.json` only if
  nothing inside the file or its name identifies the unit. Where the capture came
  from and what was measured off it go in its committed placement's `note`
  (step 3), which is committed where the capture is not. Nothing under
  `data/` is committed, its READMEs included, so the note is the only record
  a clone has.
- Settings ▸ Targets needs nothing: it lists `identifiable_targets()`.

## Same target, new specimen

Steps 0, 1, 3, 4, 5 and 6, and a line in the definition's notes. No new id. The
committed placement is still worth adding: it is the only evidence the
definition fits more than the one unit it was measured on.
