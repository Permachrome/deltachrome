# Deltachrome (Δχ)

Open tooling for cultural heritage imaging QC. Deltachrome analyzes photographic
test target captures against digitization standards — FADGI, Metamorfoze,
ISO 19264 — and tells you, per patch and per metric, whether a capture system is
meeting the specification you claim to be meeting.

Deltachrome is the first public component of **Permachrome**.

## Status

Pre-alpha. Two front ends over one engine: a command line, and a desktop
application (`deltachrome-gui`, PySide6, installed with the `gui` extra).

The engine remains a standalone library with **no UI code inside it** — the
frontend is a separate top-level package, `deltachrome_ui`, and
`tests/test_ui_boundary.py` asserts by source scan that nothing under
`deltachrome` imports a GUI toolkit. It stays installable, importable and
testable with no toolkit present.

It runs **a folder at a time** as well as a capture at a time: point Batch
Analysis at a directory, check the chart's placement once on its first frame, and
every capture is measured against one guideline into a dated folder of ordinary
records. See "Batch runs" below.

The desktop application covers opening a master, choosing target, profile and
reference, placing the chart by hand on a zoomable canvas with a full-resolution
loupe, checking which pixels the sampler will actually read, running the analysis
and saving the record to a library — which can be pointed at any folder, since a
record references its capture by path and never sits beside it.

A saved capture **opens back into its report** without decoding the pixels: the
record holds every per-patch value and every edge of every chart, so the whole report
renders from it, and a frame holding five charts steps between them instantly.

The report is one metric at a time. It opens on an **overview** — every
metric as a row in the same table the library uses, with its value, its unit, what
it was judged against and its verdict — and a flat list at the left reaches any one of
them in a click, grouped under headings read from each metric's own declared
capability. A metric's page is a line saying what it read, whatever qualifies that
reading immediately under it, and then a picture chosen by the data rather than by
the family: patches get the chart drawn from what was measured, each patch split
between the reading and its reference; a ramp gets its tone curve; edges get the
field of MTF curves, averaged per field location or shown one by one. Its readings are a tab beside the picture, as a table.
Where a reading physically sits is answered on the capture, not on a diagram: the
capture can be loaded alongside, with every sampled region marked on the pixels it
was measured from and the focused one at full resolution, which is the difference
between a soft system and a speck of dust.

Nothing on any screen explains itself: standing prose that is the same on every
run is read once and read past forever, and a beta with none of it is the only
chance to learn which parts a technician can read unaided. What qualifies *this*
run — a metric's reason, the conformance caveat, the run's warnings — was never
that, and stays. Several saved runs can be selected and
read together as a **session** — the question a QC run is really asking, since
almost no chart measures everything a standard wants. Combining re-measures
nothing: every number a session reduces is already in each record.

Opening a capture works out **which chart it is** as well as where, and the two are
one pass rather than two. It names a chart only when one fits, never as the best of a
bad field: where two fit equally it names both and leaves the choice, which is the
only correct answer for the two IT8 targets, whose layouts are identical and whose
aim values agree to about a ΔE. Whatever it decides is a combo box you can change.

**A capture may hold more than one chart, and the record is the capture.** The Applied
Image QA-62 is a small card reading one field position, shot four to a board so that
sharpness across the frame is four measurements rather than one. Each chart is placed,
checked and measured on its own — the placement panel steps through them, and the primary
action commits all of them at once. It lands on the first chart's **report**, with a
`◀ Target 2 of 5 ▶` stepper in the header and the board's **Combined Verdict** one press
away in the same bar — because a board is laid out to measure a system rather than a
card, but the report is what the run produces and the combined verdict is a question
asked about several of them. Every chart may be a different model, each with its own
target and its own reference file, and a region that held a chart and could not be named
is reported rather than dropped.

What is **saved** is one record per capture holding every chart in it, because a board is
one capture of one system at one moment and five library rows is five things to keep
together by memory. Nothing about a run widened to do it: each chart's record inside the
container is an ordinary run record, byte-for-byte what it always was, so re-scoring takes
one and the report renders one. Switching charts on the report is therefore a read rather
than a re-analysis.

The search is not the last word, either. A chart it missed can be **added by hand** and
dragged onto; a region it wrongly took for a chart, or a chart you simply do not want
measured, can be **ignored** without being removed — and put back with the same press.

`--detect-all` does all of this in one command, with `--identify` naming each chart.
What is still beyond it: two charts sitting closer together than the search can separate
read as one, and the frame reports fewer charts than it holds.

```
image + embedded ICC → place target on its printed fiducials → sample patches and
    slanted edges → color-managed Lab and ISO 12233 SFR → compare against a
    per-unit reference → evaluate a standard's profile → pass / fail / incomplete
```

Implemented:

| | |
|---|---|
| **Color** | ΔE2000/94/76 mean, maximum and 90th percentile, white balance as Euclidean ΔE(a\*b\*) or CIEDE2000 without luminance |
| **Tone** | ΔL\*2000 OECF, gain modulation (ISO 19264-1), noise as σ(L\*) |
| **Resolution** | Slanted-edge SFR per ISO 12233 — SFR10, SFR50, response at Nyquist, sharpening, spatial overshoot. MTF50 in lp/mm, cycles/px or % of Nyquist |
| **Registration** | Color-channel misregistration per Burns & Williams, per-record edge location off the same edges |
| **Uniformity** | Lightness uniformity as σ/mean over the target's areal coverage, from a declared sample set or one inferred from repeated reference neutrals |
| **Image plane** | Illumination and white balance across the whole frame, measured on a frame-filling white sheet rather than on a chart |
| **Geometry** | Geometric distortion, from the spread of scale across the field and between axes, read off the rules a chart prints at several positions |
| **File** | Master format, bit depth, color space, sampling frequency |
| **Scale** | Sampling frequency and reproduction scale accuracy, measured off the chart's own printed rule — or off a printed **checkerboard**, which marks the same divisions with filled cells and fits about thirty times more regularly. A declared band is a **search region**, not the band read: the band is swept within it and the reading is the median of every band that works, because a band tuned to one specimen does not survive the next |
| **Transmissive** | ISA 4x5, 35mm and 120 film targets (the 35mm definition also serves the slide-mounted one) — neutral wedges and quarter-disc slanted edges — plus a 35mm microform resolution chart in both polarities |
| **Grayscale** | Single-record captures are read, not refused: grayscale ICC profiles are evaluated through their own transform, and the metrics that need three colour records report **incomplete** rather than a perfect score |
| **Sessions** | Several charts judged together against one profile — the charts of one frame, or captures selected from the library, which costs no re-measurement. Coverage is unioned, the worst reading governs where two charts overlap, and where they disagree is reported rather than averaged |
| **Standards** | FADGI 3rd Ed. — **all fifteen material categories**, 55 profiles, read out of the published PDF rather than typed in. Metamorfoze 2.0 — Full, Light and Extra Light |
| **Inputs** | TIFF, JPEG 2000 at 8 or 16 bits (ICC read from the `colr` box; decoded through OpenJPEG, checked bit-for-bit against a second binding), matrix/TRC and LUT ICC profiles, CGATS and CxF3 references carrying Lab, XYZ or spectra |

Every metric the shipped FADGI and Metamorfoze profiles declares now has a plugin
behind it; a test
asserts that, because a metric reporting NOT IMPLEMENTED can never report a pass and
so makes its whole profile unusable. That test is what kept the Metamorfoze profiles
from shipping until geometric distortion existed.

A threshold can depend on something the capture cannot show. Metamorfoze bands its
illumination tolerance by the size of the originals a system is being qualified for,
which is a property of the job rather than of the image — so the profile carries the
band table and `--original-size` supplies the answer. Without one the metric is
measured but not judged, and reports which bands the reading meets.

A **local guideline** can be built from a standard one with thresholds
overridden, metrics measured but not judged, metrics omitted and metrics added —
projects are rarely uniformly one star level — and saved for re-use. Once it is
saved it is its own specification: it carries the name its author gave it and
records nothing about the table it was built from. What it does record is
`inherited`, the criteria nobody chose, because moving one threshold carries
every other one across unexamined and nothing on the screen says so.

A local guideline is not a conformance test against any published standard. It is
not thereby **advisory**, and that distinction is deliberate. A run loses its
conformance claim two ways — the reference is not a measurement of the chart in
front of the camera, or the criteria are the site's own — and they are not the
same kind of thing. The first is a fact about the evidence that no decision makes
untrue; the second is a decision, recorded and named on the report, and the run
*met* it. So a run against a department's own written standard, on a chart that
was itself measured, reads a plain `PASS` and says which guideline it is a verdict
about. Demoting it too meant a site that used the guideline builder could never
see an unqualified verdict, which taught the qualification to mean nothing.

Several captures can be judged together against one profile — an SG for color and
a resolution chart for edges — with coverage unioned, disagreement between charts
reported, and any metric no capture reached named rather than dropped.

## Design commitments

These are load-bearing, not stylistic:

- **The reference basis is never assumed.** A per-unit measurement is always
  preferred and always wins: manufacturing tolerance means every physical chart
  differs, and where a chart carries a printed code that code pairs the capture
  with its own unit's file, with a mismatch reported rather than assumed harmless.
  But some targets are effectively never sold measured and cannot practically be
  measured by their owner, so where the manufacturer publishes aim values the
  definition ships them and the run falls back to them — labeled as nominal, in
  the header, in the warnings, in the saved record, and at the cost of the run's
  conformance claim. A verdict reached that way reads `ADVISORY`, never a bare
  `PASS`, because the chart's own print tolerance is inside every ΔE. That is
  about the *evidence*: judging against a local guideline withholds the claim
  too and does not make the reading doubtful — see above.
- **Real color management.** Master files in this domain are routinely Adobe RGB,
  ProPhoto or ECI RGB. Deltachrome reads the embedded ICC profile and uses its
  actual primaries and tone curves at full float precision. It never applies
  sRGB formulas to non-sRGB data, and it refuses images with no embedded profile.
  Both ICC transform structures are evaluated exactly: analytic matrix/TRC
  profiles, and LUT-based (`A2B`) profiles of the kind scanners and cameras use.
- **Raw data first.** Per-patch measurements are persisted, not just aggregate
  pass/fail. That is what makes time-series tracking and re-scoring against a
  different profile possible without re-shooting the target.
- **Metadata over categorical naming.** Where targets differ by degree — sample
  density, capability strength — that is reported as numbers alongside the
  result, rather than encoded as invented capability tiers.
- **No overclaiming.** Where a measurement is useful but not a strict match to a
  standard's literal wording, the report says so. A metric with no plugin reports
  NOT IMPLEMENTED and blocks a pass rather than being quietly omitted.
- **Algorithms are parameters, not constants.** Edge fit, ESF bin width, window,
  derivative kernel, percentile method, ROI inset and the rest are all settable and
  are recorded with every result — two runs under different settings are different
  measurements.

## Install (development)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Add the desktop application. The toolkit is an optional extra so that the engine
stays installable and testable without it:

```bash
.venv/bin/pip install -e ".[dev,gui]"
```

## Use

### Desktop

```bash
deltachrome-gui
```

Opens on the run library. Each row offers **Open** to read the report, **Re-run**
to place the capture again from the corners it used, and **Copy Command** to put
the whole `deltachrome analyze` invocation on the clipboard.

Your work — the run library, the bank of per-unit reference files, and any
guidelines written here — lives under `~/Documents/Deltachrome`, and the
application's own settings where the platform keeps settings (`~/Library/Application
Support/Deltachrome` on macOS, `%APPDATA%\Deltachrome` on Windows). A folder
already in use under the older hidden `~/.deltachrome` goes on being used.
**Settings** moves each of the three, remembering the choice, with
**Default** on any row that has moved. A
remembered folder that is no longer there falls back to the default rather than
being recreated, so a library pointed at a scratch folder or an unmounted share
does not come back as an empty one at an address nobody recognises.
`DELTACHROME_RUN_DIR`, `DELTACHROME_REFERENCE_DIR` and `DELTACHROME_PROFILE_DIR`
pin one folder each for a scripted or shared-drive deployment, and
`DELTACHROME_HOME` moves all three defaults at once — the engine reads it too, so nothing is left writing into a person's own home when the rest has moved.

**Settings ▸ Targets** lists every target identification can name, reflective and
transmissive, each with a checkbox. Untick the ones you do not own and a capture
is no longer searched for them; the target list on the placement screen still
offers every chart for choosing by hand. Where a search names nothing, it says how
many targets were turned off, since a borrowed chart is the likeliest reason a
chart that is plainly there was not found.

**New Analysis** works out **which chart the capture is of** and **finds it in
the frame** before it asks you anything, and opens one screen with both answers
on it: the capture with its four corners already placed, and beside it the target,
the standard it is judged against, and where its reference measurements come
from. Two chips say what was decided — **Target Detected** and **Placed
Automatically** — and what each was decided on is on hover. Change the target and
the chart is found again without leaving the screen. Where a chart cannot be
found, the corners are left alone, the chip says so, and the chart's own
placement instruction appears.

**The target's printed code is read in the same pass**, so the setup screen
already knows which per-unit measurement this capture could be judged against and
says which — a chip, and the code and file beside it. Where the code matches
nothing in the bank, or the target prints none, that is said here rather than
raised after the decode, and in every reference mode rather than only in the one
that pairs. **Choose from bank…** lists what the bank holds with the codes each
file answers to, and offers to bind this capture's code to whichever is chosen so
the chart pairs itself from then on.

**A code can say where its own measurement lives.** Under the OpenQualia label
standard — Digital Transitions, Image Science Associates, ImageZebra, Gray Sky
Imaging, Munsell at RIT — a chart's QR or Data Matrix encodes a web address for
that individual unit's measurement. Where the bank does not already hold it,
**Fetch Measurement** appears beside the reference options; pressing it downloads
the file, files it, and binds it to that unit so every later capture of the chart
pairs itself. Nothing leaves the machine unless it is pressed: reading a code is
not permission to tell a third party what you are shooting.

For a chart that prints a serial and no label — every Golden Thread made before
2025, of which some three thousand are published — Settings ▸ Reference Files does
the same from a typed serial. You pick the chart by name and never type a
manufacturer or a target type; those spellings are not guessable and are declared
by the target.

**A per-unit file matters more than it looks.** Two units from one production
batch differ by 0.2 to 0.9 ΔE2000; two units of one model from *different* batches
differ by 7 to 14. So a batch average is only as good as its batch, and a run
against one can carry several times the system's own error. Where a target ships
aim values Deltachrome uses them rather than refuse, and reports `ADVISORY`.

**A region that is a decoded code is not offered as a target.** A compact block of
high-contrast structure is exactly what the chart search looks for, so a chart's
own QR is otherwise put forward as a second target in its own frame; knowing a
code was decoded there settles it, and only a decoder can. Where the search
returns something that is not a chart anyway, **Remove This Target** takes it and
its placement out of the frame, and the record still says the capture held more
than was measured.

Whichever way the corners arrive, the canvas pans and zooms over the
full-resolution master, with a loupe showing the pixels under the handle being
placed, and the regions this run will actually read are **drawn on it from the
start** — patch ROIs and slanted-edge ROIs alike. A placement can be entirely
plausible and still have every region straddling two patches, or an edge ROI
holding two transitions, and that is invisible until the outlines are on the
pixels. **Hide Sampling** turns them off.

Identification runs a finder per placement method and ranks whatever survives by the
chart itself rather than its outline: the proportions the target's grid implies,
whether its neutral ramp reads light to dark in the declared order, how much of its
array lands on patches, and how far its colors sit from whatever aim values it
ships. It names a chart only when one fits, and only when its regions came back looking
like patches: a 35mm slide read as a microform chart flagged 90% of them where the
right answer flagged none.

Automatic placement locates whatever the target says a placement is made on —
printed crosshairs, a keyline, four discs, the array of patches itself — and then
snaps the corners onto the chart's own printed features. Measured against every
capture in this repository it re-derives the committed placement to within 0.63%
of the chart's shorter side, and fifteen of twenty-three to under a pixel. **It
refuses rather than guessing**: a wrong placement samples every region from the
wrong pixels and returns a full set of plausible numbers, so where the chart
cannot be found the corners are left alone and the reason is shown. That a
placement was automatic is recorded in the run, and stops being claimed the
moment you drag a handle.

**Refine Placement** does the second half of that on its own, for a placement you
made by hand, and reports how far the corners moved. Worth pressing: on the
Golden Thread sample a 20 px error — 0.2% of an 11600 × 8700 frame — is the
difference between 6 printed rules being readable and none, and geometric
distortion needs four. It is never applied silently, and it refuses rather than
guessing when too few features are found or the correction is too large to be
one.

#### Reading a report

A run opens on its **Overview**: every metric the profile declares, one to a row
— verdict dot, name, value, unit, and what it was judged against. That is the
whole answer, and there is no paragraph above it restating it. Where a run
cannot be quoted as conformance one line says so, in the open rather than behind
a click, with the full wording on hover.

A flat list on the left reaches **one metric at a time**, grouped under headings
that come from what each metric measures. A metric page is a line saying what it
read, whatever qualifies that reading immediately underneath it, and then the
picture — and which picture is decided by the data rather than by the metric's
name. Patches get the chart; the neutral ramp gets its OECF curve; a field of
slanted edges gets its MTF curves. Every chart carries the profile's own limits,
drawn as a band across the readings only where the profile actually judges per
reading, and its own name is drawn *over* the readings rather than under them.

**Nothing dims until you ask it to.** Pointing at a reading — on the plot, in the
table, on the capture — lights it and leaves the rest alone; clicking a mark on
the plot isolates it, which is a request to see fewer and is made where seeing
fewer is the point. The field of edges can also be **averaged rather than
narrowed**, which is the question a field is shot to ask: one curve per field
location, or one per direction the response was measured in, over whatever the
location, orientation and colour-record chips have left showing.

Two tabs at most, and they mean the same thing on every metric: **Charts** is the
picture, **Values** is the readings as a table, worst first. Hovering a row
lights that reading on the capture panel.

Nothing is coloured by pass or fail. Those thresholds are conditions on an
aggregate, and a per-reading verdict is one the profile never formed; what is
marked is a sampling flag, which is a fact about the measurement rather than a
judgement of it.

**Every table sorts**, on the key the column declares rather than on the text it
shows — so a reading of `10.500` does not sort before `9.100`. A table opens in
the order that answers the question it exists for, worst reading first, and a
click on a heading asks a different one: which patch is worst, whether the
flagged patches are the failing ones, which colour record is soft.

**Load Capture** decodes the master into a panel showing where the focused
reading sits in the frame — the marks drawn on the pixels they were measured
from — and its pixels at full resolution. It is on request because opening a
saved run otherwise costs no decode at all.

The **profile in the header is a control**, not a caption. Choosing another
performance level or material category judges the same reading again without
re-reading the capture — 0.2 s against 16 s on a JPEG 2000 master — which is the
commonest thing a report needs to answer: not "what did it measure" but "what
would it have taken". A run reopened from the library holds numbers and no
pixels, so there it is a label saying the same thing.

**Nothing on any screen explains itself.** Explanation that reads the same on
every run — what a mark size means, why a curve rising past Nyquist is aliasing —
is documentation, kept apart from the application. A control that answers its own
question can never be shown to be unclear, and finding out which parts of this
application a digitisation technician can read unaided is worth more than the
eight seconds a panel would have saved them.

What changed on *this* run stays on the screen, always and in the open: every
reason a verdict cannot be quoted as conformance, why a metric could not be
finished, which patch was clipped.

### Command line

```bash
deltachrome analyze CAPTURE.tif \
  --target colorchecker_sg_v1 \
  --reference data/reference/my_sg_unit.txt \
  --profile fadgi_4star_prints_photographs_2023 \
  --detect
```

`--detect` finds the chart and places it, then refines the corners onto its own
printed features. The run **stops** rather than guessing if the chart cannot be
found, and says why; that the placement was automatic goes into the record.

`--identify` goes one further and works out *which* chart it is, so `--target` is not
needed either. It stops rather than guessing where more than one target fits, and
names them. `--candidates ID [ID ...]` narrows what it chooses among to the charts
you own: fewer candidates is fewer chances for the wrong one to fit, and the two
IT8s and the two ISA FADGI-ISO 19264 versions, which share a layout and cannot be
told apart, are named outright for anybody who owns one of each pair. The record
says which set it chose from, as `geometry.searched_targets`.

`--corners "1025,1921 6966,1935 6954,6164 1008,6151"` is the other way, and one of
the two is required. Corners are the target's printed fiducials in image pixels —
for a ColorChecker SG, the four crosshair centers. Give them in the **target's own
order**, clockwise from the corner nearest patch A1, not the image's: a chart
photographed rotated then needs no special handling, because the homography absorbs
it.

Add `--dump-overlay overlay.png` to see exactly which pixels were sampled. With
geometry that is refined per capture, that is the only way to confirm placement
before trusting a marginal result.

Add `--refine-placement` to snap the corners onto the chart's printed features
first. The correction is reported and recorded, never applied silently.

Other commands:

```bash
deltachrome targets                       # installed target definitions
deltachrome profiles --body FADGI --star 4    # filter by body, edition, material, level
deltachrome references --library data/reference   # the reference library and its identifiers
deltachrome identify CAPTURE.jp2 --library data/reference   # read the chart's code and pair it
deltachrome inspect-reference FILE.cxf    # parse and summarize a CGATS or CxF3 file
deltachrome inspect-icc CAPTURE.tif       # report the embedded ICC profile
```

Charts that carry a QR code or barcode can pair themselves with the right per-unit
reference file. QR is read by OpenCV and zxing-cpp covers most 1-D symbologies;
Standard 2 of 5 — which the general decoders dropped and which LaserSoft's targets
use — is decoded here, in a pass shaped for it, since a barcode wider than a tile
decodes to a confident fragment otherwise:

```bash
deltachrome analyze CAPTURE.jp2 --target colorchecker_sg_v1 --reference-library data/reference --profile fadgi_4star_prints_photographs_2023 --corners "..."
```

### Batch runs

A folder of captures, judged one at a time against one guideline. **Not a
session**: a session pools several charts to ask whether the *system* passed,
while a batch asks whether *each* of a hundred production captures did — pooling
them would let one bad frame fail the lot and lose which one, and conformance is
stated per image anyway. Every record a batch writes is an ordinary one that
opens, re-scores and combines like any other.

```bash
deltachrome batch CAPTURES/ \
  --target isa_object_level_05x_v1 \
  --profile fadgi_3star_bound_volumes_general_2023 \
  --seed-from a-checked-run.json \
  --into RUNS/batch-2026-09-10
```

`--seed-from` is where to look, taken from a placement already made — a saved run
record whose corners somebody checked, or a capture the chart can be found in on
its own. The chart is still found and refined in **every** frame; the seed only
narrows the search, which is what makes a chart occupying a sixth of its frame
findable at all. Seeded from one frame across twenty, the corners actually found
moved a median of 21 px and as much as 95. A capture holding several charts of the
batch's model cannot be a seed — it cannot say which one to follow — so seed from a
saved run of the one you want.

Before a frame is measured, the chart's own grey ramp is read at the placement
found. Where it runs out of order the placement is on something else — a quad of
marks from neighbouring cards, say — and the frame is refused with that reason
rather than measured confidently on nothing.

A batch measures **one chart in each capture** — the one its target and placement
name. Where the first capture holds more than that, the run says so once:
*"The first capture held 4 charts and a batch measures one in each."* It looks
at the first capture only. To measure every chart in one frame, run that capture
on its own, where all of them are placed and combined.

A failure never stops the run: every capture is measured or refused with the
reason, recorded either way. Records are written as they are made and the index
rewritten after each, so `--resume` costs ten frames rather than a hundred when a
run stopped at frame ninety.

**A batch reports how its own captures differ from each other**, which is the one
thing a hundred single runs cannot say. Across twenty captures of one chart every
frame met the sharpness requirement and one read 4.7 standard deviations below the
batch on SFR50 and 3.5 on sampling efficiency — one soft frame, corroborated on two
metrics, that no per-image verdict mentions because nothing failed. A finding also
has to be a fair share of the profile's own criterion: statistical unusualness
alone reported eleven findings from twenty captures, of which seven had moved a few
percent across a criterion and meant nothing.

In the desktop application it is **Batch Analysis** on the library bar. It opens
the folder's first capture on the ordinary placement screen — a folder cannot run
until somebody has said where the chart is — and the primary action reads *Run
Batch of 20*. The run screen is one table that gains rows: each capture as it
finishes, its verdict, the reading that makes it unlike the rest of the batch
beside what the rest of the batch read, and a count of what it flagged. Above it
sit the two things the table cannot say about itself — how far through the run is,
and a running tally of how the rows are coming out.

**In the library a batch is one row that opens**, because a thousand records
arriving at one instant would otherwise be a thousand rows between a person and
everything else they have. The row carries the batch's target, guideline, date and
total warnings, and a **tally** of its captures' verdicts rather than a verdict of
its own — a batch is a hundred independent answers and pooling them is the one
thing it must not become. Searching reaches inside a folder that is shut, and
deleting one moves the whole folder to the library's trash.

### Exporting values

A capture's measured values can be written out as **text**, **CSV** or **XML**,
the extension choosing which. In the desktop application it is the export square
in a report's bar, or on All Targets; on the command line it is `--export` on
`analyze` (repeatable), or `export` for anything already saved:

```bash
deltachrome export RUNS/f38505f48793.json values.csv values.xml
```

All three hold the same data: a short header — software version, date of
analysis, image, target, guideline and verdict — and one table with a row per reading, giving
the metric, its verdict, value, units and condition, the reference it was
compared against, the reading itself, and a patch's measured and aim L\*a\*b\*.
Names are written out (*White Balance*, *ΔE\*ab*, *Upper Left Horizontal*), never
as internal ids. A capture holding several targets exports as one file, with a
*Target* column saying which. The XML nests each reading inside its metric, and
each metric inside its target. Warnings and explanations stay in the report; the
full raw record stays the JSON it already is.

## Supported targets

| Target | Patches | Regions | Notes |
|---|---|---|---|
| ColorChecker Digital SG | 140 | one 14×10 lattice, 1 rule | Reflective; places on four printed crosshairs. Its printed millimetre scale is read, so sampling frequency comes from the chart in front of you rather than from a declared extent — the reading reproduces a hand measurement to 0.02% on both units this project has captures of |
| LaserSoft ISO 12641-2 (4x5, 120) | 864 | one 36×24 lattice | Transmissive; places on the printed frame corners. Prints no rule and records no extent, so sampling frequency reports N/A — the requirement still applies to the capture, this target cannot verify it. Its barcode reads `1930021` where the card and its own CxF say `T1910021`; the parse is unambiguous, so the two are bound by hand once |
| ISA Golden Thread DLT | 50 | 9 lattices, 5 edge modules, 20 rules, 5 markers | Reflective; places on the four printed **discs**, which form a diamond about the chart's centre rather than a rectangle at its corners — chosen because they are findable to 0.16 px, not because a diamond fits better (it is 25% worse conditioned than the corner modules it replaced). Ten rules per axis at five field positions, which is what makes geometric distortion measurable |
| ISA Transmissive Film, 4×5 | 24 | 2 wedges, 5 edge modules, 4 rules | Transmissive; places on the corners of its printed keyline. A metric and an imperial rule above the chart and the same pair down the right-hand side — the best geometric distortion measurement of the three, its floor being 0.036%. Runs on manufacturer nominal values |
| ISA Transmissive Film, 35mm | 24 | 2 wedges, 5 edge modules, 4 rules | Transmissive; places on the corners of its printed keyline. A metric and an imperial rule above the chart and the same pair down the left-hand side, agreeing on the sampling rate to 0.04%. The definition also serves the slide-mounted target, which prints the same pattern |
| ISA Transmissive Film, 120 | 24 | 2 wedges, 5 edge modules, 4 rules | The same product line in its 6×6 cm cut, so its keyline is square. The wedges sit above and below the centre as two 3×4 blocks, and it prints a metric and an imperial rule at each of two positions — as, it turns out, both its siblings do |
| 35mm microform resolution target (positive) | 20 | 2 L-shaped ladders, 4 straight edges, 4 rules | Transmissive; places on four printed crosses one inch apart. Ten density steps in an L, declared as ten occupied cells of a 5×6 lattice. Its ticks are **light on dark**, which is what the ruler reader's polarity detection is for |
| 35mm microform resolution target (negative) | 20 | 2 L-shaped ladders, 4 straight edges, 4 rules | The same chart as a negative, and the sample capture is **grayscale** — one colour record, tagged Gray Gamma 2.2. Same geometry, opposite ladder: the two reference files are **not** interchangeable |
| Image Engineering UTT, A3 | 159 | 7 lattices, 9 edge modules, 12 rules | Reflective; places on the four corner modules' outermost printed crosshairs. The Metamorfoze and ISO 19264-1 chart: two colour bars, four 20-step grey scales, 36 slanted edges at nine field positions, and a 10 mm checkerboard whose white squares are the uniformity set |
| ISA FADGI – ISO 19264 (early) | 168 | one 14×21 lattice, 5 edge modules, 6 rule segments, 12 markers | Reflective; places on its three black corner squares and the dot standing where a fourth square would be — it prints no pluses, unlike its successor. That dot is one of 75 printed on a grid, so the one above it is the mistake to avoid, and the placement page says how to tell them apart. Three black corner squares and eight registration dots are declared as markers, which is what lets it be refined at all -- eleven features, enough to over-determine a homography. Ten slanted edges declared one at a time, five field positions in both orientations. Prints its patch numbers 1–168 |
| ISA FADGI – ISO 19264 (commercial) | 168 | one 14×21 lattice, 5 edge modules, 6 rule segments, 13 markers | Places on the four printed pluses just inside its corners; refines on eight registration dots and three corner squares. Prints an 18-step keylined tone ladder the early chart does not, declared as a marker and deliberately not measured. Same layout as the early chart, re-chosen colours, grid labelled A–N / 1–16 — the two reference files are **not** interchangeable. Each module stacks two separate figures, so its horizontal and vertical edges are declared individually — the resolution field's own borders are higher contrast and axis-aligned, and are not edges |
| Avian Rochester NGT | 130 | one 13×10 lattice | Reflective; no printed marks at all, so it places on the patch array's own corners. Colour and tone only — the twelve grey steps are two rows running in opposite directions, so the ladder is ordered by density rather than by position |
| DT Next Generation Target V2 | 130 | one 13×10 lattice | Reflective; places on four printed crosshairs. The same lattice and the same patch ids as the NGT above, printed to different values — the two reference files are **not** interchangeable |
| IT8.7/1 transmissive | 288 | one 22×12 lattice, plus a 24-step grayscale strip | Transmissive film; places on the colour array's corners. Patches are butted, so the sampling inset is the only thing holding an ROI off the boundary. Reference data is per batch and the batch number is printed and barcoded on the chart; the shipped nominal file covers the colour array only, so the strip needs a batch file to be judged |
| IT8.7/2 reflective | 288 | one 22×12 lattice, plus a 24-step grayscale strip | The reflective member of the same pair, same layout and same caveats. Kept as a separate definition so a print cannot be judged against a film chart's batch file |
| ISA Golden Thread Object Level, 1X | 30 | one row of 34 slots, 1 edge module, 2 rules, 1 checkerboard | Reflective strip that lies beside the object. Prints its own reference table beside the patches. One definition serves both versions — the revamped one dropped the older card's centre fiducial disc. The only target here measured on two physical units. Its centimetre rule cannot be read on every specimen — one capture's millimetre comb scatters 10% — so the **checkerboard above it** is declared as a third scale, and is the better feature anyway |
| ISA Golden Thread Object Level, 0.5X | 30 | 14 + 1 + 15 in a row, 1 edge module, 1 rule | The same strip half the length, and **not** the 1X scaled: its patches stand 1.49 times taller than wide where the 1X's are square, its two blocks sit at pitches 1.4% apart, and the module between them spans 4.37 pitches rather than a whole number, and its **patch 15 is cut short** where the left block meets that module — declared apart at the width it really is, because sampling it as a fifteenth equal cell put the ROI on the printed keyline and the reading 6.63 ΔE2000 out. It is the chart that took the strip finder off a slot index and onto the positions a definition declares. Prints the same batch table as the older 1X unit |
| Applied Image QA-62 Scanner SFR & OEC #2 | 20 | one 6×6 lattice with 20 cells occupied, 1 edge module | Reflective; places on four printed crosshairs bounding the measurement field rather than the card. **The first target here meant to be shot several to a frame** — it is a small card reading one field position, so it is laid out four or more to a board and the frame is read as several charts. One 5° slanted square at the centre, ringed by a 20-step neutral ladder running clockwise from the top-left. Applied Image's aim values for those steps are not in hand, so it ships none and claims only what its square measures: sharpness and colour channel misregistration. Its crosshairs are 2.625 in (66.675 mm) apart by design, which is enough for sampling frequency and lp/mm and deliberately not enough for reproduction scale accuracy |
| DT Nexus Transmissive Profiling, Rev A | 129 | one 13×10 lattice, 129 circular filters | Transmissive glass-and-metal chart for building film and glass-plate profiles, designed with Gray Sky Imaging using colorants not tied to any emulsion. The top-right cell holds an OpenQualia QR rather than a patch, rows are 0-indexed and ids carry a dash (`A-0`). Column A is a ten-step neutral ladder ending in an opaque flare check at `A-9`. Places on the centres of four circles one in from the corners — A-1, L-0, M-8 and C-9, the same four on every DT chart — because the metal frame is opaque in a transmissive capture and, on Rev B, floats in it, and one corner cell is opaque on every unit. Geometry measured from DT's own developer capture of unit TP24010 |
| DT Nexus Transmissive Profiling, Rev B | 129 | one 13×10 lattice, 129 circular filters | The 2026 revision, serial TP26008 and higher. Identical in layout and colour; the patch pitch is 8% tighter. Both lattices fill the same frame, the patch array's own bounding box, so the relative geometry barely moves and the reason both ship is the physical extent — 110.5 mm against 102.5, measured off this unit's own 2133 ppi capture, which scales sampling frequency by 7.3% |
| DT Nexus Transmissive Validation, Rev A | 129 | one 13×10 lattice, 129 circular filters | The independent counterpart to the Profiling target — same form factor, different colours, serial prefix TV. Its neutral series is nine scattered patches rather than a column, which for a chart meant to validate a profile built elsewhere is reasonable; the positions are identical on all five published units and were declared from those rather than inferred, because Revision B's are chromatic enough that a neutral test drops three of them |
| DT Nexus Transmissive Validation, Rev B | 129 | one 13×10 lattice, 129 circular filters | The 2026 revision, serial TV26001 and higher, on Rev B's tighter pitch and 7.3% smaller physical extent. The one definition here whose lattice is borrowed — no capture of this model and revision exists — which its runs say out loud |
| Frame-filling white field | — | none at all | A white sheet filling the frame. Measures illumination and white balance across the image plane, which a chart sitting in the middle of the frame cannot |

Every target carrying patches has its geometry measured from a physical unit rather
than from a spec sheet. A target is described as a list of **regions**, each with its
own geometry — a single lattice is just the one-region case, so simple targets stay
simple to write, and a chart with wedges, edge modules and printed rules is the same
format with more entries.

Reference values are never hardcoded in a definition. A per-unit measurement file is
preferred and always wins; where one cannot be had, the definition may ship batch
values as a declared fallback and the run reports `ADVISORY` rather than
conformance. That fallback matters less than it once did: the ISA Golden Thread
charts were the reason it exists, and Digital Transitions now publish per-unit
measurements for over three thousand of those units, which Deltachrome fetches from
the chart's own code or from a typed serial.

## Documentation

- [Understanding your results](understanding-your-results.md) — what each
  metric means, why tone response is ΔL*2000 rather than plain ΔL*, and which
  warnings to act on.

Adding a target means writing one definition file in
`src/deltachrome/targets/definitions/` and nothing else — no code. The
`capabilities` / `not_capable_of` lists tell the engine what the chart can and
cannot measure, and any profile metric it cannot reach is reported as such rather
than counted as a failure, so a partial target is a first-class one.

## License

MIT. See [LICENSE](../LICENSE).
