# Understanding your results

Plain-language guide to what Deltachrome reports and what each number actually
means. If you only read one section, read [Tone response](#tone-response-is-measured-in-δl2000-not-plain-δl) —
it is the number people most often misread.

---

## The verdict line

```
RESULT: PASS
COVERAGE: 4 of 9 metrics judged, 5 not measurable by this target
All 4 judged metrics met their thresholds.
```

Three verdicts are possible:

| Verdict | Meaning |
|---|---|
| **PASS** | Every metric that applies to your target ran and met its threshold. |
| **FAIL** | At least one metric ran and missed its threshold. |
| **INCOMPLETE** | Nothing failed, but the run does not *demonstrate* conformance. |

INCOMPLETE is the important one. It appears when a metric the profile requires
could not be evaluated — usually because Deltachrome has no plugin for it yet, or
because too few patches were usable. Everything that ran may have passed, but the
run did not test everything the standard asks for, so calling it a pass would be
overclaiming. Read the explanation line; it names what was missing.

### Why a metric might not have run

Each metric reports its own outcome, and two of them look similar but are not:

| Status | Meaning | Affects verdict? |
|---|---|---|
| `PASS` / `FAIL` | Ran and was judged | Yes |
| `N/A` | **Your target cannot measure this.** A ColorChecker SG has no slanted edges, so it cannot measure sharpness. This is not a shortcoming of your capture. | No |
| `NOT IMPLEMENTED` | **Your target could measure this, but Deltachrome cannot yet.** | Yes — forces INCOMPLETE |
| `INCOMPLETE` | The metric ran but had too little usable data to trust | Yes — forces INCOMPLETE |

Keeping `N/A` and `NOT IMPLEMENTED` separate is deliberate. If they were merged, a
report could claim full coverage of a standard while quietly skipping a metric your
equipment was perfectly capable of being tested on.

### If your capture is grayscale

Deltachrome reads grayscale masters — they are what a microfilm scanner produces,
and FADGI's microfilm tables list Gray Gamma 2.2 as a permitted working space.
Tone response, noise, gain modulation, sharpness, the printed rules and geometric
distortion all read from a grayscale capture exactly as they do from a colour one.

Three metrics do not, and they will report `INCOMPLETE`:

* **colour accuracy**
* **white balance**
* **colour channel misregistration**

They report incomplete rather than `N/A` because the limitation is in the *file*,
not in your chart. If you scanned a colour target in grayscale, the chart could have
answered these questions and the capture cannot.

**This matters more than it looks.** White balance is a ΔE with lightness excluded,
and a grayscale file has no chroma at all — so if it were allowed to run, it would
report **0.00 on every patch and pass**, on a system that might be rendering
neutrals with a heavy cast. Colour channel misregistration would likewise report no
displacement between records that do not exist. A perfect score for a measurement
that never happened is the worst thing a QC tool can print, so a run that needs any
of these three on a grayscale capture can never come back `PASS`.

If your workflow is genuinely monochrome, the answer is a profile whose category
expects it — FADGI's microfilm tables do not require white balance in the same way —
or a target that disclaims colour, as the microform charts here do. If your workflow
is meant to be colour, a grayscale master is the finding.

### When the verdict says ADVISORY

If the verdict line reads `PASS (ADVISORY -- NOT A CONFORMANCE TEST)` — or `FAIL`
the same way — your capture was measured against the **manufacturer's nominal values
for your model of chart**, not against a measurement of the chart you photographed.
The report header shows this too: the reference line reads `MANUFACTURER NOMINAL`.

Every chart differs slightly from the aim values it was printed to match. So a ΔE
measured this way contains two things added together — your capture system's error,
and your chart's own print tolerance — and there is no way to separate them
afterwards. That cuts both ways: a marginal pass might really be a fail, and a
marginal fail might really be a pass.

**What it is good for**: setting a system up, tracking it over time, catching real
problems. A capture reading ΔE 8 against nominal has something wrong with it that no
print tolerance explains. Comparing the same chart to itself across months is
perfectly sound, because the chart's offset is the same every time.

**What it is not good for**: certifying conformance to a star level for an auditor,
or arguing about a tenth of a ΔE.

Deltachrome falls back to nominal values only when the target definition ships them
and you did not supply a reference file. It exists because some charts have patches
too small to measure yourself and were seldom sold measured — the ISA Golden Thread
was the case in point — and refusing to analyze them at all would help nobody.

**Try to get your own unit's file before accepting an advisory run.** For the
Golden Thread charts in particular this is now usually possible: Digital
Transitions publish per-unit measurements for over three thousand of them, and
Deltachrome will fetch yours from the chart's own printed code, or from a serial
you type in Settings. See *Your chart may be able to fetch its own file* below.

To turn an advisory run into a conformance one, supply a per-unit measurement file
with `--reference`; it always takes precedence. To refuse the fallback and fail
loudly instead, pass `--no-nominal-fallback`.

### Not all nominal values are the same kind of number

"Nominal" covers two things, and the report's own reference file says which you have.
The difference matters because one is much closer to your chart than the other.

**Design aims** are the values the model was printed to match. They give themselves
away by being round: the Universal Test Target's aims are whole L\* units throughout.
A ΔE against an aim carries the full print tolerance of your particular chart.

**Batch averages** are a measurement — of a print run, not of your card. They give
themselves away too, by *not* being round: they quantize to 0.002 rather than 0.01,
which is what a mean of five readings taken to two decimals does. Most of the targets
Deltachrome ships nominal data for are this kind, and Image Science Associates print
the convention on the chart itself: *"All values are batch averages."*

A batch average sits closer to your card than an aim does, so those numbers are better
data. It is still not a measurement of your card, which is why the run is `ADVISORY`
either way — the difference is how much of the ΔE is yours and how much is the chart's,
not whether any of it is.

### Some charts print their own reference values

Worth knowing before you go looking for a file. The Golden Thread Object Level and
Device Level Targets print a table of L\*a\*b\* values beside the patches, headed
`D50 Illuminant, 2 degree observer`. That table is your card's own data, and it is
right there.

It also means two cards of the same model can carry different tables, because they came
from different print runs — and they do: the two Object Level specimens Deltachrome's
geometry was measured on differ by about 1 ΔE2000 on average. If your card's printed
table does not match the numbers in the report's reference file, that is not an error.
It means your card is from another batch, and **your card's table is the one to use**.

---

## Tone response is measured in ΔL*2000, not plain ΔL*

This is the most commonly misunderstood number in the report, and Deltachrome
follows FADGI's wording exactly.

FADGI states its tone response criterion in **ΔL\*2000** units. That is *not* the
raw difference between the measured and reference L\* values. ΔL\*2000 is the
lightness term of the CIEDE2000 color-difference formula, which divides the raw
difference by a weighting factor called **S_L**:

```
ΔL*2000  =  |measured L* − reference L*|  ÷  S_L
```

S_L is 1.0 in the mid-tones and grows toward both ends of the lightness scale. The
practical effect is that **the same physical error scores lower in the deep shadows
and bright highlights than it does in the mid-tones**.

Here is what that means across the ColorChecker SG's twelve-step ramp, against
FADGI's 3-star threshold of 3.0:

| Ramp L\* | S_L | A 3.0 L\* error scores | Error needed to reach 3.0 |
|---:|---:|---:|---:|
| 96 (highlight) | 1.687 | 1.78 | 5.06 |
| 88 | 1.566 | 1.92 | 4.70 |
| 80 | 1.445 | 2.08 | 4.34 |
| 72 | 1.323 | 2.27 | 3.97 |
| 64 | 1.200 | 2.50 | 3.60 |
| 57 | 1.088 | 2.76 | 3.27 |
| 49 (mid) | 1.003 | 2.99 | 3.01 |
| 41 | 1.121 | 2.68 | 3.36 |
| 33 | 1.247 | 2.41 | 3.74 |
| 26 | 1.354 | 2.22 | 4.06 |
| 18 | 1.475 | 2.03 | 4.43 |
| 10 (shadow) | 1.596 | 1.88 | 4.79 |

**What this means in practice:**

- Near mid-gray, the threshold behaves almost exactly like "3.0 L\* of error".
- At the ends of the ramp it is more permissive: about 5.1 L\* of error at the
  highlight end, 4.8 at the shadow end.
- This is intentional in the standard, not a loophole. The CIEDE2000 weighting
  models the fact that human observers are less sensitive to a given lightness
  difference near black and near white than in the mid-tones.

**Why it matters that Deltachrome does this correctly:** a tool that computed plain
|ΔL\*| and compared it to 3.0 would fail captures at the ends of the ramp that FADGI
passes. On a twelve-step ramp, eight of the twelve steps sit in the region where the
two calculations differ by more than 10%. If you have used a tool that reported
harsher tone response numbers than Deltachrome does, this weighting is the likely
reason — check which quantity it reported.

The report's tonal ramp table shows both, so you can see the raw and weighted
values side by side:

```
  TONAL RAMP (OECF)
    step  patch    ref L*  meas L*  ΔL*2000  ΔE(a*b*)
    1     E5        96.00    95.31     0.41      0.12
    ...
```

`ref L*` and `meas L*` are the raw measurements; `ΔL*2000` is what the threshold
is applied to.

### Ramp order

The report's `step` column is derived from your reference file's measured L\*
values, sorted from lightest to darkest — not assumed from patch position. If that
disagrees with the expected reading order, the report says so. A ramp read
backwards would still pass a worst-case threshold while making the OECF curve
meaningless, so the ordering is established rather than assumed.

---

## Color accuracy: two different numbers

FADGI specifies two color accuracy metrics, and they are not the same measurement:

**Average Color Accuracy** — the mean ΔE2000 across all patches. A general
indicator of how close the whole capture is.

"All patches" means every color patch the chart carries. On most charts the grey
ramp sits inside the color array and counts toward it. The Universal Test Target is
the exception. Its four grey scales are printed apart from its two color bars and
make up 80 of its 134 reference values, so only the 54 color-bar patches count,
as in OpenDICE and Image Engineering's iQ-Analyzer. The grey scales are judged by
tone response and white balance instead.

**Color Accuracy 90th Percentile** — sort every patch's ΔE2000 from best to worst
and take the value 90% of the way up. **This is not the maximum.** With 140 patches
it is roughly the 126th-worst patch, meaning about 14 patches are allowed to exceed
it.

Implementing the 90th percentile as a maximum is a common error, and it is far
stricter than the standard — it can fail a fully compliant system on one dusty
patch. Deltachrome uses the actual percentile.

> **A note on interpolation.** FADGI does not specify how to interpolate a
> percentile when it falls between two measured patches. Deltachrome defaults to
> `linear` (the same convention as Excel's `PERCENTILE` and NumPy) and **states
> the method in the report**, because it is a choice rather than a fact. If you
> prefer the classical definition, which always returns a value some patch actually
> measured, use `--percentile-method nearest_rank`. It is slightly more
> conservative.

**Color Accuracy Maximum** — a third number, and Metamorfoze's rather than FADGI's.
Where FADGI pairs the mean with a 90th percentile, Metamorfoze pairs it with a
**maximum**, so a single bad patch decides it. That is exactly the strictness the
90th percentile was designed to avoid, and it is deliberate on Metamorfoze's part —
which is why the two are separate metric names in a report rather than one row whose
meaning depends on the profile. Extra Light specifies no color accuracy at all.

The **WORST PATCHES** table lists the highest-ΔE patches by name so you can see
whether a bad result is systematic (a color cast affecting everything) or local
(one or two patches, often dust or a specular reflection). Under a maximum, that
table is not a diagnostic aid but the whole verdict.

---

## White balance excludes lightness on purpose

White balance is reported as a color difference in a\* and b\* only, with L\* left
out entirely: **ΔE2000(a\*b\*)**, CIEDE2000 with its lightness term removed.

That is FADGI's definition. Section 2.4.4 calls it "similar to the color accuracy
ΔE2000 metric, with the exception that the L\* component is eliminated", and it is
what OpenDICE computes. The reasoning behind leaving lightness out is diagnostic. A gray patch
reproduced too dark is a **tone response** problem. A gray patch reproduced with a
color cast is a **white balance** problem. They have different causes and different
fixes. Folding lightness into the white balance number would make a pure exposure
error show up as a white balance failure, sending you after the wrong thing.

Deltachrome reports them separately, and they stay independent: an exposure or
gamma error moves tone response without touching white balance.

The threshold is applied **per patch, not on average**. FADGI's wording is "for any
given gray patch", so the worst gray patch on the ramp has to clear the limit — one
badly cast step is a failure even if the other eleven are perfect.

### Two formulas that are not interchangeable

Both FADGI and Metamorfoze ask for **ΔE2000(a\*b\*)**. FADGI's tables label the
column "ΔE(a\*b\*)", which reads like the plain Euclidean distance in the a\*b\*
plane. That is a different number, because CIEDE2000 keeps its chroma and hue
weighting and the Euclidean form has none. Deltachrome follows FADGI's own
definition of the metric, not the shorter label.

How different the two are depends on where the color sits. Each of these is
exactly 5.0 Euclidean units:

| | ΔE(a\*b\*) | ΔE2000(a\*b\*) |
|---|---|---|
| near-neutral | 5.000 | **5.310** |
| saturated red | 5.000 | 1.697 |
| saturated blue | 5.000 | 1.028 |

The difference across the range is a factor of five, and its direction
**reverses**. Near the neutral axis, which is where white balance is actually
measured, CIEDE2000 reads about 6% *higher*. A tool that uses the Euclidean form
therefore reports white balance slightly better than it is.

A guideline you have built yourself keeps whichever formula it was built with. The
units column says which you are looking at. If you are comparing a Deltachrome
figure with another tool's, check that column first.

---

## Lightness uniformity: what it does and does not measure

**What it measures:** FADGI states the units explicitly — *standard deviation
divided by mean* — so the reported value is a **percentage**: the standard
deviation of measured L\* across a set of nominally identical neutral patches,
divided by their mean L\*. Thresholds are < 8% / 5% / 3% / 1% for 1- to 4-star.

**Where the patches come from.** Two ways, and the report says which. On a chart
whose reference file shows a neutral repeated across its area — a ColorChecker SG —
the set is found in the measurements, so nothing has to be hand-listed. On a chart
that names its own, the definition declares it: the ISA Golden Thread's twenty
registration squares, four at the corners of each of its five modules, reaching
every corner of the chart and its center.

**Uniformity needs no reference values.** It is the spread of a set about its own
mean, so the set is its own reference — which matters, because the Golden Thread
prints no values for those twenty squares. When they are absent you lose two things,
and the report names both rather than omitting them quietly: the split into neutral
families, and the chart-corrected figure that separates the chart's own spread from
your system's. Without that correction, the number you get is an **upper bound** on
your system's non-uniformity: some of the spread belongs to the chart.

**Which patches, and why it matters more than you would think.** FADGI does not
name them. A ColorChecker SG's border is three interleaved neutral families —
near-white, mid-gray and near-black. Standard deviation over mean is far harsher on
a dark family because the denominator is small: the same 1 L\* spread scores about
1% at L\*96 and about 10% at L\*10. Deltachrome reports the **lightest** family by
default, which follows the convention for uniformity references, and lists every
family's value so the choice hides nothing. Use `family_selection` in the profile
to report the worst or darkest instead.

Your chart's own patches are not perfectly identical. The report gives the raw
FADGI ratio as the headline value, because that is what the standard asks for, and
the chart-corrected figure alongside it.

**What it does not measure:** uniformity across your full camera frame. Your chart
covers part of the frame, so the measurement covers that part. FADGI's own
GoldenThread tooling scopes device-level target uniformity the same way, so this is
consistent with existing practice — but it is worth being clear about, because
"uniformity" sounds like it should mean the whole frame.

The report tells you exactly what the number rests on:

```
  LIGHTNESS UNIFORMITY SAMPLING
    13 points, 13 on the chart perimeter, 0 interior
    covering 89% of the chart area, centroid at [0.5, 0.5]
```

Read those numbers before trusting the result. Thirteen points spread across 89% of
the chart is a reasonable sample. Five points clustered along one edge is not, and
the report will show you that rather than hiding it behind a single figure.

Deltachrome deliberately does **not** invent separate "sparse" and "dense"
capability tiers. Sample count and distribution are reported as numbers, so you can
judge confidence directly.

**Manufacturing spread is subtracted.** The "identical" neutral patches on a real
chart are not truly identical, and your reference file records what they actually
measure. That known spread is removed before reporting, so you see your camera's
non-uniformity rather than your chart's. If the reference patches themselves differ
by more than 0.5 L\*, the report says so — that patch set is a weaker uniformity
reference than usual.

---

## Illumination across the image plane

This is a **different measurement from lightness uniformity above**, on a different
kind of capture, and the two are easy to confuse because both are about evenness.

| | Lightness uniformity | Illumination non-uniformity |
|---|---|---|
| Asked for by | FADGI | Metamorfoze |
| Measured on | a chart's repeated neutral patches | a frame-filling white sheet |
| Covers | the area the chart occupies | the whole image plane |
| Reported as | standard deviation / mean, a **percentage** | a **ΔL\***, the spread between spots |
| Typical threshold | < 3% (3-star) | < 3 (originals up to A3) |

Because the numbers land in a similar range and neither carries an obvious unit,
quoting one against the other's threshold is an easy mistake and Deltachrome keeps
them under separate names for that reason.

### Why a separate capture at all

A chart sits in the middle of the frame. So a uniformity figure computed from a
chart's patches describes the middle of the frame — and illumination fall-off is
worst at the corners, which are outside it. Metamorfoze is explicit that the
measurement wants *"a frame-filling UTT or a frame-filling L\*a\*b\* measured white
sheet (made of paper, cardboard or plastic)"*, at roughly L\* 95.

Deltachrome models that as a target in its own right — `flat_field_v1` — with no
patches at all. You place the four corners on the area you want measured, which is
normally the whole image. The two metrics it supports:

- **`illumination_non_uniformity`** — the ΔL\* between the lightest and darkest
  spot on the plane. The report names both, and where they are.
- **`white_balance_non_uniformity`** — the largest white balance difference between
  *any two* spots. Measured spot against spot rather than against a reference,
  because the fault being looked for is a cast that varies across the frame — one
  side warmer than the other — which a white balance read at the center cannot see.

Nothing is compared against a reference, so the sheet's own color cancels and only
its evenness is measured. That is why an unmeasured sheet of paper is a legitimate
target here and would not be anywhere else.

### The tolerance depends on the size of your originals

Metamorfoze bands it, because holding a large platen even is harder:

| Largest original | Illumination ΔL\* | 
|---|---|
| up to A3 | 3 |
| over A3, up to A2 | 4 |
| over A2, up to A1 | 5 |
| over A1, up to A0 | 6 |

So the profile has to be the one for the size you are digitising — a run against the
A3 profile is not a pass for A0 work, and vice versa.

### Two things that will flatter a bad capture

**A clipped sheet reads as perfectly even.** If the white is blown, every spot
returns the same maximum value and the spread is zero — the measurement scores
perfect exactly when the capture is worst. Deltachrome flags every clipped spot and
says so in the notes rather than reporting the zero quietly. If you see that
warning, the number is meaningless; re-shoot with the sheet below the encoding limit.

**A dark sheet compresses the result.** The same lighting fall-off measured on an
L\* 60 sheet produces a smaller ΔL\* than on an L\* 95 one, because L\* is
non-linear in luminance. The report calls out a field averaging below L\* 80.

### What it cannot tell you

Unevenness of the *sheet* is measured as if it were unevenness of the *lighting*,
and one capture cannot separate them. Rotating the sheet 180° and re-shooting does:
the lighting's pattern stays where it is, the sheet's turns with it.

Spots at the frame edge are half-spots, clipped rather than nudged inward, so the
sample still reaches the very corner — which is the whole point of measuring. The
cost is that a clipped spot's average sits inward by about a quarter of its width,
so a gradient is under-reported slightly: a synthetic ramp of exactly 3.0 ΔL\*
comes back as 2.94. The error is always in the conservative direction for the
sheet and the permissive direction for your verdict, and a smaller `spot_size`
reduces it at the cost of more noise.

---

## Sharpness: the slanted-edge measurements

Four numbers come off the same measured curve, and they answer different questions.
All are read from edges the target prints at a deliberate few-degree tilt — the tilt
is what makes the measurement possible, because each row then catches the edge at a
different sub-pixel phase.

| Metric | Question it answers |
|---|---|
| **SFR10 sampling efficiency** | How much of the resolution your pixel count promises is actually delivered? |
| **SFR50** | Where does response fall to half? Tracks perceived sharpness better than the 10% point. |
| **Response at Nyquist** | How much survives *at* the sampling limit? |
| **Sharpening** | Is the system boosting contrast it did not capture? |

### Sampling efficiency above 100%

Not an error. It means the optics still resolve beyond the limit of the sampling
grid, so detail finer than the pixels can record is **aliasing** into the image
rather than being filtered out. FADGI's criterion is a floor (`> 90%` at 4-star), so
a high value passes — but it is worth knowing, because aliasing shows as moiré on
fabric, screens and fine rules, and no amount of later processing removes it.

SFR50 failing its *upper* bound says the same thing in a different way.

### MTF50 in lp/mm, and why the unit changes what it means

The measurement off a slanted edge is a frequency in **cycles per pixel**, and every
other unit is that number times a property of the capture. FADGI states MTF50 as a
percentage of the half-sampling frequency; Metamorfoze states its resolution
requirement in **line pairs per millimeter**. Same measurement, different unit —
`--frequency-units` switches between `lp/mm`, `cycles/mm`, `lp/inch`, `cycles/inch`,
`cycles/px` and `% of half-sampling freq`. One line pair is one cycle, so the two mm
forms are the same number under two names.

The choice matters more than it looks:

- **Per pixel, or as a percentage of Nyquist**, the figure is about the *system's*
  sharpness relative to its own sampling grid. It needs nothing else to compute.
- **Per millimeter**, it is a joint statement about sharpness *and* sampling rate,
  referred to the original. A softer capture at a higher rate beats a sharper one at
  a lower rate — which is usually what a preservation requirement intends.

So anything per unit of distance needs a **sampling rate at the target plane**: a
chart carrying a printed scale, or `--fiducial-extent`. Without one, Deltachrome
reports the metric as incomplete and points you at `cycles/px` rather than inventing
a rate and printing a confident lp/mm figure that means nothing.

**A lp/mm requirement can be unreachable.** A line pair needs two samples, so nothing
above `ppi ÷ 25.4 ÷ 2` is resolvable however good the lens. Deltachrome reports that
ceiling beside the reading, and if the requirement sits above it says so and gives
the sampling rate that would change the answer — because a bare FAIL there sends you
shopping for optics when the problem is the resolution you chose.

### Sharpening, and why two numbers are reported

Sharpening is the peak of the MTF curve. The curve is 1.0 at zero frequency by
definition — that is the normalization — so an unsharpened system reads exactly
**1.000**, and anything above that is contrast being added. FADGI's limits sit just
above 1.0 for exactly that reason.

The classic oversharpening signature is: 1.0 at DC, a lift shortly after, then the
normal downward slope. That lift is the halo.

Alongside it the run record keeps **spatial overshoot**, measured on the edge profile
rather than its transform: how far the profile shoots past its own plateau before
settling. That excursion *is* the bright rim on the light side of a sharpened edge
and the dark one on the dark side. Calibrated against known sharpening: a clean edge
gives **0%**, a mild sharpen about **10%**, a strong one about **27%**. Imatest treat
roughly 25% as the limit of acceptability.

Why both: they fail differently. The MTF peak is a Fourier quantity, and the first
couple of transform bins are where an analysis error shows up before anything
optical does — so a small excursion there deserves the second opinion. Overshoot is
measured on the profile itself, and no amount of transform trouble can put a rim on
an edge that has none.

This is not hypothetical. Deltachrome used to report up to **1.105** here, on a
capture that Image Engineering's iQ-Analyzer and FADGI's OpenDICE both read as
1.00 — two steps in its own edge processing were moving the value the curve is
normalised against, which scales the whole curve. Both are gone. An unsharpened
system now reads **exactly 1.000**, on every edge of every capture checked against
those two tools. If yours does not, the number means something.

### Comparing an MTF50 with another tool's

Two differences will move the number, and both are method rather than error.

**Linearization.** This measures the edge after converting it through the capture's
own ICC transform, because the method assumes a profile linear in luminance. A tool
that works on the encoded values instead reads lower, and by a varying amount --
how much depends on where the edge's two levels sit on the tone curve. Measured
against OpenDICE on two captures, that difference alone runs from 0.1 to 1.9 points
and accounts for most of the disagreement between them and us.

**Which record "luminance" means.** Here it is one MTF of a luminance-weighted
image, so green dominates. OpenDICE instead averages the three separate RGB curves
with equal weight. Neither is wrong; they are different quantities and they differ
whenever the records resolve differently.

Measured L\* agrees with OpenDICE to a few hundredths across the grey scale,
per-patch noise agrees closely, and maximum SFR amplitude agrees exactly. Among
these, the sharpness figures are the ones that carry the linearization difference.
[Comparing with OpenDICE](#comparing-with-opendice) has the full picture, including
the metrics where the two tools calculate different things.

So a figure from another tool is worth comparing for *shape and pattern* -- which
corner is soft, whether the field is even -- more than digit for digit. Where a
digit-for-digit match matters, check that both tools linearize the same way first.

### The same edge in each color record

Every edge is also measured separately in red, green and blue, and all three curves
are kept. **None of it can change a verdict** — every sharpness number above is read
from the luminance measurement — but it answers a question luminance cannot.

If the three records resolve alike, their curves lie on top of one another. If they
separate, one record is soft. Whether that matters, and what it means, depends on
where it happens: a difference that **grows towards the corners** is lateral
chromatic aberration, which is nil on the axis by definition. A record that is
softer **everywhere by the same amount** is something else — a defocused channel in
a tri-linear array, or per-channel processing — and needs a different fix. The
report correlates the per-edge spread against distance from the center of the field
and says which of the two it looks like.

The per-edge table carries a column per record — **R**, **G**, **B** — each
holding that record's own SFR50 at that position, which is the layout OpenDICE
prints. Sort on one of them and a lens focusing blue short is one click: the blue
column runs low at the corners and level at the centre, which is the classic
radial signature. A spread that is the same everywhere is the other fault.

One caution that applies to both: a chart whose own printed edges are chromatically
offset — ink registration on the print itself — produces exactly this picture with
nothing wrong with your system. If it matters, confirm against a second target.

**Reading them together:**

| MTF peak | Overshoot | Reading |
|---|---|---|
| 1.000 | ~0% | No sharpening. This is what a clean capture reads. |
| slightly > 1, peak below ~0.05 c/px | ~0% | Not a halo — sharpening lives near 0.09 c/px. Suspect the ROI: a gradient across it, or an edge too close to another feature. |
| > 1, peak near 0.09 c/px | several % | Real sharpening |
| well > 1 | > 25% | Heavy sharpening; visible halos |

The report says where the peak sits, so you can tell which row you are in.

### Every edge is kept — this is the diagnostic part

The pass/fail reduces a field of edges to one number and tells you nothing about
*why*. The differences between edges are the diagnosis, so every edge is recorded
with its **orientation** and **field position**.

**Orientation names the direction the response was measured in, not the way the
printed edge runs.** That is ISO 12233's convention: horizontal resolution is the
value measured in the longer image dimension, "typically using a vertical or near
vertical oriented test-edge". So an edge listed as *horizontal* was read across a
near-vertical printed edge, and tells you how the system resolves detail from side
to side. The saved record stores the other thing — the edge's own geometry, under
`orientation`, because that is what decides which way the ROI is cut — so a record
read back with your own tools uses the opposite word. The report and the console
both say the direction, and the cell's tooltip says which edge it came from.

With that settled:

- A consistent gap between edge orientations at the same positions is what
  **astigmatism** looks like — sagittal and tangential rays focusing at different
  distances. It is also what motion during capture and line-scan artefacts look
  like, so treat it as a pointer.
- Softness growing from center to corner is **field curvature** or a decenterd
  element. Softness on one side only is **tilt** — of the sensor, the lens, or the
  copy stand.
- Four edges of one square disagreeing means the local focus or the measurement is
  unstable.

The FIELD PATTERN block reports the horizontal-minus-vertical difference **against
the spread across the whole field**. A difference smaller than the field's own
scatter is not evidence of anything.

A worked example: a capture reading 71.36% in one direction against 70.82% in the
other — a gap of only 0.22× the field spread, so no evidence of astigmatism —
while position ran 69.0% at top-left to 74.0% at bottom-right. That diagonal
gradient is a plane tilt, not a lens aberration, which would be symmetric about
the optical axis.

### How several edges become one verdict

**Neither FADGI nor ISO 19264 says.** Both state per-edge criteria and stop there.
Deltachrome defaults to the **mean**, following OpenDICE — the open FADGI-aligned
implementation, which reports "the RGB average sampling efficiency values of all the
edge regions" — so your figures are comparable with a GoldenThread report of the
same capture without conversion.

Be aware of the trade: **a mean can pass a system with one badly soft corner.** That
is what the reported spread and the per-edge table are for. Set `edge_aggregation`
to `worst` in a profile for a conservative gate; worst, mean, median and spread are
reported either way.

### If the numbers look unstable

Slanted-edge SFR is sensitive when a system is very sharp relative to its own
sampling. If the measured line spread function is around one pixel wide, the choice
of analysis bin width can move MTF at 0.25 cycles/pixel by more than 0.1, and two
legitimate binning methods can differ per-edge by a similar amount. The *mean over a
field* of edges is far steadier than any single edge. Do not read single-edge SFR to
three figures on a system that sharp.

## Sampling frequency and reproduction scale: measured, not declared

Two numbers in the report come from the chart's own printed rule rather than from
anything you or the file tell Deltachrome.

**Sampling frequency** is how finely your system actually resolved the original, in
pixels per inch. Deltachrome measures it by reading the tick pitch off the rule
printed on the target. It does not read it from the file header, and it does not
work it out from a declared chart size — a header can say anything, and a chart's
physical dimensions vary between batches and cuts.

**Reproduction scale accuracy** compares that measured rate against what your file's
header claims, as a percentage. It is the only number in the report that tests
*metadata*, and that is exactly why it earns its place: a wrong header resolution is
invisible to every other check, and everything downstream believes it — print size,
on-screen scale, any measurement taken from the file, and the catalogue record that
says what was digitised at what resolution.

The report shows the working:

```
  PRINTED SCALE (sampling rate measured from the chart, not the header)
    ruler                       division  divisions   pitch px  residual       ppi
    ruler_metric_top            1.0000mm         24     78.801     0.444    2001.5
    ruler_imperial_top          1.5875mm         15    125.155     1.055    2002.5
    combined 2002.0 ppi; the chart's scales agree to 0.047%
    the file's header claims 600.0 x 600.0 ppi
```

### Why two rules, and why their agreement matters

Both ISA targets print a metric and an imperial scale, engraved separately.
Deltachrome reads both. Their agreement is the only check available that each was
read the way the chart intends — because a rule misread by a factor of two still
fits a perfectly straight line through its own ticks and reports a small residual
while doing it. If the report says the scales disagree, treat *both* numbers as
suspect rather than averaging them.

**A small disagreement between units is worth reading twice.** A rule misread by a
factor is obvious. A chart whose two engravings differ by a fraction of a percent is
not, and it happens: the ISA FADGI – ISO 19264's millimetre comb reads 0.74% above
its own inch comb on one capture and 1.26% above on another, with both combs located
cleanly and the millimetre comb agreeing with its own decade markers to 0.014%. Two
scales, both read correctly, printed at different pitches. When Deltachrome sees a
between-unit difference bigger than the readings' own fits can account for, it says
so — and what it means is that **only one of your chart's scales can be a
dimensional reference**, not that the capture did anything wrong. The sampling rate
quoted is still the mean of both, so it sits between them.

### If reproduction scale fails

Check the header before you check the camera. It is metadata and costs nothing to be
wrong, while the measured rate is corroborated by the fit residual and by the second
rule. A failure of a few percent is a genuine scale error worth chasing. A failure of
tens or hundreds of percent almost always means the header was never set for this
capture — one of the sample files here claims 600 ppi for a capture the chart
measures at 2002.

The value is reported as an absolute percentage, because FADGI writes the limit
two-sided (±2%). The signed difference is in the saved run record: positive means
your capture resolves finer than the file claims, negative means coarser.

---

## Density and L\*: related, but not the same measurement

Transmissive targets are specified in **optical density**. Every metric here works
in **L\***. The two describe the same physical thing — how much light gets through —
and they are still not interchangeable, so it is worth being precise about where
they meet and where they part.

### Where they meet

Both start from the fraction of light a sample passes, and differ only in what they
do with it:

| | | |
|---|---|---|
| **Density** | `D = -log10(T)` | logarithmic, unbounded, 0 at full transmission |
| **L\*** | `L* = 116·f(Y/Yn) - 16` | roughly cube-root, bounded 0–100, relative to a white |

For a neutral sample, put one into the other and you get a closed form:

```
L* = 116 · 10^(-D/3) - 16          (while Y/Yn is above 216/24389)
```

So there is a real relationship, and Deltachrome uses it —
`density_to_lightness()` in `imaging/color.py`.

### Where they part

**1. Equal steps in one are not equal steps in the other.** Differentiating the
above gives `dL*/dD = -(ln10/3)·(L*+16)`: about **8.6 L\* per 0.1 D near white, but
only 2.8 near L\* 20**. A wedge with even density steps is not a wedge with even
lightness steps, and a tolerance quoted in one means something quite different at
each end when read in the other.

**2. L\* has a floor and a kink; density does not.** Below `Y/Yn = 216/24389` — about
**D 2.05** measured from a white anchor — L\* leaves its cube-root branch for a
linear one. The ISA 4x5 ladder reaches D 2.25, so **its darkest step is past that
break** and the simple formula above no longer applies to it. Density just keeps
going; there is no density at which the scale changes character.

**3. Density has no color.** It is one number. Lab is three. Two patches of
identical visual density can differ in hue by any amount at all, so a density
reference can constrain tone response and can *never* constrain color accuracy.
This is the plainest sense in which the two are apples and oranges.

**4. They weight the spectrum differently.** `Y` is the transmittance weighted by
the illuminant and the observer's ȳ color-matching function. Visual density is
weighted by the densitometer's own response, which is a different standard (and
"density" without a status — visual, status A, status M, printing — is ambiguous to
begin with). The two agree exactly only for a **spectrally non-selective** sample.
Silver-halide neutral scales are close to that, which is why the conversion is
usable here at all, but "close" is doing real work in that sentence.

**5. They are anchored differently.** Density is measured against the beam with no
sample; film ladders often quote it including the base, which is why the ISA 4x5's
lightest step is D 0.05 rather than 0. L\* is measured against a reference **white**.
Relating them means choosing which density counts as white, and that choice moves
every value in the ladder.

**6. Density describes the object; L\* describes a reproduction.** This is the one
that matters most. The density of a piece of film is a fact about the film, the same
under any scanner. There is no matching fact about its L\*: that depends on how your
system maps transmittance to output — which is precisely the tone response the
measurement is testing.

### What that means for your report

When you analyze a transmissive target against a density ladder, converting it to
aim L\* is **choosing a rendering intent, not converting units**. Deltachrome anchors
at FADGI's own transparency highlight aim of 96 ± 2 L\*, so the choice is the
standard's rather than ours, and says so in the reference file, but it is still a
choice. Two consequences:

- A tone-response figure against a density-derived reference tells you how closely
  your system matches *that* rendering. A different, equally defensible anchor moves
  the numbers.
- Such a run is reported as `ADVISORY` for two reasons stacked on top of each
  other: the values are nominal for the model rather than measured from your chart,
  **and** the L\* aims were derived rather than measured.

White balance is the exception, and cleanly so. It is ΔE(a\*b\*) with L\* excluded, so
it never touches the anchor at all — on a target whose scales are spectrally neutral
by design, an aim of a\*=b\*=0 measures exactly what the metric is for.

---

## Geometric distortion

**What it measures:** magnification is supposed to be the same everywhere in the
frame and in every direction. This is how much it is not, as a percentage.

Metamorfoze names three ways that fails — pincushion (the sides of a square bow
inward), barrel (they bow outward), and lines shifted relative to each other so
squares become diamonds — and gives one tolerance for all of them: ≤ 2% for Full and
Light, ≤ 4% for Extra Light.

### Where the number comes from

Off the **rules the chart prints**, not off the chart's own geometry. Each rule is a
local measurement of scale, at a known place, along a known axis. The ISA Golden
Thread prints twenty of them — a centimeter and an inch rule on all four sides of
each of its five modules — so each axis is measured five times, and twice over at
each position through independently engraved scales.

Reading it off the printed rules rather than by fitting the chart's fiducials is
deliberate. The Golden Thread's declared geometry was itself measured from a
capture, so fitting it and reporting the residual would be comparing a capture
against its own distortion. A printed rule has a physical pitch that no fit absorbs.

### The two components, and why both are reported

- **Field spread** — the difference between the highest and lowest scale reading
  anywhere on the chart, over their mean. Pincushion and barrel show up here, as the
  corners reading a different scale from the center.
- **Per-position anisotropy** — at one position, how much the horizontal scale
  differs from the vertical. This is the shear that turns squares into diamonds, and
  a pooled figure hides it: four positions can be perfect while one is sheared badly.

The reported value is the **worse of the two**, because the standard gives one
tolerance for every form it names. Both are always shown, along with every
individual rule and a per-position table, because the differences between positions
are the diagnostic — a single number tells you that something is wrong, not where.

### A chart whose own scales disagree

Distortion is the magnification *varying* — with where you look, and with which
direction you look along. A chart's metric and imperial engravings disagreeing is
neither: it moves every rule of one unit together, so it has no place and no
direction. It belongs to the chart, not to your lens.

Deltachrome takes that offset out before measuring anything, and reports it
separately as `unit_disagreement`. Without that, a chart printing two scales 0.2%
apart would report 0.2% of pincushion on a flawless capture — on the ISA 120 that was
two thirds of the figure.

There is one case where it cannot: a chart printing millimetres only across and
inches only down. Then "the two engravings differ" and "the two axes differ" are the
same measurement, and choosing either would be inventing the answer. Deltachrome
refuses the correction there and says so in the notes, so you know the offset is
inside the number.

### What it cannot see

**Curvature inside one rule's span.** A rule is short and a comb fitted through it
absorbs mild bowing into its own pitch. Between modules the same distortion shows up
as the corners reading differently from the center, which is what the field spread
measures — but a chart with only one rule, or with all its rules in one place,
cannot measure distortion at all and the metric says so rather than returning a
number. The fit residual reported beside every slanted edge is the other half of
this picture, and `edge_fit: quadratic` exists for it.

### The measurement floor

Each rule's pitch carries a standard error from its own comb fit, and a spread
smaller than that is the reading rather than the lens. The floor is kept in the run
record rather than announced, since it never changes the verdict: a value under
about three times it is better read as an upper bound than as a measurement. How
low the floor can go is mostly the chart's doing — long rules with many ticks give
a low one — so there is little an operator can do about it beyond a sharper,
higher-resolution capture.

The floor is itself a lower bound on the uncertainty: an error common to every rule
— the chart printed slightly off-size — moves all twenty readings together and so
never appears in the spread between them.

---

## Color channel registration

All lenses focus red, green and blue slightly differently, and any system that uses
motion to capture — a linear or planetary scanner — can lay the three records down
out of step. FADGI measures the result in pixels of dislocation, with limits of
1.2 / 0.8 / 0.5 / 0.33 px at 1 to 4 stars. Seen in an image it is color fringing at
hard, high-contrast edges.

It is measured off the same slanted edges the sharpness metrics use: the edge is
located separately in each color record, and the differences between those
locations are the misregistration. That means it needs a target with slanted edge
features — an ISA Golden Thread or DICE-style chart. A patch-only chart such as the
ColorChecker SG reports `N/A`, which is not a failure.

### Reading the table

The report lays it out the way OpenDICE does, so the two can be compared directly:
one column per channel pair (**G-B**, **G-R**, **B-R**), positions down the page,
and the two axes separated. The row labels here name the direction a record
*moved*, which is the same convention the edge table uses: a vertical printed edge
can only reveal a sideways displacement, and a horizontal one only an up-and-down
displacement.

Field positions are abbreviated to one or two letters — `UL`, `UC`, `UR`, `CL`,
`C`, `CR`, `LL`, `LC`, `LR` — because a chart with a full three-by-three of edge
modules puts nine of them on an axis a couple of hundred pixels wide, six times
over.

The values are **signed**, and the sign is half the diagnosis:

- A displacement that **grows with distance from the frame center** is the lens —
  lateral chromatic aberration, which is radial and vanishes on axis.
- A displacement that is **the same everywhere, center included**, is not the lens.
  Look at the capture mechanism: a tri-linear scanner's arrays, a stitching step, a
  sharpening filter applied per channel.
- Only two of the three pairs are independent. The third is shown because the
  extremes of the spectrum are often the worst pair.

### The arrows, and the radial split

That distinction — the lens against the capture mechanism — used to be left to you
to make by reading down a column of signed numbers. It is now measured and drawn.

Each position that gave both axes has a displacement **vector**, and the report
draws it as an arrow where the reading was taken. The two cases look completely
different at a glance:

- Arrows **pointing out of (or into) the frame center and growing towards the
  corners**: one record imaged at a slightly different magnification from another.
  This is the radial, lens-side case.
- Arrows **the same length and direction everywhere, center included**: the records
  are offset bodily. Look at the capture mechanism.

Alongside them the report gives the split as numbers: the average radial and
tangential components, how strongly the radial part correlates with distance from
the center, and how closely the arrows agree on a direction. A note reads them back
and names the pattern.

Two cautions. The arrows are **scaled to be visible** — these displacements are
sub-pixel on a master tens of thousands of pixels wide, so at true scale none of
them would be longer than the dot it starts from. Their *relative* lengths are
true and the caption says what the longest one measures; nothing else about their
length means anything. And with five positions, which is what an ISA target gives,
these are weak statistics: when the report says neither pattern is clear, that is
the honest answer rather than a hedge.

### The line marked "antisymmetric term"

This one is worth understanding, because it is the number most likely to be
mistaken for misregistration.

Each square feature gives two edges per axis, and they face opposite ways. A genuine
displacement moves both the same way. Something that makes the feature measure a
slightly different **size** in one record moves them oppositely. Deltachrome
separates the two: the verdict uses the displacement, and the size term is reported
under its own name.

It is usually the larger of the two. On the reference ISA capture the displacement
is 0.003–0.028 px while the size term is 0.030–0.048 px, almost entirely on the
pairs involving blue. Blue scatters most, so veiling flare — light bouncing around
inside the lens and the copy setup — reaches into a dark feature more in that record
and moves where its edge appears to be. **That is a property of your lighting and
your chart, not of your camera's channel registration**, and folding it into the
verdict would fail systems that are registered correctly.

If that term is large, chase flare: stray light, lens surfaces, a bright surround,
a missing lens hood. If the *displacement* is large, chase the lens or the capture
mechanism.

### If the number is very small

The run record keeps the capture's own measurement floor. A reading within about
three times it is "no measurable misregistration" rather than a displacement of
that size — the digits after the decimal point are describing the
measurement, not your system.

### How several positions become one verdict

As with sharpness, **neither FADGI nor ISO 19264-1 says**. Registration defaults to
the **worst** position, the opposite of the sharpness default, for two reasons:
OpenDICE plots every position against the limit lines rather than displaying an
average, so the worst point is what a reader judges from that screen; and fringing
in one corner is a fault whether or not the rest of the frame is clean. Worst, mean,
median and spread are all reported.

The verdict is also judged **per axis**, as OpenDICE plots it. The dislocation at a
point is really a two-dimensional vector, and its length is never shorter than
either axis alone — a 0.4 px diagonal displacement reads 0.28 px on each axis. The
vector figure is printed alongside, and a profile can be set to judge on it instead.

## Beyond color: the file itself has to conform

FADGI's tables specify more than colorimetry. Every star level also fixes the
master file format, bit depth, color space and sampling frequency, and a capture
that misses any of those does not meet the level however good its ΔE is. So these
are real metrics with real pass/fail, not report decoration:

| Metric | What it checks |
|---|---|
| `master_file_format` | The container, against the permitted list (TIFF, JPEG 2000) |
| `bit_depth` | 4-star Prints and Photographs requires 16-bit; 3-star permits 8 or 16 |
| `color_space` | 4-star drops sRGB and Gray Gamma 2.2, permitting only Adobe RGB (1998), ProPhoto and ECIRGB_v2 |
| `sampling_frequency` | Pixels per inch **measured at the target plane**, not read from the file header — a header ppi can say anything |

A failing `bit_depth` cannot be fixed by re-encoding. Precision that was never
captured does not come back; the material has to be re-shot. The report says this,
because converting the file is most people's first instinct and it wastes a day.

`sampling_frequency` is measured at the target plane from the placed fiducials, so
it needs to know the physical size of the area those fiducials enclose. On a
ColorChecker SG the crosshairs enclose 247.6 mm of a 292.1 mm card — deriving ppi
from the card would under-state it by about 15%, enough to fail a capture that
actually conforms.

Not every target can supply that. A chart with a printed scale (the SG has one below
columns K–N) can be verified from its own capture. A chart with no dimensional
markings cannot, and Deltachrome will not carry a figure measured from one specimen:
film targets are imaged onto stock and then cut to format, so neither the imaging
scale across batches nor the placement of the cut is guaranteed.

When a target cannot establish its own scale, the metric reports **INCOMPLETE, not
N/A** — the resolution requirement still applies and your capture may well miss it;
only the verification is impossible. Two ways forward:

- Use a target that carries dimensional markers.
- Measure your own specimen and pass `--fiducial-extent WxH` in mm. The report
  records that the figure was operator-supplied and notes that it describes the unit
  you measured, not the model.

## Noise

Reported as the standard deviation of L\* inside each flat neutral patch, computed
on color-managed pixels rather than device values (a fixed device-value spread
means very different L\* noise at the two ends of the tone scale).

FADGI sets two limits. The **upper** one tightens with star level — < 4 / 3 / 2 / 1.
The **lower** one is fixed at 0.25 across every level and is described in the
guidelines as raising a warning rather than a failure. That lower limit is not a
quality floor, it is a tell: a capture with almost no measurable noise has usually
had noise reduction applied, which destroys the fine detail the digitisation was
meant to preserve. Deltachrome reports it exactly that way — a note, not a FAIL.

## Color management: why Deltachrome refuses some images

**Every image must have an embedded ICC profile.** Deltachrome will not analyze one
without it.

This is not pedantry. Master files in cultural heritage digitisation are routinely
Adobe RGB, ProPhoto, or ECI RGB — not sRGB. If a tool assumes sRGB for a ProPhoto
file, every ΔE it reports is wrong, sometimes by many units, and *nothing in the
output looks wrong*. A confidently incorrect QC report is worse than no report.

Deltachrome reads the profile's own primaries and tone curves and works in full
float precision throughout. It handles both ways an ICC profile can describe a
transform:

- **Matrix/TRC profiles** — every RGB working space (sRGB, Adobe RGB 1998,
  ProPhoto, ECI RGB v2, Rec.2020, Display P3). The transform is stated
  analytically, and Deltachrome evaluates it exactly.
- **LUT-based profiles** — typical of scanner and camera profiles, where the
  device's response cannot be reduced to per-channel curves plus a matrix.
  Deltachrome evaluates the lookup pipeline directly in float.

When a LUT profile is used, the report says so:

```
  ICC        MyScanner 2026-03 profile
             LUT transform via A2B1 (mft2, CLUT 33x33x33, lab_legacy16)
```

**Rendering intent.** For LUT profiles Deltachrome uses the **A2B1** table
(media-relative colorimetric), which describes the device's actual response. If a
profile only offers **A2B0** (perceptual), Deltachrome uses it but warns — perceptual
tables include gamut mapping, so the measurement then reflects the profile's
rendering choices as well as the device.

**Profiles with both structures.** A few profiles carry both. Deltachrome prefers
the LUT for device profiles (scanner, camera, printer), because that is what such a
profile exists to express; and prefers the matrix for working-space and display
profiles, where the matrix is the exact definition and a LUT only re-samples it.
The report notes which was used whenever there was a choice.

### White point

Reference measurements and image values must share a white point. Deltachrome
evaluates both against **CIE D50**, which is what a spectrophotometer reports Lab
against. If your reference file was measured under a different illuminant, values
are chromatically adapted (Bradford) and the report warns — adaptation is a model,
not a measurement, and a D50 reference is always preferable.

---

## Clipping: two different faults that look identical in the report

A `clipped_high` or `clipped_low` flag means pixels in that patch sat at the very
top or bottom of the encoding range. The measurement is a ceiling or a floor, not
the patch's real color. But there are **two quite different causes**, with
different remedies, and telling them apart is the whole game.

### Exposure clipping

The capture is too bright or too dark, so signal runs off the end of the range.

**How to recognize it:** the affected patches are the extremes of the tonal scale —
the lightest few or the darkest few. All three channels clip together, or nearly
so. `clipped_high` on near-whites means over-exposure; `clipped_low` on near-blacks
means under-exposure or a black point set too high.

**The fix is exposure.** Adjust until the lightest neutral sits below the top of
the range and the darkest sits above the bottom. FADGI's own guidance for film is a
useful rule of thumb: on an 8-bit scale, keep values above about 5 in the shadows
and below about 250 in the highlights, so nothing is resting on a limit.

### Gamut clipping

The color is real and correctly exposed, but it does not **fit inside the working
space** you encoded into. To reproduce it, one channel would need a negative
amount of a primary — and negative light does not exist, so it pins at zero.

**How to recognize it:** the affected patches are *saturated*, not dark. One
channel clips while the others sit at healthy values. The patch's reference L\* can
be perfectly respectable — a vivid yellow at L\*76 clipping hard in blue is gamut
clipping, not exposure.

**Exposure will not fix this.** Changing lighting or aperture moves everything
together; it cannot bring a color inside a container it does not fit. The fix is a
**wider working space**.

### Reading your own report

**The report now tells you which it is.** A patch whose channel sits on the floor
*while another channel is bright* is flagged `gamut_clipped(B)` — with the channel
named — alongside the `clipped_low` that always accompanies it. The color metrics
group their flagged patches by kind and give the remedy for each, so you do not have
to work it out:

```
19 gamut_clipped -- a channel is held at the encoding floor while the patch is
    bright in another -- the colour is more saturated than the capture's working
    space can encode... Not an exposure fault: re-shoot in a wider space.
2 clipped_high -- pixels at the top of the encoding range... Reduce exposure.
```

The test is on the **brightest** other channel, not on all of them, and that
matters: a saturated red has green at zero *and* blue low, so asking for every other
channel to be bright would file it under exposure. What separates the two cases is
whether anything in the patch is bright at all. A grayscale capture never gets the
flag — one color record has nothing to be out of gamut against. Nor does a capture
still in a camera or scanner profile: its encoding *is* the device's own, with nothing
wider to move to, so the same pattern there is flagged `channel_floor` and means the
channel received almost no light.

If you want to check the reasoning yourself, the per-patch flags carry the clipped
fraction and the run record (`--save-run`) holds every patch's reference Lab:

| What you see | Likely cause |
|---|---|
| Clipping only on the lightest / darkest patches, all channels | Exposure |
| Clipping on saturated patches at mid or high L\*, one channel only | Gamut |
| Green channel on the floor, reference a\* strongly positive | Saturated reds exceed the space |
| Blue channel on the floor, reference b\* strongly positive | Saturated yellows exceed the space |

A worked example from a real scan of an ISO 12641-2 film target encoded in
**eciRGB v2**. **127 of its 864 patches carry a clipping flag**, and 120 of those are
`gamut_clipped` — the engine's own classification, and it agrees with the manual one
below. Not one of the 127 is underexposed: their mean levels reach 0.60 and their
median measured chroma is C\*ab 57 against 23 for the other 737.

Which channel is on the floor depends on how you count, and both counts are worth
having. At the flag's own threshold — more than 0.1% of an ROI's pixels — red is
pinned on 33 patches, green on 48 and blue on 46. Counting only patches **more than
half** clipped gives 61 patches: red **never**, green 36, blue 25. That second view is
the one that matches the gamut arithmetic, and 51 of those 61 have a reference L\*
above 20 — they were not shadows.

Converting every reference value into the working space settled it: **67 of the
864 patches fall outside eciRGB v2's gamut** (44 needing negative green, 21
negative blue), which matches the observed clipping closely. In **ProPhoto RGB,
zero patches fall outside**. No amount of exposure work would have helped; the
container was the problem.

### Choosing a working space

Wider is safer for capture, at a cost. Roughly:

- **sRGB** — too small for this work. FADGI drops it above 2-star for prints and
  photographs for exactly this reason.
- **Adobe RGB (1998)** — comfortable for most reflective originals; can clip
  saturated cyans and yellows.
- **eciRGB v2** — a good general archival space, but as above it does not contain
  everything a saturated transparency can produce.
- **ProPhoto RGB** — very large; holds essentially anything a reflective or
  transmissive original can produce. Because it is so large, **use it at 16 bits**:
  spreading a huge gamut over 8 bits puts visible banding in smooth gradients.

FADGI permits Adobe RGB (1998), ProPhoto and ECIRGB_v2 at 3- and 4-star for prints
and photographs. If your material includes saturated transparencies, ProPhoto at 16
bits is the safe choice.

One caveat worth knowing: a target's reference file describes the *chart*, not your
originals. A chart deliberately includes colors near the edge of what film or ink
can do, so some gamut clipping on a target does not automatically mean your actual
collection is clipping. It does tell you where your pipeline's limits are.

## Gain modulation: is the curve the right shape?

Tone response asks whether each gray patch landed at the right lightness. Gain
modulation asks a different question: **is the curve moving at the right rate?**

The distinction matters because a capture can pass tone response at every patch
while the curve flattens somewhere along the scale. Where it flattens, tones that
differ in the original come back nearly identical — detail is gone, and no amount
of later adjustment brings it back. ISO/TS 19264-1 puts it directly: *"even with
the tone reproduction being within tolerances the slope is too flat to differentiate
tonal values that are close to each other."*

**Gain** is the local slope of the tone curve:

```
gain = (measured L*ᵢ − measured L*ⱼ) ÷ (reference L*ᵢ − reference L*ⱼ)
```

The aim is **1.0** — the capture separates two tones by the same amount the original
does. ISO specifies the pairing: patches at least **10 L\* apart**, so the gain is
local without being dominated by noise on two adjacent steps. Deltachrome pairs
each patch with the nearest one at least 10 L\* away, which generalises ISO's
"compare patch 1 with 3, 2 with 4" example to ramps that are not evenly spaced.

| Gain | Meaning | Consequence |
|---|---|---|
| ≈ 1.0 | Correct slope | Tonal separation preserved |
| < 1.0 | Too flat | Tones compress together; detail lost in that range |
| > 1.0 | Too steep | Differences exaggerated; less headroom, clips sooner |

Because it fails in **both** directions, the criterion is a band, not a ceiling.
FADGI splits it in two, tightening with star level, and uses the pair's average L\*
to decide which band applies:

| | Highlight pairs (avg L\* 85–95) | All other pairs |
|---|---|---|
| 3-star | 0.7 – 1.2 | 0.6 – 1.4 |
| 4-star | 0.8 – 1.1 | 0.7 – 1.3 |

The reported value is the gain **furthest from 1.0**, not the average. An average
would hide exactly the fault the metric exists to catch: a single flat stretch in
an otherwise well-behaved curve.

If your target's neutral scale does not reach into the 85–95 band, the highlight
metric reports INCOMPLETE rather than a number — that is a property of the target,
not a failure of the capture.

## Reference files

### If your file holds only spectra

Many do, and it is the better kind of file rather than a lesser one. A
spectrophotometer records a **spectrum** — a reflectance or transmittance at every
few nanometres — and Lab is an integration someone's software performed on it
afterwards, under an illuminant and observer they chose and may not have written
down. Some vendors ship files that carry spectra and nothing else, which is the
purest form of the measurement and the one a reader should be able to use.

Deltachrome integrates them itself, under the illuminant and observer **the file
declares**, and says so in the warnings. Where a file states Lab as well, the stated
Lab is used: a reader that silently disagreed with a number the file asserts would
be worse than one that could not read spectra.

CxF files are read whether they spell their elements `Color` or `Colour`. ISO
17972-1 says the first and at least one vendor ships the second, and a file spelled
the other way is otherwise perfectly good — its Lab, its XYZ and its measurement
conditions are all there under names a strict reader walks straight past.

It refuses rather than guessing on two things. A file naming an illuminant it does
not have a table for is not integrated under D50 anyway — a value integrated under
the wrong illuminant is wrong by several ΔE and looks entirely reasonable. And a
handful of bands is not a spectrum; interpolating four of them onto a 5 nm grid
would invent structure the instrument never measured.

### If your file is an OpenDICE profile

The target profiles OpenDICE reads, and FADGI's website distributes, can be given
to Deltachrome as they are. They are tab-separated tables headed `L a b D`,
`L* a* b* D` or `D`. Deltachrome recognises them by their contents, whatever the
file is named.

Such a file states only values, not how they were measured. Deltachrome reads them
as D50 and 2°, which is FADGI's convention and OpenDICE's assumption. A step listed
by density alone, as on the film targets, takes its L\* from that density with the
lightest step at L\* 96. Deltachrome's own film values use the same rule.

Where a step lists both, its L\* is used. The density stays in the record as
density. That is one place the two tools part company; see
[Tone response: different aims](#tone-response-different-aims).

Two charts' profiles name their patches in a way that names alone cannot settle.
The UTT's calls a color patch and a grey step both `B1`, and the film charts' number
two ladders 1 to 24 straight through. Those targets declare the order OpenDICE
lists their patches in. A file that does not match that order exactly is refused
rather than matched by guesswork.

### If your file holds more than one table

Most do not, and if yours does not you will never see this. A CGATS file can hold
several data tables, and the commonest reason is an instrument writing the same
chart measured under several conditions — M0 and M1 — into one export.

Deltachrome uses the **first table** and says so whenever there is more than one,
listing what the others were. If the tables differ in measurement condition it says
something stronger, because then the choice changes every ΔE in your report rather
than being a formatting detail.

To name the one you want:

```bash
deltachrome analyze capture.tif --target colorchecker_sg_v1 --reference ref.txt --reference-table M1 --profile fadgi_3star_prints_photographs_2023 --corners "..."
```

`--reference-table` takes a number (`2`) or a measurement condition (`M1`). If what
you name is not in the file, the run stops rather than falling back to the first —
otherwise you would get a different measurement from the one you asked for, in a
report showing the number you named.


### Why the file has to be your chart's

A **per-unit** file measures the chart in front of you. Anything else — a
manufacturer's aim values, a batch average, a file for another unit — measures a
chart *like* yours, and the difference lands in your report as though it were your
scanner's error.

Two numbers say how much that matters, both measured from published measurements
of real charts:

* Two units **from one production batch** differ by **0.2 to 0.9 ΔE2000** across
  their patches. That is the irreducible part, and it is small.
* Two units of the same model **from different batches** differ by far more:
  **7.1 ΔE2000** on the ISA FADGI 19264, about **8** on the DT Nexus Transmissive
  Validation target, **13.7** on the Profiling target.

So a batch average is only as good as the batch it came from, and a run against
one carries an error that can be several times your system's own. Where a target
ships aim values Deltachrome will use them rather than refuse — the ISA Golden
Thread charts would otherwise be unusable — but the run reports `ADVISORY` and
makes no conformance claim, and the report says so in the header.

### Your chart may be able to fetch its own file

Increasingly a chart carries a **QR code or Data Matrix that is a web address**,
under the OpenQualia label standard — Digital Transitions, Image Science
Associates, ImageZebra, Gray Sky Imaging and Munsell at RIT are behind it. The
code says where that individual unit's measurement is published.

Deltachrome reads it while it is working out which chart is in the frame, so by
the time you see the placement screen it already knows. If the measurement is not
in your bank, **Fetch Measurement** appears beside the reference options: press
it, and the file is downloaded, filed, and bound to that unit so every future
capture of the chart pairs itself.

Nothing is downloaded unless you press it. Reading a code is not permission to
tell a third party what you are shooting.

**If your chart prints a serial but no label** — every Golden Thread made before
2025, and there are over three thousand of those published — Settings ▸ Reference
Files ▸ ↓ does the same thing from a serial you type. Pick the chart by name; you
never type a manufacturer or a target type, because the spellings are not
guessable and Deltachrome holds them.

Not every unit is published. If yours is not, the refusal says so and you add the
measurement that came with the chart by hand.

### One warning worth stopping for

If the host returns a file that **states a different serial** than the unit you
asked for, Deltachrome says so and keeps the note with the file. This is rare and
real: one published measurement, asked for under one serial, states another inside
— and the two are genuinely different measurements. Pairing still works, because
the file is filed under the unit you asked for. But if the chart in front of you is
not that unit, every colour number in the report is wrong in a way nothing else
will reveal. Check the serial printed on the chart against both.

Check a file before relying on it:

```bash
deltachrome inspect-reference YOUR_FILE.txt --target colorchecker_sg_v1
```

**Duplicate patches are refused, not merged.** If a `SAMPLE_NAME` appears twice
with different values, Deltachrome stops and shows you the conflicting fields
rather than silently keeping one. This is a real failure mode: a reference file for
a ColorChecker SG was found listing K8 twice, the second entry being a mislabelled
K9. A parser that kept the last value would have corrupted every result involving
that patch, invisibly.

---

## Warnings worth acting on

**"Chart geometry is provisional"** — patch spacing comes from published chart
dimensions rather than a measurement of a physical chart. No shipped chart carries
it now; it is there for a definition you write yourself. Every ROI stays on
the chart, but they may not be perfectly centerd on the patches. Run with
`--dump-overlay overlay.png` and look at it before trusting a marginal result.

**"N patches too small to measure reliably"** — each sampled fewer than 100 pixels.
Your capture resolution is low relative to the chart, or the placement is wrong.
Per-patch noise will dominate.

**"Placement doesn't match the chart's shape"** — the placed outline's proportions
differ from those of what the chart's corner marks bound — the patch strip on an
Object Level Target, the crosshairs' square on a QA-62 — by more than 8%. A correct
placement sits within about 1%, and one placed a quarter turn out 40% or more.
Almost always the placement points were set in the wrong order or on the wrong
printed features; check the placement before trusting anything else on the report.

**How the chart was placed and named** is no longer a warning. The run record
says it in `geometry.placed` — `by_hand`, `automatic`, or `automatic_then_moved`
when a found chart's corners were then dragged — and `geometry.identified`, which
says whether Deltachrome worked out which chart it was rather than being told. The
console report prints both as one **Placed** line under the target. They are there
so the record says where the geometry came from, which a reader a year later has
no other way to find out.

There is one thing worth checking on any placement, automatic or your own: that
the sampled regions are where they belong. The placement screen draws them for you
— patch regions on the patches, slanted-edge regions across the edges — and
**Hide Sampling** turns them off; `--dump-overlay overlay.png` writes the same
picture from the command line. A placement that is wrong does not announce itself
— it samples 140 regions from slightly the wrong places and returns 140 plausible
numbers — which is why automatic placement refuses rather than guessing, and why
the one glance is worth taking anyway.

**`gamut_clipped(<channel>)` flag** — that channel sits on the encoding floor while
the patch is bright in another: the color is more saturated than your working space
can hold, so its value is truncated and its chroma understated. Exposure will not fix
it; a wider space will. Always accompanied by `clipped_low`, which it refines.

**`channel_floor(<channel>)` flag** — the same pattern in a capture still encoded in
the camera's or scanner's own profile rather than a working space. There is no wider
space to move to: the encoding is the device's own, so the channel is at the floor
because it received almost no light. More exposure lifts it, if the brightest channel
has room.

**`clipped_high` / `clipped_low` flags** — pixels in that patch hit the top or bottom
of the encoding range. The measurement is a floor or ceiling, not the patch's real
color. Adjust exposure and re-shoot.

**`mean_median_divergence`** — the average and median of that ROI disagree, which
usually means dust, a scratch, or a specular highlight in the sampled area. The
patch is still included; check it before accepting a result that hinges on it.

---

## Comparing with OpenDICE

OpenDICE is the Library of Congress's open FADGI analysis tool, and the one you are
most likely to check Deltachrome against. Both tools were run on the same eight
captures with the same reference values. Anything read straight off the pixels
agreed, usually to the second decimal:

| | Typical agreement |
|---|---|
| ΔE2000 for each patch | 0.01–0.06 |
| Measured L\* for each grey step | 0.01–0.17 |
| Noise for each grey step | 0.01–0.10 |
| Mean and 90th-percentile ΔE2000 | within 0.1 |
| White balance for each grey step, ΔE2000(a\*b\*) | within 0.06 on average |
| Maximum SFR amplitude | identical |
| MTF50, with linearization matched | within 1 point of the field mean |

When the two reports disagree, it is almost always for one of the reasons below.
Check these before concluding that either tool is wrong.

### Sharpness: Deltachrome linearizes first

MTF50 and the response at Nyquist read up to two points higher here, because
Deltachrome converts each edge through the capture's ICC profile before analysing
it, and OpenDICE's sharpness analysis works on the encoded values. See
[Comparing an MTF50 with another tool's](#comparing-an-mtf50-with-another-tools).
Sampling efficiency agrees only loosely between the tools. It is read where the
curve has fallen to 10%, which is its noisiest point.

### Tone response: different aims

OpenDICE does not use the L\* values in your reference file. It reads the density
column (`D`) and converts each density back to L\*, then judges each step against a
band of ±1.5 L\* (at 4 stars). Deltachrome judges each step as ΔL\*2000 against the
L\* the reference file states, as FADGI's tables specify.

The density column is usually just the same measurement written another way: the
visual density is −log₁₀(Y), so it carries the same information as L\*. But it is
usually written to two decimal places, and near white a difference of 0.01 in
density is about 0.9 L\*. Some files' densities also disagree with their own L\*
outright; one ColorChecker SG file states L\* 87.78 for a step whose density
converts to 91.43. **So the two tools can measure identical L\* and still reach
opposite verdicts on tone response.** If they disagree, compare the measured L\*
values first. If those match, the difference is in the aims, not in your capture.

### Lightness uniformity: different quantities

FADGI states uniformity as the standard deviation divided by the mean L\*, and that
is what Deltachrome reports. OpenDICE reports (maximum − minimum) / mean of the raw
pixel values at five locations. The two are not interchangeable, and they can fall
on opposite sides of a 1% threshold.

### Sampling rate

Deltachrome reads the sampling rate from the rules printed on the chart.
OpenDICE's figures have differed from it by 0.4–0.6%. On a chart printing several
rules, the report says how closely they agree with each other. That spread is how
far Deltachrome's own figure can be trusted.

### When a sampling area leaves its patch

A sampling area that falls partly off its patch shows as high noise on a patch
whose L\* has moved toward its neighbour's. It is likeliest toward the far columns
of a large butted array, at the ends of a long grey scale, and on small captures
of a strip target. Deltachrome draws every sampled region on the capture, so check
where the regions sit in each tool before trusting either number.

---

## Re-scoring without re-shooting

`--save-run results.json` writes every per-patch measurement, not just the
aggregates, along with the ICC profile used, the reference file's provenance, the
ROI inset, and the percentile method.

That means a capture can be re-scored against a different or revised profile later,
and captures can be tracked over time for a given device, without going back and
re-shooting the target.
