"""The standing explanation. **Nothing imports this, and that is the point.**

Every string here is the same on every run. That was once the test for putting a
sentence behind a ⓘ rather than on the screen, and it is now the test for it
being documentation rather than interface: the application ships with no
explanation panel at all.

The reason is not that the prose was wrong. It is that a screen which explains
itself cannot be shown to be unclear. An operator meeting a control they do not
understand opens the panel, the confusion is resolved in eight seconds, and
nobody -- least of all the person who built it -- learns that the control needed
explaining. A beta run without any of it says which parts of this application a
person in the trade can read and which parts they cannot, and that answer is
worth more than the eight seconds.

So these are kept, verbatim, as the source for user documentation written apart
from the application. They are not dead text to be deleted at the next tidy-up;
they are the only place several of these arguments are written down, and
:mod:`tools.export_copy` will not find them once nothing shows them.

What is *not* here, and never was, is the other kind of sentence: how many
readings carried a position, which ones were flagged, that this chart has no
slanted edges, that a run was judged against nominal values. Those are findings
rather than explanations of findings, and they are still on screen.

Each entry is ``(title, body)``: the title is the heading a documentation page
would carry, the body its text.
"""

from __future__ import annotations

Explanation = tuple[str, str]

METRIC_NOTES: Explanation = (
    "Reading a Metric",
    "The number beside the name is what this run measured; the badge is how it "
    "was judged. What it was judged *against* is drawn on the chart below as a "
    "band — the range the guidelines ask the reading to stay inside — so how "
    "much room is left is a glance rather than a subtraction.\n\n"
    "A metric that carries notes of its own shows them here instead of this. "
    "What never moves behind this icon is why a run could not finish a "
    "measurement: that is part of the answer, and it stays on the page.",
)

PATCH_TABLE: Explanation = (
    "Reading the Per-Patch Values",
    "Sorted worst first, against the condition the guidelines set.\n\n"
    "The shape of the distribution is the finding. A result driven by one or two "
    "named patches is usually dust or a specular reflection — a target that wants "
    "cleaning and a re-shoot. One spread evenly across all of them is usually a "
    "cast, which is a system out of adjustment. The aggregate cannot tell those "
    "apart, and they need entirely different responses. That is the reason the "
    "engine persists per-patch values at all.\n\n"
    "A flagged patch was clipped or contaminated at the moment of sampling, so "
    "its value means less than the others' whatever the metric made of it.",
)

EDGE_FIELD: Explanation = (
    "Reading the Edge Field",
    "Every slanted edge is measured and kept individually, whatever the verdict.\n\n"
    "The aggregate cannot show that one corner is soft, or that horizontals "
    "differ from verticals, and those differences are what point at astigmatism, "
    "field curvature or a tilted focal plane. A mean can pass a system with one "
    "badly soft corner; the spread is what shows it.\n\n"
    "Point at a mark or a curve to identify it in both plots at once. Click to "
    "pin it and compare the rest against it, and click empty space to let go. "
    "The keys hide an orientation: with the verticals off, softness that stays "
    "on one side of the field is a tilted plane, where softness that follows "
    "orientation everywhere is astigmatism.",
)

MTF_CURVES: Explanation = (
    "Reading the MTF Curves",
    "One curve per edge, colored by orientation.\n\n"
    "An outlier is visible as an outlier here; in an average it is only a moved "
    "mean. That is the argument for plotting every edge rather than the "
    "aggregate the verdict is formed on.\n\n"
    "Past the dashed line the system is sampling below the detail it is being "
    "shown — a line pair needs two samples, so nothing sharper can be resolved "
    "at this rate however good the optics. A curve that rises again beyond it is "
    "aliasing rather than resolution.",
)

CHANNEL_CURVES: Explanation = (
    "The Same Edge in Each Color Record",
    "Every edge is additionally measured in all three color records on every "
    "run, and the curves kept. Nothing here can move a verdict: every metric is "
    "formed on the luminance measurement.\n\n"
    "This is how lateral chromatic aberration reads. A lens that focuses red and "
    "blue at different distances from the axis resolves them differently, and "
    "the difference grows towards the corners — which one aggregate curve cannot "
    "show and three can. It is also the other half of channel registration: that "
    "measures how far apart the records sit, this measures whether they are "
    "equally sharp, and a system can fail either alone.",
)

TONE_CURVE: Explanation = (
    "Reading the Tone Response Curve",
    "The dashed line is the aim. Sitting below it means the reproduction is "
    "darker than the original at that lightness, above it means lighter. Point "
    "at a step to name the patch it came from.\n\n"
    "The shape names the fault. A bow across the middle with the ends still on "
    "the line is a gamma error. Ends that flatten onto a horizontal run are "
    "shadows or highlights being clipped, and no amount of tone correction "
    "recovers those — the values are simply not in the file.\n\n"
    "Where a departure sits therefore matters as much as how large it is, and "
    "the metric reports the worst step without saying which.",
)

TONE_RESIDUAL: Explanation = (
    "Reading the Residual",
    "The same measurements with the aim subtracted, which is the only way the "
    "small departures are visible at all beside a hundred-point axis.\n\n"
    "This is the signed L* difference, so its sign tells you which direction the "
    "tone moved. It is deliberately not the quantity the profile judges: tone "
    "response is scored on ΔL*2000, which is unsigned and perceptually "
    "weighted.\n\n"
    "So no threshold line is drawn here. A plot that draws a limit implies a "
    "judgement, and this plot is not showing the quantity that was judged — the "
    "metric's own tile states the condition it actually met or missed.",
)

CHART_SWATCHES: Explanation = (
    "Reading the Chart",
    "Each patch is drawn where it physically sits, split: what the camera "
    "recorded on one side, what the target is supposed to be on the other. A seam "
    "that disappears is a patch that matched, so the eye finds the ones that did "
    "not without reading a single figure.\n\n"
    "A table of differences answers how far off; it never answers in which "
    "direction, and direction is what separates a warm cast from a green one, or "
    "one contaminated patch from a target photographed under the wrong light.\n\n"
    "Patches are drawn at their full size. The area actually measured is an "
    "inset fraction of each one, which the saved run record states.\n\n"
    "A dashed outline marks a patch carrying a sampling flag. Nothing is "
    "colored by whether it passes, for the same reason the field maps refuse "
    "it: these thresholds are conditions on an aggregate.\n\n"
    "The conversion to screen color is for looking at only. It is not "
    "color-managed to your monitor, and no number in the report comes from it.",
)

CAPTURE_DOCK: Explanation = (
    "The Capture Beside the Numbers",
    "The frame with every sampled region drawn on it, the size of each mark "
    "following its reading, and the focused region at the resolution this panel "
    "can show, padded so its border is visible.\n\n"
    "Position is not decoration, and this is where it is answered. A mean says "
    "the illumination is uneven; the frame says the left side is dark, and those "
    "have different causes and different fixes. Drawing it here rather than as a "
    "diagram of the target means the marks sit on the pixels they were measured "
    "from.\n\n"
    "The size range spans the middle 90% of the readings, so one contaminated "
    "patch cannot flatten the rest into identical marks; anything beyond that "
    "sits at the end of the ramp.\n\n"
    "Nothing is colored by whether it passes. The thresholds these metrics carry "
    "are conditions on the aggregate \u2014 a mean, a 90th percentile, a worst "
    "case \u2014 and coloring each mark against one would show a per-reading "
    "verdict the guidelines never formed. What does mark a reading is a sampling "
    "flag, which is a fact about how it was measured rather than a judgement of "
    "it.\n\n"
    "This is the gap between “edge 7 is soft” and “edge 7 sits on a dust speck”, "
    "or between “patch K3 reads 8.4” and “K3 has a fingerprint on it”. Those "
    "need different responses — one is a system out of specification, the other "
    "is a target that wants cleaning — and no number distinguishes them. The "
    "pixels do, immediately.\n\n"
    "How much of the frame the target fills is itself a finding: it is the "
    "sampling rate the capture achieves at the target plane.\n\n"
    "Color-managed: device values go through the master's own embedded ICC "
    "profile to sRGB, so a linear-light master no longer looks dark here. Still "
    "not a rendering to your own monitor — sRGB is assumed as its display "
    "space — and no number in the report comes from what is drawn.",
)


PLACEMENT_TOOLS: Explanation = (
    "The Placement Tools",
    "Fit, 1:1 and Reset move the view — except Reset, which puts the four "
    "handles back on the starting rectangle.\n\n"
    "Refine Placement snaps the corners onto the target's own printed features. "
    "A placement can look right and still be far enough out to stop a printed "
    "scale being readable, which silently costs a metric: on the Golden Thread "
    "sample a 20 px error is the difference between six printed rules being "
    "readable and none, and geometric distortion needs four. It is offered only "
    "where the target prints features it can use, and it refuses rather than "
    "guessing.\n\n"
    "Hide Sampling turns off the outlines showing the regions the sampler will "
    "actually read. They are drawn by default, because a placement can be "
    "entirely plausible and still have every region straddling two patches, and "
    "that is invisible until the regions are on screen. A region drawn "
    "differently was clipped or contaminated where it sits — the report says "
    "which and how badly, with the reading beside it.\n\n"
    "Copy Command puts the whole deltachrome analyze invocation on the clipboard "
    "— capture, target, guidelines, reference and this placement — so the run "
    "can be repeated, scripted, or quoted in a report.\n\n"
    "A frame may hold more than one target — a QA-62 is a small card shot four "
    "to a board, and a device-level target beside a color one is an ordinary "
    "thing to shoot. Each is placed, checked and run on its own, and the "
    "selector in the bar steps through them.\n\n"
    "Add a Target places another by hand, starting from a rectangle in the "
    "middle of the frame that you drag onto the target. It exists because the "
    "search does not find everything: two targets sitting closer together than "
    "it can separate come back as one, and it says so rather than reporting the "
    "second as absent. A target added this way is recorded as placed by hand, "
    "which is what the archive needs to say about it a year later.\n\n"
    "Ignore This Target leaves one out of the run without removing it from the "
    "list — for a region that is not a target at all, or one you simply do not "
    "want measured this time. It is a toggle, so it is also the undo: an "
    "ignored target stays on screen, greyed in the selector, and comes back "
    "with one press. A target you added yourself is removed outright instead, "
    "because there is no detection underneath it to preserve.",
)


PLACEMENT: Explanation = (
    "Placing the Target",
    "Drag each handle onto the matching corner of the target. The loupe shows the "
    "pixels underneath at full resolution.\n\n"
    "TL, TR, BR, BL are the target's corners, not the image's — clockwise from "
    "the one nearest the first patch. A target photographed rotated therefore "
    "needs no special handling, because the homography absorbs it.\n\n"
    "Which feature counts as a corner is the target's business and differs "
    "between them: the ColorChecker SG uses the centers of four printed "
    "crosshairs, and the ISA Golden Thread uses marks inside its corner modules, "
    "nowhere near the corners of the card. Where a target says, its instruction "
    "is shown on this panel.\n\n"
    "Placement is a measurement input, not a convenience. On the Golden Thread "
    "sample a 20 px error — two tenths of one percent of the frame — is the "
    "difference between six printed rules being readable and none, and geometric "
    "distortion needs four.\n\n"
    "What the canvas shows goes through the master's own embedded ICC profile "
    "to sRGB, so a linear-light master no longer looks dark here. sRGB is "
    "assumed as your display's own space rather than measured, which is the "
    "one thing still not accounted for. Nothing on this canvas is a "
    "measurement either way — every value in the report is taken from the "
    "file through the embedded profile at full float precision, independently "
    "of what is drawn.",
)


AUTOMATIC_PLACEMENT: Explanation = (
    "Finding the Target Automatically",
    "The target is located by whatever it says a placement is made on — printed "
    "crosshairs, a keyline, four discs, the array of patches itself — and the "
    "four corners are then snapped onto its own printed features, exactly as "
    "the Refine button does by hand.\n\n"
    "It does not have to be accurate to be useful, and that is why it works. "
    "Refinement recovers a placement from errors of tens of pixels, so what the "
    "search has to deliver is a target found within a few percent of its own "
    "size, not a target found exactly. Measured against every target in this "
    "repository, it lands within two thirds of one percent and usually under a "
    "pixel.\n\n"
    "It refuses rather than guessing. A wrong placement is worse than none, "
    "because it samples every region from the wrong pixels and returns a full set "
    "of plausible numbers — so where the target cannot be found with confidence, "
    "the corners are left where they were and the reason is shown here.\n\n"
    "Check it before running. The result is a placement you did not make, and the "
    "one thing worth a glance is that the handles are on the features the panel "
    "names and the sampled regions land on patches.",
)


IDENTIFYING: Explanation = (
    "Working Out Which Target This Is",
    "Each target says what a placement is made on \u2014 printed crosshairs, a "
    "keyline, four discs, its own array of patches \u2014 so the capture is searched "
    "for each of those in turn, and most candidates rule themselves out by simply "
    "not being found.\n\n"
    "What ranks the ones that survive is the target itself rather than its "
    "outline: "
    "the proportions the target's patch grid implies against the proportions "
    "actually placed, whether that target's neutral ramp reads light to dark in the "
    "order it is declared in, how much of its array landed on patches, and \u2014 "
    "where the target ships aim values \u2014 how far its colors sit from them.\n\n"
    "It names a target only when one fits, never as the best of a bad field. Where "
    "two fit equally it names both and leaves the choice, which is the honest answer "
    "rather than a coin toss: two targets can share a layout exactly and differ only "
    "in their colors, and the two IT8 targets here do.\n\n"
    "Whatever it says, the choice is yours to change \u2014 and changing it here "
    "changes everything downstream, including the placement.",
)
