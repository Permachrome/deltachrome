"""Snapping a hand placement onto the features the chart actually prints.

Placement is manual (automatic detection is roadmap phase 4), and the cost of
being slightly wrong is not what it looks like. On the Golden Thread capture in
this repository a **20 pixel** error on an 11600 x 8700 master -- 0.2% of the
frame -- moves the number of printed rules that can be read from 6 to 0, and
geometric distortion needs at least four. So a placement can look right, pass the
aspect-ratio check, and produce perfectly good color and SFR figures across
eighteen metrics while silently disabling a nineteenth.

That is the failure this project cares most about: not a wrong number, but a
missing measurement nobody noticed.

**The method reuses what the engine already does.** Slanted-edge regions declare
their geometry in target-relative coordinates, and
:func:`deltachrome.imaging.edges._refine_square` already snaps each declared
square onto the square actually in the capture -- that is how edges are located
at all. Each successful snap is therefore a correspondence between a *known
relative* position and a *measured pixel* position.

**A ``marker`` region counts too, and that is not a stretch.** Refinement does not
need a feature to be *measurable*, only *locatable*, and a marker is by definition a
feature the chart carries that no metric reads. The Golden Thread is the case that
made the point: each of its five modules prints a white "dots per inch" field inside
a dark mount, and the mount alone gave five features where both give ten -- which is
the difference between a similarity and a homography, and on that chart between
recovering a misplacement to 15 px and to 0.12.

Using the fiducial's own feature is not circular. The operator places four corners on
those white fields *by eye*; refinement locates the same fields *by algorithm* and
corrects the difference. That is automating the placement the operator approximated,
which is what this is for.

**Only the centers of those squares are used.** That restriction was forced by the
data. On the Golden Thread the declared half-sizes are 4-6% larger than the printed
squares and inconsistent between modules -- five modules declare sides of 790 to 809
px where every measured side is 760.8, agreeing to 0.6 px. Corner-to-corner
correspondence therefore asks the fit to reconcile sizes that do not match, and a
transform with enough freedom will do it by distorting the whole chart. Centers
carry the error that matters and none of the error that does not.

**Which transform is fitted depends on how many features were found**, and the
rule is that a model may only be used when the features *over-determine* it: at
least as many features as the model has degrees of freedom, so at least twice as
many equations as unknowns. That single rule reproduces two findings that look
contradictory, without being tuned to either.

On the Golden Thread, five modules. A homography has eight degrees of freedom, so
five features -- ten equations -- barely determine it, and the spare freedom goes
into fitting the *declaration* errors rather than the placement: measured on that
capture, centers with a similarity took readable rules from 2 of 20 to 10 of 20,
against 1 of 20 for the homography. So five features get a **similarity**: shift,
rotation, uniform scale. The four hand-placed corners already establish the chart's
perspective, what is left over is a small rigid misplacement, and four degrees of
freedom cannot absorb a mislocated feature by distorting the frame.

On the ISA transmissive targets, ten -- each of five apex-to-apex pairs contributes
both of its quarter discs. Twenty equations against eight unknowns, and the declared
positions are good to a fraction of a pixel. There a **homography** is the complete
description of where a planar chart sits, and it is not close: across three captures
and five kinds of misplacement it recovered every corner to within 0.44 px, while
the similarity left up to 22 px whenever the four corners were placed
*independently* wrong -- which is a perspective error, exactly what a similarity
cannot express and what a hand placement actually produces.

The UTT A3, with nine, promotes too, and it is the case that tested the rule rather
than illustrated it: nine features against eight degrees of freedom is the tightest
promotion the rule allows. Measured, the homography is never meaningfully worse and
is often far better -- 3.7 px against 25.0 for the similarity when the four corners
are placed independently wrong, and a dead heat on an exact placement. Both are
floored at about 3.7 px there by the *declared* module positions, which agree with
each other to only 2 px RMS, and no fit can do better than the numbers it is
given.

An affine fit sits between them and is used for neither. It has enough freedom to
distort the frame and not enough to describe a projection, and it measured worse
than *both* neighbours on one of the captures -- 6.1 px on an exact placement where
the similarity gave 0.35 and the homography 0.44.

**It is never applied silently.** The result carries how far each corner moved
and how many features voted, and it refuses rather than guessing when too few
features were found or when the correction is too large to be a correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from deltachrome.imaging.edges import (
    _refine_quarter_disc,
    _refine_square,
    _square_corners,
)
from deltachrome.imaging.geometry import ChartPlacement
from deltachrome.imaging.loader import LoadedImage
from deltachrome.targets.schema import TargetDefinition

#: Fewest located features a correction may be solved from. A similarity has four
#: degrees of freedom and two points fix it exactly, so three is the first count
#: that is over-determined -- the first at which the features can disagree with
#: each other, which is what makes the reported residual mean anything.
MIN_FEATURES = 3

#: Region kinds that can supply a correspondence.
#:
#: ``marker`` is here because refinement needs a feature to be locatable and not to
#: be measurable, and a marker is precisely a feature the chart carries that no
#: metric reads. Excluding it cost the Golden Thread half its evidence.
FEATURE_KINDS = frozenset({"slanted_edge", "marker"})

#: The transforms a correction may be fitted as, richest first, with the degrees of
#: freedom each costs.
#:
#: A model is admissible only when the located features over-determine it -- at least
#: as many features as it has degrees of freedom, so at least twice as many equations
#: as unknowns. See the module docstring for the two measurements that rule
#: reproduces. Affine is deliberately absent: six degrees of freedom is enough to
#: distort the frame and not enough to describe a projection, and it measured worse
#: than both its neighbours on one of the three captures it was tried on.
MODELS = (("homography", 8), ("similarity", 4))

#: Largest correction, as a fraction of the chart's shorter placed dimension,
#: that is credible as a correction rather than a misdetection. A hand placement
#: is out by pixels or tens of pixels; a snap onto the wrong feature entirely is
#: out by a fraction of the chart.
MAX_SHIFT_FRACTION = 0.05

#: Feature disagreement, as the same fraction, above which the fit is *reported* as
#: resting on a majority rather than on all of its evidence.
#:
#: A note and not a gate, and that was tested rather than assumed. **A gate does not
#: help.** Measured over 511 seeded refinements across all seven charts that carry
#: features -- corners perturbed by 0 to 40 px, which is the shape of error a hand
#: placement makes -- 464 were applied, and of those 93% IMPROVED the placement, by a
#: median of 15 px. Only 27 (5.8%) made it worse by more than a pixel, and those by a
#: median of 3.8 px.
#:
#: The reason a gate cannot pay is that refusing does not give you a good placement --
#: it gives you back the one you had. Of the 48 refinements that landed more than 5 px
#: from the truth, 25 were STILL an improvement on what they replaced (median 31.2 px
#: in, 28.6 px out). The single worst outcome in the whole set, 93.7 px on the UTT,
#: came from a placement that was 101.8 px out: a residual gate at half a percent of
#: the chart would have caught it and left the operator worse off.
#:
#: Swept as a policy, no threshold improves the final corner error. Against no gate at
#: all -- median 0.37 px, 90th percentile 6.07, worst 93.7 -- gating on the residual
#: leaves the median untouched and moves the 90th percentile to between 6.9 and 34.7
#: and the worst to 101.8. Gating on the fraction of declared features found
#: discriminates better (harmful fits find a median of 55% of them, harmless ones
#: 100%) and still does not pay: refusing below 70% catches 23 of the 27 harmful cases
#: but loses 16 good corrections, and the distribution comes out identical.
#:
#: The residual cannot separate the two even in principle, which is the older finding
#: restated with numbers: expressed as a fraction of the chart, good outcomes reach
#: 0.0088 and bad ones start at 0.0006, so the ranges overlap by more than a decade.
#: The documented case behind that is a chart where four of ten discs snapped onto the
#: wrong neighbour by 50 px: the residual rose from 0.15 px to 32 and the recovered
#: corners were still right to 0.21 px.
#:
#: What the residual IS good for is saying that something in the capture is not where
#: the definition says -- worth knowing, and not a reason to discard the fit.
#:
#: On a chart fitted by least squares (see :func:`_estimator`) the note carries more
#: weight, because least squares IS dragged by a bad feature. Measured on the Device
#: Level Target with features displaced 50 px by hand: one bad feature moves the
#: corners 15.9 px and lifts the residual to 12.9, two move them 28.3 and lift it to
#: 19.0 -- so the note fires in every case where the fit is actually harmed, and stays
#: silent at 0.17 px when nothing is wrong.
#:
#: CAVEAT ON THE MEASUREMENT: the seeds are independent random perturbations of the
#: four corners. That is what a hand placement does; a systematically wrong placement
#: -- a chart clicked one patch over, say -- is a different distribution and is not
#: covered here. It is also the case ``MAX_SHIFT_FRACTION`` already refuses.
NOISY_FEATURE_FRACTION = 0.005

#: Closest pair of declared features, as a fraction of the widest pair, below which
#: the features count as *clustered* and the fit is made by least squares instead of
#: LMEDS. See :func:`_estimator` for the whole argument and the measurements behind
#: this number; in short, LMEDS draws minimal four-point subsets and a subset holding
#: two near-coincident points is degenerate, which it cannot detect because it is
#: entitled to ignore half the data.
#:
#: Every shipped target reads 0.14 or above except the Golden Thread Device Level
#: Target, which reads 0.0014 -- so the value sits in two decades of empty space and
#: is not a tuned constant. It is a property of a DEFINITION, not of a capture, so a
#: chart always takes the same path and no run can silently switch estimators.
CLUSTERED_FEATURE_FRACTION = 0.02


@dataclass
class Refinement:
    """A proposed placement, and everything needed to judge it."""

    corners: list[tuple[float, float]]
    #: How far each corner moved, in pixels, in declared corner order.
    moved_px: list[float]
    #: How many printed features were located and agreed on the result.
    features_used: int
    #: RMS disagreement between the located features and the fitted transform.
    #: A small number means the features are consistent with one projection of
    #: the chart; a large one means something was mislocated.
    residual_px: float
    applied: bool
    #: Which transform the correction was fitted as. Chosen by how many features
    #: were found rather than by which chart it is -- see :data:`MODELS` -- and
    #: reported because a similarity and a homography are different claims about
    #: what was wrong with the placement.
    model: str = "similarity"
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def max_shift_px(self) -> float:
        return max(self.moved_px) if self.moved_px else 0.0


def _declared_features(region):
    """Every correspondence one region declares, as (label, centre, half, tilt, shape).

    Declared, not located: a feature here is one the *definition* states, and
    whether it can be found in a given capture is a question for the pixels.

    Split out from :func:`_region_features` because two callers read this branching
    and they must not drift -- the locating below, and :func:`locatable_features`,
    which answers "could this chart be refined at all" without touching a pixel.
    """
    geometry = region.geometry or {}
    tilt = float(geometry.get("tilt_deg", 0.0))

    discs = geometry.get("discs")
    if discs:
        for index, disc in enumerate(discs):
            yield (
                f"{region.region_id}_d{index}",
                np.asarray(disc["center"], dtype=np.float64),
                np.asarray(disc["half_size"], dtype=np.float64),
                float(disc.get("tilt_deg", tilt)),
                "disc",
            )
        return

    center, half = geometry.get("center"), geometry.get("half_size")
    if center is None or half is None:
        return
    yield (
        region.region_id,
        np.asarray(center, dtype=np.float64),
        np.asarray(half, dtype=np.float64),
        tilt,
        "square",
    )


def locatable_features(target: TargetDefinition) -> int:
    """How many correspondences ``target`` declares, without reading a pixel.

    An upper bound on ``Refinement.features_used``: every one of these still has to
    be found in the capture. That is exactly what makes it the right question for a
    *control* and the wrong one for a verdict -- whether refinement is possible is a
    property of the definition, where whether it succeeded is a property of the
    capture.
    """
    return sum(
        1
        for region in target.regions
        if region.kind in FEATURE_KINDS
        for _ in _declared_features(region)
    )


def refinable(
    target: TargetDefinition, *, min_features: int = MIN_FEATURES
) -> tuple[bool, str]:
    """Whether refinement could ever succeed on ``target``, and why not if it cannot.

    Here rather than in the frontend because the answer has to be the same rule the
    run-time refusal applies, in the same words. A gate the frontend computes for
    itself is a second statement of the rule, and the two had drifted in both
    directions: the frontend counted only rotated squares, so the three ISA
    transmissive charts -- ten disc features apiece, and the charts this module
    recovers every corner of to 0.44 px -- were shown a dead button saying they
    print no marks to snap to, while both Object Level Targets, which declare one
    feature and can never reach ``MIN_FEATURES``, were shown a live one that refuses
    on every press.
    """
    features = locatable_features(target)
    if features >= min_features:
        return True, ""
    if features == 0:
        return False, "No marks to refine on."
    mark = "mark" if features == 1 else "marks"
    return False, (
        f"Only {features} {mark} to refine on; {min_features} are needed."
    )


def _region_features(image, placement, region):
    """Every locatable feature in one slanted-edge region, as (label, declared, found).

    Two shapes of feature, and the difference is only in how each is found in the
    capture -- what comes back from both is the centre of a printed shape whose
    relative position the definition states, which is what the fit consumes.

    **A rotated square** is snapped by :func:`_refine_square`, which thresholds the
    feature out of its surround and takes the minimum-area rectangle.

    **A quarter-disc pair** contributes each disc separately, through
    :func:`_refine_quarter_disc`. That was left out when refinement was first built,
    on the grounds that a quarter disc needs its apex identified before its geometry
    means anything -- which is true of *extracting its edges* and irrelevant here.
    The bounding square's centre needs no apex at all, and the disc finder was
    already computing it: it fits a minimum-area rectangle and checks the mask fills
    pi/4 of it, which is a stronger identity check than a square gets, because
    nothing else on these charts has that fill ratio.

    Each disc votes on its own rather than each pair voting once. Five pairs give
    **ten** correspondences instead of five, which is the difference between a fit
    that is over-determined and one that is barely determined -- and the residual
    then reports whether the two discs of a pair agree with each other, which is the
    only signal available that one of them was mislocated.
    """
    for label, center, half, tilt, shape in _declared_features(region):
        declared_px = placement.to_pixel(_square_corners(center, half, tilt))
        if shape == "disc":
            found = _refine_quarter_disc(image, declared_px)
            if found is None:
                continue
            box, _apex = found
            yield label, center, np.asarray(box, dtype=np.float64).mean(axis=0)
            continue
        found_px = _refine_square(image, declared_px)
        # Declining is told from succeeding by object identity, not by value -- see
        # :func:`_feature_centers`.
        if found_px is declared_px:
            continue
        yield label, center, np.asarray(found_px, dtype=np.float64).mean(axis=0)


def _feature_centers(
    image: LoadedImage, placement: ChartPlacement, target: TargetDefinition
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Declared and measured centers, both in target-relative coordinates.

    Measured centers are pulled back through the current placement rather than
    left in pixels, so the correction being fitted is a correction *to the
    placement* and the chart's perspective -- which the four given corners
    already established -- is left alone.

    A region whose refinement *declined* is excluded rather than counted as
    agreement: ``_refine_square`` returns its input when it cannot identify the
    feature confidently, and letting those vote would mean uncertain regions
    endorsing the placement they were derived from.

    Declining is told from succeeding by **object identity, not by value**. Every
    refusal path in ``_refine_square`` returns the array it was given, while a
    success returns a new one -- and value equality cannot tell the two apart,
    because a correct placement makes a successful snap land exactly where it was
    declared. Comparing values instead found zero features on a perfect
    placement, which is the one case that must not need correcting.
    ``test_refine_placement`` pins that contract so a change to ``_refine_square``
    breaks loudly here rather than quietly disabling refinement.
    """
    inverse = np.linalg.inv(placement.matrix)
    declared: list[np.ndarray] = []
    measured: list[np.ndarray] = []
    used: list[str] = []

    for region in target.regions:
        if region.kind not in FEATURE_KINDS:
            continue
        for label, center, found_center in _region_features(
            image, placement, region
        ):
            homogeneous = inverse @ np.array([found_center[0], found_center[1], 1.0])
            declared.append(np.asarray(center, dtype=np.float64))
            measured.append(homogeneous[:2] / homogeneous[2])
            used.append(label)

    if not declared:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), used
    return (
        np.asarray(declared, dtype=np.float32),
        np.asarray(measured, dtype=np.float32),
        used,
    )


def refine_placement(
    image: LoadedImage,
    placement: ChartPlacement,
    target: TargetDefinition,
    *,
    min_features: int = MIN_FEATURES,
    max_shift_fraction: float = MAX_SHIFT_FRACTION,
) -> Refinement:
    """Propose a better placement from the chart's own printed features."""
    corner_positions = target.fiducials.corner_positions
    original = [tuple(map(float, point)) for point in placement.corners_px]

    def refuse(reason: str, features: int = 0, model: str = "none") -> Refinement:
        return Refinement(
            corners=original, moved_px=[0.0] * len(original), features_used=features,
            model=model, residual_px=float("nan"), applied=False, reason=reason,
        )

    if not corner_positions:
        return refuse(
            f"{target.display_name} declares no fiducial positions to solve for."
        )

    declared, measured, used = _feature_centers(image, placement, target)
    features = len(used)
    if features < min_features:
        prints_features = any(r.kind in FEATURE_KINDS for r in target.regions)
        return refuse(
            f"{features} printed feature(s) could be located, and at least "
            f"{min_features} are needed before a correction is better determined "
            "than the placement it would replace. "
            + (
                "Check that the corners are roughly right before refining."
                if prints_features
                else f"{target.display_name} prints none this can use."
            ),
            features,
        )

    # Which MODEL is chosen by how many features were found (see the module
    # docstring); which ESTIMATOR is chosen by how they are spread (see _estimator).
    _, estimator = _estimator(declared)
    model, transform = _fit(declared, measured, features)
    if transform is None:
        return refuse(
            "The located features do not agree on any single correction, so at "
            "least one of them is not the feature it was taken for.",
            features,
        )

    corrected = cv2.perspectiveTransform(
        declared.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)
    residual = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        placement.to_pixel(corrected.astype(np.float64))
                        - placement.to_pixel(measured.astype(np.float64))
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )

    corner_relative = np.asarray(corner_positions, dtype=np.float32).reshape(-1, 1, 2)
    refined = placement.to_pixel(
        cv2.perspectiveTransform(corner_relative, transform)
        .reshape(-1, 2).astype(np.float64)
    )
    moved = [
        float(np.linalg.norm(np.asarray(new) - np.asarray(old)))
        for new, old in zip(refined, original)
    ]

    # A correction is out by pixels or tens of pixels. Anything approaching a
    # fraction of the chart means a feature was mislocated, and moving the
    # placement there would be worse than leaving it alone.
    placed = np.asarray(original, dtype=np.float64)
    short_side = float(
        min(
            np.linalg.norm(placed[1] - placed[0]),
            np.linalg.norm(placed[3] - placed[0]),
        )
    )
    limit = short_side * max_shift_fraction
    if max(moved) > limit:
        return refuse(
            f"The correction would move a corner {max(moved):.0f} px, more than "
            f"{max_shift_fraction:.0%} of the chart ({limit:.0f} px). That is a "
            "misdetection rather than a correction; place the corners closer and "
            "try again.",
            features, model,
        )

    notes = [
        f"Solved from {features} printed feature(s) as a {model}: "
        f"{', '.join(sorted(used))}.",
        f"Fitted by {estimator}.",
        f"The located features agree with the fitted placement to {residual:.2f} px RMS.",
    ]
    if residual > short_side * NOISY_FEATURE_FRACTION:
        notes.append(
            "Some located features don't match the chart's expected layout. "
            "Check placement."
        )
    return Refinement(
        corners=[(float(x), float(y)) for x, y in refined],
        moved_px=moved,
        features_used=features,
        model=model,
        residual_px=residual,
        applied=True,
        reason=(
            f"Corners moved by {min(moved):.1f}-{max(moved):.1f} px, solved from "
            f"{features} printed features as a {model}."
        ),
        notes=notes,
    )


def feature_separation(declared) -> float:
    """The closest pair of declared features, as a fraction of the widest pair.

    A shape statistic about the *definition*, not about any capture: it asks
    whether the chart's features are spread over its area or bunched into a few
    places. See :func:`_estimator` for what turns on it.
    """
    points = np.asarray(declared, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 1.0
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    upper = gaps[np.triu_indices(len(points), 1)]
    widest = float(upper.max())
    return 1.0 if widest <= 0.0 else float(upper.min()) / widest


def _estimator(declared) -> tuple[int, str]:
    """Which robust estimator to fit with, decided by how the features are spread.

    **LMEDS unless the features are clustered.** It is the right default and it
    earns its keep: it is median-based, so it survives a minority of mislocated
    features, and on a chart where four of ten discs snapped onto the wrong
    neighbour by 50 px it recovered the corners to 0.21 px while the residual rose
    from 0.15 px to 32. Least squares would have been dragged badly there.

    **Least squares when they are clustered**, because LMEDS then fails in a way
    that is invisible. It works by drawing random *minimal* four-point subsets and
    keeping whichever scores the best median error. When two features nearly
    coincide, a subset containing both is degenerate for a homography -- and it
    still scores acceptably, because LMEDS is entitled to ignore half the data. It
    then returns a wild transform that most of the features disagree with.

    The Golden Thread Device Level Target is the case. It declares ten features
    that sit in only FIVE places: each module contributes its dark mount and the
    white resolution field inside it, whose centres are 0.0009 apart in relative
    units -- 5.4 px on an 11600 px master. Over 60 runs seeded with 8 px of
    placement error, LMEDS put the corners more than 5 px out in **19** of them,
    the worst by 58 px, while least squares did so in **none** and matched LMEDS's
    median to a thousandth of a pixel. Refitting on only the features that agreed
    did not help, which is what rules out the obvious reading that outliers were
    dragging the fit: they were not, and the features LMEDS discarded were the
    correct ones.

    The threshold is set from measurement rather than taste. Across every shipped
    target with four or more features the statistic reads 0.34 (UTT), 0.18 (ISA
    4x5), 0.18 (ISA 120), 0.14 (ISA 35mm) and 0.0014 (Device Level) -- two decades
    of empty space between the one clustered chart and the rest, so anything in it
    separates them and the exact value cannot matter. On the three film targets
    the two estimators agree to three decimal places anyway.
    """
    ratio = feature_separation(declared)
    if ratio < CLUSTERED_FEATURE_FRACTION:
        return 0, f"least squares (features cluster: closest pair is {ratio:.4f} of the widest)"
    return cv2.LMEDS, "LMEDS"


def _fit(declared, measured, features: int):
    """The richest transform the features over-determine, and its matrix.

    Returned as a 3x3 either way, so the caller has one shape of thing to apply.
    """
    method, _ = _estimator(declared)
    for name, dof in MODELS:
        if features < dof:
            continue
        if name == "homography":
            matrix, _ = cv2.findHomography(declared, measured, method=method)
            if matrix is not None:
                return name, matrix
        else:
            partial, _ = cv2.estimateAffinePartial2D(
                declared, measured, method=method
            )
            if partial is not None:
                return name, np.vstack([partial, [0.0, 0.0, 1.0]])
    return MODELS[-1][0], None
