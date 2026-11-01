"""File-level conformance metrics.

FADGI's tables specify more than colorimetry. Every material category and star
level also fixes the master file format, bit depth, color space and sampling
frequency, and a capture that misses any of those does not meet the star level
however good its Delta E is.

A profile that evaluated only the color metrics and still printed "FADGI 4-star:
PASS" for an 8-bit sRGB file would be overclaiming in exactly the way this project
set out to avoid -- so these are real metrics with real pass/fail, not report
decoration.
"""

from __future__ import annotations

import re
from typing import Any

from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    evaluate_condition,
    register_metric,
)
from deltachrome.metrics.resolution import resolve_sampling_ppi

#: Color space names normalize before comparison: profile descriptions in the wild
#: read "eciRGB v2", "ECI RGB v2" and "ECIRGB_v2" for the same space.
_NORMALISE = re.compile(r"[^a-z0-9]+")

CONTAINER_NAMES = {
    "tiff": "TIFF",
    "jp2": "JPEG 2000",
    "jpeg2000": "JPEG 2000",
    "jpeg": "JPEG",
    "png": "PNG",
}


def permitted_key(text: str) -> str:
    """A permitted value normalised for comparison.

    Profile descriptions in the wild read "eciRGB v2", "ECI RGB v2" and
    "ECIRGB_v2" for the same space, and a table naming one of them means all
    three. Public because anything *offering* a list of permitted values has to
    compare them the same way this does, or a value could be added twice under
    two spellings and match once.
    """
    return _NORMALISE.sub("", text.lower())


#: The in-module spelling, kept because it reads better at the call sites here.
_key = permitted_key


def _image(ctx: MetricContext):
    image = ctx.extras.get("image")
    if image is None:
        raise KeyError("image")
    return image


def _skip(spec: MetricSpec, reason: str) -> MetricResult:
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="incomplete",
        units=spec.units,
        condition=spec.pass_condition,
        reason=reason,
    )


@register_metric("bit_depth")
def bit_depth(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """The master's bit depth against the star level's requirement.

    Stated as the depths the level permits rather than as a floor. ``value >= 8``
    was the older encoding and it says something the tables do not: that a level
    asking for 8 bits would take any depth above it. A master file is 8-bit or
    16-bit, those are the two the tables name, and a criterion reading ``8 or 16``
    is the one a reader can check against the published row. A profile that still
    carries the older ``pass_condition`` is judged by it exactly as before, since
    a definition is a file a user may have written.
    """
    try:
        image = _image(ctx)
    except KeyError:
        return _skip(spec, "No image was supplied to the metric context.")

    value = float(image.bit_depth)
    allowed = [int(a) for a in spec.algorithm.get("allowed", [])]
    if allowed:
        passed: bool | None = image.bit_depth in allowed
        condition = " or ".join(str(a) for a in allowed)
        threshold = float(min(allowed))
    else:
        passed, threshold = evaluate_condition(value, spec.pass_condition)
        condition = spec.pass_condition

    notes = []
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units or "bits",
        value=value,
        # A depth is counted, not measured. "8.000" claims five figures of
        # precision for a number that has one.
        value_text=str(image.bit_depth),
        passed=passed,
        condition=condition,
        notes=notes,
        metadata={"bit_depth": image.bit_depth, "dtype": image.dtype,
                  "threshold": threshold, "allowed": allowed or None},
    )


@register_metric("master_file_format")
def master_file_format(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """The container the master is stored in, against the permitted list."""
    try:
        image = _image(ctx)
    except KeyError:
        return _skip(spec, "No image was supplied to the metric context.")

    allowed = spec.algorithm.get("allowed", [])
    actual = CONTAINER_NAMES.get(image.container.lower(), image.container.upper())
    permitted = {_key(a) for a in allowed}
    passed = _key(actual) in permitted if allowed else None

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units,
        value=None,
        value_text=actual,
        passed=passed,
        condition=", ".join(allowed) if allowed else None,
        notes=[] if passed is not False else [
            f"Master is {actual}; this level permits {', '.join(allowed)}."
        ],
        metadata={"format": actual, "allowed": allowed},
    )


@register_metric("color_space")
def color_space(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """The embedded ICC profile against the permitted color spaces."""
    try:
        image = _image(ctx)
    except KeyError:
        return _skip(spec, "No image was supplied to the metric context.")

    allowed = spec.algorithm.get("allowed", [])
    description = image.profile.description
    key = _key(description)
    # Substring matching both ways: a profile may be named "Adobe RGB (1998)" while
    # the table says "Adobe RGB", or carry a vendor prefix around the space name.
    matched = next(
        (a for a in allowed if _key(a) in key or key in _key(a)),
        None,
    )
    passed = matched is not None if allowed else None

    notes = []
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units,
        value=None,
        value_text=description,
        passed=passed,
        condition=", ".join(allowed) if allowed else None,
        notes=notes,
        metadata={
            "icc_profile": description,
            "matched": matched,
            "allowed": allowed,
            "transform_kind": image.profile.transform_kind,
        },
    )


@register_metric("sampling_frequency")
def sampling_frequency(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """Measured sampling rate at the target plane, in pixels per inch.

    Derived from the placed fiducial quadrilateral and the physical size of what
    that quadrilateral bounds -- or, in preference to both, read off a scale the
    chart prints. For a chart whose fiducials are the corners of its patch array,
    the extent is the *array* size, not the card size: using the card would
    under-report the sampling rate by the width of the chart's border, which on a
    ColorChecker SG is enough to fail a level the capture actually meets.

    The resolution itself lives in :mod:`deltachrome.metrics.resolution`, because
    MTF50 in line pairs per millimetre needs exactly the same number and the two
    must not be able to disagree about it.
    """
    rate, reason = resolve_sampling_ppi(ctx)
    if rate is None:
        if (ctx.target.dimensional_reference or {}).get("available") is False:
            # The target has *declared* that nothing in a capture of it can
            # establish its physical size, which is the same kind of statement
            # ``not_capable_of`` makes and is answered the same way. It used to
            # report incomplete, on the argument that the resolution requirement
            # applies to the capture whatever chart was in front of it -- true,
            # and true of sharpness too, which a chart with no slanted edges
            # reports as N/A without anybody thinking the requirement has gone
            # away. One metric holding the opposite convention made INCOMPLETE
            # mean two different things in one report: "this could not be
            # measured and should have been" on some rows and "this chart cannot
            # measure it" on others.
            #
            # What the requirement's continued force rests on is not this
            # outcome. A run that measures nothing about resolution says so --
            # the coverage of a profile is reported, and a session distinguishes
            # a metric no chart carried from one a chart carried and failed on.
            return MetricResult(
                metric_id=spec.metric_id, outcome="not_applicable",
                units=spec.units, condition=spec.pass_condition,
                reason=reason + " The resolution requirement still applies to the "
                                "capture; it cannot be verified from this target.",
            )
        return _skip(spec, reason)

    passed, threshold = evaluate_condition(rate.ppi, spec.pass_condition)
    metadata: dict[str, Any] = {"threshold": threshold, "source": rate.source}

    if rate.source == "printed ruler":
        measured = rate.detail or {}
        notes = [
            "Measured from the chart's own printed ruler, not read from the file "
            "header and not derived from a declared physical size. A header ppi can "
            "be set to anything, and a declared extent is a property of a batch "
            "rather than of the specimen in front of the camera.",
            "Rulers: " + ", ".join(
                f"{name} {entry['ppi']:.1f} ppi from {entry['divisions_used']} "
                f"divisions, fit residual {entry['fit_residual_px']:.2f} px"
                for name, entry in measured.get("rulers", {}).items()
            ),
            *rate.notes[1:],
        ]
        metadata.update({"rulers": measured.get("rulers"),
                         "ruler_agreement": measured.get("agreement")})
    else:
        placement = ctx.extras["placement"]
        notes = [
            "Measured at the target plane from the placed fiducials, not read from "
            "the file header. A header ppi can be set to anything; this is what the "
            "capture actually resolved.",
            *rate.notes,
        ]
        metadata.update({
            "fiducial_extent_mm": (rate.detail or {}).get("fiducial_extent_mm"),
            "extent_source": ("operator-supplied"
                              if rate.source == "operator-supplied extent"
                              else "target definition"),
            "image_pixels": [placement.image_size[0], placement.image_size[1]],
        })

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units or "ppi",
        value=rate.ppi,
        passed=passed,
        condition=spec.pass_condition,
        notes=notes,
        metadata=metadata,
    )