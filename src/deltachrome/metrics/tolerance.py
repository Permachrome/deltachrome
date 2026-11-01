"""Highlight/Shadow Tolerance and Dynamic Range -- FADGI 2.4.15 and 2.4.16.

Both are informative-only in every FADGI table that carries them: the standard's
own words are "this is an emerging area and the final values are
provisional/informational" for Highlight/Shadow Tolerance, and "...informative
only" for Dynamic Range. Neither gates a verdict; both are read and reported
for the reason every other informative row here is -- ``sfr_response_at_nyquist``
already carries a real FADGI figure and no ``pass_condition``, and these follow
it rather than being left out of the profile's ``metrics`` list entirely, which
is what they were before this module existed.

**Highlight/Shadow Tolerance** checks the two ends of the tonal ramp against a
fixed aim FADGI states directly in its own table -- 96 +/- 2 L* for the
highlight end, 2.7 +/- 1.5 L* for the shadow end -- identical across every star
level and every category that carries the row (checked against all six: the
five FADGI Photographic Negatives / Transparencies / Microfilm / X-Ray Film
categories that declare it print the same two numbers everywhere). That is why
this needs no aim from the target's own reference file the way white balance or
tone response do: the aim is the standard's, not the chart's, and it is checked
against the *measured* L* directly rather than against a difference from a
reference. Modelled as two metrics rather than one row with two numbers,
because the engine's ``MetricResult`` is one reading, and a single scalar
combining two different patches at two different aims would not be either
reading.

**Dynamic Range** is "D max minus D min", and FADGI's own text gives no
measurement procedure -- "the dynamic range is calculated for both capture
systems and source material" describes what the number means, not how to
derive it from one capture, and the guideline explicitly calls this "an
emerging area". Inventing an algorithm the standard does not specify would be
exactly the failure this project's own culture refuses -- a plausible-looking
number nobody can trace back to a definition. What is reported instead is
traceable to one: the density span of the target's own declared reference
ramp, restricted to the patches this capture actually resolved without
clipping. That is a real reading of "source material... calculated for
capture systems" -- the declared span, cut down by whatever this particular
capture could not discriminate -- built entirely from numbers already in the
record (a reference density, already read generically off the CGATS file; a
clipping flag, already computed by the sampler) rather than from a new
threshold. It needs a target whose reference carries a declared density per
patch; a reflective target's ramp has no density at all and reports N/A.
"""

from __future__ import annotations

from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchDatum,
    PatchValue,
    register_metric,
    select_patches,
)

#: FADGI 2.4.15's own aim. Identical across every star level and every category
#: that carries the row -- not a profile parameter, because it is the standard's
#: fixed target rather than a per-chart or per-level threshold.
HIGHLIGHT_AIM_L = 96.0
HIGHLIGHT_TOLERANCE_L = 2.0
SHADOW_AIM_L = 2.7
SHADOW_TOLERANCE_L = 1.5

#: Sampling flags that mean the patch's own density (or lightness) was not
#: faithfully captured, so it must not stand for what the system *could*
#: resolve. Matches how :mod:`deltachrome.metrics.difference` names the same
#: two flags; a divergence or gamut flag is about colour fidelity rather than
#: about whether the tone itself was recorded at all, so neither excludes a
#: patch here.
_CLIPPING_FLAGS = ("clipped_high", "clipped_low")


def _is_clipped(datum: PatchDatum) -> bool:
    return any(f.split("(")[0] in _CLIPPING_FLAGS for f in datum.flags)


def _ramp_extreme(
    ctx: MetricContext, spec: MetricSpec, *, want_min: bool
) -> PatchDatum | None:
    """The lightest (``want_min``) or darkest patch of the selected ramp.

    By declared ``ramp_position`` rather than by measured L* -- that order is
    the target's own statement of which end is which, and it is the order
    every other ramp reading here (the OECF curve, gain modulation) already
    trusts. Measured L* would beg the question this metric exists to check.
    """
    selected = [
        d for d in select_patches(ctx.data, spec.patch_selection)
        if d.patch.ramp_position is not None
    ]
    if not selected:
        return None
    key = (lambda d: d.patch.ramp_position)
    return min(selected, key=key) if want_min else max(selected, key=key)


def _tolerance_metric(
    ctx: MetricContext, spec: MetricSpec, *, want_min: bool,
    aim: float, tolerance: float, end_name: str,
) -> MetricResult:
    datum = _ramp_extreme(ctx, spec, want_min=want_min)
    if datum is None:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units="L*",
            condition=spec.pass_condition,
            aggregation=spec.aggregation,
            reason=(
                f"No patch matching {spec.patch_selection!r} declares a ramp "
                f"position, so the ramp's {end_name} end could not be identified."
            ),
        )
    value = float(datum.measured_lab[0])
    within_aim = abs(value - aim) <= tolerance
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units="L*",
        value=value,
        # Informative: FADGI's own words mark this row provisional/informational,
        # so ``passed`` stays unset rather than inventing a verdict the standard
        # does not ask for -- see MetricResult.status, which reads that as MEASURED.
        passed=None,
        condition=spec.pass_condition,
        aggregation=spec.aggregation,
        per_patch=[
            PatchValue(
                patch_id=datum.patch_id,
                role=datum.role,
                value=value,
                measured_lab=tuple(float(v) for v in datum.measured_lab),
                reference_lab=(
                    tuple(float(v) for v in datum.reference_lab)
                    if datum.reference_lab is not None else None
                ),
                flags=list(datum.flags),
                position_relative=datum.position_relative,
            )
        ],
        metadata={
            "fadgi_aim_L": aim,
            "fadgi_tolerance_L": tolerance,
            "within_fadgi_aim": within_aim,
            "ramp_position": datum.patch.ramp_position,
        },
    )


@register_metric("highlight_tolerance")
def highlight_tolerance(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    return _tolerance_metric(
        ctx, spec, want_min=True, aim=HIGHLIGHT_AIM_L,
        tolerance=HIGHLIGHT_TOLERANCE_L, end_name="highlight",
    )


@register_metric("shadow_tolerance")
def shadow_tolerance(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    return _tolerance_metric(
        ctx, spec, want_min=False, aim=SHADOW_AIM_L,
        tolerance=SHADOW_TOLERANCE_L, end_name="shadow",
    )


@register_metric("dynamic_range")
def dynamic_range(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    selected = select_patches(ctx.data, spec.patch_selection)
    with_density = [d for d in selected if d.reference_density is not None]
    usable = [d for d in with_density if not _is_clipped(d)]
    if not usable:
        reason = (
            "No patch in the selected ramp declares a reference density -- "
            "reflective targets have none, and this reads only a transmissive "
            "target's own density-referenced ramp."
            if not with_density else
            "Every patch that declares a reference density was flagged clipped "
            "in this capture, so no density this capture resolved can be reported."
        )
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units="D",
            condition=spec.pass_condition,
            aggregation=spec.aggregation,
            reason=reason,
        )
    densities = [float(d.reference_density) for d in usable]  # type: ignore[arg-type]
    d_max, d_min = max(densities), min(densities)
    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units="D",
        value=d_max - d_min,
        passed=None,  # informative -- FADGI marks this row informative-only
        condition=spec.pass_condition,
        aggregation=spec.aggregation,
        per_patch=[
            PatchValue(
                patch_id=d.patch_id, role=d.role, value=float(d.reference_density),  # type: ignore[arg-type]
                measured_lab=(0.0, 0.0, 0.0), reference_lab=None,
                flags=list(d.flags), position_relative=d.position_relative,
            )
            for d in usable
        ],
        metadata={
            "d_max": d_max, "d_min": d_min,
            "n_unclipped": len(usable),
            "n_excluded_clipped": len(with_density) - len(usable),
        },
    )
