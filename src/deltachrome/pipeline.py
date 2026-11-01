"""The analysis pipeline.

    image + ICC -> corner placement -> homography -> patch sampling
                -> color-managed Lab -> join to per-unit reference
                -> profile metrics -> pass / fail

Each stage is a separate module and independently testable; this file only wires
them together and records what it did. No UI code lives here or below.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from deltachrome.errors import MissingReferenceError
from deltachrome.imaging.color import (
    WHITE_POINTS,
    chromatic_adapt,
    lab_to_xyz,
    rgb_to_lab,
    xyz_to_lab,
)
from deltachrome.imaging.geometry import ChartPlacement, place_chart
from deltachrome.imaging.icc import CIE_D50
from deltachrome.imaging.loader import LoadedImage, load_image
from deltachrome.imaging.sampling import DEFAULT_ROI_INSET, SamplingResult, sample_patches
from deltachrome.metrics.base import (
    ROLES_WITHOUT_REFERENCE,
    MetricContext,
    PatchDatum,
)
from deltachrome.profiles.evaluator import (
    ProfileEvaluation,
    evaluate_profile,
    required_capability_note,
)
from deltachrome.profiles.schema import AnalysisProfile, load_profile
from deltachrome.reference import MeasurementSet, read_reference
from deltachrome.reference.crosscheck import crosscheck_against_nominal
from deltachrome.reference.sample_ids import reconcile_sample_ids
from deltachrome.results.edges import (
    build_edge_record,
    build_registration_records,
    field_summary,
    registration_summary,
)
from deltachrome.results.models import (
    AnalysisRun,
    PatchRecord,
    environment_info,
    file_fingerprint,
    now_utc,
)
from deltachrome.targets.registry import load_target
from deltachrome.targets.roles import RoleResolution, resolve_roles
from deltachrome.targets.schema import TargetDefinition
from deltachrome.text import plural

#: Aspect-ratio disagreement between the placed quadrilateral and the proportions
#: its fiducials bound, above which the placement is almost certainly wrong. See
#: :func:`deltachrome.imaging.detect.shape_deviation`: correct placements sit
#: within 1% and a quarter turn 40% or more out.
ASPECT_WARN_THRESHOLD = 0.08

#: Stands in for a reference file on a target that carries no patches. Not a
#: fallback and not a default: there is nothing to compare, so there is nothing to
#: import, and an empty set says that more honestly than a made-up one would.
EMPTY_REFERENCE = MeasurementSet(samples={}, metadata={}, fields=[])


@dataclass
class Measurement:
    """What the capture yielded, before any profile judged it.

    The pipeline has two halves and they cost wildly different amounts. Reading a
    capture -- decode, place, sample, extract every slanted edge, read every
    printed rule -- is where the seconds go: 16.4 s on the UTT's JPEG 2000 master.
    Judging what was read is a re-run of every metric plugin against data already
    in memory, and that is **0.21 s** on the same capture, 0.48 s on the Golden
    Thread Device Level Target with its twenty edges in three colour records.

    So this is the first half, kept, and it is what makes asking "would this have
    passed at three star" cost a fifth of a second rather than another decode.

    **It is not a cache of the verdict.** Every metric genuinely runs again, on the
    new profile's own algorithm parameters -- flat-field resamples the image on its
    own grid, SFR re-reduces the edges under its own aggregation. What is reused is
    the *reading*, which no profile can change. That distinction is the whole
    warrant for re-scoring: a profile chooses what to measure and what to demand,
    and it does not choose what the capture says.
    """

    context: MetricContext
    image: LoadedImage
    target: TargetDefinition
    reference: MeasurementSet
    placement: ChartPlacement
    sampling: SamplingResult
    roles: RoleResolution
    white_point: tuple[float, float, float]
    patch_records: list[PatchRecord] = field(default_factory=list)
    edges: list = field(default_factory=list)
    edge_records: list = field(default_factory=list)
    registration_records: list = field(default_factory=list)
    registration_reduced: dict[str, Any] = field(default_factory=dict)
    ruler_readings: list = field(default_factory=list)
    #: Everything the reading raised, before a profile was applied. A re-score
    #: starts from a copy of this, so a warning about the reference file survives
    #: into every verdict made from the same measurement and a warning about the
    #: previous profile's coverage does not.
    warnings: list[str] = field(default_factory=list)
    #: Run overrides that belong to the run rather than to the profile, kept so a
    #: re-score applies the same ones and does not silently answer under different
    #: conventions from the run it replaces.
    percentile_method: str | None = None
    frequency_units: str | None = None
    original_size: str | None = None
    #: Who put the chart where it is, and who said which chart it was. Facts about
    #: how the geometry came to be, kept on the measurement so a re-score writes
    #: them into its record as well.
    placed: str = "by_hand"
    identified: bool = False
    #: The targets identification chose among, where that was narrower than every
    #: installed target. ``None`` is all of them, or no identification at all.
    searched_targets: list[str] | None = None


#: How a run's chart came to be where it is. A placement nobody chose has to be
#: visible in the record, and so does one somebody corrected: "automatic" claims
#: nobody touched it, which stops being true the moment a corner is dragged.
PLACED_BY = ("by_hand", "automatic", "automatic_then_moved")


@dataclass
class AnalysisResult:
    """Everything the pipeline produced, live objects plus the persistable record."""

    run: AnalysisRun
    evaluation: ProfileEvaluation
    image: LoadedImage
    target: TargetDefinition
    reference: MeasurementSet
    placement: ChartPlacement
    sampling: SamplingResult
    roles: RoleResolution
    edges: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The reading this verdict was made from, where the caller still holds it.
    #: ``None`` for a result rebuilt from a saved record, which has numbers and no
    #: pixels -- and that is exactly the distinction :func:`rescore` needs, so it
    #: is read rather than assumed.
    measurement: Measurement | None = None

    @property
    def overall(self) -> str:
        return self.evaluation.overall

    @property
    def conformance_claim(self) -> bool:
        """False when the run was scored against nominal rather than measured data."""
        return self.evaluation.conformance_claim


def analyze(
    image: str | Path | LoadedImage,
    *,
    target: str | Path | TargetDefinition,
    reference_path: str | Path | None = None,
    profile: str | Path | AnalysisProfile,
    corners: Sequence[tuple[float, float]],
    roi_inset: float = DEFAULT_ROI_INSET,
    white_point: tuple[float, float, float] = CIE_D50,
    percentile_method: str | None = None,
    frequency_units: str | None = None,
    original_size: str | None = None,
    extra_warnings: Sequence[str] = (),
    fiducial_extent_mm: tuple[float, float] | None = None,
    allow_nominal_reference: bool = True,
    reference_table: int | str | None = None,
    placed: str = "by_hand",
    identified: bool = False,
    searched_targets: Sequence[str] | None = None,
) -> AnalysisResult:
    """Run one image through the full pipeline.

    ``image`` is a path, or an already-decoded
    :class:`~deltachrome.imaging.loader.LoadedImage` when the caller has one.

    ``placed`` and ``identified`` say how the geometry came to be -- one of
    :data:`PLACED_BY`, and whether the chart was worked out from the capture
    rather than named. They go into the record's ``geometry``, which is where a
    reader a year later looks; they used to be warnings, printed on every
    automatic run and saying nothing was wrong.

    ``searched_targets`` is the set identification chose among, where the operator
    had narrowed it in Settings. A chart named from five candidates is a different
    claim from one named from twenty-three, so the record says which; ``None``
    writes nothing, which is every installed target or no identification at all.
    """
    if placed not in PLACED_BY:
        raise ValueError(f"placed must be one of {PLACED_BY}; got {placed!r}")
    warnings: list[str] = list(extra_warnings)

    target_def = target if isinstance(target, TargetDefinition) else load_target(target)
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)

    _apply_run_overrides(profile_def, percentile_method, frequency_units)

    #: Whether the definition says outright that its patches carry no reference
    #: values. ``source_type: "none"`` is the existing spelling of that -- the
    #: frame-filling white field has always used it -- and reading it once here
    #: keeps the two places that must honour it from disagreeing.
    reference_free = target_def.reference_data.get("source_type") == "none"

    # 1. Reference first: without it there is nothing to compare against, and the
    #    target definition's roles cannot be resolved. Unless the target has no
    #    patches -- a dedicated resolution chart measures nothing colorimetric, so
    #    requiring reference data for it would be requiring data about nothing.
    #
    #    A target may also *carry* patches and declare that none of them is compared
    #    against anything, which is not the same thing and is why it is a second test
    #    rather than a widening of the first. The QA-62 is the case: it prints a
    #    twenty-step neutral ring, and the two capabilities it claims are both read
    #    from its slanted square. The ring is declared because it is there and
    #    because its order is what tells the placement which way up the chart is --
    #    the quad is square and so are the cells, so nothing geometric can. Refusing
    #    to run because those twenty patches have no aim values would refuse a
    #    measurement that never wanted them.
    if not target_def.patches or reference_free:
        reference = EMPTY_REFERENCE
        if reference_path is not None and not target_def.patches:
            warnings.append(
                "The reference file was not read: this chart has no patches."
            )
        elif reference_path is not None:
            warnings.append(
                "The reference file was not read: this chart compares no patches."
            )
    else:
        reference_path = _reference_source(
            reference_path, target_def, allow_nominal_reference, warnings
        )
        reference = read_reference(reference_path, table=reference_table)
        warnings.extend(reference.warnings)
        # One chart, two vocabularies. A file may name the chart's patches
        # differently from the chart -- the measurements published for the ISA
        # Golden Thread targets name them 'A1'-'A30' where the card prints
        # '1'-'30' -- so the join is reconciled here, once, before role
        # resolution and the reference Lab table both read it. Only ever when a
        # declared correspondence matches more patches than the file as written,
        # and never silently: the note goes into the run's warnings.
        reference, alias_note = reconcile_sample_ids(reference, target_def)
        if alias_note:
            warnings.append(alias_note)
        # A file can name every patch correctly and still describe a different
        # object: manufacturers revise patch colours without renaming the product,
        # and the ISA FADGI - ISO 19264 is a case in hand. Only worth asking of a
        # file the operator supplied -- the nominal compared with itself says
        # nothing.
        if not reference.is_nominal:
            disagreement = crosscheck_against_nominal(reference, target_def)
            if disagreement:
                warnings.append(disagreement)
        if target_def.reference_data.get("required", True) and len(reference) == 0:
            raise MissingReferenceError(f"{reference_path} parsed but contains no samples.")

    # 2. Image, with its ICC profile. Refuses if absent.
    #
    #    An already-decoded LoadedImage is accepted so that a caller holding the
    #    master in memory does not pay to decode it again. That is not a
    #    micro-optimisation: a JPEG 2000 preservation master takes around 25
    #    seconds to decode, and code verification needs the pixels too, so the
    #    path-only form used to cost two full decodes per analysis.
    image = image if isinstance(image, LoadedImage) else load_image(image)

    # 3. Roles from measured data, not assumption.
    roles = resolve_roles(target_def, reference)
    target_def = roles.target
    warnings.extend(roles.notes)

    # 4. Geometry.
    placement = place_chart(
        corners,
        image_size=(image.width, image.height),
        corner_positions_relative=target_def.fiducials.corner_positions or None,
    )
    from deltachrome.imaging.detect import shape_deviation

    deviation = shape_deviation(target_def, corners)
    if deviation is not None and deviation > ASPECT_WARN_THRESHOLD:
        warnings.append(
            "Placement doesn't match the chart's shape."
        )
    if target_def.geometry_is_provisional:
        for reason in target_def.provisional_reasons():
            warnings.append(f"Chart geometry is provisional: {reason}")

    # 5. Sampling.
    if target_def.grid_regions:
        sampling = sample_patches(image, placement, target_def, inset=roi_inset)
        warnings.extend(sampling.warnings)

        # 6/7. Color-managed conversion and the join to reference values.
        data, patch_records, join_warnings = _build_patch_data(
            image, target_def, reference, sampling, white_point
        )
        warnings.extend(join_warnings)
    else:
        sampling = SamplingResult(samples={}, inset=roi_inset, warnings=[])
        data, patch_records = [], []

    # This refusal guards against a reference file for the wrong model of chart,
    # which leaves every patch unmatched and would otherwise pass an empty
    # comparison off as a result. It can only do that because a patch without a
    # reference is normally dropped -- so it must not fire where no reference was
    # ever in play, or a chart that declares its patches uncompared could not run
    # at all.
    if not data and target_def.patches and not reference_free:
        raise MissingReferenceError(
            "No patch could be both sampled and matched to a reference value. "
            "Check that the reference file is for this chart model and that the "
            "corner placement is correct."
        )

    # 8. Slanted edges, where the target carries them.
    edges: list = []
    edge_records: list = []
    registration_records: list = []
    registration_reduced: dict = {}
    if any(r.kind == "slanted_edge" for r in target_def.regions):
        from deltachrome.imaging.edges import extract_edges
        from deltachrome.imaging.registration import measure_edges as measure_registration

        edges = extract_edges(image, placement, target_def)
        edge_records = _measure_edges(edges, placement)
        if image.channels >= 3:
            measured_registration, _ = measure_registration(edges)
            registration_records = build_registration_records(measured_registration)
            registration_reduced = registration_summary(measured_registration)
        elif edges:
            # Not attempted rather than attempted and failed: differencing one
            # record against itself has no answer, and a list of twenty per-edge
            # failures would read as a fault in the capture's edges.
            warnings.append(
                "Grayscale capture: color and registration not measured."
            )
        if not edges:
            warnings.append(
                "No slanted edges found. SFR not analyzed."
            )

    # 9. The chart's own printed scale, where it has one. Read here rather than
    #    inside a metric, for the same reason the edges are: two metrics want it,
    #    and the reading is worth keeping whether or not a profile asks for either.
    ruler_readings: list = []
    if any(r.kind == "scale" for r in target_def.regions):
        from deltachrome.imaging.rulers import read_rulers

        ruler_readings, ruler_notes = read_rulers(image, placement, target_def)
        # One line however many failed: the rule's name and the reason are in
        # the reading notes, and four rulers failing once each is one fact.
        if ruler_notes:
            warnings.append(
                f"{plural(len(ruler_notes), 'printed scale')} could not be read."
            )

    # 10. Everything read from the capture, in one object, because the next step
    #     is the only one a profile has any say in and it can be repeated.
    measurement = Measurement(
        context=MetricContext(
            target=target_def,
            data=data,
            capture_records=image.channels,
            reference_provenance=reference.provenance(),
            reference_basis=reference.basis,
            extras={
                "placement": placement,
                "image": image,
                "sampling": sampling,
                "fiducial_extent_mm": fiducial_extent_mm,
                "edges": edges,
                "rulers": ruler_readings,
                # Not a property of the capture: what the system is being qualified
                # for. Metamorfoze's illumination tolerance bands on it, and the
                # illumination capture is a blank sheet with no size of its own.
                "original_size": original_size,
            },
        ),
        image=image,
        target=target_def,
        reference=reference,
        placement=placement,
        sampling=sampling,
        roles=roles,
        white_point=white_point,
        patch_records=patch_records,
        edges=edges,
        edge_records=edge_records,
        registration_records=registration_records,
        registration_reduced=registration_reduced,
        ruler_readings=ruler_readings,
        warnings=warnings,
        percentile_method=percentile_method,
        frequency_units=frequency_units,
        original_size=original_size,
        placed=placed,
        identified=identified,
        searched_targets=(
            sorted(searched_targets) if searched_targets is not None else None
        ),
    )

    # 11. Metrics, and the verdict.
    return _judge(measurement, profile_def)


def rescore(
    result: AnalysisResult,
    profile: str | Path | AnalysisProfile,
) -> AnalysisResult:
    """Judge a capture that has already been read, against a different profile.

    The question this answers is an ordinary one and used to cost a whole
    re-analysis: *it failed at four star -- does it pass at three?* Or the plainer
    version, which is picking the wrong material category and noticing from the
    report. Neither is a reason to decode a preservation master again.

    **It is a re-measurement, not a re-comparison.** Every metric plugin runs
    again, on the new profile's own algorithm parameters, because a profile
    chooses far more than thresholds -- which Delta E variant, which patches, how
    a field of edges is reduced, what grid the illumination is sampled on. What it
    cannot change is what the capture said, and that is precisely what is reused.

    The result is a *new run*, with its own id and its own record. Two profiles
    applied to one capture are two findings and the archive holds both; quietly
    overwriting the first would make the library's history depend on which
    question was asked last.

    Raises :class:`ValueError` where the measurement is gone -- a run rebuilt from
    a saved record has every number and no pixels, so it can be read and combined
    into a session but not judged again. Re-open the capture for that.
    """
    if result.measurement is None:
        raise ValueError(
            "This run was rebuilt from its saved record, which holds the readings "
            "but not the capture they came from. Re-open the image to judge it "
            "against another profile."
        )
    profile_def = profile if isinstance(profile, AnalysisProfile) else load_profile(profile)
    return _judge(result.measurement, profile_def)


def _judge(measurement: Measurement, profile: AnalysisProfile) -> AnalysisResult:
    """Run a profile's metrics against a measurement and build the record.

    The second half of :func:`analyze`, and the whole of :func:`rescore`. One
    function rather than two so that a re-scored run cannot drift from a fresh one
    -- a report reading differently depending on which route produced it is the
    same failure as two report renderers, one screen further back.
    """
    _apply_run_overrides(
        profile, measurement.percentile_method, measurement.frequency_units
    )
    # A copy: the coverage note below belongs to this profile, and appending it to
    # the measurement's own list would carry one profile's warning into the next
    # verdict made from the same reading.
    warnings = list(measurement.warnings)

    evaluation = evaluate_profile(profile, measurement.target, measurement.context)
    required_note = required_capability_note(profile, measurement.target)
    if required_note:
        warnings.append(required_note)

    run = _build_run(
        image=measurement.image,
        target=measurement.target,
        reference=measurement.reference,
        profile=profile,
        placement=measurement.placement,
        sampling=measurement.sampling,
        roles=measurement.roles,
        white_point=measurement.white_point,
        patch_records=measurement.patch_records,
        evaluation=evaluation,
        coverage_notes=[required_note] if required_note else [],
        edge_records=measurement.edge_records,
        registration_records=measurement.registration_records,
        registration_reduced=measurement.registration_reduced,
        ruler_readings=measurement.ruler_readings,
        warnings=warnings,
        original_size=measurement.original_size,
        percentile_method=measurement.percentile_method,
        frequency_units=measurement.frequency_units,
    )
    run.geometry["placed"] = measurement.placed
    run.geometry["identified"] = measurement.identified
    if measurement.searched_targets is not None:
        run.geometry["searched_targets"] = list(measurement.searched_targets)

    return AnalysisResult(
        run=run,
        evaluation=evaluation,
        image=measurement.image,
        target=measurement.target,
        reference=measurement.reference,
        placement=measurement.placement,
        sampling=measurement.sampling,
        roles=measurement.roles,
        edges=measurement.edges,
        warnings=warnings,
        measurement=measurement,
    )



def _ruler_record(readings: list) -> dict[str, Any]:
    if not readings:
        return {}
    from deltachrome.imaging.rulers import sampling_rate

    return sampling_rate(readings)


def _reference_source(
    reference_path: str | Path | None,
    target: TargetDefinition,
    allow_nominal: bool,
    warnings: list[str],
) -> Path:
    """Decide what the capture is measured against, and say so out loud.

    A per-unit measurement is always preferred and is used whenever one is supplied.
    The fallback exists because some targets are effectively never available
    measured -- the ISA Golden Thread's patches are too small for a hand
    spectrophotometer and the units are seldom bought measured -- so a hard refusal
    puts the target out of reach rather than protecting anyone. Where the
    manufacturer publishes aim values, using them is what OpenDICE does with its
    preloaded sample profiles, and it reports the substitution rather than hiding it.

    The fallback is never silent: it warns here, it marks the run's reference basis,
    and it costs the run its conformance claim.
    """
    if reference_path is not None:
        return Path(reference_path)

    nominal = target.nominal_reference_path
    if nominal is not None and allow_nominal and not nominal.exists():
        # Distinct from "this target has no nominal values", which reads as a
        # deliberate choice. A definition that names a file it does not ship is
        # broken, and saying so is the difference between a typo and a policy.
        raise MissingReferenceError(
            f"{target.display_name} declares manufacturer nominal values in "
            f"{nominal.name}, but no such file sits beside its definition "
            f"({nominal.parent}). The nominal file travels with the definition that "
            "names it."
        )
    if nominal is not None and not allow_nominal:
        raise MissingReferenceError(
            f"No reference file was supplied. {target.display_name} ships manufacturer "
            f"nominal values ({nominal.name}) that would have been used, but the "
            "nominal fallback was refused for this run. Import the per-unit "
            "measurement file for this chart, or allow the fallback and read the "
            "result as advisory."
        )
    if nominal is None:
        raise MissingReferenceError(
            f"No reference file was supplied and {target.display_name} declares no "
            "manufacturer nominal values, so there is nothing to measure against. "
            "Import the per-unit measurement file for this chart."
        )
    return nominal

def _reference_lab(
    reference: MeasurementSet, white_point: tuple[float, float, float]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Reference Lab values, adapted to ``white_point`` if the file used another.

    Lab is only meaningful relative to a white point. A reference measured under
    D65 and an image evaluated against D50 differ by several Delta E on saturated
    colors, and nothing downstream can notice. Where adaptation is applied it is
    reported, because adaptation is a model rather than a measurement.
    """
    warnings: list[str] = []
    declared = (reference.illuminant or "D50").strip().upper()
    source_white = WHITE_POINTS.get(declared)

    if source_white is None:
        warnings.append(
            f"Unknown reference illuminant {declared!r}; assumed D50."
        )
        source_white = white_point

    observer = (reference.observer or "2").strip()
    if observer not in ("2", "2.0", "1931", ""):
        warnings.append(
            f"Reference uses {observer}° observer; capture uses 2°."
        )

    out: dict[str, np.ndarray] = {}
    needs_adaptation = not np.allclose(source_white, white_point, atol=1e-4)
    for measurement in reference:
        if measurement.lab is not None:
            lab = np.asarray(measurement.lab, dtype=np.float64)
            if needs_adaptation:
                xyz = chromatic_adapt(
                    lab_to_xyz(lab, white_point=source_white), source_white, white_point
                )
                lab = xyz_to_lab(xyz, white_point=white_point)
        elif measurement.xyz is not None:
            xyz = np.asarray(measurement.xyz, dtype=np.float64)
            # CGATS XYZ is conventionally scaled 0-100.
            if float(np.max(xyz)) > 3.0:
                xyz = xyz / 100.0
            if needs_adaptation:
                xyz = chromatic_adapt(xyz, source_white, white_point)
            lab = xyz_to_lab(xyz, white_point=white_point)
        else:
            continue
        out[measurement.sample_name] = lab

    return out, warnings


def _reference_density(reference: MeasurementSet) -> dict[str, float]:
    """A reference file's own declared density per sample, where it has one.

    Read generically off ``PatchMeasurement.raw`` -- the CGATS ``D`` column the
    transmissive film targets' nominal files carry beside Lab -- rather than
    adding a dedicated field to the parser. Only :func:`dynamic_range` reads
    this; every reflective target's reference simply has no such column and the
    dict comes back empty for it.
    """
    out: dict[str, float] = {}
    for measurement in reference:
        raw_d = measurement.raw.get("D")
        if isinstance(raw_d, (int, float)):
            out[measurement.sample_name] = float(raw_d)
    return out


def _build_patch_data(
    image: LoadedImage,
    target: TargetDefinition,
    reference: MeasurementSet,
    sampling: SamplingResult,
    white_point: tuple[float, float, float],
) -> tuple[list[PatchDatum], list[PatchRecord], list[str]]:
    reference_labs, warnings = _reference_lab(reference, white_point)
    reference_densities = _reference_density(reference)

    data: list[PatchDatum] = []
    records: list[PatchRecord] = []
    missing_reference: list[str] = []
    unusable: list[str] = []

    for patch in target.patches:
        sample = sampling.samples.get(patch.patch_id)
        ref = reference_labs.get(patch.patch_id)

        measured_lab: np.ndarray | None = None
        if sample is not None and sample.usable:
            measured_lab = rgb_to_lab(sample.rgb_mean, image.profile, white_point=white_point)

        if sample is None or not sample.usable:
            unusable.append(patch.patch_id)
        elif ref is None:
            missing_reference.append(patch.patch_id)

        records.append(
            PatchRecord(
                patch_id=patch.patch_id,
                row=patch.row,
                col=patch.col,
                role=patch.role,
                ramp_position=patch.ramp_position,
                rgb_mean=list(map(float, sample.rgb_mean)) if sample else [],
                rgb_median=list(map(float, sample.rgb_median)) if sample else [],
                rgb_std=list(map(float, sample.rgb_std)) if sample else [],
                measured_lab=list(map(float, measured_lab)) if measured_lab is not None else None,
                reference_lab=list(map(float, ref)) if ref is not None else None,
                pixel_count=sample.pixel_count if sample else 0,
                clipped_high_fraction=sample.clipped_high_fraction if sample else float("nan"),
                clipped_low_fraction=sample.clipped_low_fraction if sample else float("nan"),
                flags=list(sample.flags) if sample else ["not_sampled"],
                roi_px=(
                    [[round(float(x), 2), round(float(y), 2)]
                     for x, y in sample.polygon_px]
                    if sample is not None else None
                ),
            )
        )

        if measured_lab is None:
            continue
        if ref is None and patch.role not in ROLES_WITHOUT_REFERENCE:
            continue

        data.append(
            PatchDatum(
                patch=patch,
                measured_lab=measured_lab,
                reference_lab=ref,
                flags=list(sample.flags) if sample else [],
                position_relative=_patch_position(target, patch),
                reference_density=reference_densities.get(patch.patch_id),
            )
        )

    carried = [
        pid for pid in missing_reference
        if (found := target.get(pid)) is not None
        and found.role in ROLES_WITHOUT_REFERENCE
    ]
    dropped = [pid for pid in missing_reference if pid not in set(carried)]
    # A target declaring no reference values at all (the QA-62) states that as a
    # fact about the chart, so it is not reported on every run as a shortfall.
    # The patches are still sampled and recorded, so the report can draw them.
    if dropped and target.reference_data.get("source_type") != "none":
        warnings.append(
            f"{len(dropped)} patch(es) were sampled but have no reference value and "
            "are excluded from every metric: "
            + ", ".join(dropped[:12])
            + (" ..." if len(dropped) > 12 else "")
        )
    if unusable:
        warnings.append(
            f"{len(unusable)} patch(es) could not be sampled and are excluded: "
            + ", ".join(unusable[:12])
            + (" ..." if len(unusable) > 12 else "")
        )

    return data, records, warnings


def _apply_run_overrides(
    profile: AnalysisProfile,
    percentile_method: str | None,
    frequency_units: str | None,
) -> AnalysisProfile:
    """Settle the two choices that belong to the run rather than to the profile.

    Both are presentation conventions rather than criteria, which is why neither
    forks the profile into a local specification -- and why both have to be
    reapplied when a measurement is judged again, or a re-score would quietly
    answer under different conventions from the run it replaces.
    """
    if percentile_method is not None:
        for spec in profile.metrics:
            if spec.aggregation.startswith("percentile_"):
                spec.algorithm = {**spec.algorithm, "percentile_method": percentile_method}

    # MTF50 is a frequency in cycles per pixel until a unit is chosen, and which one
    # is a presentation choice rather than a change to the measurement -- so it is
    # overridable per run without the profile becoming a local specification.
    if frequency_units is not None:
        for spec in profile.metrics:
            if spec.metric_id == "mtf50":
                # Recorded *beside* the profile's own declaration rather than over
                # it. The threshold is written in the declared unit, so converting
                # the criterion needs it -- overwriting it is what let a change of
                # unit change the verdict.
                spec.algorithm = {
                    **spec.algorithm, "frequency_units_override": frequency_units,
                }
    return profile


def _build_run(
    *,
    image: LoadedImage,
    target: TargetDefinition,
    reference: MeasurementSet,
    profile: AnalysisProfile,
    placement: ChartPlacement,
    sampling: SamplingResult,
    roles: RoleResolution,
    white_point: tuple[float, float, float],
    patch_records: list[PatchRecord],
    evaluation: ProfileEvaluation,
    coverage_notes: list[str],
    edge_records: list,
    registration_records: list,
    registration_reduced: dict[str, Any],
    ruler_readings: list,
    warnings: list[str],
    original_size: str | None,
    percentile_method: str | None,
    frequency_units: str | None,
) -> AnalysisRun:
    metrics: list[dict[str, Any]] = []
    for result in evaluation.results:
        metrics.append(
            {
                "metric_id": result.metric_id,
                # Which capability this metric reads. Persisted so a report can
                # group metrics by what they measure without keeping its own list
                # of metric ids -- the same read-the-data rule that decides which
                # metrics get a field map.
                "requires_capability": _spec_capability(profile, result.metric_id),
                "outcome": result.outcome,
                "status": result.status,
                "value": result.value,
                # Present only where the number is not the reading -- the
                # master's container, its colour space, its bit depth. Absent
                # everywhere else, and a record written before schema 7 simply
                # omits it, which reads back as "format it from the number".
                "value_text": result.value_text,
                "units": result.units,
                # Only where a reading is not in the aggregate's unit; see
                # MetricResult.reading_units. Records before schema 9 lack it and
                # are filled in on read by AnalysisRun.from_dict.
                "reading_units": result.reading_units,
                "passed": result.passed,
                "condition": result.condition,
                "aggregation": result.aggregation,
                "reason": result.reason,
                "notes": result.notes,
                "metadata": result.metadata,
                "per_patch": [
                    {
                        "patch_id": pv.patch_id,
                        "role": pv.role,
                        "value": pv.value,
                        "flags": pv.flags,
                        # Kept even when null. For a patch this could in
                        # principle be re-derived from the target definition, but
                        # most of what flows through here is not a patch -- a
                        # flat-field spot, a printed rule, a position at which the
                        # color records were differenced -- and for those the
                        # record is the only place the location exists at all.
                        "position_relative": (
                            list(pv.position_relative)
                            if pv.position_relative else None
                        ),
                        # Present only for samples that are not patches; a
                        # patch's region is on its own record, joined by id.
                        "roi_px": pv.roi_px,
                    }
                    for pv in result.per_patch
                ],
                "vectors": [
                    {
                        "label": v.label,
                        "position_relative": list(v.position_relative),
                        "vector_px": list(v.vector_px),
                        "source": v.source,
                    }
                    for v in result.vectors
                ],
            }
        )

    return AnalysisRun(
        run_id=uuid.uuid4().hex[:12],
        created_utc=now_utc(),
        image={
            "path": str(image.path.resolve()),
            "filename": image.path.name,
            # So a run whose capture has moved can be offered a candidate rather
            # than only the path it used to be at.
            "fingerprint": file_fingerprint(image.path),
            "width": image.width,
            "height": image.height,
            "bit_depth": image.bit_depth,
            "dtype": image.dtype,
            # How many colour records the file carried. A reopened run has no
            # pixels, so without this it could not say why a colour metric came
            # back incomplete -- and "incomplete" with no reason is the state this
            # record exists to prevent.
            "color_records": image.channels,
            # What the file claims, kept beside what was measured. Never used to
            # derive anything -- see loader.read_header_resolution.
            "header_ppi": list(image.header_ppi) if image.header_ppi else None,
        },
        target={
            "target_id": target.target_id,
            "display_name": target.display_name,
            "capabilities": sorted(target.capabilities),
            "not_capable_of": sorted(target.not_capable_of),
            "patch_count": len(target.patches),
            "role_counts": roles.counts,
            "uniformity_patches": roles.uniformity_patches,
            "geometry_provisional": target.geometry_is_provisional,
            "regions": [
                {"region_id": r.region_id, "kind": r.kind, "notes": r.notes}
                for r in target.regions
            ],
        },
        reference={
            "path": str(reference.source_path) if reference.source_path else None,
            "sample_count": len(reference),
            "fields": reference.fields,
            "provenance": reference.provenance(),
            "measurement_mode": reference.measurement_mode,
            "basis": reference.basis,
            "nominal_caveat": reference.nominal_caveat,
            "table_index": reference.table_index,
            "tables_available": reference.tables_available,
        },
        profile={
            "profile_id": profile.profile_id,
            "display_name": profile.display_name,
            "standard_reference": profile.standard_reference,
            # The fact, not the parent's id. A record written before this
            # carries "derived_from" and is read by `criteria_are_local`.
            "local_criteria": profile.local_criteria,
            "overall_pass_rule": profile.overall_pass_rule,
            "capability_report": evaluation.capability_report,
            "coverage": evaluation.coverage,
            "coverage_line": evaluation.coverage_line,
            "coverage_notes": coverage_notes,
        },
        geometry=_geometry_record(target, placement),
        sampling={
            "roi_inset": sampling.inset,
            "patches_sampled": len(sampling),
            "flagged": sampling.flagged(),
        },
        color_management=_color_management_record(image, reference, white_point),
        patches=patch_records,
        edges=edge_records,
        edge_field_summary=field_summary(edge_records),
        registration=registration_records,
        registration_summary=registration_reduced,
        rulers=_ruler_record(ruler_readings),
        metrics=metrics,
        overall=evaluation.overall,
        overall_explanation=evaluation.overall_explanation,
        conformance_claim=evaluation.conformance_claim,
        conformance_caveats=list(evaluation.conformance_caveats),
        warnings=warnings,
        environment=environment_info(),
        original_size=original_size,
        percentile_method=percentile_method,
        frequency_units=frequency_units,
    )


def _spec_capability(profile, metric_id: str) -> str | None:
    for spec in profile.metrics:
        if spec.metric_id == metric_id:
            return spec.requires_capability
    return None


def _color_management_record(
    image: LoadedImage,
    reference: MeasurementSet,
    white_point: tuple[float, float, float],
) -> dict[str, Any]:
    """Record how device values became Lab, in enough detail to reproduce it.

    Which ICC structure was evaluated is part of the result, not an implementation
    detail: a matrix/TRC profile and a LUT profile describe the device differently,
    and a reader comparing two runs needs to know which was used.
    """
    profile = image.profile
    record: dict[str, Any] = {
        "icc_profile": profile.description,
        "icc_class": profile.profile_class,
        "icc_pcs": profile.pcs,
        "transform_kind": profile.transform_kind,
        "transforms_available": list(profile.available_transforms),
        "lab_white_point": list(white_point),
        "reference_illuminant": reference.illuminant,
        "reference_observer": reference.observer,
        "icc_notes": list(profile.notes),
    }

    if profile.transform_kind == "matrix_trc":
        assert profile.trc is not None
        record["trc"] = [curve.describe() for curve in profile.trc]
        record["method"] = (
            "Device RGB linearized through the embedded ICC tone curves, converted to "
            "PCS XYZ by the profile's own colorant matrix, then to L*a*b* against the "
            "stated white point. Full float precision throughout."
        )
    else:
        assert profile.pipeline is not None
        pipeline = profile.pipeline
        record["lut"] = {
            "tag": pipeline.tag,
            "tag_type": pipeline.tag_type.strip(),
            "pcs_encoding": pipeline.pcs_encoding,
            "clut_grid": list(pipeline.clut.grid) if pipeline.clut else None,
            "stages": pipeline.describe(),
        }
        record["method"] = (
            f"Device RGB evaluated through the embedded ICC {pipeline.tag} lookup "
            f"pipeline ({pipeline.tag_type.strip()}), with tetrahedral CLUT "
            "interpolation, then to L*a*b* against the stated white point. Full float "
            "precision throughout."
        )
    return record


def _geometry_record(target: TargetDefinition, placement: ChartPlacement) -> dict[str, Any]:
    """Record the placement, deriving sampling rate from the right dimension.

    The fiducials bound a specific area, and on most charts that is smaller than the
    card -- a ColorChecker SG's crosshairs enclose 247.6 x 176.1 mm of a 292.1 x
    203.2 mm card. Dividing the fiducial quad's pixel size by the *card* dimension
    under-reports the sampling rate by about 15%, which is the difference between
    passing and failing a star level. The extent the fiducials actually bound is
    used where the target records it, and the report says which was used.
    """
    extent = target.fiducial_extent_mm
    basis = "fiducial_extent_mm" if extent else "physical_size_mm"
    size = extent or target.physical_size_mm

    record: dict[str, Any] = {
        "corners_px": placement.corners_px.tolist(),
        "corner_order": target.fiducials.corner_ids,
        "placement_method": target.fiducials.method,
        "chart_area_px": placement.area_px,
        "aspect_ratio": placement.aspect_ratio,
        "aspect_deviation": placement.aspect_deviation(size),
        "effective_ppi": placement.effective_ppi(size),
        "sampling_rate_basis": basis,
        "sampling_rate_reference_mm": list(size) if size else None,
    }
    if extent is None and target.physical_size_mm:
        record["sampling_rate_caveat"] = (
            "Derived from the whole card, because this target does not record the "
            "physical size of the area its fiducials bound. If the fiducials sit "
            "inside the card edges, this under-reports the true sampling rate."
        )
    return record


def _measure_edges(edges: list, placement: ChartPlacement) -> list:
    """Measure every edge once and keep the full record, pass or fail.

    The verdict reduces the field to one number; the differences between edges are
    what diagnose the system. Recorded here rather than inside a metric so the data
    exists even for a profile that declares no SFR criteria at all.
    """
    from deltachrome.imaging.sfr import SFRError, compute_sfr

    records = []
    for edge in edges:
        try:
            result = compute_sfr(edge.pixels, linearize=getattr(edge, "linearize", None))
        except SFRError:
            continue
        relative = placement.to_relative(np.asarray(edge.center_px, dtype=np.float64))
        records.append(
            build_edge_record(edge, result, relative, _channel_sfr(edge))
        )
    return records


def _channel_sfr(edge) -> dict[str, Any]:
    """The same edge measured again in each color record.

    Run for every edge whatever the profile asks for, on the same grounds as the
    per-edge records themselves: the differences are the diagnosis, and re-shooting
    to get them back is not an option. Measured at about 3.5 ms per edge per
    record, so three more passes over twenty edges costs a fifth of a second --
    which is why this is not a switch. It was worth measuring before assuming;
    the guess going in was that it might cost seconds.

    Linearized **per channel**, never through the luminance transform. That maps
    to XYZ and takes Y, so each output is a blend of all three records and any
    difference between them is averaged away -- the same reason channel
    registration insists on it.

    A record that cannot be measured is left out rather than recorded as a
    failure: the luminance measurement is the one the verdict rests on, and it
    already succeeded by the time this runs.
    """
    from deltachrome.imaging.sfr import SFRError, compute_sfr

    linearize = getattr(edge, "linearize_channels", None)
    out: dict[str, Any] = {}
    if edge.pixels.ndim != 3 or edge.pixels.shape[-1] < 3:
        return out
    for channel in ("r", "g", "b"):
        try:
            out[channel] = compute_sfr(
                edge.pixels, channel=channel, linearize=linearize
            )
        except SFRError:
            continue
    return out


def _patch_position(target: TargetDefinition, patch) -> tuple[float, float] | None:
    """Where a patch sits on the target, or nothing if its grid cannot say.

    A target may describe patches it has no regular geometry for, and a guessed
    position is worse than no position: it would put a mark on a field map at a
    place nothing was measured.
    """
    try:
        return tuple(float(v) for v in target.center_relative(patch))  # type: ignore[return-value]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
