"""Plain-text report.

Written for someone who has to defend the result. Provenance first (which chart,
which reference file, which illuminant, which ICC profile), then the verdict, then
the numbers behind it, then everything that could undermine it. Warnings are never
buried below the pass/fail line.
"""

from __future__ import annotations

from typing import Iterable

from deltachrome.pipeline import AnalysisResult
from deltachrome.imaging.edges import sfr_direction
from deltachrome.metrics.base import MetricResult
from deltachrome.profiles.evaluator import criteria_are_local

RULE = "=" * 78
THIN = "-" * 78


#: How the verdict line qualifies itself. Two states rather than one, because a run
#: loses its conformance claim two ways and they mean different things to do next.
#: A nominal reference is a fact about the evidence and is fixed by measuring the
#: chart; a local guideline is a decision about what to demand. The second says
#: what the verdict is *against* rather than that it was met -- it qualifies a FAIL
#: exactly as it qualifies a PASS, and "MEETS A LOCAL GUIDELINE" beside the word
#: FAIL says the opposite of what happened.
_VERDICT_ADVISORY = "  (ADVISORY -- NOT A CONFORMANCE TEST)"
_VERDICT_LOCAL = "  (AGAINST A LOCAL GUIDELINE, NOT A PUBLISHED STANDARD)"


def _verdict_suffix(*, claim: bool, evidence_sound: bool) -> str:
    if claim:
        return ""
    return _VERDICT_LOCAL if evidence_sound else _VERDICT_ADVISORY


def render(result: AnalysisResult, *, worst_n: int = 8, show_all_patches: bool = False) -> str:
    run = result.run
    out: list[str] = []
    add = out.append

    add(RULE)
    add(f"  Deltachrome  Δχ    analysis {run.run_id}    {run.created_utc}")
    add(RULE)
    add("")
    add(f"  Image      {run.image['filename']}  "
        f"({run.image['width']}x{run.image['height']}, {run.image['bit_depth']}-bit)")
    cm = run.color_management
    add(f"  ICC        {cm['icc_profile']}")
    if cm.get("transform_kind") == "lut":
        lut = cm.get("lut", {})
        grid = "x".join(str(g) for g in (lut.get("clut_grid") or []))
        add(f"             LUT transform via {lut.get('tag')} "
            f"({lut.get('tag_type')}, CLUT {grid}, {lut.get('pcs_encoding')})")
    add(f"  Target     {run.target['display_name']}  ({run.target['patch_count']} patches)")
    # How the geometry came to be: a fact for the header, where it used to be a
    # warning on every automatic run. Absent from records written before it was.
    placed = (run.geometry or {}).get("placed")
    if placed is not None:
        how = {
            "automatic": "automatically",
            "automatic_then_moved": "automatically, then moved by hand",
        }.get(placed, "by hand")
        if (run.geometry or {}).get("identified"):
            how += "; chart identified from the capture"
            searched = (run.geometry or {}).get("searched_targets")
            if searched is not None:
                how += f" among {len(searched)} owned target(s)"
        add(f"  Placed     {how}")
    reference_basis = run.reference.get("basis", "per_unit_measurement")
    # Three states, not two. A population mean is not a measurement of this chart
    # and is not a printed aim either, and a reader who could act on the difference
    # is told which one they have. Anything unrecognised reads as the nominal case,
    # which is the safe direction for a label as much as for the claim.
    label = {
        "per_unit_measurement": "per-unit measurement",
        "population_mean": "BATCH MEAN OF MEASURED UNITS",
    }.get(reference_basis, "MANUFACTURER NOMINAL")
    add(f"  Reference  {_basename(run.reference['path'])}  "
        f"({run.reference['sample_count']} samples, {label})")
    tables = run.reference.get("tables_available") or []
    if len(tables) > 1:
        add(f"             table {run.reference.get('table_index')} of {len(tables)}: "
            + "; ".join(
                f"{t['table']}={t['rows']} rows"
                + (f" {t['measurement_mode']}" if t.get("measurement_mode") else "")
                for t in tables
            ))
    prov = run.reference["provenance"]
    if prov:
        add(f"             {_provenance_line(prov, run.reference['measurement_mode'])}")
    add(f"  Profile    {run.profile['display_name']}")
    reference = run.profile.get("standard_reference") or {}
    if criteria_are_local(run.profile):
        # The guideline's own display name is on the line above. What follows it is
        # that it is one -- not which standard it was derived from, and not the
        # list of departures from that standard, both of which are a connection to
        # a table the operator may have picked only as a convenient starting point.
        # Nothing is lost that this report does not print anyway: every metric's
        # own criterion is in the table below, which is what the run was judged by.
        add("             LOCAL GUIDELINE -- not a published standard")
    roles = run.target.get("role_counts", {})
    if roles:
        add("  Roles      "
            + ", ".join(f"{n} {role}" for role, n in sorted(roles.items(), key=lambda kv: -kv[1])))
        uniformity = run.target.get("uniformity_patches") or []
        if uniformity:
            add(f"             uniformity set resolved from reference data: "
                f"{len(uniformity)} patches ({', '.join(uniformity[:8])}"
                + (" ...)" if len(uniformity) > 8 else ")"))
    ppi = run.geometry.get("effective_ppi")
    if ppi:
        basis = run.geometry.get("sampling_rate_basis")
        note = " (from the whole card -- may under-report)" if basis == "physical_size_mm" else ""
        add(f"  Placement  {run.geometry['placement_method']}, ~{ppi:.0f} ppi{note}")
    add("")

    add(THIN)
    add(f"  RESULT: {run.overall}" + _verdict_suffix(
        claim=getattr(run, "conformance_claim", True),
        evidence_sound=(run.reference or {}).get("basis", "per_unit_measurement")
        == "per_unit_measurement",
    ))
    coverage = (run.profile or {}).get("coverage_line")
    if coverage:
        add(f"  COVERAGE: {coverage}")
    add(f"  {_wrap(run.overall_explanation, indent=2)}")
    for note in (run.profile or {}).get("coverage_notes") or []:
        add(f"  {_wrap(note, indent=2)}")
    add(THIN)
    add("")

    add("  METRICS")
    add(f"  {'metric':<32}{'value':>10} {'units':<14} {'requirement':<22}{'status':>9}")
    add("  " + "-" * 87)
    for metric in result.evaluation.results:
        add("  " + _metric_line(metric))
    add("")

    long_conditions = [
        (m.metric_id, m.condition)
        for m in result.evaluation.results
        if m.condition and len(m.condition) > 22
    ]
    if long_conditions:
        add("  REQUIREMENTS IN FULL")
        for metric_id, condition in long_conditions:
            add(f"    {metric_id}: {_wrap(condition, indent=6)}")
        add("")

    for metric in result.evaluation.results:
        for note in metric.notes:
            add(f"  note [{metric.metric_id}] {_wrap(note, indent=8)}")
    if any(m.notes for m in result.evaluation.results):
        add("")

    skipped = [m for m in result.evaluation.results if m.outcome != "evaluated"]
    if skipped:
        add("  NOT EVALUATED")
        for metric in skipped:
            add(f"    {metric.metric_id}  [{metric.status}]")
            add(f"      {_wrap(metric.reason or '', indent=6)}")
        add("")

    worst_metric, worst = _worst_patches(result.evaluation.results, worst_n)
    if worst and worst_metric is not None:
        units = worst_metric.units or "value"
        add(
            f"  WORST PATCHES (top {len(worst)} by {units} against reference, "
            f"under {worst_metric.metric_id})"
        )
        add(f"    {'patch':<7}{'role':<20}{units:>8}   measured Lab            reference Lab")
        for pv in worst:
            add(
                f"    {pv.patch_id:<7}{pv.role:<20}{pv.value:>8.2f}   "
                f"{_lab(pv.measured_lab):<22}  {_lab(pv.reference_lab)}"
                + (f"  {','.join(pv.flags)}" if pv.flags else "")
            )
        add("")

    ramp = _ramp_table(result)
    if ramp:
        add("  TONAL RAMP (OECF)")
        out.extend(ramp)
        add("")

    uniformity = next(
        (m for m in result.evaluation.results if m.metric_id == "lightness_uniformity"), None
    )
    if uniformity and uniformity.outcome == "evaluated":
        dist = uniformity.metadata.get("sample_distribution", {})
        add("  LIGHTNESS UNIFORMITY SAMPLING")
        line = f"    {uniformity.metadata.get('sample_point_count')} points"
        if "on_perimeter" in dist:
            line += (f", {dist['on_perimeter']} on the chart perimeter, "
                     f"{dist['in_interior']} interior")
        add(line)
        add(f"    spanning {dist.get('covered_fraction_of_fiducial_quad', 0):.0%} of the "
            f"fiducial quad's area, centroid at {dist.get('centroid')}")
        if dist.get("perimeter_split"):
            add(f"      {_wrap(dist['perimeter_split'], indent=6)}")
        if uniformity.metadata.get("reference_values_available") is False:
            add("    no reference values for these patches: the raw ratio only, with no "
                "chart-corrected figure")
        add("")

    if show_all_patches:
        add("  ALL PATCHES")
        add(f"    {'patch':<7}{'role':<20}{'measured Lab':<24}{'reference Lab':<24}flags")
        for record in run.patches:
            add(
                f"    {record.patch_id:<7}{record.role:<20}"
                f"{_lab(record.measured_lab):<24}{_lab(record.reference_lab):<24}"
                f"{','.join(record.flags)}"
            )
        add("")

    if run.edges:
        add(f"  EDGES ({len(run.edges)} measured; every one kept, whatever the verdict)")
        # ``reads`` rather than ``orient``, and the direction rather than the
        # edge's own geometry -- see ``imaging.edges.sfr_direction``. The column
        # name says which of the two it is, because the record spells the other.
        add(f"    {'edge':<18}{'reads':<11}{'position':<16}{'SFR50%':>8}{'@Nyq':>8}"
            f"{'peak':>7}{'over%':>7}")
        for e in sorted(run.edges, key=lambda r: r.edge_id):
            pos = f"({e.center_relative[0]:+.2f},{e.center_relative[1]:+.2f})"
            fifty = "-" if e.sfr50_pct is None else f"{e.sfr50_pct:.1f}"
            add(f"    {e.edge_id:<18}{sfr_direction(e.orientation):<11}{pos:<16}"
                f"{fifty:>8}"
                f"{e.mtf_at_nyquist:>8.3f}{e.peak_modulation:>7.3f}"
                f"{e.overshoot_pct:>7.2f}")
        add("")

        summary = run.edge_field_summary or {}
        by_orientation = summary.get("by_orientation") or {}
        if by_orientation:
            add("  FIELD PATTERN")
            for orientation, metrics in by_orientation.items():
                fifty = metrics.get("sfr50_pct") or {}
                add(f"    {sfr_direction(orientation):<12}"
                    f"SFR50 mean {fifty.get('mean', 0):.2f}%  "
                    f"range {fifty.get('min', 0):.1f}-{fifty.get('max', 0):.1f}  "
                    f"(n={fifty.get('n', 0)})")
            regions = summary.get("by_region") or {}
            if len(regions) > 1:
                add("    by position: " + ", ".join(
                    f"{name} {(vals.get('sfr50_pct') or {}).get('mean', 0):.1f}%"
                    for name, vals in regions.items()))
            difference = summary.get("orientation_difference")
            if difference:
                ratio = difference.get("as_fraction_of_field_spread")
                # The words swap with the convention and the number does not:
                # the record's "horizontal minus vertical" is a difference
                # between *edges*, and an edge that runs horizontally is the one
                # that reads the vertical resolution.
                add(f"    vertical - horizontal: "
                    f"{difference['sfr50_horizontal_minus_vertical_pct']:+.2f}% SFR50"
                    + (f", {ratio}x the field spread" if ratio is not None else ""))
                add(f"      {_wrap(difference['note'], indent=6)}")
            add("")

    rulers = getattr(run, "rulers", None) or {}
    if rulers.get("rulers"):
        add("  PRINTED SCALE (sampling rate measured from the chart, not the header)")
        add(f"    {'ruler':<26}{'division':>10}{'divisions':>11}{'pitch px':>11}"
            f"{'residual':>10}{'ppi':>10}")
        for name, entry in rulers["rulers"].items():
            add(f"    {name:<26}{entry['division_mm']:>8.4f}mm{entry['divisions_used']:>11}"
                f"{entry['pitch_px']:>11.3f}{entry['fit_residual_px']:>10.3f}"
                f"{entry['ppi']:>10.1f}")
        line = f"    combined {rulers['ppi']:.1f} ppi"
        if len(rulers["rulers"]) > 1:
            line += f"; the chart's scales agree to {rulers['agreement']:.3%}"
        add(line)
        header = (run.image or {}).get("header_ppi")
        if header:
            add(f"    the file's header claims {header[0]:.1f} x {header[1]:.1f} ppi")
        if rulers.get("warning"):
            add(f"    {_wrap(rulers['warning'], indent=4)}")
        add("")

    registration = run.registration_summary or {}
    if registration.get("positions"):
        out.extend(_registration_block(registration))

    caveat = run.reference.get("nominal_caveat")
    if caveat and not getattr(run, "conformance_claim", True):
        add("  ABOUT THE REFERENCE VALUES")
        add(f"    {_wrap(caveat, indent=4)}")
        add("")

    icc_notes = run.color_management.get("icc_notes") or []
    if icc_notes:
        add("  COLOR MANAGEMENT NOTES")
        for note in icc_notes:
            add(f"    * {_wrap(note, indent=6)}")
        add("")

    if run.warnings:
        add("  WARNINGS")
        for warning in run.warnings:
            add(f"    * {_wrap(warning, indent=6)}")
        add("")

    add(THIN)
    add("  Per-patch raw measurements are retained in the saved run record, so this")
    add("  capture can be re-scored against a different profile without re-shooting.")
    add(RULE)
    return "\n".join(out)



def _registration_block(summary: dict) -> list[str]:
    """Channel registration, laid out the way OpenDICE lays it out.

    One column per channel pair in OpenDICE's own order and sign, positions down
    the page, the two axes separated -- so a reader can put this next to a
    GoldenThread screen and compare column for column rather than deriving
    anything. Signed, because the direction is half the diagnosis: a displacement
    that grows with distance from the frame center is the lens, one that is the
    same everywhere is the capture mechanism.
    """
    out = ["  CHANNEL REGISTRATION (displacement between color records, px)"]
    pairs = ["G-B", "G-R", "B-R"]
    out.append(f"    {'axis':<6}{'position':<18}"
               + "".join(f"{p:>9}" for p in pairs)
               + f"{'+/- s.e.':>10}")
    for axis in ("x", "y"):
        for position in summary["positions"]:
            if position["axis"] != axis:
                continue
            values = position["translation_px"]
            flag = "" if position["paired"] else "  unpaired"
            out.append(
                f"    {axis:<6}{position['region']:<18}"
                + "".join(f"{values.get(p, 0.0):>+9.3f}" for p in pairs)
                + f"{position['standard_error_px']:>10.3f}{flag}"
            )
    antisymmetric: dict[str, list[float]] = {p: [] for p in pairs}
    for position in summary["positions"]:
        for pair, value in (position.get("antisymmetric_px") or {}).items():
            antisymmetric[pair].append(abs(value))
    if any(antisymmetric.values()):
        out.append(
            "    antisymmetric term (apparent size difference between records, "
            "not displacement):"
        )
        out.append("      " + ", ".join(
            f"{pair} {sum(v) / len(v):.3f}" for pair, v in antisymmetric.items() if v
        ) + " px mean magnitude")
    vectors = summary.get("displacement_vectors") or {}
    if vectors:
        worst = max(vectors.items(), key=lambda kv: kv[1]["worst_magnitude_px"])
        out.append(
            f"    worst displacement vector: {worst[1]['worst_magnitude_px']:.3f} px "
            f"at {worst[0]} (the per-axis figures above are its projections)"
        )
    rotation = summary.get("rotation_between_records_deg") or {}
    if rotation:
        out.append("    rotation between records: " + ", ".join(
            f"{k.replace('_mean_deg', '')} {v:+.4f} deg"
            for k, v in rotation.items() if k.endswith("mean_deg")
        ))
    out.append("")
    return out


def _metric_line(metric: MetricResult) -> str:
    """One fixed-width row. Long units and conditions are elided rather than allowed
    to push the status column out of alignment -- a report that does not line up is
    a report people stop reading."""
    if metric.outcome != "evaluated":
        return (
            f"{_elide(metric.metric_id, 32):<32}{'-':>10} {'':<14} "
            f"{'':<22}{metric.status:>9}"
        )
    # ``reading`` rather than the number: the file-level metrics read as a word
    # or a whole number, and "-" for a container the metric identified perfectly
    # well is a report saying it did not.
    value = _elide(metric.reading, 10)
    units = _elide(metric.units or "", 14)
    condition = _elide(metric.condition or "(informational)", 22)
    return (
        f"{_elide(metric.metric_id, 32):<32}{value:>10} {units:<14} "
        f"{condition:<22}{metric.status:>9}"
    )


def _elide(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def _worst_patches(
    results: Iterable[MetricResult], n: int
) -> tuple[MetricResult | None, list]:
    """The largest per-patch deviations, and the metric they were measured under.

    Only metrics whose per-patch values are deviations *against a reference* are
    eligible. A metric whose per-patch value is an absolute reading -- the spots of
    a flat field are L*, not an error -- has no worst patch to show: ranking them
    puts the brightest at the top under a heading saying they are the worst.

    The metric is returned with them because the heading has to name what was
    ranked. Tone response is Delta L*2000, not Delta E2000, and the two were
    previously both labeled Delta E2000.
    """
    eligible = [
        metric for metric in results
        if metric.per_patch and any(pv.reference_lab is not None for pv in metric.per_patch)
    ]
    if not eligible:
        return None, []
    chosen = next(
        (m for m in eligible if m.metric_id == "average_color_accuracy"), eligible[0]
    )
    patches = [pv for pv in chosen.per_patch if pv.reference_lab is not None]
    return chosen, sorted(patches, key=lambda pv: -pv.value)[:n]


def _ramp_table(result: AnalysisResult) -> list[str]:
    ramp = [r for r in result.run.patches if r.role == "achromatic_ramp"]
    if not ramp:
        return []
    ramp.sort(key=lambda r: (r.ramp_position is None, r.ramp_position))

    tone = next(
        (m for m in result.evaluation.results if m.metric_id == "tone_response_oecf"), None
    )
    tone_values = {pv.patch_id: pv.value for pv in tone.per_patch} if tone else {}
    wb = next((m for m in result.evaluation.results if m.metric_id == "white_balance"), None)
    wb_values = {pv.patch_id: pv.value for pv in wb.per_patch} if wb else {}

    lines = [f"    {'step':<6}{'patch':<7}{'ref L*':>8}{'meas L*':>9}{'ΔL*2000':>9}{'ΔE(a*b*)':>10}"]
    for record in ramp:
        ref_L = record.reference_lab[0] if record.reference_lab else float("nan")
        meas_L = record.measured_lab[0] if record.measured_lab else float("nan")
        lines.append(
            f"    {str(record.ramp_position or '-'):<6}{record.patch_id:<7}"
            f"{ref_L:>8.2f}{meas_L:>9.2f}"
            f"{tone_values.get(record.patch_id, float('nan')):>9.2f}"
            f"{wb_values.get(record.patch_id, float('nan')):>10.2f}"
        )
    return lines


def _lab(lab: Iterable[float] | None) -> str:
    if lab is None:
        return "-"
    values = list(lab)
    if not values:
        return "-"
    return "(" + ", ".join(f"{v:6.2f}" for v in values) + ")"


def _provenance_line(prov: dict[str, str], mode: str | None) -> str:
    bits = []
    if prov.get("INSTRUMENTATION"):
        bits.append(prov["INSTRUMENTATION"])
    if mode:
        bits.append(mode)
    if prov.get("ILLUMINATION_NAME"):
        bits.append(prov["ILLUMINATION_NAME"])
    if prov.get("OBSERVER_ANGLE"):
        bits.append(f"{prov['OBSERVER_ANGLE']}°")
    if prov.get("CREATED"):
        bits.append(f"measured {prov['CREATED']}")
    return " / ".join(bits) if bits else "(no provenance metadata in file)"


def _basename(path: str | None) -> str:
    if not path:
        return "(none)"
    return path.rsplit("/", 1)[-1]


def _wrap(text: str, indent: int = 0, width: int = 74) -> str:
    import textwrap

    lines = textwrap.wrap(text, width=width - indent)
    if not lines:
        return ""
    pad = " " * indent
    return ("\n" + pad).join(lines)


def render_session(session) -> str:
    """The session report: what each capture contributed, and what none of them did.

    Arranged around coverage rather than around captures, because the question a
    session answers is whether the *system* met the specification -- which chart
    happened to supply which number is the detail underneath.
    """
    out: list[str] = []
    add = out.append

    add(RULE)
    add(f"  Deltachrome  Δχ    session of {len(session.captures)} capture(s)"
        f"    {session.created_utc}")
    add(RULE)
    add("")
    add(f"  Profile    {session.profile.display_name}")
    for capture in session.captures:
        label = ("" if capture.reference_basis == "per_unit_measurement"
                 else "  [NOMINAL REFERENCE]")
        add(f"  Capture    {capture.name}  ->  {capture.target_name}{label}")
    add("")

    add(THIN)
    add(f"  RESULT: {session.overall}" + _verdict_suffix(
        claim=session.conformance_claim,
        evidence_sound=getattr(session, "evidence_is_sound", False),
    ))
    add(f"  COVERAGE: {session.coverage_line}"
        + ("" if session.covers_profile else "  -- profile NOT fully covered"))
    add(f"  {_wrap(session.explanation, indent=2)}")
    add(THIN)
    add("")

    add("  METRICS ACROSS CAPTURES  (* judged per capture, not across them)")
    add(f"  {'metric':<32}{'status':<12}{'measured by':<34}value(s)")
    add("  " + "-" * 92)
    for metric in session.metrics:
        values = ", ".join(
            f"{name}={value:.3f}" for name, value in metric.summary()["values"].items()
        )
        by = ", ".join(metric.summary()["covered_by"]) or "-"
        name = metric.metric_id + ("*" if metric.per_capture else "")
        add(f"  {name:<32}{metric.status:<12}{by[:32]:<34}{values}")
    add("")

    disagreeing = [m for m in session.metrics if (m.disagreement or 0) > 0]
    if disagreeing:
        add("  WHERE CAPTURES DISAGREE")
        add("    Two charts measuring one system should agree about it. Where they do"
            " not, the")
        add("    question is which is wrong -- a misplaced quad, the wrong reference"
            " file, a")
        add("    chart that is not what it claims -- and not what their average is."
            " Metrics")
        add("    describing each file rather than the system are left out: two"
            " captures differ")
        add("    in bit depth or sampling rate by design, and each is judged on its"
            " own.")
        for metric in sorted(disagreeing, key=lambda m: -(m.disagreement or 0)):
            values = metric.summary()["values"]
            add(f"    {metric.metric_id:<32}spread {metric.disagreement:.3f}  "
                + ", ".join(f"{k}={v:.3f}" for k, v in values.items()))
        add("")

    if session.not_carried:
        add("  REACHED BY NO CAPTURE")
        for metric_id in session.not_carried:
            add(f"    {metric_id}")
        add("    Nothing failed on these. No chart in this sitting carries the")
        add("    feature they read; shooting one that does would complete the profile.")
        add("")

    if session.incomplete:
        add("  ATTEMPTED AND INCOMPLETE")
        for metric_id in session.incomplete:
            add(f"    {metric_id}")
        add("    A chart here does carry the feature and the measurement did not")
        add("    complete. Another target would not help; the capture or the")
        add("    placement is where to look.")
        add("")

    if session.warnings:
        add("  WARNINGS")
        for warning in session.warnings:
            add(f"    * {_wrap(warning, indent=6)}")
        add("")

    add(THIN)
    add("  Each capture's own report is unchanged and still available; this combines")
    add("  them without altering any of them.")
    add(RULE)
    return "\n".join(out)
