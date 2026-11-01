"""Generate FADGI profile JSON from the transcribed conformance tables.

The tables live in ``fadgi_tables_2023.json``, read out of the published PDF by
``extract_fadgi_tables.py`` rather than typed in by hand. Keeping them as data
means a star level cannot drift from its neighbours by a typo, a future edition
is a diff against a table, and the transcription can be re-derived and checked
against the source instead of trusted.

Run: python tools/generate_fadgi_profiles.py

**Not every category is shaped alike**, and the differences are the whole reason
this reads a table rather than assuming one:

* Bound Volumes: Rare and Special Materials, and its unbound equivalent, print a
  1-Star column and leave every cell in it empty. They offer 2, 3 and 4-star
  only, and the level that is missing is the *lowest*.
* Documents (Unbound): Modern Textual Records has no performance levels at all --
  one specification, and an aim range of 20 <= L* <= 100 that no other category
  states.
* X-Ray Film accepts DICOM as a master format; most categories accept PDF/A.
* Several categories specify no color accuracy, no noise, no gain modulation, no
  misregistration or no reproduction scale at all. An absent row is not a
  threshold of zero and not a threshold to invent -- the metric is simply not part
  of that category's conformance criteria.
* The film categories mark some rows **informative only**. See ``INFORMATIVE``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src/deltachrome/profiles/definitions"
TABLES = HERE / "fadgi_tables_2023.json"
EDITION = "3rd Edition, May 2023"

#: Names real ICC profiles use for the spaces the tables list, added beside the
#: table's own spelling so an embedded profile description matches.
ALIASES = {
    "ProPhoto": ["ProPhoto", "ProPhoto RGB"],
    "ECIRGB_v2": ["ECIRGB_v2", "eciRGB v2"],
}

#: Profile id stems, pinned per section rather than derived from the title, so a
#: change of wording in a future edition cannot silently rename a profile.
SLUGS = {
    "3.1": "bound_volumes_rare_special",
    "3.2": "bound_volumes_general",
    "3.3": "documents_unbound_manuscripts_rare_special",
    "3.4": "documents_unbound_general",
    "3.5": "documents_unbound_modern_textual",
    "3.6": "oversize_items",
    "3.7": "newspapers",
    "3.8": "prints_photographs",
    "3.9": "photographic_transparencies_35mm_to_4x5",
    "3.10": "photographic_transparencies_4x5_and_larger",
    "3.11": "photographic_negatives_35mm_to_4x5",
    "3.12": "photographic_negatives_4x5_and_larger",
    "3.13": "paintings_two_dimensional_art",
    "3.14": "xray_film",
    "3.15": "microfilm",
}

#: Table rows FADGI marks with an asterisk, described in the tables' own note as
#: "informative only, and are not required parameters for each FADGI star level".
#: Only the six film categories use the convention. Such a metric is still
#: measured and reported -- the reading is useful -- but it carries no pass
#: condition, so it reads MEASURED and cannot fail a run against a criterion the
#: standard does not impose.
INFORMATIVE = {
    "SFR Response at Nyquist": ("sfr_response_at_nyquist",),
    "Highlight/Shadow, Tolerance": ("highlight_tolerance", "shadow_tolerance"),
    "Dynamic Range": ("dynamic_range",),
}

#: The order the notes name informative rows in. Not alphabetical: the
#: tolerance rows were added after sfr_response_at_nyquist and the shipped notes
#: list them in the order they were added, which this reproduces.
INFORMATIVE_ORDER = (
    "sfr_response_at_nyquist", "highlight_tolerance", "shadow_tolerance",
    "dynamic_range",
)


def load_tables() -> dict:
    return json.loads(TABLES.read_text(encoding="utf-8"))["categories"]


def spaces_for(cell: list[str]) -> list[str]:
    out: list[str] = []
    for name in cell:
        for alias in ALIASES.get(name, [name]):
            if alias not in out:
                out.append(alias)
    return out


def informative_metrics(spec: dict) -> set[str]:
    return {
        metric
        for label, metrics in INFORMATIVE.items()
        if any(label in row for row in spec.get("informative", []))
        for metric in metrics
    }


def number(value: float) -> str:
    return f"{value:g}"


def metrics_for(spec: dict, i: int) -> list[dict]:
    """Every metric this category specifies at level index ``i``.

    A row the table does not carry produces no metric. That is not the same as a
    metric with no threshold: one says the standard is silent about it, the other
    that the standard measures it without requiring anything.
    """
    unjudged = informative_metrics(spec)

    def row(key):
        values = spec.get(key)
        return None if values is None else values[i]

    out: list[dict] = [
        {
            "metric_id": "master_file_format",
            "algorithm": {"allowed": list(spec["master_formats"])},
            "aggregation": "none",
        }
    ]
    if row("bit_depth") is not None:
        # The depths the table permits, not a floor: "value >= 8" admits 12 and
        # 32, which no FADGI table does. Corrected by hand in every shipped file
        # in 80c8ae7 and only now here, so regenerating no longer reverts it.
        out.append({
            "metric_id": "bit_depth",
            "aggregation": "none",
            "units": "bits",
            "algorithm": {"allowed": [8, 16] if row("bit_depth") <= 8 else [16]},
        })
    if spec.get("color_space"):
        out.append({
            "metric_id": "color_space",
            "algorithm": {"allowed": spaces_for(spec["color_space"][i])},
            "aggregation": "none",
        })
    if row("resolution_ppi") is not None:
        out.append({
            "metric_id": "sampling_frequency",
            "aggregation": "none",
            "pass_condition": f"value >= {number(row('resolution_ppi'))}",
            "units": "ppi",
        })

    # FADGI 2.4.15 and 2.4.16. Informative-only wherever a table carries them,
    # so they are measured and never judged. Added to the shipped profiles by
    # hand, read off the PDF, and generated here since.
    if {"highlight_tolerance", "shadow_tolerance"} <= unjudged:
        for metric_id, end, aim in (
            ("highlight_tolerance", "highlight", "96 +/- 2"),
            ("shadow_tolerance", "shadow", "2.7 +/- 1.5"),
        ):
            out.append({
                "metric_id": metric_id,
                "requires_capability": "tonal_response",
                "algorithm": {"patch_selection": "role == 'achromatic_ramp'"},
                "aggregation": "none",
                "units": "L*",
                "notes": (
                    "FADGI 2.4.15, informative-only for this category. The table's "
                    f"own aim is {aim} L* for the {end} end of the ramp, checked here "
                    "against the measured value directly rather than judged."
                ),
            })
    if row("dynamic_range") is not None:
        out.append({
            "metric_id": "dynamic_range",
            "requires_capability": "tonal_response",
            "algorithm": {"patch_selection": "role == 'achromatic_ramp'"},
            "aggregation": "none",
            "units": "D",
            "notes": (
                "FADGI 2.4.16, informative-only. The table's own figure for this "
                f"level is {row('dynamic_range'):.1f}. What is reported is the "
                "density span of this target's own declared reference ramp, "
                "restricted to patches this capture resolved without clipping -- "
                "FADGI gives no measurement procedure of its own, calling this 'an "
                "emerging area'. N/A for a target whose reference carries no "
                "density column, which is every reflective target."
            ),
        })

    if row("average_color_accuracy") is not None:
        out += [
            {
                "metric_id": "average_color_accuracy",
                "requires_capability": "color_accuracy",
                "algorithm": {"delta_e_variant": "delta_e_2000",
                              "patch_selection": "all patches"},
                "aggregation": "mean",
                "pass_condition": f"value < {number(row('average_color_accuracy'))}",
                "units": "dE2000",
            },
            {
                "metric_id": "color_accuracy_90th_percentile",
                "requires_capability": "color_accuracy",
                "algorithm": {"delta_e_variant": "delta_e_2000",
                              "patch_selection": "all patches",
                              "percentile_method": "linear"},
                "aggregation": "percentile_90",
                "pass_condition": f"value < {number(row('color_accuracy_90th'))}",
                "units": "dE2000",
                "notes": "NOT the maximum. FADGI does not specify an interpolation "
                         "method, so it is stated explicitly and reported.",
            },
        ]

    if row("white_balance") is not None:
        out.append({
            "metric_id": "white_balance",
            "requires_capability": "white_balance",
            "algorithm": {"delta_e_variant": "delta_e_2000_ab",
                          "patch_selection": "role == 'achromatic_ramp'"},
            "aggregation": "max_per_patch",
            "pass_condition": f"value < {number(row('white_balance'))}",
            "units": "dE2000(a*b*)",
            "notes": "'For any given gray patch', so the worst patch must clear the "
                     "threshold. CIEDE2000 with the L* component eliminated, as "
                     "section 2.4.4 defines it; the tables' 'dE(a*b*)' names the "
                     "same quantity.",
        })
    if row("tone_response") is not None:
        out.append({
            "metric_id": "tone_response_oecf",
            "requires_capability": "tonal_response",
            "algorithm": {"measure": "delta_l_2000",
                          "patch_selection": "role == 'achromatic_ramp'"},
            "aggregation": "max_per_patch",
            "pass_condition": f"value < {number(row('tone_response'))}",
            "units": "dL*2000",
            "notes": "dL*2000 includes the CIEDE2000 S_L weighting; it is not plain |dL*|.",
        })
    if row("lightness_uniformity") is not None:
        out.append({
            "metric_id": "lightness_uniformity",
            "requires_capability": "lightness_uniformity",
            "algorithm": {"patch_selection": {"roles": ["achromatic_border"]},
                          "family_selection": "lightest"},
            "aggregation": "none",
            "pass_condition": f"value < {number(row('lightness_uniformity'))}",
            "units": "% (std dev / mean L*)",
            "notes": "FADGI states the units as standard deviation divided by mean, so "
                     "this is a percentage.",
        })

    if row("noise_upper") is not None:
        out.append({
            "metric_id": "noise",
            "requires_capability": "tonal_response",
            "algorithm": {"patch_selection": {"roles": ["achromatic_border",
                                                        "achromatic_ramp",
                                                        "achromatic_other"]},
                          "lower_limit": 0.25},
            "aggregation": "max",
            "pass_condition": f"value < {number(row('noise_upper'))}",
            "units": "std dev of L*",
            "notes": "FADGI also sets a lower limit of > 0.25 at every star level, raised "
                     "as a warning rather than a failure: near-zero noise indicates noise "
                     "reduction, which removes detail.",
        })

    for key, metric_id, label in (
        ("gain_highlight", "gain_modulation_highlight",
         "highlight patches (average L* 85-95)"),
        ("gain_other", "gain_modulation_other", "all other patches"),
    ):
        if row(key) is None:
            continue
        low, high = row(key)
        out.append({
            "metric_id": metric_id,
            "requires_capability": "tonal_response",
            "algorithm": {"patch_selection": "role == 'achromatic_ramp'",
                          "min_delta_l": 10.0,
                          "evaluation_range": [5.0, 95.0],
                          "l_star_range": [85.0, 95.0]},
            "aggregation": "worst_case",
            "pass_condition": f"{number(low)} <= value <= {number(high)}",
            "units": "gain",
            "notes": f"FADGI requires gain between {number(low)} and {number(high)} for "
                     f"{label}. Gain is the local slope of the OECF; pairing follows "
                     "ISO/TS 19264-1 6.2, which computes incremental gain between gray "
                     "steps at least 10 L* apart. Aim is 1.0, and the band fails in both "
                     "directions -- too flat loses tonal separation, too steep clips "
                     "sooner.",
        })

    if row("misregistration") is not None:
        out.append({
            "metric_id": "color_channel_misregistration",
            "requires_capability": "color_registration",
            "aggregation": "max",
            "pass_condition": f"value < {number(row('misregistration'))}",
            "units": "pixels",
        })
    if row("sfr10") is not None:
        out.append({
            "metric_id": "sfr10_sampling_efficiency",
            "requires_capability": "sfr",
            "algorithm": {"edge_fit": "linear", "esf_supersample_bin_width": 0.25,
                          "roi_size_px": 100},
            "aggregation": "worst_edge",
            "pass_condition": f"value > {row('sfr10'):.0f}",
            "units": "% of Nyquist",
            "notes": "FADGI states this as a percentage ratio; the frequency at 10% "
                     "modulation as a percentage of the Nyquist (half-sampling) "
                     "frequency.",
        })
    if row("sfr_nyquist") is not None:
        metric = {
            "metric_id": "sfr_response_at_nyquist",
            "requires_capability": "sfr",
            "aggregation": "worst_edge",
            "units": "modulation",
        }
        if "sfr_response_at_nyquist" in unjudged:
            metric["notes"] = (
                f"FADGI marks this row informative only for this category: 'not a "
                f"required parameter for each FADGI star level'. The reading is "
                f"reported and the table's figure for this level is < "
                f"{number(row('sfr_nyquist'))}, but it is not judged, because failing a "
                "run on a criterion the standard does not impose would misrepresent the "
                "standard just as surely as passing one it does."
            )
        else:
            metric["pass_condition"] = f"value < {number(row('sfr_nyquist'))}"
        out.append(metric)
    if spec.get("sfr50"):
        low, high = spec["sfr50"][i]
        out.append({
            "metric_id": "sfr50",
            "requires_capability": "sfr",
            "algorithm": {"edge_fit": "linear", "esf_supersample_bin_width": 0.25},
            "aggregation": "worst_edge",
            "pass_condition": f"{number(low)} <= value <= {number(high)}",
            "units": "% of half-sampling freq",
            "notes": "A two-sided band: too low is soft, too high means the response "
                     "extends so far that detail above Nyquist aliases into the image.",
        })
    if row("sharpening") is not None:
        out.append({
            "metric_id": "sharpening",
            "requires_capability": "sfr",
            "aggregation": "max",
            "pass_condition": f"value < {number(row('sharpening'))}",
            "units": "max modulation",
        })
    if row("reproduction_scale") is not None:
        out.append({
            "metric_id": "reproduction_scale_accuracy",
            "requires_capability": "reproduction_scale",
            "aggregation": "max",
            "pass_condition": f"value < {number(row('reproduction_scale'))}",
            "units": "% difference from header ppi",
        })
    return out


def notes_for(spec: dict, section: str, i: int) -> str:
    unjudged = informative_metrics(spec)
    notes = (
        f"Thresholds read from FADGI {EDITION}, section {section} "
        f"({spec['title']}), pages {spec['page']}-{spec['last_page']}. "
    )
    if spec.get("sfr50"):
        low, high = spec["sfr50"][i]
        notes += (
            f"SFR50 is modeled as the two-sided band [>{number(low)}%, "
            f"<{number(high)}%] of half sampling frequency. "
        )
    if unjudged:
        notes += (
            "Rows this category marks informative-only are measured and reported "
            "without a pass condition, so they read MEASURED: "
            + ", ".join(m for m in INFORMATIVE_ORDER if m in unjudged)
            + ". "
        )
    else:
        notes += "This category marks no row informative-only. "
    notes += (
        "A row the table does not carry produces no metric at all, which is different "
        "from one measured without a threshold. A metric with no plugin reports NOT "
        "IMPLEMENTED and forces INCOMPLETE rather than a pass."
    )
    if spec.get("average_color_accuracy") is None:
        notes += (
            " This category specifies NO color accuracy metric: FADGI notes that film "
            "content is not scene referred, so there is no unambiguous original to judge "
            "colorimetric accuracy against. Delta E measured against a target's own "
            "reference file is still useful and can be run with a custom profile, but it "
            "is not a FADGI conformance criterion here and must not be reported as one."
        )
    if spec.get("provisional"):
        notes += (
            " FADGI marks this category's measurement parameters provisional: the values "
            "were compared against optical density equivalents up to 2.2 and are to be "
            "adopted until there is evidence to the contrary."
        )
    return notes


def build() -> list[Path]:
    tables = load_tables()
    written: list[Path] = []
    for section, spec in tables.items():
        slug = SLUGS[section]
        levels = spec["stars"] or [None]
        for i, star in enumerate(levels):
            profile_id = (
                f"fadgi_{star}star_{slug}_2023" if star else f"fadgi_{slug}_2023"
            )
            display_level = f"{star}-Star — " if star else ""
            reference = {
                "body": "FADGI",
                "document_version": EDITION,
                "edition": "3rd",
                "published": "2023-05-09",
                "material_category": spec["title"],
                "section": section,
                "source_url": json.loads(TABLES.read_text(encoding="utf-8"))["_source_url"],
                "thresholds_transcribed_from": (
                    f"Section {section} conformance table, pages {spec['page']}-{spec['last_page']}" if spec['last_page'] > spec['page'] else f"Section {section} conformance table, page {spec['page']}"
                ),
                "provisional": bool(spec.get("provisional")),
            }
            if star:
                reference["star_level"] = star
            else:
                # No performance levels at all: one specification. Recorded as a
                # named level so a list has something to show beside the others.
                reference["quality_level"] = "Single level"

            metrics = metrics_for(spec, i)
            has_colour = any(m["metric_id"] == "average_color_accuracy" for m in metrics)
            profile = {
                "$schema_version": 1,
                "profile_id": profile_id,
                "display_name": (
                    f"FADGI {display_level}{spec['title']} (2023 3rd Ed.)"
                ),
                "standard_reference": reference,
                "applicable_target_capabilities_required": (
                    ["color_accuracy", "tonal_response", "white_balance"]
                    if has_colour else ["tonal_response", "white_balance"]
                ),
                "applicable_target_capabilities_optional": [
                    "sfr", "color_registration", "lightness_uniformity",
                    "reproduction_scale",
                ],
                "metrics": metrics,
                "overall_pass_rule": "all applicable metrics must pass",
                "profile_editable": True,
                "profile_locked_fields": ["standard_reference"],
                "notes": notes_for(spec, section, i),
            }
            path = OUT / f"{profile_id}.json"
            path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
            written.append(path)
    return written


if __name__ == "__main__":
    for path in build():
        print("wrote", path.name)
