"""Generate the three Metamorfoze profiles from its transcribed specification table.

Values are transcribed from Metamorfoze Preservation Imaging Guidelines, Image
Quality, version 2.0 (April 2025), section 1.6. As with the FADGI generator,
keeping the table here rather than hand-writing the JSON means one level cannot
drift from its neighbours by a typo.

Two entries in that table were readings rather than quotations when this file was
first written, marked as uncertain in the profile's own notes. Both are now
**confirmed against the guideline's own prose**, not merely against the table:

**MTF50.** Stated in the table as ">= 45% of the minimum required lp/mm at MTF10".
Section 2.14 settles the reading that mattered: "Regardless of the sampling rate
size, there are always minimum requirements for sampling efficiency (>= 85%) and
MTF50 performance (>= 45% of the required MTF10 performance, expressed in lp/mm)"
-- "regardless of the sampling rate size" is explicit that the requirement scales
with whatever rate the capture was actually made at, not a constant anchored to
the 300 ppi standard rate. Section 2.17 gives the other half: MTF10 performance is
itself defined as a *percentage* of Nyquist (85%), with "5 lp/mm at 300 ppi" given
only as what that percentage happens to equal at that one rate ("If a measured
MTF10 performance at Nyquist is 5 line pairs (cycles) per millimetre, the sampling
efficiency is ... 85%"). So both halves of the chain are rate-relative and the
sampling rate cancels: 0.45 x 0.85 = 38.25% of the half-sampling frequency, at
every sampling rate. The alternative this file used to flag as untested -- anchoring
"minimum required" to a flat 300 ppi-derived 2.259 lp/mm regardless of the
capture's own rate, which would let a 600 ppi capture pass with half the relative
sharpness -- is the one the guideline's own "regardless of the sampling rate size"
rules out. Shipped in percent-of-Nyquist because that is the form in which the
ambiguity disappears.

**"Maximum L* value for assessing neutrals"** (L* 5 / 20 / 30). Confirmed against
the guideline's own prose beside the table: "For Metamorfoze Full, the neutrals
are assessed up to L* 5. For Metamorfoze Light, the neutrals are assessed up to
L* 20 ... For Metamorfoze Extra Light, neutrals are assessed up to L* 30." Read as
the darkest tone the neutral ladder is judged down to, so it becomes a
``min_reference_L`` on the patch selection for the metrics assessed on neutrals.
The lighter levels do not hold the deep shadows to the same standard.

Run: python tools/generate_metamorfoze_profiles.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/deltachrome/profiles/definitions"
#: The bare directory this pointed to for some time resolves to nothing --
#: found while re-confirming the two readings above against the document itself,
#: which needed the real file.
SOURCE_URL = (
    "https://www.metamorfoze.nl/sites/default/files/documents/"
    "Preservation%20Imaging%20Guidelines%20English%202.0%2C%20April%202025.pdf"
)
EDITION = "version 2.0, April 2025"

WORKING = ["ECIRGB_v2", "eciRGB v2"]
TIFF_JP2_JPEG = ["TIFF", "JPEG 2000", "JPEG"]

#: 0.45 of the MTF10 requirement, which is itself 0.85 of Nyquist. See the module
#: docstring: the sampling rate cancels, so this is the same number at any rate.
MTF50_PERCENT_OF_NYQUIST = 0.45 * 0.85 * 100.0

#: Illumination across the image plane, banded by the size of the originals the
#: system is being qualified for. Not a property of any capture -- the measurement
#: is made on a blank white sheet -- so it is declared per run.
A_SERIES_MM = {"A3": 420.0, "A2": 594.0, "A1": 841.0, "A0": 1189.0}
BAND_LABELS = {
    "A3": "originals up to A3",
    "A2": "originals over A3, up to A2",
    "A1": "originals over A2, up to A1",
    "A0": "originals over A1, up to A0",
}

LEVELS = {
    "full": {
        "display": "Metamorfoze Full",
        "file_formats": TIFF_JP2_JPEG,
        "color_space": WORKING,
        "bit_depth": 8,               # "16/8" -- 8 is the floor the table states
        "neutral_floor_L": 5.0,
        "white_balance": 3.0,
        "exposure": 2.0,
        "noise": 1.6,
        "illumination": {"A3": 3.0, "A2": 4.0, "A1": 5.0, "A0": 6.0},
        "color_accuracy_mean": 3.0,
        "color_accuracy_max": 7.0,
        "misregistration": 0.35,
        "distortion": 2.0,
    },
    "light": {
        "display": "Metamorfoze Light",
        "file_formats": TIFF_JP2_JPEG,
        "color_space": WORKING + ["Adobe RGB (1998)"],
        "bit_depth": 8,
        "neutral_floor_L": 20.0,
        "white_balance": 3.0,
        "exposure": 2.0,
        "noise": 1.6,
        "illumination": {"A3": 3.0, "A2": 4.0, "A1": 5.0, "A0": 6.0},
        "color_accuracy_mean": 4.0,
        "color_accuracy_max": 14.0,
        "misregistration": 0.50,
        "distortion": 2.0,
    },
    "extra_light": {
        "display": "Metamorfoze Extra Light",
        "file_formats": TIFF_JP2_JPEG + ["PDF"],
        "color_space": WORKING + ["Adobe RGB (1998)", "sRGB"],
        "bit_depth": 8,
        "neutral_floor_L": 30.0,
        "white_balance": 5.0,
        "exposure": 4.0,
        "noise": 2.0,
        # "Not specified" above A3, so only the one band is declared. A run
        # declaring a larger original is refused rather than judged against a
        # threshold the standard does not state.
        "illumination": {"A3": 5.0},
        "color_accuracy_mean": None,   # "Not specified"
        "color_accuracy_max": None,
        "misregistration": 0.50,
        "distortion": 4.0,
    },
}

NEUTRALS = ["achromatic_ramp"]


def neutral_selection(level: dict) -> dict:
    return {"roles": NEUTRALS, "min_reference_L": level["neutral_floor_L"]}


def illumination_bands(level: dict) -> dict:
    return {
        "parameter": "original_size",
        "description": (
            "Largest originals this system is being qualified for. Metamorfoze bands "
            "the illumination tolerance by it, and the measurement is made on a blank "
            "white sheet that has no size of its own."
        ),
        "bands": [
            {
                "band": band,
                "label": BAND_LABELS[band],
                "max_mm": A_SERIES_MM[band],
                "pass_condition": f"value <= {limit:g}",
            }
            for band, limit in level["illumination"].items()
        ],
    }


def metrics_for(level: dict) -> list[dict]:
    specs: list[dict] = [
        {
            "metric_id": "master_file_format", "aggregation": "none",
            "algorithm": {"allowed": level["file_formats"]},
            "notes": "TIFF 6.0 uncompressed is the assumed master; the others need "
                     "the client's approval, which this cannot check.",
        },
        {
            "metric_id": "color_space", "aggregation": "none",
            "algorithm": {"allowed": level["color_space"]},
        },
        {
            "metric_id": "bit_depth", "aggregation": "none", "units": "bits",
            # A floor admits any depth above it, which the table does not: "16/8"
            # names two depths, not a range. The generator
            # writes it so, matching what is shipped.
            "algorithm": {"allowed": [8, 16] if level["bit_depth"] <= 8 else [16]},
        },
        {
            "metric_id": "white_balance", "requires_capability": "white_balance",
            "algorithm": {
                "delta_e_variant": "delta_e_2000_ab",
                "patch_selection": neutral_selection(level),
            },
            "aggregation": "max_per_patch", "units": "dE2000(a*b*)",
            "pass_condition": f"value <= {level['white_balance']:g}",
            "notes": "CIEDE2000 without the luminance term, which is what Metamorfoze "
                     "specifies -- not the Euclidean dE(a*b*). Near the neutral axis "
                     "the two differ by about 6%, in the permissive direction if the "
                     "Euclidean one is used.",
        },
        {
            "metric_id": "tone_response_oecf", "requires_capability": "tonal_response",
            "algorithm": {"measure": "delta_l", "patch_selection": neutral_selection(level)},
            "aggregation": "max_per_patch", "units": "dL*",
            "pass_condition": f"value <= {level['exposure']:g}",
            "notes": "Plain dL* (CIE2000 with S_L = 1), not FADGI's dL*2000. At the "
                     "ends of the ramp the two differ by more than a factor of 1.5.",
        },
        {
            "metric_id": "gain_modulation_highlight", "requires_capability": "tonal_response",
            "algorithm": {
                "patch_selection": neutral_selection(level),
                "l_star_range": [85.0, 95.0], "min_delta_l": 10.0,
                "evaluation_range": [5.0, 95.0],
            },
            "aggregation": "worst_case", "units": "gain",
            "pass_condition": "0.8 <= value <= 1.1",
        },
        {
            "metric_id": "gain_modulation_other", "requires_capability": "tonal_response",
            "algorithm": {
                "patch_selection": neutral_selection(level),
                "min_delta_l": 10.0, "evaluation_range": [5.0, 95.0],
            },
            "aggregation": "worst_case", "units": "gain",
            "pass_condition": "0.6 <= value <= 1.4",
        },
        {
            "metric_id": "noise", "requires_capability": "tonal_response",
            "algorithm": {"patch_selection": neutral_selection(level), "lower_limit": 0.25},
            "aggregation": "max", "units": "std dev of L*",
            "pass_condition": f"value <= {level['noise']:g}",
        },
        {
            "metric_id": "illumination_non_uniformity", "requires_capability": "flat_field",
            "aggregation": "max", "units": "dL*",
            "pass_condition_by": illumination_bands(level),
            "notes": "Measured on a frame-filling white sheet at about L* 95, not on a "
                     "chart. A chart occupies the middle of the frame; the fall-off "
                     "this is about is at the corners.",
        },
        {
            "metric_id": "white_balance_non_uniformity", "requires_capability": "flat_field",
            "aggregation": "max", "units": "dE2000(a*b*)",
            "pass_condition": f"value <= {level['white_balance']:g}",
            "notes": "White balance across the entire image plane, on the same "
                     "frame-filling white sheet. Catches a cast that varies across the "
                     "frame, which a white balance read at the center cannot see.",
        },
        {
            "metric_id": "sampling_frequency", "aggregation": "none", "units": "ppi",
            "pass_condition": "value >= 300",
            "notes": "300 ppi for originals of DIN A5 and larger. Originals smaller "
                     "than A5 may go up to 600 ppi; that ceiling is not modeled here, "
                     "since it is a permission rather than a requirement.",
        },
        {
            "metric_id": "reproduction_scale_accuracy",
            "requires_capability": "reproduction_scale",
            "aggregation": "max", "units": "% difference from header ppi",
            "pass_condition": "value <= 2",
            "notes": "Metamorfoze's 'difference between claimed and obtained sampling "
                     "rate'. The claimed rate is the file header's; the obtained rate "
                     "is measured off the chart's own printed scale.",
        },
        {
            "metric_id": "sfr10_sampling_efficiency", "requires_capability": "sfr",
            "aggregation": "worst_edge", "units": "% of Nyquist",
            "pass_condition": "value >= 85",
            "algorithm": {"edge_aggregation": "worst"},
            "notes": "Metamorfoze applies this to both dimensions and to every MTF "
                     "target in the image plane, so the worst edge governs rather "
                     "than the mean OpenDICE reports.",
        },
        {
            "metric_id": "mtf50", "requires_capability": "sfr",
            "aggregation": "worst_edge", "units": "% of half-sampling freq",
            "algorithm": {"frequency_units": "% of half-sampling freq",
                          "edge_aggregation": "worst"},
            "pass_condition": f"value >= {MTF50_PERCENT_OF_NYQUIST:g}",
            "notes": "Stated as '>= 45% of the minimum required lp/mm at MTF10'. That "
                     "MTF10 requirement is sampling efficiency >= 85% of Nyquist, so "
                     "the sampling rate cancels and the requirement is 0.45 x 0.85 = "
                     "38.25% of the half-sampling frequency at any rate. CONFIRMED "
                     "against the guideline's own section 2.14: 'regardless of the "
                     "sampling rate size' the requirement is stated as a fraction of "
                     "MTF10 performance, not anchored to the 300 ppi standard rate -- "
                     "which is what would make it a flat 2.259 lp/mm instead. Set "
                     "frequency_units to lp/mm to see the same measurement in the "
                     "unit the standard writes it in.",
        },
        {
            "metric_id": "sharpening", "requires_capability": "sfr",
            "aggregation": "worst_edge", "units": "max modulation",
            "algorithm": {"edge_aggregation": "worst"},
            "pass_condition": "value <= 1.05",
        },
        {
            "metric_id": "color_channel_misregistration",
            "requires_capability": "color_registration",
            "aggregation": "max", "units": "pixels",
            "algorithm": {"registration_aggregation": "worst", "component": "per_axis"},
            "pass_condition": f"value <= {level['misregistration']:g}",
        },
        {
            "metric_id": "geometric_distortion",
            "requires_capability": "geometric_distortion",
            "aggregation": "max", "units": "%",
            "pass_condition": f"value <= {level['distortion']:g}",
        },
    ]

    if level["color_accuracy_mean"] is not None:
        specs.insert(9, {
            "metric_id": "average_color_accuracy", "requires_capability": "color_accuracy",
            "algorithm": {"delta_e_variant": "delta_e_2000", "patch_selection": "all patches"},
            "aggregation": "mean", "units": "Delta E2000",
            "pass_condition": f"value <= {level['color_accuracy_mean']:g}",
        })
        specs.insert(10, {
            "metric_id": "color_accuracy_max", "requires_capability": "color_accuracy",
            "algorithm": {"delta_e_variant": "delta_e_2000", "patch_selection": "all patches"},
            "aggregation": "max", "units": "Delta E2000",
            "pass_condition": f"value <= {level['color_accuracy_max']:g}",
            "notes": "Metamorfoze states a maximum where FADGI states a 90th "
                     "percentile. One bad patch decides it.",
        })
    return specs


def build(key: str, level: dict) -> dict:
    capabilities = sorted({
        spec["requires_capability"] for spec in metrics_for(level)
        if spec.get("requires_capability")
    })
    return {
        "$schema_version": 1,
        "profile_id": f"metamorfoze_{key}_2025",
        "display_name": f"{level['display']} (2.0, April 2025)",
        "standard_reference": {
            "body": "Metamorfoze",
            "document_version": EDITION,
            "quality_level": level["display"],
            "source_url": SOURCE_URL,
        },
        "applicable_target_capabilities_required": [
            c for c in capabilities if c in {"tonal_response", "white_balance"}
        ],
        "applicable_target_capabilities_optional": [
            c for c in capabilities
            if c not in {"tonal_response", "white_balance"}
        ],
        "metrics": metrics_for(level),
        "overall_pass_rule": "all applicable metrics must pass",
        "profile_editable": True,
        "profile_locked_fields": ["standard_reference"],
        "notes": (
            f"{level['display']}, from the specification table in section 1.6 of "
            f"Metamorfoze Preservation Imaging Guidelines {EDITION}.\n\n"
            "No single target carries everything this profile asks for. Illumination "
            "and white balance across the image plane are measured on a frame-filling "
            "white sheet (target flat_field_v1); color and tone come from a patch "
            "chart; sharpness, registration and distortion from a chart carrying "
            "slanted edges and printed scales. Run them together as a session -- "
            "metrics no capture reaches are reported as such and never counted as "
            "failures.\n\n"
            "Two entries are readings rather than quotations and say so in their own "
            "notes: the MTF50 requirement, and the 'maximum L* value for assessing "
            "neutrals' which is applied here as the darkest tone the ladder is judged "
            "down to.\n\n"
            "'Artefacts' is not modeled at any level: Full and Light state N/A, and "
            "Extra Light states a prose requirement that text remain readable, which "
            "is not machine-checkable."
        ),
    }


def main() -> int:
    for key, level in LEVELS.items():
        profile = build(key, level)
        path = OUT / f"{profile['profile_id']}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(f"  {path.name}: {len(profile['metrics'])} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
