"""FADGI's two transparency categories, and what separates them.

Thresholds are transcribed from a published table, and a mistyped one is invisible
in use: it silently passes or fails real captures with no symptom to notice. So the
numbers are pinned here against the document rather than against the code that
generated them, and the relationship between the two categories -- which differ in
exactly one row -- is asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from deltachrome.profiles.schema import available_profiles, load_profile

SMALL = "fadgi_{star}star_photographic_transparencies_35mm_to_4x5_2023"
LARGE = "fadgi_{star}star_photographic_transparencies_4x5_and_larger_2023"

# FADGI 3rd Edition, May 2023, section 3.9, from page 54.
#
# SFR Response at Nyquist is deliberately absent from every level below. FADGI
# marks that row with an asterisk in all six film categories, and the tables' own
# note says starred rows are "informative only, and are not required parameters
# for each FADGI star level". It is measured and reported; it is not judged. See
# ``test_the_informative_row_is_measured_but_not_judged``.
PUBLISHED = {
    1: {"sampling_frequency": "value >= 970", "bit_depth": [8, 16],
        "tone_response_oecf": "value < 5.25", "white_balance": "value < 5.25",
        "lightness_uniformity": "value < 8", "color_channel_misregistration": "value < 1.2",
        "sfr10_sampling_efficiency": "value > 60",
        "sfr50": "30 <= value <= 95", "sharpening": "value < 1.15",
        "reproduction_scale_accuracy": "value < 3"},
    2: {"sampling_frequency": "value >= 1950", "bit_depth": [8, 16],
        "tone_response_oecf": "value < 4", "white_balance": "value < 4",
        "lightness_uniformity": "value < 5", "color_channel_misregistration": "value < 0.8",
        "sfr10_sampling_efficiency": "value > 70",
        "sfr50": "35 <= value <= 85", "sharpening": "value < 1.1",
        "reproduction_scale_accuracy": "value < 2.5"},
    3: {"sampling_frequency": "value >= 2940", "bit_depth": [16],
        "tone_response_oecf": "value < 2.75", "white_balance": "value < 2.75",
        "lightness_uniformity": "value < 3", "color_channel_misregistration": "value < 0.5",
        "sfr10_sampling_efficiency": "value > 80",
        "sfr50": "40 <= value <= 75", "sharpening": "value < 1.05",
        "reproduction_scale_accuracy": "value < 2"},
    4: {"sampling_frequency": "value >= 3960", "bit_depth": [16],
        "tone_response_oecf": "value < 1.5", "white_balance": "value < 1.5",
        "lightness_uniformity": "value < 1", "color_channel_misregistration": "value < 0.33",
        "sfr10_sampling_efficiency": "value > 90",
        "sfr50": "45 <= value <= 65", "sharpening": "value < 1.02",
        "reproduction_scale_accuracy": "value < 1"},
}


@pytest.mark.parametrize("star", [1, 2, 3, 4])
def test_thresholds_match_the_published_table(star):
    profile = load_profile(SMALL.format(star=star))
    for metric_id, condition in PUBLISHED[star].items():
        spec = profile.metric(metric_id)
        if isinstance(condition, list):
            # Bit depth is a set of permitted depths rather than a floor: the
            # tables name 8 and 16 and nothing between or beyond them.
            assert spec.algorithm["allowed"] == condition, metric_id
        else:
            assert spec.pass_condition == condition, metric_id


#: What the table prints for the informative row, level by level. Recorded so the
#: figure is not lost -- it is reported beside the reading -- but never applied.
INFORMATIVE_NYQUIST = {1: "0.5", 2: "0.4", 3: "0.3", 4: "0.2"}


@pytest.mark.parametrize("star", [1, 2, 3, 4])
@pytest.mark.parametrize("template", [SMALL, LARGE])
def test_the_informative_row_is_measured_but_not_judged(star, template):
    """Failing a run on a criterion the standard does not impose misrepresents the
    standard just as surely as passing one it does.

    These profiles judged this row until the asterisks were read off the PDF, which
    made them stricter than FADGI. The reading is still taken and still shown --
    response at Nyquist is a useful number -- it simply carries no verdict.
    """
    metric = load_profile(template.format(star=star)).metric("sfr_response_at_nyquist")
    assert metric.pass_condition is None
    assert "informative" in (metric.notes or "").lower()
    assert INFORMATIVE_NYQUIST[star] in (metric.notes or "")


@pytest.mark.parametrize("star", [1, 2, 3, 4])
def test_a_reflective_category_still_judges_response_at_nyquist(star):
    """The convention is confined to the film tables. Prints and Photographs does
    not mark the row, so there it remains a requirement -- which is what makes the
    transparency change a reading of the document rather than a preference."""
    prints = load_profile(f"fadgi_{star}star_prints_photographs_2023")
    assert prints.metric("sfr_response_at_nyquist").pass_condition is not None


@pytest.mark.parametrize("star", [1, 2, 3, 4])
def test_the_two_transparency_categories_differ_only_in_sampling_frequency(star):
    """Same measurement parameters throughout; a smaller original simply has to be
    sampled far more finely to reach the same detail on the page it prints to."""
    small = load_profile(SMALL.format(star=star))
    large = load_profile(LARGE.format(star=star))
    ids = {s.metric_id for s in small.metrics}
    assert ids == {s.metric_id for s in large.metrics}
    differing = {
        metric_id for metric_id in ids
        if small.metric(metric_id).pass_condition != large.metric(metric_id).pass_condition
    }
    assert differing == {"sampling_frequency"}


@pytest.mark.parametrize("star", [1, 2, 3, 4])
def test_the_smaller_format_asks_for_exactly_twice_the_sampling(star):
    """970/485, 1950/975, 2940/1470, 3960/1980 -- a clean factor of two at every
    level, which is a useful check on the transcription as well as a fact about the
    table: half the linear dimension needs twice the rate for the same detail."""
    def rate(profile):
        return float(profile.metric("sampling_frequency").pass_condition.split(">=")[1])

    assert rate(load_profile(SMALL.format(star=star))) == pytest.approx(
        2 * rate(load_profile(LARGE.format(star=star)))
    )


def test_neither_transparency_category_specifies_color_accuracy():
    """FADGI's reasoning is that film content is not scene referred, so there is no
    unambiguous original to judge colorimetric accuracy against. Inventing a
    threshold to fill the gap would be exactly the overclaiming this project
    refuses."""
    for template in (SMALL, LARGE):
        declared = {s.metric_id for s in load_profile(template.format(star=3)).metrics}
        assert not declared & {
            "average_color_accuracy", "color_accuracy_90th_percentile", "noise",
            "gain_modulation_highlight", "gain_modulation_other",
        }


def test_the_provisional_status_is_carried_into_the_profile():
    """FADGI marks this category's parameters provisional -- compared against optical
    density equivalents up to 2.2, to be adopted until there is evidence otherwise."""
    notes = load_profile(SMALL.format(star=3)).notes
    assert "provisional" in notes.lower()
    assert "2.2" in notes


def test_the_category_is_named_the_way_the_document_names_it():
    reference = load_profile(SMALL.format(star=3)).standard_reference
    assert reference["material_category"] == "Photographic Transparencies: 35mm up to 4”x5”"
    assert reference["star_level"] == 3
    assert "3rd Edition" in reference["document_version"]


def test_every_shipped_profile_still_has_a_plugin_for_everything_it_declares():
    from deltachrome.metrics import available_metrics

    registered = set(available_metrics())
    for profile_id in available_profiles():
        declared = {s.metric_id for s in load_profile(profile_id).metrics}
        assert declared <= registered, f"{profile_id} declares {declared - registered}"
