"""Thresholds that depend on something the capture cannot tell you.

Metamorfoze's illumination tolerance is 3 / 4 / 5 / 6 delta-L* for originals up to
A3 / A2 / A1 / A0. The size of the original is not in the capture -- the measurement
is made on a blank white sheet -- and it is not really part of the standard's
identity either. It describes the job the system is being qualified for.

So it is a run input against a table the profile carries. These tests pin the two
consequences that follow: the band has to travel with the result, and a run that
declares nothing must not have a band guessed for it.
"""

from __future__ import annotations

import pytest

from deltachrome.errors import ProfileError
from deltachrome.imaging.loader import load_image
from deltachrome.pipeline import analyze
from deltachrome.profiles.bands import BandedCondition
from deltachrome.profiles.schema import profile_from_dict
from fixtures import render_flat_field

BANDS = {
    "parameter": "original_size",
    "description": "Largest originals this system is being qualified for.",
    "bands": [
        {"band": "A3", "max_mm": 420, "label": "originals up to A3",
         "pass_condition": "value < 3"},
        {"band": "A2", "max_mm": 594, "label": "originals over A3, up to A2",
         "pass_condition": "value < 4"},
        {"band": "A1", "max_mm": 841, "label": "originals over A2, up to A1",
         "pass_condition": "value < 5"},
        {"band": "A0", "max_mm": 1189, "label": "originals over A1, up to A0",
         "pass_condition": "value < 6"},
    ],
}

PROFILE = profile_from_dict({
    "profile_id": "banded", "display_name": "Banded",
    "metrics": [{
        "metric_id": "illumination_non_uniformity", "requires_capability": "flat_field",
        "units": "dL*", "pass_condition_by": BANDS,
    }],
})


@pytest.fixture
def field(tmp_path):
    """A field whose spread lands at about 3.7 dL* -- between the A3 and A2 bands,
    which is where the interesting behavior is."""
    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 91.24))
    image = load_image(path)
    return path, [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]


def measure(field, size=None):
    path, corners = field
    result = analyze(path, target="flat_field_v1", profile=PROFILE, corners=corners,
                     original_size=size)
    return result.evaluation.results[0]


# -- reading a declared size ------------------------------------------------


def test_a_band_name_selects_its_threshold(field):
    assert measure(field, "A3").passed is False
    assert measure(field, "A2").passed is True


@pytest.mark.parametrize(
    "declared,band",
    [
        ("A2", "A2"),           # a band name as written
        ("A4", "A3"),           # a paper the smallest band brackets
        ("500x700", "A1"),      # a measurement, judged on its longest side
        ("450", "A2"),          # just the longest dimension
        ("a1", "A1"),           # case is not a trap
    ],
)
def test_a_size_can_be_given_several_ways(field, declared, band):
    assert measure(field, declared).metadata["band"] == band


def test_a_size_beyond_every_band_is_refused_rather_than_clamped():
    """The standard states no requirement above A0, so there is nothing to judge
    against -- silently using the A0 row would invent one."""
    banded = BandedCondition.from_dict(BANDS)
    with pytest.raises(ProfileError, match="beyond the largest band"):
        banded.named("1500x900")


def test_an_unreadable_size_says_what_would_work():
    banded = BandedCondition.from_dict(BANDS)
    with pytest.raises(ProfileError, match="Cannot read"):
        banded.named("quite large")


# -- the band travels with the result ---------------------------------------


def test_the_declared_band_is_recorded(field):
    """A profile no longer determines the verdict on its own, so the run has to say
    which band it was judged under or the result is not reproducible."""
    result = measure(field, "500x700")
    assert result.metadata["band"] == "A1"
    assert result.metadata["declared"] == "500x700"
    assert result.metadata["banded_by"] == "original_size"
    assert result.condition == "value < 5"
    assert any("different measurement" in note for note in result.notes)


def test_the_available_bands_are_listed_either_way(field):
    for size in (None, "A2"):
        assert "A3 (value < 3)" in measure(field, size).metadata["bands_available"]


# -- declaring nothing ------------------------------------------------------


def test_no_declared_size_is_measured_but_not_judged(field):
    result = measure(field)
    assert result.outcome == "evaluated"
    assert result.passed is None
    assert result.status == "MEASURED"
    assert result.condition is None


def test_it_reports_which_bands_the_reading_meets(field):
    """The question the measurement actually answers, and more useful than a pass
    against a band someone had to pick in advance."""
    note = next(n for n in measure(field).notes if "meets the requirement" in n)
    assert "A2, A1 and A0" in note
    assert "but not A3" in note


def test_the_bands_are_not_described_as_nesting(field):
    """The trap: the tolerance *loosens* as originals get larger, so meeting A0 does
    not carry A3 with it. 'Qualifies for originals up to A0' would say it does."""
    notes = " ".join(measure(field).notes)
    assert "do not nest" in notes
    assert "up to A0" not in notes


def test_a_reading_that_meets_everything_says_so(tmp_path):
    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 92.9))
    image = load_image(path)
    result = analyze(path, target="flat_field_v1", profile=PROFILE,
                     corners=[(0, 0), (image.width, 0), (image.width, image.height),
                              (0, image.height)])
    note = next(n for n in result.evaluation.results[0].notes if "meets the" in n)
    assert "A3, A2, A1 and A0" in note
    assert "but not A" not in note   # no exclusion clause; "but not judged" is fine


def test_a_reading_that_meets_nothing_says_declaring_a_size_would_not_help(tmp_path):
    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 86.0))
    image = load_image(path)
    result = analyze(path, target="flat_field_v1", profile=PROFILE,
                     corners=[(0, 0), (image.width, 0), (image.width, image.height),
                              (0, image.height)])
    notes = " ".join(result.evaluation.results[0].notes)
    assert "meets none of the bands" in notes
    assert "would give a FAIL" in notes


# -- the table survives a round trip ----------------------------------------


def test_a_banded_profile_saves_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path))
    from deltachrome.profiles import schema

    schema._index.cache_clear()
    derived = PROFILE.derive("banded_local", thresholds={
        "illumination_non_uniformity": "value < 2"})
    derived.save()
    schema._index.cache_clear()

    reloaded = schema.load_profile("banded_local")
    spec = reloaded.metric("illumination_non_uniformity")
    assert spec.pass_condition_by is not None
    assert [b.band for b in spec.pass_condition_by.bands] == ["A3", "A2", "A1", "A0"]
    schema._index.cache_clear()
