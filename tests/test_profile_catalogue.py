"""Presenting the profile list.

Choosing a profile is a three-way decision -- body and edition, material
category, performance level -- and two of the three are easy to get wrong in ways
the resulting report will not explain. FADGI's transparency categories specify no
color accuracy metric at all, so a transparency profile chosen for a print gives
a report that never mentions color. Metamorfoze bands by a named level rather
than a star count, and a summary that only knew about stars called all three of
its levels "custom".

Both facts are properties of the shipped data, not of any front end, so they are
computed once here and tested once here.
"""

from __future__ import annotations

from deltachrome.profiles.catalogue import LOCAL_GROUP, catalogue, grouped, summarise
from deltachrome.profiles.schema import available_profiles, load_profile


def find(profile_id: str):
    return next(s for s in catalogue() if s.profile_id == profile_id)


def test_every_installed_profile_is_summarised():
    assert len(catalogue()) == len(available_profiles())


def test_a_star_level_reads_as_a_star_level():
    assert find("fadgi_3star_prints_photographs_2023").level_label == "3-star"


def test_metamorfoze_levels_are_named_not_called_custom():
    """They are named levels, and 'custom' is both wrong and unhelpful."""
    for profile_id, expected in (
        ("metamorfoze_full_2025", "Metamorfoze Full"),
        ("metamorfoze_light_2025", "Metamorfoze Light"),
        ("metamorfoze_extra_light_2025", "Metamorfoze Extra Light"),
    ):
        assert find(profile_id).level_label == expected


def test_levels_are_ordered_least_to_most_demanding():
    """Alphabetical would put Metamorfoze Full in the middle, which reads as
    meaningless next to a 1-to-4-star list that is ordered by strictness."""
    for (body, _, _), group in grouped():
        labels = [s.level_label for s in group]
        if body == "Metamorfoze":
            assert labels == [
                "Metamorfoze Extra Light", "Metamorfoze Light", "Metamorfoze Full"
            ]
        stars = [s.star_level for s in group if s.star_level]
        assert stars == sorted(stars)


def test_a_category_without_color_accuracy_says_so():
    """The trap this module exists for: a report that never mentions color, and
    never says why, because the category specifies no color metric."""
    summary = find("fadgi_3star_photographic_transparencies_4x5_and_larger_2023")
    assert any("no color accuracy" in flag for flag in summary.flags)


def test_a_category_with_color_accuracy_is_not_flagged():
    assert not any(
        "color accuracy" in flag
        for flag in find("fadgi_3star_prints_photographs_2023").flags
    )


def test_the_flag_wording_suits_both_standards():
    """For FADGI it is a property of the material category, for Metamorfoze of the
    level, so the wording must not name either."""
    summary = find("metamorfoze_extra_light_2025")
    flag = next(f for f in summary.flags if "color accuracy" in f)
    assert "category" not in flag and "level" not in flag


def test_a_provisional_category_is_flagged():
    summary = find("fadgi_1star_photographic_transparencies_35mm_to_4x5_2023")
    assert any("provisional" in flag for flag in summary.flags)


def test_a_missing_material_category_is_empty_rather_than_a_placeholder():
    """Metamorfoze does not band by material at all. A presenter should omit the
    line, not print a dash that looks like missing data."""
    assert find("metamorfoze_full_2025").group[2] == ""


def test_a_saved_guideline_files_under_its_own_name_and_nothing_else(
    tmp_path, monkeypatch
):
    """Once saved, a guideline is its own specification.

    It used to file under the category and the standard it was derived *from*, so
    a guideline called "Our 4-star" appeared as "LOCAL SPECIFICATIONS — Prints and
    Photographs · derived from fadgi_1star_prints_photographs_2023" -- three facts
    about a table the operator may have started from only because it was
    convenient, and not the one fact they gave it. The parent stays in the record;
    it is off every list.
    """
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path))
    base = load_profile("fadgi_1star_prints_photographs_2023")
    derived = base.derive(
        "local_test_spec", thresholds={"noise": "value < 9"},
        display_name="Our 4-star",
    )
    derived.save(tmp_path)

    summary = summarise(load_profile(tmp_path / "local_test_spec.json"))
    assert summary.group == (LOCAL_GROUP, "", "Our 4-star")
    assert summary.material is None
    assert not any("derived" in flag for flag in summary.flags)
    assert not any(base.profile_id in flag for flag in summary.flags)


def test_grouping_covers_every_profile_exactly_once():
    seen = [s.profile_id for _, group in grouped() for s in group]
    assert sorted(seen) == sorted(available_profiles())
