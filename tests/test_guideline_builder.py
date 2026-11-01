"""The Guideline Builder: what it collects, and what it refuses to invent.

The whole screen is a way of reaching `AnalysisProfile.derive`, so almost nothing
here tests arithmetic -- the engine's own tests do that. What these pin is the
boundary between the two: that the builder hands `derive` what the operator
typed and shows back exactly what `derive` returned, including the sentence that
will end up in every record, rather than composing a second version of it that
can drift.

The other half is what the screen will not offer. A threshold's operator is not
editable, because `compare_conditions` can only say `relaxed` or `tightened` when
the two conditions are the same shape. A threshold banded by original size has no
single number to type. And a guideline may never shadow a published standard by
accident, nor be derived from another guideline, which are the only two ways
provenance could go wrong from here.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui.guideline_builder import JUDGED, MEASURED, OMITTED  # noqa: E402

PARENT = "fadgi_4star_prints_photographs_2023"
BANDED = "metamorfoze_full_2025"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def guidelines(qt_app, tmp_path, monkeypatch):
    """A builder writing into an empty guideline folder."""
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path / "profiles"))
    from deltachrome.profiles.schema import _index

    _index.cache_clear()
    from deltachrome_ui.guidelines_panel import GuidelinesPanel

    panel = GuidelinesPanel()
    yield panel
    _index.cache_clear()


def _rows(builder) -> dict:
    return {edit.spec.metric_id: edit for edit in builder._edits}


def _set(edit, mode: str) -> None:
    edit.state.setCurrentIndex([JUDGED, MEASURED, OMITTED].index(mode))


def _start(panel, parent: str = PARENT, name: str = "Our 4-star"):
    panel._new()
    builder = panel.builder
    assert builder.picker.select_profile(parent), parent
    builder.name.setText(name)
    return builder


# -- what it collects ----------------------------------------------------


def test_the_id_is_made_from_the_name_and_is_never_shown(guidelines):
    """An id is a filename and a key in every record that used it. Asking a person
    to invent one is asking for a decision whose consequences are invisible at the
    moment they make it -- and showing them one they cannot edit invites the
    question of whether they were supposed to."""
    builder = _start(guidelines, name="Our 4-star, relaxed sampling")
    assert builder.profile_id == "our_4_star_relaxed_sampling"
    assert not hasattr(builder, "id_label")


def test_the_id_is_fixed_once_a_guideline_exists(guidelines):
    """Renaming would orphan every record that names it."""
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    builder.name.setText("Something Else Entirely")
    assert builder.profile_id == "our_4_star"


def test_a_guideline_that_changes_nothing_can_still_be_saved(guidelines):
    """"Our standard is FADGI 3-star, for now" is a real thing to record, and the
    guideline is where the site's later decisions will go. `derive` refuses an
    identical derivation by default -- right for a caller that meant to change
    something and did not -- and the builder asks for it explicitly."""
    builder = _start(guidelines)
    assert builder.save_button.isEnabled()
    builder._save()

    from deltachrome.profiles.schema import load_profile

    saved = load_profile(builder.editing.profile_id)
    assert saved.departures == []
    assert saved.local_criteria
    # Adopted unchanged means every criterion is inherited: nothing was decided.
    assert saved.inherited == sorted(m.metric_id for m in saved.metrics)
    # And it costs the conformance claim exactly as a changed one does, which is
    # what the button says before it is pressed.
    assert "not conformance results" in builder.save_button.toolTip()


def test_an_unnamed_guideline_cannot_be_saved(guidelines):
    """The one thing a guideline cannot do without. Its name is how every record
    that used it will refer to it."""
    builder = _start(guidelines, name="")
    _rows(builder)["noise"].fields[0].setText("3")
    assert not builder.save_button.isEnabled()
    assert "Name the guideline" in builder.save_button.toolTip()


def test_the_departure_sentence_is_the_engine_s_and_is_not_drawn_as_a_warning(
    guidelines,
):
    """It used to be an amber callout above the save button, which is the shape
    this application uses for a *warning* -- the wrong signal for something the
    operator has just deliberately done, and redundant beside a table that shows
    every departure column against column. The sentence itself is unchanged and
    the engine still composes it, but it is not shown on the save button either:
    it is written in metric ids and engine syntax ("noise relaxed from 'value <
    3' to ..."), and the table directly above already shows every value beside
    the standard's own."""
    builder = _start(guidelines)
    _rows(builder)["noise"].fields[0].setText("3")
    assert not hasattr(builder, "summary")
    summary = builder._preview().departure_summary
    assert "noise" in summary
    assert summary not in builder.save_button.toolTip()

    builder._save()
    # Gone from the screen and gone from the file: a departure reads "relaxed
    # from 'value < 3' to 'value < 9'", which means nothing except beside the
    # table it departs from, and that table is no longer recorded.
    assert summary not in builder.save_button.toolTip()
    assert "FADGI" not in builder.save_button.toolTip()
    assert "summary" not in json.loads(_saved_path(builder).read_text())[
        "standard_reference"
    ]


def _saved_path(builder):
    from deltachrome.profiles.schema import user_profile_dir

    return user_profile_dir() / f"{builder.editing.profile_id}.json"


def test_all_four_kinds_of_departure_reach_the_engine(guidelines):
    """A moved threshold, a changed list of permitted values, a metric measured
    but not judged, and a metric left out. Four distinct things, kept apart
    because the engine keeps them apart."""
    builder = _start(guidelines)
    rows = _rows(builder)
    rows["sampling_frequency"].fields[0].setText("150")
    rows["master_file_format"].allowed = ["TIFF"]
    _set(rows["noise"], MEASURED)
    _set(rows["reproduction_scale_accuracy"], OMITTED)
    builder._save()

    # Checked on what came out rather than on the departure list, which is no
    # longer written down: all four kinds have to have reached `derive` for the
    # saved guideline to say these four things.
    saved = builder.editing
    assert saved.metric("sampling_frequency").pass_condition == "value >= 150"
    assert saved.metric("master_file_format").algorithm["allowed"] == ["TIFF"]
    assert saved.metric("noise").pass_condition is None
    assert not any(m.metric_id == "reproduction_scale_accuracy" for m in saved.metrics)
    # And none of the four counts as inherited, all four having been decided.
    assert not {"sampling_frequency", "master_file_format", "noise"} & set(
        saved.inherited
    )

    # ...and on disk, which is what a run will actually be judged against.
    kept = {m["metric_id"]: m for m in
            json.loads(_saved_path(builder).read_text())["metrics"]}
    assert "reproduction_scale_accuracy" not in kept
    assert kept["master_file_format"]["algorithm"]["allowed"] == ["TIFF"]


def test_measured_is_not_the_same_as_omitted(guidelines):
    """The distinction the three-state control exists for. A metric set to
    Measured still runs and reports its number, carrying no criterion so it reads
    MEASURED rather than PASS or FAIL; an omitted one is not there at all."""
    builder = _start(guidelines)
    _set(_rows(builder)["noise"], MEASURED)
    builder._save()

    from deltachrome.profiles.schema import load_profile

    saved = load_profile(builder.editing.profile_id)
    noise = saved.metric("noise")
    assert noise.pass_condition is None
    assert noise.pass_condition_by is None
    # Still declared, so it still runs and still reports.
    assert noise.metric_id == "noise"


def test_unjudging_clears_a_criterion_of_any_shape(guidelines):
    """Three shapes count as one criterion -- a threshold, a threshold banded by a
    run input, and a list of permitted values -- and leaving one behind would give
    a metric that says it is not judged and then judges."""
    builder = _start(guidelines)
    _set(_rows(builder)["master_file_format"], MEASURED)
    builder._save()

    from deltachrome.profiles.schema import load_profile

    saved = load_profile(builder.editing.profile_id)
    assert not saved.metric("master_file_format").algorithm.get("allowed")


def test_a_band_keeps_both_of_its_numbers(guidelines):
    """Collapsing a band to one bound would only ever catch half the fault --
    gain modulation fails for being too flat *or* too steep."""
    builder = _start(guidelines)
    row = _rows(builder)["gain_modulation_highlight"]
    assert len(row.fields) == 2
    row.fields[0].setText("0.7")
    row.fields[1].setText("1.2")
    assert row.condition() == "0.7 <= value <= 1.2"


def test_the_operator_is_carried_through_untouched(guidelines):
    """Not editable, so a departure can always be reported as relaxed or
    tightened rather than as the useless `changed`."""
    builder = _start(guidelines)
    for edit in builder._edits:
        # A metric the parent does not declare starts with empty fields, so it
        # has no condition yet -- its operator is checked below by the shape it
        # was given rather than by a string it cannot yet build.
        if edit.template is None or not edit.declared:
            continue
        condition = edit.condition()
        assert condition is not None
        # Every operator in the standard's own condition survives.
        for token in ("<=", ">=", "<", ">"):
            if token in (edit.spec.pass_condition or ""):
                assert token in condition
                break


# -- what it refuses -----------------------------------------------------


def test_a_list_of_permitted_values_is_edited_as_a_list(guidelines):
    """The three file-level criteria are the tables' own requirements rather than
    decoration -- a level permitting TIFF and JPEG 2000 is not met by a JPEG --
    so restricting or extending one is the same kind of edit as moving a
    threshold."""
    builder = _start(guidelines)
    for metric_id in ("master_file_format", "bit_depth", "color_space"):
        row = _rows(builder)[metric_id]
        assert row.fields == []
        assert row.allowed_button is not None
        assert row.allowed == [
            str(a) for a in row.spec.algorithm.get("allowed", [])
        ]


def test_a_threshold_banded_by_a_run_input_has_no_number_to_type(guidelines):
    builder = _start(guidelines)
    assert builder.picker.select_profile(BANDED)
    banded = _rows(builder)["illumination_non_uniformity"]
    assert banded.fields == []
    assert banded.allowed_button is None
    assert banded.fixed_reason == "banded by original size"


def test_a_metric_with_no_typeable_threshold_can_still_be_omitted(guidelines):
    """Which is the whole of what this screen can do for a banded criterion, and
    is not nothing."""
    builder = _start(guidelines, parent=BANDED, name="Our Metamorfoze")
    _set(_rows(builder)["illumination_non_uniformity"], OMITTED)
    builder._save()
    data = json.loads(_saved_path(builder).read_text())
    assert "illumination_non_uniformity" not in {m["metric_id"] for m in data["metrics"]}


def test_a_published_standard_can_never_be_shadowed_by_accident(guidelines):
    """Shadowing a shipped id is a real feature for a site that has decided its
    own means something particular. From here it could only ever be an accident."""
    from deltachrome_ui.guideline_builder import free_id, slugify

    assert slugify("FADGI 4star Prints Photographs 2023") == (
        "fadgi_4star_prints_photographs_2023"
    )
    assert free_id("fadgi_4star_prints_photographs_2023") != (
        "fadgi_4star_prints_photographs_2023"
    )


def test_the_parent_control_is_gone_once_a_guideline_is_saved(guidelines):
    """Hidden rather than disabled. A heavily modified guideline is its own
    specification, and a combo box naming FADGI 4-star above it invites exactly
    the association it should not have -- while letting the parent move under a
    saved guideline would silently rewrite what every record naming it was judged
    against."""
    builder = _start(guidelines)
    assert builder.picker.isVisibleTo(builder)
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    assert not builder.picker.isVisibleTo(builder)


def test_a_guideline_is_never_offered_as_a_parent(guidelines):
    """Deriving from a guideline nests the provenance -- the departure list would
    be measured against the intermediate rather than against the published table.
    It is also how a guideline saved in an earlier session came to be offered as a
    parent, which nobody meant to ask for."""
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()

    from deltachrome_ui.guideline_builder import GuidelineBuilder, published_profiles

    assert "our_4_star" not in published_profiles()
    fresh = GuidelineBuilder()
    assert not fresh.picker.select_profile("our_4_star")


# -- the round trip ------------------------------------------------------


def test_a_saved_guideline_reopens_with_its_edits_on_it(guidelines):
    builder = _start(guidelines, name="Our 4-star")
    rows = _rows(builder)
    rows["noise"].fields[0].setText("3")
    _set(rows["white_balance"], MEASURED)
    _set(rows["reproduction_scale_accuracy"], OMITTED)
    rows["master_file_format"].allowed = ["TIFF"]
    builder._save()
    guidelines.reload()

    guidelines.list.setCurrentRow(0)
    reopened = _rows(guidelines.builder)
    # Amended in place: the guideline is its own baseline, there being no parent
    # to re-derive it from.
    assert guidelines.builder.parent_profile.profile_id == "our_4_star"
    assert reopened["noise"].fields[0].text() == "3"
    assert reopened["white_balance"].mode == MEASURED
    assert reopened["reproduction_scale_accuracy"].mode == OMITTED
    assert reopened["master_file_format"].allowed == ["TIFF"]


def test_the_standard_column_is_shown_while_making_and_not_after(guidelines):
    """A saved guideline is its own specification -- the same reason its parent is
    no longer a control -- and a column of somebody else's numbers beside yours
    invites the comparison it should not."""
    builder = _start(guidelines, name="Our 4-star")
    assert builder.standard_heading.isVisibleTo(builder)
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    guidelines.reload()
    guidelines.list.setCurrentRow(0)
    assert not guidelines.builder.standard_heading.isVisibleTo(guidelines.builder)


def test_a_row_edited_in_any_session_stops_being_inherited(guidelines):
    """The property the old re-derivation was protecting, by the means that
    survives the parent being severed.

    A guideline used to be re-derived from its parent on every edit so that a
    threshold moved in an earlier session was still recorded as a departure. With
    no parent there is nothing to re-derive from, so ``inherited`` narrows
    instead: it holds what the operator has *never* touched, across every session,
    and a row leaves it the moment it is edited and never rejoins it by being left
    alone afterwards.
    """
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    first = set(builder.editing.inherited)
    assert "noise" not in first
    assert "white_balance" in first

    guidelines.reload()
    guidelines.list.setCurrentRow(0)
    again = guidelines.builder
    _rows(again)["white_balance"].fields[0].setText("5")
    again._save()

    from deltachrome.profiles.schema import load_profile

    reloaded = load_profile("our_4_star")
    assert "noise" not in reloaded.inherited
    assert "white_balance" not in reloaded.inherited
    # And the one touched in the first session did not come back when the second
    # session left it alone.
    assert set(reloaded.inherited) == first - {"white_balance"}


def test_a_saved_guideline_is_a_profile_the_engine_can_load(guidelines):
    """The point of routing everything through `derive`: what comes out is not a
    special kind of object, so it is chosen, run and re-scored by code that knows
    nothing about this screen."""
    from deltachrome.profiles.catalogue import LOCAL_GROUP, summarise
    from deltachrome.profiles.schema import load_profile

    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()

    profile = load_profile("our_4_star")
    assert profile.local_criteria
    assert profile.metric("noise").pass_condition == "value < 3"
    # And nothing in the file reaches back to what it was built from.
    assert PARENT not in _saved_path(builder).read_text()
    # And it files itself where a person scanning the list would look for it.
    assert summarise(profile).group[0] == LOCAL_GROUP


def test_deleting_a_guideline_removes_it_from_every_list(guidelines):
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    guidelines.reload()

    from deltachrome.profiles.schema import available_profiles
    from deltachrome_ui.guideline_builder import delete_guideline

    assert "our_4_star" in available_profiles()
    delete_guideline(builder.editing)
    assert "our_4_star" not in available_profiles()


def test_a_published_standard_is_not_one_of_the_users_own(guidelines):
    """Listing them here would put a delete button beside a standard."""
    from deltachrome.profiles.schema import ProfileError, load_profile
    from deltachrome_ui.guideline_builder import delete_guideline

    with pytest.raises(ProfileError):
        delete_guideline(load_profile(PARENT))


def test_the_picker_picks_up_a_guideline_saved_after_it_was_built(guidelines):
    """Every profile control in the window reads the catalogue when it is
    constructed, and the placement panel's lives for the whole session."""
    from deltachrome_ui.profile_picker import ProfilePicker

    picker = ProfilePicker()
    assert not picker.select_profile("our_4_star")

    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()

    assert not picker.select_profile("our_4_star")
    picker.reload()
    assert picker.select_profile("our_4_star")


def test_the_permitted_values_dialog_matches_the_way_the_engine_does(qt_app):
    """`eciRGB v2` and `ECIRGB_v2` are one value to the metric that reads them, so
    a guideline carrying either has to tick the one entry the list offers rather
    than adding a second that could never disagree with it."""
    from deltachrome_ui.guideline_builder import AllowedValuesDialog

    dialog = AllowedValuesDialog("Color Space", ["ECIRGB_v2", "sRGB"], ["eciRGB v2"])
    assert dialog.values() == ["ECIRGB_v2"]
    assert len(dialog._boxes) == 2


def test_one_spelling_per_space_is_offered(qt_app):
    """The shipped tables list both `ProPhoto` and `ProPhoto RGB` because a table
    names the spellings a file might carry. The metric matches by substring both
    ways, so offering both would be two checkboxes that cannot disagree."""
    from deltachrome_ui.guideline_builder import KNOWN_ALLOWED

    spaces = KNOWN_ALLOWED["color_space"]
    assert "ProPhoto RGB" in spaces
    assert "ProPhoto" not in spaces


def test_the_list_is_closed(qt_app):
    """Nothing on this screen could tell somebody whether a name they typed will
    match the ICC description their own files carry, so a free-text box would
    offer a guideline that silently fails every capture."""
    from deltachrome_ui.guideline_builder import AllowedValuesDialog

    dialog = AllowedValuesDialog("Color Space", ["sRGB"], [])
    assert not hasattr(dialog, "_entry")
    assert [b.text() for b in dialog._boxes] == ["sRGB"]


def test_a_value_already_on_the_guideline_is_shown_even_if_no_table_names_it(qt_app):
    """A profile written by hand is entitled to say what it says, and opening it
    here must not silently drop what this list does not offer."""
    from deltachrome_ui.guideline_builder import AllowedValuesDialog

    dialog = AllowedValuesDialog("Color Space", ["sRGB"], ["Our House RGB"])
    assert dialog.values() == ["Our House RGB"]


# -- every guideline has the same rows ------------------------------------


def test_every_guideline_shows_every_metric(guidelines):
    """A table showing only what its parent declares gives FADGI X-Ray thirteen
    rows and FADGI Prints eighteen, so a site comparing its own two house
    standards would be comparing two shapes. What differs between guidelines is
    the *state* of a row, not whether the row is there."""
    from deltachrome.profiles.catalogue import metric_templates

    expected = len(metric_templates("FADGI"))
    xray = _start(guidelines, parent="fadgi_3star_xray_film_2023", name="X")
    assert len(xray._edits) == expected
    prints = _start(guidelines, parent=PARENT, name="P")
    assert len(prints._edits) == expected
    assert [e.spec.metric_id for e in xray._edits] == [
        e.spec.metric_id for e in prints._edits
    ]


def test_a_metric_the_standard_never_states_starts_blank_and_omitted(guidelines):
    """FADGI's X-Ray category states no colour accuracy metric at all. There is no
    value to default to, and inventing one would be this application stating a
    requirement nobody published."""
    builder = _start(guidelines, parent="fadgi_3star_xray_film_2023", name="X")
    row = _rows(builder)["average_color_accuracy"]
    assert not row.declared
    assert row.mode == OMITTED
    assert [f.text() for f in row.fields] == [""]


def test_omitting_what_was_never_there_is_not_a_departure(guidelines):
    """Saying `average_color_accuracy omitted` of a table that never mentioned it
    would be a departure from something that was never there."""
    builder = _start(guidelines, parent="fadgi_3star_xray_film_2023", name="X")
    assert not any(builder._departures().values())
    builder._save()

    from deltachrome.profiles.schema import load_profile

    assert load_profile(builder.editing.profile_id).departures == []


def test_a_metric_can_be_added_to_a_standard_that_does_not_state_it(guidelines):
    """A department shooting radiographs on a rig it also uses for prints may
    reasonably want colour measured anyway."""
    builder = _start(guidelines, parent="fadgi_3star_xray_film_2023", name="X")
    row = _rows(builder)["average_color_accuracy"]
    _set(row, JUDGED)
    row.fields[0].setText("5")
    builder._save()

    from deltachrome.profiles.schema import load_profile

    saved = load_profile(builder.editing.profile_id)
    added = saved.metric("average_color_accuracy")
    assert added.pass_condition == "value < 5"
    # And it carries the shape the published tables give it, not a guess.
    assert added.requires_capability == "color_accuracy"
    assert added.aggregation == "mean"
    assert added.units


def test_an_added_metric_with_no_value_is_refused_rather_than_dropped(guidelines):
    """A row set to Judged and then quietly ignored is the control lying about
    what it did."""
    builder = _start(guidelines, parent="fadgi_3star_xray_film_2023", name="X")
    _set(_rows(builder)["average_color_accuracy"], JUDGED)
    assert not builder.save_button.isEnabled()
    assert "Average Color Accuracy" in builder.save_button.toolTip()


def test_a_metric_template_never_crosses_bodies(guidelines):
    """Metamorfoze states exposure as a plain delta-L* judged down to a lightness
    floor, and FADGI as delta-L*2000 over the whole ramp. Taking one body's shape
    into the other's guideline would change the measurement while looking like a
    tidy-up."""
    from deltachrome.profiles.catalogue import metric_templates

    assert metric_templates("FADGI")["tone_response_oecf"].units == "dL*2000"
    assert metric_templates("Metamorfoze")["tone_response_oecf"].units == "dL*"


# -- the list, and the row a guideline gets before it exists ---------------


def test_starting_a_new_guideline_gives_it_a_row(guidelines):
    """It was represented by *no selection*, and `QAbstractItemView` restores the
    current row to the first item when a view with none regains focus -- which is
    exactly what clicking the button beside the list causes. The form was built
    and Qt put the first saved guideline straight back over it."""
    from PySide6.QtCore import Qt as QtCore
    from PySide6.QtGui import QFocusEvent

    builder = _start(guidelines, name="Saved One")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    guidelines.reload()

    guidelines._new()
    assert guidelines.list.count() == 2
    assert guidelines.list.currentRow() == 1
    assert guidelines.builder.editing is None

    QApplication.sendEvent(
        guidelines.list,
        QFocusEvent(QFocusEvent.Type.FocusIn, QtCore.FocusReason.MouseFocusReason),
    )
    assert guidelines.builder.editing is None
    assert guidelines.list.currentRow() == 1


def test_the_new_row_follows_the_name_being_typed(guidelines):
    guidelines._new()
    assert guidelines.list.item(guidelines.list.count() - 1).text() == "New Guideline"
    guidelines.builder.name.setText("Half Typed")
    assert guidelines.list.item(guidelines.list.count() - 1).text() == "Half Typed"


def test_moving_off_an_unsaved_guideline_abandons_its_row(guidelines):
    """Keeping it would offer a second entry that is not a guideline and cannot be
    deleted."""
    builder = _start(guidelines, name="Saved One")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()
    guidelines.reload()

    guidelines._new()
    assert guidelines.list.count() == 2
    guidelines.list.setCurrentRow(0)
    assert guidelines.list.count() == 1
    assert guidelines.builder.editing.profile_id == "saved_one"


# -- saying that a save happened ----------------------------------------------


def test_saving_is_confirmed_on_the_button(guidelines):
    """Amending a guideline changes nothing on screen otherwise.

    The name is already in the field, the parent combo is already gone, and the
    row in the list already reads what it read -- so without this the only way to
    find out whether Save worked is to leave the screen and come back. It is the
    same case the copy button was given `IconButton.flash` for, and it lasts the
    same 1400 ms; a word rather than a glyph, this button being worded because a
    shape cannot say which of several things it commits.
    """
    from deltachrome_ui.guideline_builder import SAVE_LABEL, SAVED_LABEL

    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    assert builder.save_button.text() == SAVE_LABEL

    builder._save()
    assert builder.save_button.text() == SAVED_LABEL

    # And it puts itself back, rather than reading "Saved" for the rest of the
    # session over a form that has since been edited.
    builder._saved_timer.timeout.emit()
    assert builder.save_button.text() == SAVE_LABEL


def test_the_save_button_says_whether_anything_is_outstanding(guidelines):
    """The question an editor is actually asked, and the one a Save button that
    stays live forever cannot answer: is there anything of mine not written
    down."""
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    assert builder.save_button.isEnabled()

    builder._save()
    assert not builder.save_button.isEnabled()
    assert "No unsaved changes." in builder.save_button.toolTip()
    # The conformance sentence is about the guideline, not about the button, so
    # it does not come and go with whether there is anything to press.
    # And it must not say "advisory": a local guideline does not demote a verdict.
    assert "not conformance results" in builder.save_button.toolTip()
    assert "advisory" not in builder.save_button.toolTip().lower()

    # Live again on the next edit, of a criterion...
    _rows(builder)["white_balance"].fields[0].setText("5")
    assert builder.save_button.isEnabled()
    builder._save()
    assert not builder.save_button.isEnabled()

    # ...and on a rename, which keeps the guideline's id and is still an edit.
    builder.name.setText("Our 4-star, revised")
    assert builder.save_button.isEnabled()


def test_a_reopened_guideline_has_nothing_to_save(guidelines):
    """The state it should open in. Computed against what is on disk rather than
    tracked through every editor, because a dirty flag has to be cleared in as
    many places as it is set."""
    builder = _start(guidelines, name="Our 4-star")
    _rows(builder)["noise"].fields[0].setText("3")
    builder._save()

    guidelines.reload()
    guidelines.list.setCurrentRow(0)
    assert not guidelines.builder.save_button.isEnabled()
    assert _rows(guidelines.builder)["noise"].fields[0].text() == "3"
