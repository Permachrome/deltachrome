"""Getting from a verdict to a cause, and the settings that change one.

A failing aggregate says a system is out of specification. It does not say why,
and the two commonest causes look nothing alike in the raw data: a cast moves
every patch a little, dust or a specular hit moves one patch a lot. The mean
cannot tell them apart. That is the whole reason the engine persists per-patch
values, and this is the screen that finally shows them.

The options tests are about one thing the panel unblocks. A Metamorfoze
illumination tolerance bands on the size of the originals a system is being
qualified for -- a property of the job, which nothing in the capture can supply --
so before the panel existed, a run started from the desktop application could only
ever report that metric as measured-but-unjudged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from deltachrome_ui.charts import FieldMap, FieldPoint, LinePlot, Series  # noqa: E402
from deltachrome_ui.detail_view import (  # noqa: E402
    EdgeCurves,
    EdgeReadings,
    PatchTable,
)
from deltachrome_ui.options_view import OptionsView  # noqa: E402
from deltachrome_ui.result_view import MetricPage  # noqa: E402
from deltachrome_ui.nav import OVERVIEW as OVERVIEW_KEY  # noqa: E402
from deltachrome_ui.nav import PAGE as PAGE_KEY  # noqa: E402
from deltachrome_ui.tables import DataTable  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def table_rows(widget) -> list[str]:
    """Every row of every table inside a widget, as text.

    The readings were labels in a hand-laid grid and are now cells in a model,
    so what a reader sees is no longer reachable through ``findChildren``. This
    is the same question asked of the model.
    """
    rows: list[str] = []
    for table in widget.findChildren(DataTable):
        model = table.model()
        for row in range(model.rowCount()):
            rows.append(" | ".join(
                str(model.index(row, column).data() or "")
                for column in range(model.columnCount())
            ))
    return rows


def text_of(widget) -> str:
    """Everything on screen: the labels, and the tables' contents.

    Including the widget itself where it *is* a label: a view whose whole answer
    is one sentence -- "this chart carries no neutral ramp" -- returns that
    label rather than a container holding it, and ``findChildren`` never returns
    the widget it is called on.
    """
    labels = [c.text() for c in widget.findChildren(QLabel) if c.text()]
    if isinstance(widget, QLabel) and widget.text():
        labels.insert(0, widget.text())
    return "\n".join(labels + table_rows(widget))


def one_metric(record, run=None, proxy=None, key="color_accuracy"):
    """The page a single metric gets, ready to interrogate."""
    from deltachrome_ui.families import Family
    from deltachrome.results.models import AnalysisRun

    if run is None:
        run = AnalysisRun.from_dict({
            "run_id": "r", "created_utc": "", "image": {}, "target": {},
            "reference": {}, "profile": {}, "geometry": {}, "sampling": {},
            "color_management": {}, "patches": [], "edges": [],
            "edge_field_summary": {}, "registration": [],
            "registration_summary": {}, "rulers": {}, "metrics": [record],
            "overall": "PASS", "overall_explanation": "",
        })
    family = Family(key=key, name="Color", order=10, metrics=[record])
    return MetricPage(family, record, run, proxy=proxy)


#: The old name. Every caller wanted one metric; the compartment was the wrapper.
one_family = one_metric


def view_names(page):
    return [page.views.tabText(i) for i in range(page.views.count())]


def open_view(page, name):
    """Switch to a view and return the widget it built."""
    names = view_names(page)
    assert name in names, f"{name} not among {names}"
    page.views.setCurrentIndex(names.index(name))
    tab = page.views.currentWidget()
    tab.ensure()
    return tab


# -- run options -------------------------------------------------------------


def test_an_untouched_option_is_absent_rather_than_passed_as_its_default(qt_app):
    """Passing a default explicitly would silently overwrite the profile's own
    choice with a value that merely happens to match today."""
    assert OptionsView().values() == {}


def _percentile_profile():
    """A profile that aggregates on a percentile, so that control applies."""
    from deltachrome.profiles.schema import load_profile

    return load_profile("fadgi_3star_prints_photographs_2023")


def test_each_option_reaches_analyze_under_its_own_keyword(qt_app):
    from test_banded_thresholds import PROFILE

    options = OptionsView()
    options.set_profile(PROFILE)
    options.original_size.setText("A2")
    values = options.values()
    assert values["original_size"] == "A2"

    from inspect import signature

    from deltachrome.pipeline import analyze

    assert set(values) <= set(signature(analyze).parameters)


def test_the_percentile_method_is_not_a_question_put_to_an_operator(qt_app):
    """It changes the reported figure, so it belongs with the result -- and no
    standard specifies it. FADGI and Metamorfoze both say "90th percentile" and
    neither says which convention, so there is nothing to look up, no basis on
    which to prefer one, and a third-decimal difference on a 140-patch chart.

    A control with no correct answer does not become useful by being offered. The
    profile's own choice is used, it is still recorded with every result, and it
    is still overridable from the command line."""
    options = OptionsView()
    options.set_profile(_percentile_profile())
    assert not hasattr(options, "percentile_method")
    assert "percentile_method" not in options.values()

    from pathlib import Path

    import deltachrome.cli

    source = Path(deltachrome.cli.__file__).read_text(encoding="utf-8")
    assert "--percentile-method" in source


def test_a_control_the_profile_cannot_use_is_not_shown_or_sent(qt_app):
    """A run option that changes nothing is worse than a missing one: it invites
    an answer and then records it beside a result it did not affect, which makes
    two identical measurements look like different ones."""
    options = OptionsView()
    options.set_profile(_percentile_profile())
    assert options._size_applies is False
    assert not options.isVisibleTo(options.parentWidget() or options), (
        "the whole section goes when nothing in it applies -- a heading over an "
        "empty space reads as a section that failed to load"
    )

    # Even with a value sitting in the field from an earlier profile.
    options.original_size.setText("A2")
    assert "original_size" not in options.values()


def test_the_banded_control_appears_because_the_profile_bands(qt_app):
    """Read from the profile's own pass_condition_by rather than from a list of
    profile ids, so a banded criterion from any body brings its control along."""
    from test_banded_thresholds import PROFILE

    options = OptionsView()
    options.set_profile(PROFILE)
    assert options._size_applies is True
    assert options._banded_metric(PROFILE)[1] == "original_size"

    options.set_profile(None)
    assert options._size_applies is False


def test_the_size_a_banded_threshold_needs_makes_it_judgeable(qt_app, tmp_path):
    """The point of the control. Without a size the metric is measured and reports
    which bands it clears; with one it is judged."""
    from deltachrome.pipeline import analyze
    from deltachrome.imaging.loader import load_image
    from test_banded_thresholds import PROFILE, render_flat_field

    path = render_flat_field(tmp_path / "field.tif", lightness=(95.0, 91.24))
    image = load_image(path)
    corners = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]

    options = OptionsView()
    options.set_profile(PROFILE)
    unjudged = analyze(path, target="flat_field_v1", profile=PROFILE,
                       corners=corners, **options.values())
    assert unjudged.evaluation.results[0].passed is None

    options.original_size.setText("A2")
    judged = analyze(path, target="flat_field_v1", profile=PROFILE,
                     corners=corners, **options.values())
    assert judged.evaluation.results[0].passed is True


# -- metric rows -------------------------------------------------------------


def metric(metric_id="average_color_accuracy", n=5, status="FAIL",
           capability="color_accuracy", per_patch=True):
    return {
        "metric_id": metric_id, "status": status, "value": 4.2, "units": "dE2000",
        "condition": "value < 3.5", "outcome": "evaluated", "notes": [],
        # What the report groups on. Recorded per metric so a compartment is read
        # from the data rather than from a list of metric ids kept in the UI.
        "requires_capability": capability,
        "per_patch": [
            {"patch_id": f"P{i}", "role": "chromatic", "value": float(10 - i),
             "flags": ["clipped_high"] if i == 0 else []}
            for i in range(n)
        ] if per_patch else [],
    }


def test_a_metric_with_several_patches_opens_onto_them(qt_app):
    page = one_metric(metric())
    assert view_names(page) == ["Charts", "Values"]
    assert "P0" in text_of(open_view(page, "Values"))


def test_one_reading_is_still_the_measurement(qt_app):
    """Gain modulation highlight has exactly one pair by definition. Hiding it
    behind a "more than one reading" test made the metric look unfinished."""
    page = one_metric({
        "metric_id": "gain_modulation_highlight", "status": "PASS",
        "value": 1.02, "units": "gain", "condition": "0.8 <= value <= 1.1",
        "per_patch": [{"patch_id": "E5-F5", "role": "achromatic_ramp",
                       "value": 1.02, "flags": []}],
    }, key="tonal_response")
    assert "Values" in view_names(page)
    assert "E5-F5" in text_of(open_view(page, "Values"))


def test_a_metric_with_one_value_does_not_pretend_to_have_detail(qt_app):
    """A single per-patch value adds nothing over the aggregate already shown --
    and with no notes and no reason either, there is nothing to select."""
    page = one_metric(metric(n=1, status="PASS", per_patch=False))
    assert view_names(page) == []


def test_a_file_property_metric_has_nothing_to_open(qt_app):
    page = one_metric({"metric_id": "bit_depth", "status": "PASS", "value": 8.0,
                       "units": "bits", "condition": "value >= 8", "per_patch": []})
    assert view_names(page) == []


def test_an_incomplete_metric_says_why_without_being_opened(qt_app):
    """It has no readings at all, and the reason it has none is the only thing
    worth showing -- so it is on the page, not behind a Notes tab.

    That is the same line ``deltachrome_ui.info`` draws: a reason is
    qualification of this run's answer, and qualification never hides."""
    page = one_metric({
        "metric_id": "mtf50", "status": "INCOMPLETE", "value": None,
        "units": "lp/mm", "condition": "value >= 2", "per_patch": [],
        "reason": "the capture's sampling rate could not be established.",
    })
    assert "sampling rate" in text_of(page)
    assert view_names(page) == []


def test_a_view_is_built_only_when_it_is_looked_at(qt_app):
    """A report can carry 140 patches, 20 edge curves and a view that re-reads a
    full-resolution master. Building every representation of every compartment up
    front would make opening a saved run wait on views nobody switches to."""
    from deltachrome_ui.result_view import ViewTabs

    built = []

    def build():
        built.append(1)
        return PatchTable(metric(n=140))

    tabs = ViewTabs()
    tabs.add_view("First", lambda: PatchTable(metric(n=2)))
    tabs.add_view("Second", build)
    assert built == []

    tabs.setCurrentIndex(1)
    assert built == [1]
    # And kept: comparing two views is the ordinary gesture here, so switching
    # back must not rebuild.
    tabs.setCurrentIndex(0)
    tabs.setCurrentIndex(1)
    assert built == [1]


def test_patches_are_listed_worst_first(qt_app):
    """A failing aggregate is a question about the tail, so the tail is at the
    top. Asserted on rows that arrive in a different order from the one they
    should leave in: the fixture's values already descend, so a table doing
    nothing at all passed this for as long as it read them in record order --
    which is exactly what it was doing, since the model had no ``sort``."""
    record = metric(n=5)
    record["per_patch"] = list(reversed(record["per_patch"]))
    rows = table_rows(PatchTable(record))
    assert rows[0].startswith("P0") and rows[-1].startswith("P4")


def test_a_flagged_patch_says_so(qt_app):
    assert "Highlights clipped" in text_of(PatchTable(metric(n=5)))


def test_a_long_patch_list_is_not_truncated(qt_app):
    """It used to stop at twelve and apologise for the other hundred and
    twenty-eight. A table has no reason to: it scrolls, and the readings that
    prompt the question are already at the top."""
    rows = table_rows(PatchTable(metric(n=140)))
    assert len(rows) == 140
    assert any(row.startswith("P139") for row in rows)


def test_the_table_explains_what_the_distribution_means(qt_app):
    """One bad patch and a cast are different faults with different fixes, and
    the aggregate cannot tell them apart."""
    from deltachrome_ui.explanations import PATCH_TABLE

    # The count was crowding the view and saying nothing the table does not; the
    # standing explanation it was attached to now hangs off the chart instead.
    assert "5 patches" not in text_of(PatchTable(metric(n=5)))
    body = "\n".join(PATCH_TABLE)
    assert "specular" in body and "cast" in body


# -- charts ------------------------------------------------------------------


def test_an_empty_plot_does_not_raise(qt_app):
    plot = LinePlot()
    plot.set_series([])
    plot.resize(300, 200)
    plot.grab()


def test_a_flat_series_does_not_divide_by_zero(qt_app):
    from PySide6.QtGui import QColor

    plot = LinePlot(y_range=None)
    plot.set_series([Series(x=[1.0, 1.0], y=[2.0, 2.0], color=QColor(200, 200, 200))])
    plot.resize(300, 200)
    plot.grab()


def test_a_field_map_with_no_readings_still_draws(qt_app):
    field = FieldMap()
    field.set_points([FieldPoint(0.5, 0.5, None, "horizontal")])
    field.resize(300, 300)
    field.grab()


# -- edge diagnostics --------------------------------------------------------


class FakeRun:
    def __init__(self, **kwargs):
        self.edges = kwargs.get("edges", [])
        self.edge_field_summary = kwargs.get("edge_field_summary", {})
        self.patches = kwargs.get("patches", [])
        self.rulers = kwargs.get("rulers", {})
        self.geometry = kwargs.get("geometry", {})
        self.reference = kwargs.get("reference", {})
        self.color_management = kwargs.get("color_management", {})
        self.sampling = kwargs.get("sampling", {})
        self.image = kwargs.get("image", {})


def edge_record(edge_id="e0", orientation="horizontal"):
    from deltachrome.results.edges import EdgeRecord

    return EdgeRecord(
        edge_id=edge_id, region="r", orientation=orientation,
        center_px=[10.0, 10.0], center_relative=[0.5, 0.5],
        bounds_px=[0, 0, 20, 20], edge_angle_deg=-5.0, phase_coverage_px=30.0,
        fit_residual_px=0.1, rows_used=50, linearized=True, binning="distance",
        sampling_efficiency_pct=90.0, sfr50_pct=70.0, mtf_at_nyquist=0.3,
        peak_modulation=1.0, peak_frequency=0.0, overshoot_pct=0.0,
        mtf_curve={"frequency_cyc_px": [0.0, 0.5, 1.0], "mtf": [1.0, 0.3, 0.4]},
        warnings=[],
    )


def test_a_target_with_no_edges_says_that_is_the_chart_not_a_fault(qt_app):
    text = text_of(EdgeCurves(FakeRun()))
    assert "no slanted edges" in text
    assert "not a fault" in text


def test_the_curve_plot_warns_that_a_rise_past_nyquist_is_aliasing(qt_app, tmp_path):
    """The curves turn up again near 1 cycle/px, and read without the marker that
    looks like sharpness increasing with frequency. The sentence saying so is
    documentation now, so what is left on the plot is what is tested."""
    from deltachrome.results.edges import EdgeRecord

    edge = EdgeRecord(
        edge_id="e0", region="r", orientation="horizontal", center_px=[10.0, 10.0],
        center_relative=[0.5, 0.5], bounds_px=[0, 0, 20, 20], edge_angle_deg=-5.0,
        phase_coverage_px=30.0, fit_residual_px=0.1, rows_used=50, linearized=True,
        binning="distance", sampling_efficiency_pct=90.0, sfr50_pct=70.0,
        mtf_at_nyquist=0.3, peak_modulation=1.0, peak_frequency=0.0,
        overshoot_pct=0.0, mtf_curve={"frequency_cyc_px": [0.0, 0.5, 1.0],
                                      "mtf": [1.0, 0.3, 0.4]}, warnings=[],
    )
    diagnostics = EdgeCurves(FakeRun(edges=[edge]))
    plot = diagnostics.findChildren(LinePlot)[0]
    assert any(label == "Nyquist" for _at, label in plot.markers)


# -- interaction -------------------------------------------------------------
#
# A plot changes what a reader can see and never what was measured, so these
# tests are about one thing only: that pointing at a mark answers correctly, and
# that the answer arrives in every view of the same population at once.

from PySide6.QtCore import QEvent, QPointF, QRectF  # noqa: E402
from PySide6.QtGui import QColor, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402

from deltachrome_ui.charts import (  # noqa: E402
    ORIENTATION_COLORS,
    ChartSelection,
    GroupLegend,
    PlotArea,
)
from deltachrome_ui.detail_view import ramp_patches  # noqa: E402
from deltachrome_ui.metric_charts import tone_response  # noqa: E402


def tone_page(run, metric=None):
    """The tone page the report actually builds.

    ``detail_view`` carried a second, older ToneResponse widget that nothing
    reached and whose second plot was the signed L* residual rather than the
    judged ΔL*2000. These tests were pointed at it, so they went on passing
    while the live page changed underneath them. They point at the live one now.
    """
    return tone_response(run, metric or {"metric_id": "tone_response_oecf",
                                         "units": "dL*2000", "per_patch": []},
                         None)


def move_to(widget, point: QPointF) -> None:
    widget.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, point, point,
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    ))


def click_at(widget, point: QPointF) -> None:
    widget.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, point, point,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    ))


def two_curves() -> list[Series]:
    return [
        Series(x=[0.0, 1.0], y=[1.0, 0.9], color=QColor(143, 184, 255),
               name="h0", group="horizontal"),
        Series(x=[0.0, 1.0], y=[0.5, 0.1], color=QColor(255, 196, 77),
               name="v0", group="vertical"),
    ]


def test_the_mapping_runs_backwards(qt_app):
    """Hit-testing and painting must not be able to disagree about where a point
    is, which is only true while they share one invertible transform."""
    area = PlotArea(QRectF(50, 10, 300, 200), 0.0, 1.0, 0.0, 1.05)
    x, y = area.to_data(area.to_px(0.32, 0.77))
    assert x == pytest.approx(0.32)
    assert y == pytest.approx(0.77)


def test_pointing_at_a_curve_names_it(qt_app):
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(400, 300)
    move_to(plot, plot._area().to_px(1.0, 0.1))
    assert plot.selection.hovered == "v0"


def test_pointing_away_from_every_curve_names_nothing(qt_app):
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(400, 300)
    move_to(plot, plot._area().to_px(0.5, 0.75))
    assert plot.selection.hovered is None


def test_pointing_at_a_curve_does_not_fade_the_rest(qt_app):
    """A hover is a question about one reading, not a request to see fewer.

    Running the cursor down a readings table used to strobe the chart beside it,
    and the answer to "which one is this" is the readout, not the disappearance
    of the population that made the question worth asking.
    """
    selection = ChartSelection()
    selection.set_hovered("v0")
    assert selection.alpha("v0", "vertical") == 1.0
    assert selection.alpha("h0", "horizontal") == 1.0


def test_a_reading_pinned_from_a_table_does_not_fade_the_chart(qt_app):
    """Only a click on the plot itself asks for less. A row in the values table
    lights its mark and leaves the field alone."""
    selection = ChartSelection()
    selection.toggle_selected("v0")
    assert selection.selected == "v0"
    assert not selection.isolating
    assert selection.alpha("h0", "horizontal") == 1.0


def test_the_curve_clicked_on_the_plot_is_isolated(qt_app):
    selection = ChartSelection()
    selection.toggle_selected("v0", isolate=True)
    assert selection.alpha("v0", "vertical") == 1.0
    assert 0.0 < selection.alpha("h0", "horizontal") < 1.0


def test_letting_go_of_an_isolated_curve_brings_the_field_back(qt_app):
    """A field left dimmed for a mark that is no longer pinned is the state
    nobody can get out of."""
    selection = ChartSelection()
    selection.toggle_selected("v0", isolate=True)
    selection.toggle_selected("v0", isolate=True)
    assert selection.selected is None
    assert not selection.isolating
    assert selection.alpha("h0", "horizontal") == 1.0


def test_nothing_fades_while_nothing_is_being_pointed_at(qt_app):
    selection = ChartSelection()
    assert selection.alpha("h0", "horizontal") == 1.0
    assert selection.alpha("v0", "vertical") == 1.0


def test_a_pinned_curve_stays_lit_while_another_is_compared_against_it(qt_app):
    """Pin the suspect edge, then hover the others to compare. Losing the pin on
    the first hover would make the comparison impossible."""
    selection = ChartSelection()
    selection.toggle_selected("v0", isolate=True)
    selection.set_hovered("h0")
    assert selection.alpha("v0", "vertical") == 1.0
    assert selection.alpha("h0", "horizontal") == 1.0
    assert selection.alpha("h1", "horizontal") < 1.0


def test_clicking_a_curve_pins_it_and_clicking_it_again_lets_go(qt_app):
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(400, 300)
    click_at(plot, plot._area().to_px(1.0, 0.1))
    assert plot.selection.selected == "v0"
    click_at(plot, plot._area().to_px(1.0, 0.1))
    assert plot.selection.selected is None


def test_clicking_empty_space_lets_go_of_the_pin(qt_app):
    """Otherwise there is no way back to seeing the whole field."""
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(400, 300)
    plot.selection.toggle_selected("v0")
    click_at(plot, plot._area().to_px(0.5, 0.75))
    assert plot.selection.selected is None


def test_one_selection_lights_the_same_edge_in_both_plots(qt_app):
    """The mark and the curve are two renderings of one edge. Pointing at either
    has to light both, or the field map cannot be used to read the overlay."""
    selection = ChartSelection()
    field = FieldMap(selection=selection)
    field.set_points([
        FieldPoint(0.5, 0.5, 70.0, "horizontal", label="h0"),
        FieldPoint(0.1, 0.1, 40.0, "vertical", label="v0"),
    ])
    field.resize(400, 400)
    plot = LinePlot(selection=selection)
    plot.set_series(two_curves())
    plot.resize(400, 300)

    move_to(field, field._area().to_px(0.5, 0.5))
    assert selection.hovered == "h0"
    assert plot.selection.is_focus("h0")
    assert not plot.selection.is_focus("v0")
    click_at(field, field._area().to_px(0.5, 0.5))
    assert plot.selection.alpha("h0", "horizontal") == 1.0
    assert plot.selection.alpha("v0", "vertical") < 1.0


def test_a_muted_orientation_is_neither_drawn_nor_pointable(qt_app):
    """Hiding the verticals is how softness on one side is separated from
    softness that follows orientation. A hidden curve that still answers the
    cursor would put the reading back in by the side door."""
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(400, 300)
    target = plot._area().to_px(1.0, 0.1)
    plot.selection.set_muted("vertical", True)
    assert plot.selection.alpha("v0", "vertical") == 0.0
    move_to(plot, target)
    assert plot.selection.hovered is None


def test_a_legend_key_mutes_the_group_it_names(qt_app):
    selection = ChartSelection()
    legend = GroupLegend(selection, [("vertical", QColor(255, 196, 77), "□ vertical")])
    button = legend.findChild(QPushButton)
    assert button.isChecked()
    button.setChecked(False)
    assert "vertical" in selection.muted


def test_muting_every_orientation_does_not_divide_by_zero(qt_app):
    plot = LinePlot(y_range=None)
    plot.set_series(two_curves())
    plot.selection.set_muted("horizontal", True)
    plot.selection.set_muted("vertical", True)
    plot.resize(300, 200)
    plot.grab()


def test_the_field_map_scales_its_marks_to_what_is_still_shown(qt_app):
    """Mark size carries the reading. If the extent kept counting hidden edges,
    hiding the worst one would leave every remaining mark misleadingly small."""
    field = FieldMap()
    field.set_points([
        FieldPoint(0.2, 0.2, 40.0, "vertical", label="v0"),
        FieldPoint(0.8, 0.8, 90.0, "horizontal", label="h0"),
    ])
    assert field._extent() == (40.0, 90.0)
    field.selection.set_muted("vertical", True)
    assert field._extent() == (90.0, 90.0)


# -- tone response -----------------------------------------------------------


class FakePatch:
    def __init__(self, patch_id, role, measured, reference, flags=()):
        self.patch_id = patch_id
        self.role = role
        self.measured_lab = list(measured)
        self.reference_lab = list(reference)
        self.flags = list(flags)


def ramp_run(steps=6, offset=-2.0):
    patches = []
    for i in range(steps):
        aim = 95.0 - i * 15.0
        patches.append(FakePatch(
            f"G{i}", "achromatic_ramp", (aim + offset, 0.2, -0.3), (aim, 0.0, 0.0)
        ))
    patches.append(FakePatch("C1", "chromatic", (50.0, 30.0, 20.0), (52.0, 28.0, 18.0)))
    return FakeRun(patches=patches)


def test_the_ramp_is_selected_by_role_not_by_lightness(qt_app):
    """Which patches form the ramp is resolved once, at analysis time. Deriving
    it again here would be a second opinion that could disagree with the one the
    metrics were computed against."""
    ramp = ramp_patches(ramp_run())
    assert [p.patch_id for p in ramp] == ["G0", "G1", "G2", "G3", "G4", "G5"]


def test_a_target_with_no_ramp_says_that_is_the_chart_not_a_fault(qt_app):
    text = text_of(tone_page(FakeRun(patches=[])))
    assert "no neutral ramp" in text
    assert "not a fault" in text


def test_the_tone_curve_plots_measured_against_aim_in_aim_order(qt_app):
    """Joined in any other order the line crosses itself and stops being a curve."""
    tone = tone_page(ramp_run())
    plot = tone.findChildren(LinePlot)[0]
    measured = [s for s in plot.series if s.name == "measured"][0]
    assert measured.x == sorted(measured.x)
    assert measured.point_labels == ["G5", "G4", "G3", "G2", "G1", "G0"]


def test_the_tone_curve_draws_the_aim_it_is_being_read_against(qt_app):
    plot = tone_page(ramp_run()).findChildren(LinePlot)[0]
    aim = [s for s in plot.series if s.name == "aim"][0]
    assert aim.dashed and aim.x == [0.0, 100.0] and aim.y == [0.0, 100.0]


def test_the_two_tone_plots_are_headed_by_what_each_one_draws(qt_app):
    """One is measured L* against aim L*, the other is the ΔL*2000 the profile
    judges. Headings that named the comparison -- "Measured Against Aim",
    "Departure Per Step" -- said what two labelled axes already say and left the
    change of unit between the plots unstated."""
    metric = {"metric_id": "tone_response_oecf", "units": "dL*2000",
              "aggregation": "max_per_patch", "per_patch": [
                  {"patch_id": "G0", "value": 1.2}]}
    text = text_of(tone_page(ramp_run(), metric))
    assert "Tone Response" in text
    assert "ΔL*2000" in text
    assert "Measured Against Aim" not in text


def test_pointing_at_a_step_names_the_patch_rather_than_the_series(qt_app):
    tone = tone_page(ramp_run())
    plot = tone.findChildren(LinePlot)[0]
    plot.resize(400, 300)
    measured = [s for s in plot.series if s.name == "measured"][0]
    move_to(plot, plot._area().to_px(measured.x[2], measured.y[2]))
    assert plot._hover is not None
    line, _, _, index = plot._hover
    assert line.point_labels[index] == "G3"


# -- axes --------------------------------------------------------------------


def test_ticks_land_on_round_numbers(qt_app):
    """Fifths of the data's own extent give labels like 0.251958, which are
    unreadable and claim a precision the axis does not have."""
    from deltachrome_ui.charts import axis_ticks

    assert axis_ticks(0.0, 1.05) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert axis_ticks(-4.78, 1.25) == pytest.approx([-4.0, -2.0, 0.0])
    for low, high in ((0.0, 1.05), (-4.78, 1.25), (0.0, 100.0), (0.0, 0.0031)):
        ticks = axis_ticks(low, high)
        assert 3 <= len(ticks) <= 7
        assert all(low <= t <= high for t in ticks)


def test_a_tick_is_labelled_at_the_precision_its_step_justifies(qt_app):
    from deltachrome_ui.charts import tick_label

    assert tick_label(0.2, 0.2) == "0.2"
    assert tick_label(20.0, 20.0) == "20"
    assert tick_label(0.0, 0.2) == "0"


def test_a_flat_axis_does_not_loop_forever(qt_app):
    from deltachrome_ui.charts import axis_ticks

    assert axis_ticks(3.0, 3.0) == [3.0]


def test_the_field_map_is_drawn_at_the_chart_s_own_proportions(qt_app):
    """Distance from the center is what separates field curvature from a tilted
    plane, so a field stretched to fill the widget shows the wrong distances."""
    field = FieldMap(aspect=2.0)
    field.resize(400, 400)
    rect = field._area().rect
    assert rect.width() / rect.height() == pytest.approx(2.0)


def test_a_field_map_with_no_known_extent_still_fills_the_widget(qt_app):
    field = FieldMap()
    field.resize(400, 300)
    rect = field._area().rect
    assert rect.width() == pytest.approx(400 - 52)


def test_the_legend_follows_a_mute_it_did_not_cause(qt_app):
    """A key that names a hidden group as shown is worse than no key at all."""
    selection = ChartSelection()
    legend = GroupLegend(selection, [("vertical", QColor(255, 196, 77), "□ vertical")])
    selection.set_muted("vertical", True)
    assert not legend.findChild(QPushButton).isChecked()


# -- a name collision that crashed the process -------------------------------


def test_a_metric_row_does_not_shadow_a_paint_device_method(qt_app):
    """`self.metric = metric` on a QWidget is a segfault waiting for a repaint.

    Every QWidget is a QPaintDevice, which declares a virtual `metric()` that Qt
    calls during painting to ask the device for its resolution. PySide6 treats a
    Python attribute of that name as an override, so Qt called a dict, the call
    raised inside the paint, and the painter was left mid-save -- surfacing as an
    unbalanced save/restore and then a segfault on some later, unrelated paint.
    The dict was never the problem; the name was.
    """
    page = one_metric(metric())
    assert not isinstance(getattr(page, "metric", None), dict)
    assert callable(page.metric)
    assert isinstance(page.record, dict)


def test_painting_a_metric_page_leaves_the_painter_usable(qt_app):
    """The failure was never visible on the widget that caused it."""
    page = one_metric(metric(n=20))
    page.resize(600, 500)
    page.grab()
    plot = LinePlot()
    plot.set_series(two_curves())
    plot.resize(300, 200)
    plot.grab()


# -- fields for metrics that are not edges -----------------------------------


def placed_metric(n=20, placed=True, role="chromatic"):
    return {
        "metric_id": "average_color_accuracy", "status": "FAIL", "value": 2.3,
        "units": "dE2000", "condition": "value < 3.5", "outcome": "evaluated",
        "notes": [],
        "per_patch": [
            {
                "patch_id": f"P{i}", "role": role, "value": 1.0 + i * 0.1,
                "flags": ["clipped_high"] if i == 0 else [],
                **({"position_relative": [i / n, 0.5]} if placed else {}),
            }
            for i in range(n)
        ],
    }


def test_a_metric_whose_readings_are_placed_is_drawn_on_the_capture(qt_app):
    """Where a reading sits is answered on the pixels it came from.

    There used to be a Field view: the same positions as marks on an empty
    rectangle, shown beside a panel already displaying the chart. Two drawings
    of one target competing to be believed is worse than either, so the marks
    moved onto the capture and the abstract one went.
    """
    from deltachrome_ui.charts import group_styles
    from deltachrome_ui.detail_view import sample_regions

    regions = sample_regions(
        patch_run(), placed_metric(n=20), group_styles(["chromatic"])
    )
    assert len(regions) == 20


def test_a_metric_offers_a_picture_and_a_table_and_nothing_else(qt_app):
    """Two views, never three, and the same two names for every metric."""
    assert view_names(one_metric(placed_metric())) == ["Charts", "Values"]


def test_a_metric_whose_readings_have_no_place_gets_the_same_two(qt_app):
    page = one_metric(placed_metric(placed=False))
    assert view_names(page) == ["Charts", "Values"]
    assert open_view(page, "Values").findChild(PatchTable) is not None


def test_the_capture_panel_refuses_to_color_marks_by_pass_or_fail(qt_app):
    """These thresholds are conditions on the aggregate -- a mean, a 90th
    percentile, a worst case. Coloring each mark against one would show a
    per-reading verdict the profile never formed.

    The sentence saying so is documentation now, so the rule is asserted where
    it lives: a mark takes its group's colour, and no verdict tone is ever in
    that palette.
    """
    from deltachrome_ui.charts import group_styles
    from deltachrome_ui.detail_view import sample_regions

    metric = placed_metric(n=20)
    # Readings spanning two orders of magnitude, one of them flagged.
    for index, sample in enumerate(metric["per_patch"]):
        sample["value"] = 0.1 + index * 4.0
    regions = sample_regions(patch_run(), metric, group_styles(["chromatic"]))

    assert len(regions) == 20
    # One group, one colour, whatever any of them read. The region a mark is
    # drawn from carries no value and no verdict at all, which is the structural
    # form of the rule rather than a promise about a palette.
    assert len({r.color.name() for r in regions.values()}) == 1


def test_a_field_of_marks_spans_the_middle_of_its_readings(qt_app):
    """One contaminated patch must not flatten the rest into identical marks.

    The size ramp is :func:`value_extent`, which the capture panel and the field
    map share; above a dozen readings it takes the middle 90% and leaves the
    outlier at the end of the ramp rather than redefining it.
    """
    from deltachrome_ui.charts import value_extent

    readings = [1.0 + i * 0.05 for i in range(20)] + [40.0]
    low, high = value_extent(readings)
    assert high < 10.0
    # Short fields keep their whole range: with nine readings there is no middle
    # 90% worth taking, and clipping would discard the worst one.
    assert value_extent([float(i) for i in range(9)]) == (0.0, 8.0)


def test_marks_shrink_as_the_field_fills_up(qt_app):
    """Nine edges and a hundred and forty patches are both fields, and a mark
    sized for one is unreadable as the other."""
    sparse, dense = FieldMap(), FieldMap()
    sparse.set_points([
        FieldPoint(i / 9, 0.5, 1.0, "a", label=f"p{i}") for i in range(9)
    ])
    dense.set_points([
        FieldPoint((i % 14) / 14, (i // 14) / 10, 1.0, "a", label=f"p{i}")
        for i in range(140)
    ])
    sparse.resize(400, 300)
    dense.resize(400, 300)
    assert sparse._scale(sparse._area())[1] > dense._scale(dense._area())[1]


def test_a_dense_field_drops_its_captions_rather_than_smearing_them(qt_app):
    dense = FieldMap()
    dense.set_points([
        FieldPoint((i % 14) / 14, (i // 14) / 10, 1.0, "a", label=f"p{i}")
        for i in range(140)
    ])
    dense.resize(400, 300)
    dense.grab()
    assert len(dense.points) > dense.CAPTION_LIMIT


def test_a_group_keeps_its_color_between_runs(qt_app):
    """Python randomises string hashing per process, and a role that changes
    color when the application restarts is a role nobody can learn."""
    import zlib

    from deltachrome_ui.charts import _GROUP_PALETTE, group_color

    # Pinned to crc32 rather than to whatever hash() gave this process.
    expected = _GROUP_PALETTE[zlib.crc32(b"achromatic_ramp") % len(_GROUP_PALETTE)]
    assert group_color("achromatic_ramp") == expected
    assert group_color("horizontal") == ORIENTATION_COLORS["horizontal"]


# -- the edge table joins the same selection ---------------------------------


def test_hovering_a_readings_row_lights_the_edge_everywhere(qt_app):
    """The row and the curve are two renderings of one edge, so the cursor has
    to mean the same thing in both -- and now in the capture panel too."""
    selection = ChartSelection()
    readings = EdgeReadings(
        FakeRun(edges=[edge_record("e0"), edge_record("e1")]), selection
    )
    table = readings.findChild(DataTable)
    index = table.model().index(1, 0)
    assert selection.hovered is None
    table._selection.set_hovered(table._name_at(index))
    assert selection.hovered in {"e0", "e1"}


def test_clicking_a_readings_row_pins_the_edge(qt_app):
    selection = ChartSelection()
    readings = EdgeReadings(FakeRun(edges=[edge_record("e0")]), selection)
    table = readings.findChild(DataTable)
    table._on_clicked(table.model().index(0, 0))
    assert selection.selected == "e0"


def test_one_bad_reading_does_not_flatten_the_rest(qt_app):
    """The same failure a mean has when one sample is dust -- and the failure
    this whole detail view exists to expose rather than repeat."""
    field = FieldMap()
    points = [FieldPoint(i / 20, 0.5, 1.0 + i * 0.05, "a", label=f"p{i}")
              for i in range(20)]
    points.append(FieldPoint(0.5, 0.9, 40.0, "a", label="dust"))
    field.set_points(points)
    field.resize(400, 300)
    low, high = field._extent()
    assert high < 10.0
    assert field.clips_extremes

    # The outlier is still drawn, at the top of the ramp rather than off it.
    scale = field._scale(field._area())
    assert field._radius(points[-1], low, high, scale) == pytest.approx(scale[1])


def test_a_short_field_spans_its_whole_range(qt_app):
    """With nine edges there is no middle 90% worth taking, and clipping would
    throw away the worst edge, which is the one being looked for."""
    field = FieldMap()
    field.set_points([
        FieldPoint(i / 9, 0.5, float(i), "a", label=f"p{i}") for i in range(9)
    ])
    assert field._extent() == (0.0, 8.0)
    assert not field.clips_extremes


def test_groups_in_one_view_never_share_both_a_color_and_a_shape(qt_app):
    """Two groups that look identical are a broken key, and a hash cannot
    promise they will not collide."""
    from deltachrome_ui.charts import group_styles

    styles = group_styles(["achromatic_border", "achromatic_ramp", "chromatic"])
    assert len({(c.name(), g) for c, g in styles.values()}) == 3


def test_orientation_keeps_its_appearance_wherever_it_appears(qt_app):
    from deltachrome_ui.charts import group_styles

    styles = group_styles(["horizontal", "vertical"])
    assert styles["horizontal"] == (ORIENTATION_COLORS["horizontal"], "○")
    assert styles["vertical"] == (ORIENTATION_COLORS["vertical"], "□")


def test_the_key_and_the_map_cannot_disagree_about_a_group(qt_app):
    from deltachrome_ui.charts import group_styles

    styles = group_styles(["chromatic"])
    field = FieldMap(styles=styles)
    assert field._style("chromatic") == styles["chromatic"]


# -- the color records ------------------------------------------------------


def channel_edge(edge_id="e0", orientation="horizontal", spread=True):
    edge = edge_record(edge_id, orientation)
    edge.channels = {
        "r": {"mtf": [1.0, 0.24, 0.1], "sfr50_pct": 63.0 if spread else 70.0,
              "mtf_at_nyquist": 0.24},
        "g": {"mtf": [1.0, 0.31, 0.1], "sfr50_pct": 70.0, "mtf_at_nyquist": 0.31},
        "b": {"mtf": [1.0, 0.19, 0.1], "sfr50_pct": 58.0 if spread else 70.0,
              "mtf_at_nyquist": 0.19},
    }
    return edge


def test_a_channel_curve_belongs_to_its_orientation_as_well_as_its_record(qt_app):
    """Hiding the verticals has to take their red curve with it, or the two keys
    disagree about what is on screen."""
    selection = ChartSelection()
    assert selection.alpha("v0", ("vertical", "r")) == 1.0
    selection.set_muted("vertical", True)
    assert selection.alpha("v0", ("vertical", "r")) == 0.0
    selection.set_muted("vertical", False)
    selection.set_muted("r", True)
    assert selection.alpha("v0", ("vertical", "r")) == 0.0
    assert selection.alpha("v0", ("vertical", "g")) == 1.0


def test_pointing_at_one_curve_lights_all_three_records_of_that_edge(qt_app):
    """That comparison -- this edge's R against its own G and B -- is the whole
    reason the records are drawn, so the series are named for the edge."""
    diagnostics = EdgeCurves(FakeRun(edges=[channel_edge("e0"), channel_edge("e1")]))
    plot = diagnostics.findChild(LinePlot)
    for record, _ in EdgeCurves.RECORDS:
        diagnostics.selection.set_muted(record, False)
    diagnostics.selection.toggle_selected("e0", isolate=True)
    lit = {s.name for s in plot.series
           if diagnostics.selection.alpha(s.name, s.group) == 1.0}
    assert lit == {"e0"}
    # Four: luminance and the three records, all one edge.
    assert sum(1 for s in plot.series if s.name == "e0") == 4


def test_the_colour_records_are_off_until_they_are_asked_for(qt_app):
    """Twenty edges in three records is eighty curves, which is a texture and
    not a plot. It opens on the luminance spread and narrows on request."""
    diagnostics = EdgeCurves(FakeRun(edges=[channel_edge("e0"), channel_edge("e1")]))
    plot = diagnostics.findChild(LinePlot)
    drawn = [s for s in plot.series
             if diagnostics.selection.alpha(s.name, s.group) > 0]
    assert len(drawn) == 2


def test_choosing_a_position_narrows_every_rendering_of_the_edges(qt_app):
    """Muting rather than rebuilding, so the plot, the values table and the
    capture panel cannot disagree about which edges are in play."""
    edges = [edge_record("c0"), edge_record("t0")]
    edges[0].region, edges[1].region = "module_center", "module_tl"
    diagnostics = EdgeCurves(FakeRun(edges=edges))
    diagnostics._choose_region("module_center")
    assert diagnostics.selection.alpha("c0", ("horizontal", "lum", "at:module_center")) > 0
    assert diagnostics.selection.alpha("t0", ("horizontal", "lum", "at:module_tl")) == 0
    diagnostics._choose_region("")
    assert diagnostics.selection.alpha("t0", ("horizontal", "lum", "at:module_tl")) > 0


def test_the_location_chips_follow_a_narrowing_they_did_not_cause(qt_app):
    edges = [edge_record("c0"), edge_record("t0")]
    edges[0].region, edges[1].region = "module_center", "module_tl"
    diagnostics = EdgeCurves(FakeRun(edges=edges))
    diagnostics._choose_region("module_tl")
    chips = diagnostics._position_chips.chips
    assert chips["at:module_tl"].isChecked()
    assert not chips["at:module_center"].isChecked()


def test_a_field_position_cannot_collide_with_a_colour_record(qt_app):
    """They share one mute namespace, so a region called "r" would vanish the
    moment the red record was switched off."""
    edges = [edge_record("e0")]
    edges[0].region = "r"
    diagnostics = EdgeCurves(FakeRun(edges=edges))
    plot = diagnostics.findChild(LinePlot)
    drawn = [s for s in plot.series
             if diagnostics.selection.alpha(s.name, s.group) > 0]
    assert len(drawn) == 1


def test_each_sfr_metric_puts_its_aim_somewhere_else_on_the_same_curve(qt_app):
    """Four metrics, one picture. The box moves; the plot does not."""
    from deltachrome_ui.detail_view import NYQUIST, sfr_aim

    boxes, _ = sfr_aim({"metric_id": "sfr50", "condition": "45 <= value <= 65"})
    assert boxes[0][0] == pytest.approx(0.45 * NYQUIST)
    assert boxes[0][1] == pytest.approx(0.65 * NYQUIST)

    boxes, _ = sfr_aim({"metric_id": "sfr10_sampling_efficiency",
                        "condition": "value > 90"})
    assert boxes[0][0] == pytest.approx(0.90 * NYQUIST)

    boxes, _ = sfr_aim({"metric_id": "sfr_response_at_nyquist",
                        "condition": "value < 0.2"})
    assert boxes[0][3] == pytest.approx(0.2)

    _, lines = sfr_aim({"metric_id": "sharpening", "condition": "value < 1.02"})
    assert lines[0][0] == pytest.approx(1.02)


def test_a_metric_with_no_threshold_draws_no_aim_rather_than_a_guess(qt_app):
    from deltachrome_ui.detail_view import sfr_aim

    assert sfr_aim(None) == ([], [])
    assert sfr_aim({"metric_id": "sfr50", "condition": None}) == ([], [])


def test_a_target_whose_edges_carry_no_records_offers_no_record_chips(qt_app):
    diagnostics = EdgeCurves(FakeRun(edges=[edge_record("e0")]))
    assert len(diagnostics.findChildren(LinePlot)) == 1
    assert "Luminance" not in text_of(diagnostics)


def test_the_edge_table_reports_each_record_rather_than_their_spread(qt_app):
    """One "how far apart are they" number says an edge has a chromatic problem
    without saying which record is soft, which is the thing to act on -- and a
    column per record is what OpenDICE prints."""
    rows = table_rows(EdgeReadings(FakeRun(edges=[channel_edge("e0")])))
    assert "63.0" in rows[0] and "70.0" in rows[0] and "58.0" in rows[0]


def test_a_record_that_was_not_measured_shows_no_reading_for_it(qt_app):
    edge = edge_record("e0")
    edge.channels = {"g": {"mtf": [], "sfr50_pct": 70.0, "mtf_at_nyquist": 0.3}}
    assert "—" in text_of(EdgeReadings(FakeRun(edges=[edge])))


def test_an_edge_is_named_by_the_direction_it_measured(qt_app):
    """ISO 12233 names the direction the response was measured in, not the way
    the edge runs: horizontal resolution comes off a near-vertical edge. The
    record keeps the geometry, which is what it is; the report shows the
    direction, which is what the registration grid beside it already showed."""
    rows = table_rows(EdgeReadings(FakeRun(edges=[
        edge_record("e0", "vertical"), edge_record("e1", "horizontal"),
    ])))
    named = {r.split(" | ")[0]: r.split(" | ")[1] for r in rows}
    assert named["e0"] == "Horizontal"
    assert named["e1"] == "Vertical"


def test_the_edge_angle_is_unsigned(qt_app):
    """Which way an edge leans does not change what it reads; how far it is from
    the axis does."""
    rows = table_rows(EdgeReadings(FakeRun(edges=[edge_record("e0")])))
    assert "5.00" in rows[0] and "-5.00" not in rows[0]


# -- the pixels behind a mark ------------------------------------------------


class FakeProxy:
    """Enough of ImageProxy to answer a region read, and to record what it read."""

    def __init__(self, width=900, height=900):
        from PySide6.QtGui import QImage

        self.width, self.height = width, height
        self.reads = []
        self.overview = QImage(60, 60, QImage.Format.Format_RGB888)

    def stride_for(self, region, target_px):
        x0, _, x1, _ = region
        return max(1, int((x1 - x0) / max(1, target_px)))

    def tile(self, region, *, stride=1):
        from PySide6.QtGui import QImage

        from deltachrome_ui.imageproxy import Tile

        x0, y0, x1, y1 = region
        self.reads.append((region, stride))
        image = QImage(
            max(1, (x1 - x0) // stride), max(1, (y1 - y0) // stride),
            QImage.Format.Format_RGB888,
        )
        image.fill(0)
        return Tile(image=image, region=(x0, y0, x1, y1), stride=stride)


def bounded_edge(edge_id="e0", bounds=(151, 291, 251, 566), center=(200.0, 420.0)):
    edge = edge_record(edge_id)
    edge.bounds_px = list(bounds)
    edge.center_px = list(center)
    return edge


def test_the_capture_view_reads_the_roi_the_edge_was_measured_on(qt_app):
    """The gap between "edge 7 is soft" and "edge 7 sits on a dust speck", which
    no number distinguishes and the pixels distinguish immediately."""
    from deltachrome_ui.capture_dock import RegionLoupe
    from deltachrome_ui.detail_view import edge_regions

    proxy = FakeProxy()
    selection = ChartSelection()
    capture = RegionLoupe(proxy, selection, edge_regions([bounded_edge("e0")]))
    capture.resize(600, 300)
    selection.set_hovered("e0")
    # Padded, so the read reaches around the region -- the border is usually
    # what was in doubt -- and the outline drawn on top says what was measured.
    region = proxy.reads[-1][0]
    assert region[0] < 151 and region[1] < 291
    assert region[2] > 251 and region[3] > 566


def test_an_roi_that_fits_is_read_at_full_resolution(qt_app):
    from deltachrome_ui.capture_dock import RegionLoupe
    from deltachrome_ui.detail_view import edge_regions

    proxy = FakeProxy()
    selection = ChartSelection()
    capture = RegionLoupe(proxy, selection, edge_regions([bounded_edge("e0")]))
    capture.resize(600, 300)
    selection.set_hovered("e0")
    assert proxy.reads[-1][1] == 1


def test_an_roi_too_tall_for_the_panel_is_read_coarser_and_says_so(qt_app):
    """Reading finer than the panel can show and scaling down would spend the
    read on detail nobody sees; scaling a coarse read up would show interpolation
    as detail. So the read follows the panel, and the stride is stated."""
    from deltachrome_ui.capture_dock import RegionLoupe
    from deltachrome_ui.detail_view import edge_regions

    proxy = FakeProxy(width=9000, height=9000)
    selection = ChartSelection()
    capture = RegionLoupe(proxy, selection, edge_regions([
        bounded_edge("e0", bounds=(0, 0, 1200, 3000), center=(600.0, 1500.0))
    ]))
    capture.resize(600, 300)
    selection.set_hovered("e0")
    assert proxy.reads[-1][1] > 1
    # The stride itself is no longer printed. It is a property of the panel's
    # width rather than of the measurement, and it was sitting in a readout
    # beside the reading, where it read as something the run had decided.


def test_hovering_wins_over_the_pin_and_letting_go_falls_back_to_it(qt_app):
    """Anything showing one thing has to break the tie focus deliberately does
    not. Falling back to nothing would make pinning pointless."""
    selection = ChartSelection()
    selection.toggle_selected("e0")
    assert selection.primary == "e0"
    selection.set_hovered("e1")
    assert selection.primary == "e1"
    selection.set_hovered(None)
    assert selection.primary == "e0"


def test_the_capture_panel_says_why_it_is_empty_rather_than_looking_broken(qt_app):
    """Nothing is missing from the measurement -- every reading was taken from
    the file while it was open."""
    from deltachrome_ui.capture_dock import CaptureDock

    dock = CaptureDock(ChartSelection())
    text = text_of(dock)
    assert "is not open" in text
    assert "report is complete" in text


# -- opening a patch onto the capture ----------------------------------------


class FakePatchRecord:
    def __init__(self, patch_id, roi_px):
        self.patch_id = patch_id
        self.roi_px = roi_px


def patch_run(roi=True):
    return FakeRun(patches=[
        FakePatchRecord(f"P{i}", [[i * 10.0, 0.0], [i * 10.0 + 8, 0.0],
                                  [i * 10.0 + 8, 8.0], [i * 10.0, 8.0]] if roi else None)
        for i in range(20)
    ])


def test_a_patch_region_is_joined_from_its_own_record(qt_app):
    """Stored once there rather than on every PatchValue: the metrics repeat
    their per-patch lists, and a hundred and forty patches across six metrics
    would write the same four corners six times."""
    from deltachrome_ui.detail_view import sample_regions
    from deltachrome_ui.charts import group_styles

    regions = sample_regions(
        patch_run(), placed_metric(n=20), group_styles(["chromatic"])
    )
    assert len(regions) == 20
    assert regions["P3"].outline[0] == [30.0, 0.0]


def test_a_sample_that_is_not_a_patch_carries_its_own_region(qt_app):
    """A flat-field spot, a rule band and a registration feature appear in one
    metric each and number tens, so there is nothing to save by joining."""
    from deltachrome_ui.detail_view import sample_regions
    from deltachrome_ui.charts import group_styles

    metric = {
        "metric_id": "illumination_uniformity", "units": "delta L*",
        "per_patch": [{
            "patch_id": "r0c0", "role": "flat_field", "value": 96.0, "flags": [],
            "position_relative": [0.0, 0.0],
            "roi_px": [[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]],
        }],
    }
    regions = sample_regions(FakeRun(patches=[]), metric, group_styles(["flat_field"]))
    assert regions["r0c0"].outline[1] == [40.0, 0.0]


def test_a_reading_with_no_region_anywhere_is_simply_absent(qt_app):
    from deltachrome_ui.detail_view import sample_regions
    from deltachrome_ui.charts import group_styles

    regions = sample_regions(
        patch_run(roi=False), placed_metric(n=20), group_styles(["chromatic"])
    )
    assert regions == {}


def test_the_read_reaches_around_the_region_and_the_outline_says_where(qt_app):
    """Showing exactly the sampled area answers "is the middle clean", which is
    rarely what was in doubt -- a speck on the border of a patch is."""
    from deltachrome_ui.detail_view import CaptureRegion

    region = CaptureRegion(outline=[[100, 100], [200, 100], [200, 200], [100, 200]],
                           color=QColor(200, 200, 200))
    x0, y0, x1, y1 = region.bounds(0.35)
    assert x0 == 65 and y0 == 65
    assert x1 == 236 and y1 == 236


def test_a_metric_page_is_complete_without_the_capture(qt_app):
    """The report renders from the record; only the pixels are missing."""
    page = one_metric(placed_metric(n=20), run=patch_run(), proxy=None)
    assert open_view(page, "Values").findChild(PatchTable) is not None


def test_the_capture_panel_draws_the_marks_once_the_pixels_arrive(qt_app):
    from deltachrome_ui.capture_dock import CaptureDock, FrameOverview
    from deltachrome_ui.charts import group_styles
    from deltachrome_ui.detail_view import sample_regions

    dock = CaptureDock(ChartSelection())
    assert dock.findChild(FrameOverview) is None
    dock.set_proxy(FakeProxy())
    dock.set_regions(
        sample_regions(patch_run(), placed_metric(n=20), group_styles(["chromatic"])),
        {"P0": (1.0, "dE2000", [])}, "average color accuracy",
    )
    overview = dock.findChild(FrameOverview)
    assert overview is not None and len(overview.regions) == 20


# -- direction on the map ----------------------------------------------------


def vector_metric(field):
    """field: [(u, v, dx, dy)] -> a metric with per-axis readings and arrows."""
    per_patch, vectors = [], []
    for i, (u, v, dx, dy) in enumerate(field):
        for axis, value in (("x", abs(dx)), ("y", abs(dy))):
            per_patch.append({
                "patch_id": f"m{i}:{axis}", "role": "worst pair G-R",
                "value": value, "flags": [],
                "position_relative": [u, v],
            })
        vectors.append({
            "label": f"m{i}", "position_relative": [u, v],
            "vector_px": [dx, dy], "source": "G-R",
        })
    return {
        "metric_id": "color_channel_misregistration", "status": "PASS",
        "value": 0.4, "units": "pixels", "condition": "value < 0.5",
        "outcome": "evaluated", "notes": [], "per_patch": per_patch,
        "vectors": vectors,
    }


RADIAL = [(0.2, 0.2, -0.3, -0.3), (0.8, 0.2, 0.3, -0.3),
          (0.8, 0.8, 0.3, 0.3), (0.2, 0.8, -0.3, 0.3)]


def test_the_field_map_still_draws_a_direction_it_is_given(qt_app):
    """No view builds one at the moment.

    Readings that carry a direction -- registration displacements -- were drawn
    as arrows on the Field view, and the Field view is gone: the positions moved
    onto the capture, which draws marks and not yet directions. The widget stays
    a charting primitive beside LinePlot and GroupLegend, tested, for when the
    arrows go back on the capture. What is *not* true today is that a
    displacement direction is visible anywhere, and the roadmap says so rather
    than this pretending otherwise.
    """
    field = FieldMap()
    field.set_points([
        FieldPoint(u, v, None, "displacement", label=f"m{i}", vector=(dx, dy))
        for i, (u, v, dx, dy) in enumerate(RADIAL)
    ])
    field.resize(400, 300)
    field.grab()
    assert field.vector_extent == pytest.approx(0.4243, abs=1e-3)
    field.selection.set_muted("displacement", True)
    assert field.vector_extent == 0.0


def test_two_readings_at_one_place_do_not_overprint_their_captions(qt_app):
    """The normal case, not the corner case: a square presents two orientations
    and a region is measured on two axes."""
    field = FieldMap()
    field.set_points([
        FieldPoint(0.5, 0.5, 1.0, "a", label="p0"),
        FieldPoint(0.5, 0.5, 2.0, "a", label="p1"),
    ])
    field.resize(400, 400)
    area = field._area()
    center = area.to_px(0.5, 0.5)
    placed = []
    first = field._caption_rect(center, 6.0, field.points[0], placed)
    placed.append(first)
    second = field._caption_rect(center, 6.0, field.points[1], placed)
    assert second is not None
    assert not first.intersects(second)


def test_a_caption_with_nowhere_left_to_go_is_dropped_not_piled_up(qt_app):
    """The value is still in the metric's table and still named on hover."""
    field = FieldMap()
    point = FieldPoint(0.5, 0.5, 1.0, "a", label="p0")
    field.set_points([point])
    field.resize(400, 400)
    center = field._area().to_px(0.5, 0.5)
    placed = []
    for _ in range(len(FieldMap.CAPTION_OFFSETS)):
        rect = field._caption_rect(center, 6.0, point, placed)
        assert rect is not None
        placed.append(rect)
    assert field._caption_rect(center, 6.0, point, placed) is None


# -- the report's shape ------------------------------------------------------


def test_a_legend_never_dictates_how_wide_its_container_must_be(qt_app):
    """The bug this caught was invisible at the legend and obvious three
    containers up: a non-wrapping row of chips reported a minimum width equal to
    all of them side by side, which forced the whole detail pane wider than the
    space it had -- so the prose beside it was cut off at the edge rather than
    wrapped, and one legend entry was pushed off-screen entirely."""
    from PySide6.QtGui import QColor

    from deltachrome_ui.charts import ChartSelection, GroupLegend

    selection = ChartSelection()
    entries = [
        (f"group{i}", QColor(200, 200, 200), f"a fairly long legend label {i}")
        for i in range(6)
    ]
    legend = GroupLegend(selection, entries)
    one_chip = max(
        legend.chips[group].sizeHint().width() for group, _, _ in entries
    )
    # Room for one chip is enough; the rest wrap underneath.
    assert legend.minimumSizeHint().width() <= one_chip + 4
    legend.resize(one_chip + 10, legend.heightForWidth(one_chip + 10))
    legend.grab()


def test_the_flow_layout_wraps_and_reports_the_height_it_needs(qt_app):
    from PySide6.QtWidgets import QLabel, QWidget

    from deltachrome_ui.charts import FlowLayout

    holder = QWidget()
    layout = FlowLayout(holder, spacing=4)
    for _ in range(4):
        label = QLabel("chip")
        label.setFixedSize(60, 20)
        layout.addWidget(label)

    # Four 60px chips with 4px gaps span 252px. In 300 they are one row of 20;
    # in 130, two rows; in 70, four.
    assert layout.heightForWidth(300) == 20
    assert layout.heightForWidth(130) == 44
    assert layout.heightForWidth(70) == 92


def test_a_run_is_grouped_by_what_it_measures_without_being_nested(qt_app):
    """The grouping is real and comes from each metric's declared capability.

    What changed is that it is a *heading* rather than a node: a reader clicking
    the word "Color" wants the colour metrics, and a family page gave them a
    summary of the rows already listed underneath it. So the families label the
    list and every metric is one click from every other.
    """
    from deltachrome_ui.nav import METRIC, PAGE
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    targets = view.nav.targets()

    assert targets[0] == (PAGE, OVERVIEW_KEY)
    assert (METRIC, "color_accuracy", "average_color_accuracy") in targets
    # Nothing selectable is a family or a view any more.
    assert {t[0] for t in targets} == {PAGE, METRIC}


def test_every_metric_the_run_judged_is_one_click_away(qt_app):
    from deltachrome_ui.nav import METRIC
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    result = _fake_result()
    view.set_result(result, saved=True)
    listed = {t[2] for t in view.nav.targets() if t[0] == METRIC}
    assert listed == {m["metric_id"] for m in result.run.metrics}


def test_the_families_are_headings_and_cannot_be_navigated_to(qt_app):
    """A heading that answers a click is one more place to be."""
    from PySide6.QtCore import Qt as _Qt
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    headings = [
        view.nav.item(i) for i in range(view.nav.count())
        if view.nav.item(i).data(_Qt.ItemDataRole.UserRole) is None
    ]
    assert headings, "the families still label the list"
    assert all(h.flags() == _Qt.ItemFlag.NoItemFlags for h in headings)


def test_a_run_opens_on_its_overview(qt_app):
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    assert view.pages.currentIndex() == view._page_index[(PAGE_KEY, OVERVIEW_KEY)]
    assert view.metric_table.model().rowCount() == len(_fake_result().run.metrics)


def test_the_overview_lists_every_metric_with_its_reading_and_condition(qt_app):
    """A table, not fifteen boxes. Fifteen is where a FADGI profile starts."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    rows = table_rows(view)
    # The criterion without its subject: the row already says what the value is.
    assert any("Average Color Accuracy" in r and "< 3.5" in r for r in rows)
    assert not any("value <" in r for r in rows)
    assert any("Bit Depth" in r for r in rows)


def test_the_overview_sorts_on_the_key_a_column_declares(qt_app):
    """The header used to be clickable, draw a chevron, and reorder nothing.

    It sorts now, on ``Col.sort`` -- which matters because the key is rarely the
    text: a value shown as ``10.500`` sorts before ``9.100`` as a string.
    """
    from PySide6.QtCore import Qt as _Qt
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    table = view.metric_table
    assert table.horizontalHeader().sectionsClickable()

    values = next(i for i, c in enumerate(table.source.columns)
                  if c.title == "Value")
    table.sortByColumn(values, _Qt.SortOrder.AscendingOrder)
    model = table.model()
    readings = [model.index(r, values).data() for r in range(model.rowCount())]
    numeric = [float(v) for v in readings if v not in ("", "—")
               and v.replace(".", "", 1).replace("-", "", 1).isdigit()]
    assert numeric == sorted(numeric)


def test_a_report_opens_in_the_order_the_guidelines_declare(qt_app):
    """Turning sorting on makes Qt sort by the current indicator, and the default
    indicator is section 0 -- which here is the verdict dot. Left alone, every
    report would have opened with its metrics reordered by status."""
    from deltachrome_ui.result_view import ResultView, metric_name

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    model = view.metric_table.model()
    shown = [model.index(r, 1).data() for r in range(model.rowCount())]
    declared = [metric_name(m) for m in _fake_result().run.metrics]
    assert shown == declared


def test_a_column_is_never_narrower_than_what_is_in_it(qt_app):
    """'Color Accuracy 90th Percentile' and '% of half-sampling frequency' are
    both longer than any width guessed in pixels, and a width guessed against a
    font the machine chooses elides exactly the row being looked for."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    table = view.metric_table
    for index, column in enumerate(table.source.columns):
        if column.fit:
            assert table.columnWidth(index) >= min(
                table.sizeHintForColumn(index), table.FIT_MAXIMUM
            ), column.title


def test_the_value_column_reads_in_the_same_face_as_the_rest(qt_app):
    """It held a mono face to line decimal points up, and now also holds
    "JPEG 2000" and "Adobe RGB (1998)" -- a second typeface for one column of
    five, aligning numbers that are one per metric and never added up."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    values = [c for c in view.metric_table.source.columns if c.title == "Value"]
    assert values and not values[0].mono


def test_a_unit_is_written_the_way_the_standards_write_it(qt_app):
    """The record keeps the ASCII the profile stored; the screen shows the delta,
    which is the symbol every one of these measurements is written with."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    rows = table_rows(view)
    assert any("ΔE*2000" in r for r in rows)
    assert not any("dE2000" in r for r in rows)


def test_choosing_a_metric_reaches_its_page(qt_app):
    from deltachrome_ui.nav import METRIC
    from deltachrome_ui.result_view import MetricPage, ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    view._on_navigate((METRIC, "color_accuracy", "white_balance"))

    page = view.pages.currentWidget()
    assert isinstance(page, MetricPage)
    assert page.current_metric()["metric_id"] == "white_balance"


def test_a_metric_page_states_its_reading_without_a_tile(qt_app):
    from deltachrome_ui.nav import METRIC
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    view._on_navigate((METRIC, "color_accuracy", "white_balance"))
    text = text_of(view.pages.currentWidget())
    assert "White Balance" in text
    # The condition is not on the line any more -- it is the band on the chart.
    assert "value <" not in text


def test_the_page_and_the_list_cannot_disagree_about_the_metric(qt_app):
    from deltachrome_ui.nav import METRIC
    from PySide6.QtCore import Qt as _Qt
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    view.show_metric("color_accuracy", "white_balance")
    current = view.nav.currentItem().data(_Qt.ItemDataRole.UserRole)
    assert tuple(current) == (METRIC, "color_accuracy", "white_balance")


def test_a_saved_run_offers_re_running_and_not_saving_again(qt_app):
    """Saving a saved run would write a second record of one analysis under a
    new id, which is two rows in the library for one measurement.

    Save is the one action that stayed a word rather than becoming an icon: it
    is the primary control and the only one that writes anything, and a shape
    for "commit this to the archive" is the shape people hesitate over."""
    from PySide6.QtWidgets import QPushButton

    from deltachrome_ui.result_view import ResultView
    from deltachrome_ui.widgets import IconButton

    def state(view):
        return (
            {b.text() for b in view.header.findChildren(QPushButton) if b.text()},
            {b.kind for b in view.header.findChildren(IconButton)},
        )

    view = ResultView()
    view.set_result(_fake_result(), saved=True)
    words, icons = state(view)
    assert "rerun" in icons
    assert "Save to Library" not in words

    view.set_result(_fake_result(), saved=False)
    words, icons = state(view)
    assert "Save to Library" in words
    assert "rerun" not in icons


def test_the_caveats_stay_on_the_summary_and_never_move_into_a_pane(qt_app):
    """Documentation moves behind a click; a qualification does not. A reader
    fluent enough to stop opening panes is exactly the reader who must still see
    that this run cannot be quoted."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(_fake_result(qualified=True), saved=True)
    summary = text_of(view)
    # One line, and the sentences it stands for are on hover rather than gone.
    assert "Advisory" in summary
    assert "manufacturer values" in summary
    assert "nominal values" not in summary


def _fake_result(qualified: bool = False):
    """The smallest thing ResultView will render."""
    from deltachrome.profiles.evaluator import advisory_line
    from deltachrome.results.models import AnalysisRun
    from deltachrome_ui.library import SavedEvaluation, SavedResult

    run = AnalysisRun.from_dict({
        "run_id": "r", "created_utc": "",
        "image": {"filename": "c.tif"},
        "target": {"display_name": "T"}, "profile": {"display_name": "P"},
        "reference": {}, "geometry": {}, "sampling": {}, "color_management": {},
        "patches": [], "edges": [], "edge_field_summary": {}, "registration": [],
        "registration_summary": {}, "rulers": {},
        "metrics": [metric(metric_id="average_color_accuracy", n=8),
                    dict(metric(metric_id="white_balance", n=6),
                         notes=["L* is excluded; this is the a*b* difference."]),
                    {"metric_id": "bit_depth", "status": "PASS", "value": 8.0,
                     "units": "bits", "condition": "value >= 8", "per_patch": []}],
        "overall": "PASS", "overall_explanation": "Everything judged, met.",
    })
    caveats = ["judged against the manufacturer's nominal values"] if qualified else []
    line = advisory_line(
        reference_basis="manufacturer_nominal" if qualified else "per_unit_measurement",
        local_criteria=False,
    )
    return SavedResult(
        run=run,
        evaluation=SavedEvaluation("PASS", not qualified, caveats, line),
    )


# -- the chart, drawn from what was measured ---------------------------------


def _swatch(pid, measured, reference, x, flags=()):
    from deltachrome.results.models import PatchRecord

    return PatchRecord(
        patch_id=pid, row=0, col=0, role="chromatic", ramp_position=None,
        rgb_mean=[0, 0, 0], rgb_median=[0, 0, 0], rgb_std=[0, 0, 0],
        measured_lab=measured, reference_lab=reference, pixel_count=100,
        clipped_high_fraction=0.0, clipped_low_fraction=0.0, flags=list(flags),
        roi_px=[[x, 0], [x + 100, 0], [x + 100, 100], [x, 100]],
    )


def _halves(grid, index):
    """The color either side of one patch's seam.

    Sampled in *device* pixels. A grab on a retina display comes back at twice
    the widget's logical size, and reading logical coordinates out of it lands
    half as far in as intended -- which for one patch filling the widget means
    reading the background and concluding the patch was never drawn.
    """
    image = grid.grab().toImage()
    ratio = image.width() / max(grid.width(), 1)
    scale, dx, dy = grid._transform()
    points = grid._corners(grid.patches[index])
    xs = [p[0] * scale + dx for p in points]
    ys = [p[1] * scale + dy for p in points]
    y = int((min(ys) + max(ys)) / 2 * ratio)
    span = max(xs) - min(xs)
    left = int((min(xs) + span * 0.2) * ratio)
    right = int((min(xs) + span * 0.8) * ratio)
    return image.pixelColor(left, y).name(), image.pixelColor(right, y).name()


def test_a_patch_that_matched_shows_no_seam(qt_app):
    """Which is what makes the chart readable at a glance: the eye finds the
    patches that did not match without reading a single figure."""
    from deltachrome_ui.swatches import PatchGrid

    grid = PatchGrid([_swatch("A1", [54, 81, 70], [54, 81, 70], 0)])
    grid.resize(400, 240)
    left, right = _halves(grid, 0)
    assert left == right


def test_a_patch_that_did_not_match_shows_both_sides(qt_app):
    from deltachrome_ui.swatches import PatchGrid

    grid = PatchGrid([_swatch("A2", [54, 81, 70], [54, -40, 35], 0)])
    grid.resize(400, 240)
    left, right = _halves(grid, 0)
    assert left != right


def test_the_chart_is_drawn_at_patch_size_not_at_sample_size(qt_app):
    """The stored quadrilateral is the inset region actually sampled -- 0.55 of
    the patch by default -- so drawing it directly gives a field of small dots
    rather than a chart. The inset is defined as that fraction, so undoing it
    recovers the patch outline exactly."""
    from deltachrome_ui.swatches import PatchGrid

    patch = _swatch("A1", [50, 0, 0], [50, 0, 0], 0)
    tight = PatchGrid([patch])
    expanded = PatchGrid([patch], inset=0.55)
    assert expanded._expand == pytest.approx(1 / 0.55)

    width = lambda g: max(x for x, _ in g._corners(g.patches[0])) - min(
        x for x, _ in g._corners(g.patches[0])
    )
    assert width(expanded) == pytest.approx(width(tight) / 0.55)


def test_the_chart_marks_flagged_samples_but_never_verdicts(qt_app):
    """Clipping is a fact about the sample; passing is a judgement on an
    aggregate the profile formed, never on one patch. The first is drawn, the
    second is not -- the same refusal the field maps make."""
    import inspect

    from deltachrome_ui import swatches

    source = inspect.getsource(swatches)
    assert "passed" not in source
    assert "flags" in source


def test_a_patch_with_no_reference_is_still_drawn(qt_app):
    """The ISA Golden Thread's registration squares carry no printed values at
    all, and a chart that dropped them would be missing most of itself."""
    from deltachrome_ui.swatches import PatchGrid

    grid = PatchGrid([_swatch("A1", [50, 0, 0], None, 0)])
    assert len(grid.patches) == 1
    grid.resize(300, 200)
    grid.grab()


# -- explanation on demand ---------------------------------------------------


def test_a_view_states_what_this_run_did_and_nothing_standing(qt_app):
    """A count of readings changes every run and stays on screen. What a mark
    size means never changes, and is documentation now rather than a panel."""
    from deltachrome_ui.capture_dock import CaptureDock
    from deltachrome_ui.charts import group_styles
    from deltachrome_ui.detail_view import sample_regions

    dock = CaptureDock(ChartSelection())
    dock.set_regions(
        sample_regions(patch_run(), placed_metric(n=20), group_styles(["chromatic"])),
        {}, "average color accuracy",
    )
    visible = text_of(dock)

    assert "20 regions" in visible and "average color accuracy" in visible
    assert "middle 90%" not in visible


def test_a_run_specific_caveat_never_moves_behind_an_icon(qt_app):
    """A fact about THIS capture is part of the result. Filing it behind an icon
    would hide it from exactly the reader who has stopped opening icons."""
    from deltachrome_ui.detail_view import EdgeCurves

    diagnostics = EdgeCurves(FakeRun(edges=[]))
    assert "carries no slanted edges" in text_of(diagnostics)


def test_every_explanation_carries_a_title_and_a_body(qt_app):
    """The panel needs something to head itself with, and a topic with no title
    would open as an unlabelled wall of text."""
    from deltachrome_ui import explanations

    topics = [
        value for name, value in vars(explanations).items()
        if name.isupper() and isinstance(value, tuple)
    ]
    assert len(topics) >= 10
    for title_text, body in topics:
        assert title_text and body
        assert len(body) > len(title_text)
        # Written as prose to be read, not as a caption.
        assert body[0].isupper() and body.rstrip().endswith(".")


def test_no_screen_carries_an_explanation_icon(qt_app):
    """The application ships without any of it: a screen that answers its own
    questions cannot be shown to be unclear, and a beta is for finding out which
    parts of this one a person in the trade can read."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src" / "deltachrome_ui"
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "InfoButton" not in source, path.name
        assert "deltachrome_ui.info" not in source, path.name


def test_actions_and_feature_names_are_title_case(qt_app):
    """A convention worth a test because it is the kind that decays one label at
    a time. Actions and feature names take Title Case; a sentence-shaped prompt
    and a unit do not — "% of half-sampling" is a quantity, not a name."""
    import re
    from pathlib import Path

    # "bank" is the operator's own wording for the reference bank, chosen over
    # "Choose from the Bank…" on the grounds that a bank is a place rather than a
    # proper name. Listed rather than argued into the rule below, because the rule
    # is right and this is a preference about one label.
    SENTENCES = {"What is this a capture of?", "Place the Target",
                 "Choose from bank…"}
    SMALL = {"a", "an", "and", "the", "of", "to", "in", "on", "at", "for", "by",
             "from"}
    offenders = []
    for path in Path("src/deltachrome_ui").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        # The lookbehind matters: without it this matched ``subtitle(...)``
        # too, which is prose and is meant to be a sentence rather than a name.
        for match in re.finditer(
            r'(?<![A-Za-z_])(?:QPushButton|heading|title|elided_title)\("([^"]+)"\)',
            text,
        ):
            label = match.group(1)
            if label in SENTENCES or not any(c.isalpha() for c in label):
                continue
            # Only the part before a separator is a name; what follows is a unit
            # or a qualifier and stays as written.
            head = label.split("  ·  ")[0]
            words = [w for w in head.split() if w[:1].isalpha()]
            for index, word in enumerate(words):
                if index and word.lower() in SMALL:
                    continue
                if not word[0].isupper():
                    offenders.append(f"{path.name}: {label}")
                    break
    assert not offenders, offenders


# -- averaging a field of edges ---------------------------------------------


def field_of_edges():
    """Four edges: two directions at each of two field positions."""
    made = []
    for region in ("module_tl", "module_tr"):
        for orientation, mtf in (("horizontal", 0.30), ("vertical", 0.50)):
            edge = edge_record(f"{region}_{orientation[0]}", orientation)
            edge.region = region
            edge.mtf_curve = {"frequency_cyc_px": [0.0, 0.5, 1.0],
                              "mtf": [1.0, mtf, 0.1]}
            made.append(edge)
    return made


def test_a_field_of_edges_opens_averaged_per_location(qt_app):
    """The spread of every edge is a real finding and is the one a reader can go
    to; what they arrive asking is whether one part of the field is soft, and
    twenty overlaid curves is a texture that cannot be read for it."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    plot = curves.findChild(LinePlot)
    assert curves._average == "region"
    assert len(plot.series) == 2
    curves._choose_average("")
    assert len(plot.series) == 4


def test_a_frame_with_one_location_opens_on_its_edges(qt_app):
    """There is nothing to average across, so the default has nothing to say."""
    edges = field_of_edges()
    for edge in edges:
        edge.region = "module_cc"
    curves = EdgeCurves(FakeRun(edges=edges))
    assert curves._average == ""
    assert len(curves.findChild(LinePlot).series) == 4


def test_averaging_per_position_gives_one_curve_for_each_corner(qt_app):
    """Which corner is softer is the question a field is shot to ask, and twenty
    overlaid curves cannot be read for it."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    curves._choose_average("region")
    plot = curves.findChild(LinePlot)
    assert len(plot.series) == 2
    # Both directions at that position, so the mean of 0.30 and 0.50.
    assert plot.series[0].y[1] == pytest.approx(0.40)
    assert "mean of 2" in plot.series[0].name


def test_averaging_per_direction_gives_one_curve_for_each_direction(qt_app):
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    curves._choose_average("orientation")
    plot = curves.findChild(LinePlot)
    assert len(plot.series) == 2
    assert {round(s.y[1], 3) for s in plot.series} == {0.30, 0.50}


def test_an_average_covers_only_what_is_currently_shown(qt_app):
    """The two controls do the work of six: narrowing to one corner and
    averaging by direction is the horizontals against the verticals *there*."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    curves._choose_region("module_tl")
    curves._choose_average("orientation")
    plot = curves.findChild(LinePlot)
    assert len(plot.series) == 2
    assert all("mean of 1" in s.name for s in plot.series)


def test_a_hidden_direction_is_out_of_the_mean_as_well_as_off_the_plot(qt_app):
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    curves._choose_average("region")
    curves.selection.set_muted("vertical", True)
    plot = curves.findChild(LinePlot)
    assert len(plot.series) == 2
    assert plot.series[0].y[1] == pytest.approx(0.30)


def test_the_location_chips_carry_the_curve_colours_only_while_averaging(qt_app):
    """A legend is its colours, except for this one, whose curves are coloured by
    *orientation* until the plot is asked to average per location."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    keyed = QColor(curves._position_chips.chips["at:module_tl"].color)
    curves._choose_average("")
    plain = QColor(curves._position_chips.chips["at:module_tl"].color)
    assert keyed != plain
    curves._choose_average("region")
    assert curves._position_chips.chips["at:module_tl"].color == keyed


def test_more_than_one_location_can_be_shown_at_once(qt_app):
    """A filter answered "which one corner". The question a field of nine modules
    is usually asking is "these three against those three"."""
    edges = field_of_edges()
    edges[0].region = edges[1].region = "module_tl"
    edges[2].region, edges[3].region = "module_tr", "module_br"
    curves = EdgeCurves(FakeRun(edges=edges))
    curves.selection.set_muted("at:module_br", True)
    plot = curves.findChild(LinePlot)
    assert {s.name.split(" mean")[0] for s in plot.series} == {"Upper Left",
                                                               "Upper Right"}


def test_the_location_legend_toggles_them_all_both_ways(qt_app):
    """Eight presses to undo one is a control missing rather than a convenience,
    and clearing the row is how a reader gets to *one* of nine: clear, then pick,
    which is two presses where muting eight is eight."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    legend = curves._position_chips
    curves._choose_region("module_tl")
    assert not legend.all_chip.isChecked()

    legend.all_chip.setChecked(True)
    legend._toggle_all()
    assert legend.all_chip.isChecked()
    # Its own groups only: the colour records are muted by default and putting
    # every *location* back must not switch three curves per edge on as well.
    assert not any(g.startswith("at:") for g in curves.selection.muted)

    legend.all_chip.setChecked(False)
    legend._toggle_all()
    assert all(f"at:{r}" in curves.selection.muted for r in curves.regions)
    assert not curves.findChild(LinePlot).series


def test_pointing_at_an_edge_does_not_rebuild_the_curves(qt_app):
    """A hover emits the same signal a mute does, and rebuilding every curve on
    every mouse move is a cost with nothing to show for it."""
    curves = EdgeCurves(FakeRun(edges=field_of_edges()))
    plot = curves.findChild(LinePlot)
    before = plot.series
    curves.selection.set_hovered("module_tl_h")
    assert plot.series is before


# -- the reading and the limit it is judged against --------------------------


def _chroma(color) -> int:
    """How far a colour is from neutral, as the spread of its three records."""
    return max(color.red(), color.green(), color.blue()) - min(
        color.red(), color.green(), color.blue())


def test_the_aim_band_carries_no_hue_at_all(qt_app):
    """It is an annotation, not a category and not a status.

    Both constants were ``QColor(255, 214, 0, ...)`` -- the brand accent to
    within a unit of green -- while the measured series was drawn in the accent
    itself, so a reading and the limit it was being judged against were one hue
    separated only by alpha. The band's own docstring said "deliberately
    neutral" the whole time; only the value had drifted.
    """
    from deltachrome_ui.charts import BAND_EDGE, BAND_FILL

    assert _chroma(BAND_FILL) <= 12
    assert _chroma(BAND_EDGE) <= 12


def test_a_reading_is_never_the_same_colour_as_the_band_behind_it(qt_app):
    """The property, rather than either of the two values that satisfy it.

    Telling the reading from the limit is the entire reason a band is drawn, so
    what has to hold is that they differ -- here by brightness, the band having
    receded to a faint wash under a dim dashed rule while the reading takes the
    brightest neutral. A later change may pick other colours; it may not pick
    one colour for both.
    """
    from deltachrome_ui.charts import BAND_EDGE

    def brightness(color) -> float:
        return (color.red() + color.green() + color.blue()) / 3.0

    measured = [s for s in tone_page(ramp_run()).findChildren(LinePlot)[0].series
                if s.name == "measured"][0]
    assert _chroma(measured.color) <= 12, "a lone series needs no hue"
    assert brightness(measured.color) - brightness(BAND_EDGE) > 40


# -- registration readings ---------------------------------------------------


def _registration_run():
    """Two positions, both axes, all three pairs -- the grid's own population."""
    import types

    positions = []
    for region in ("module_tl", "module_center"):
        for axis in ("x", "y"):
            positions.append({
                "region": region, "axis": axis, "paired": True,
                "translation_px": {"G-B": 0.011, "G-R": -0.009, "B-R": -0.020},
                "antisymmetric_px": {"G-B": 0.030, "G-R": -0.005, "B-R": -0.035},
                "standard_error_px": 0.0036,
            })
    return types.SimpleNamespace(registration_summary={"positions": positions})


def test_the_registration_readings_carry_the_antisymmetric_term(qt_app):
    """It was three sentences of prose ending "Worst here: G-B", which is a
    sorted column read aloud. No standard asks for the quantity -- FADGI 2.4.7
    and ISO 19264-1 both state the criterion in pixels of dislocation -- but it
    is what pairing takes *out* of the verdict, and on a real capture it runs
    larger than the translation it was removed from. A reader who cannot see it
    cannot tell whether pairing earned its keep."""
    from deltachrome_ui.detail_view import RegistrationReadings
    from deltachrome_ui.tables import DataTable

    # Held, not chained: the parent going out of scope takes the table's C++
    # object with it.
    view = RegistrationReadings(_registration_run())
    table = view.findChild(DataTable)
    headings = [table.model().headerData(i, Qt.Orientation.Horizontal)
                for i in range(table.model().columnCount())]
    assert "Antisymmetric" in headings
    assert "Translation" in headings
    # Every position, axis and pair -- the same population the grid draws,
    # rather than the worst pair per axis the verdict is formed on.
    assert table.model().rowCount() == 2 * 2 * 3


def test_a_registration_reading_keeps_its_sign(qt_app):
    """Two displacements of equal size and opposite direction are opposite
    findings, and a column of magnitudes cannot say which is which."""
    from deltachrome_ui.detail_view import RegistrationReadings

    text = text_of(RegistrationReadings(_registration_run()))
    assert "-0.0200" in text and "+0.0110" in text


def test_a_run_with_no_registration_positions_says_so(qt_app):
    import types

    from deltachrome_ui.detail_view import RegistrationReadings

    empty = types.SimpleNamespace(registration_summary={})
    assert "no per-position registration readings" in text_of(
        RegistrationReadings(empty)
    )


# -- the chip rows are one control, not a row of objects ---------------------


def _radii(chip) -> tuple[int, int]:
    """The left and right corner radii a chip's own style sheet declares."""
    import re

    sheet = chip.styleSheet()
    left = int(re.search(r"border-top-left-radius: (\d+)px", sheet).group(1))
    right = int(re.search(r"border-top-right-radius: (\d+)px", sheet).group(1))
    return left, right


def test_a_legend_key_is_a_square_and_a_word_with_no_box_round_it(qt_app):
    """The rows were strips of cells that lit up in their own colour -- tinted
    fill, coloured border, coloured text -- and three of them read as a neon
    sign. A key is drawn as one: the colour on its square, nowhere else."""
    from deltachrome_ui.charts import LegendKey

    legend = GroupLegend(ChartSelection(), [("a", QColor(64, 196, 180), "A")])
    key = legend.chips["a"]
    assert isinstance(key, LegendKey)
    assert key.color == QColor(64, 196, 180)
    sheet = key.styleSheet()
    assert "border: none" in sheet and "background: transparent" in sheet
    assert "64,196,180" not in sheet.replace(" ", ""), "no glow in the style"


def test_show_all_sits_at_the_end_of_its_row_and_says_what_it_does(qt_app):
    """Beside the row's caps heading it read as part of the heading, and inside
    the row as one more location. It is the one outlined button on the row, after
    the keys, and its word is the action a press takes."""
    selection = ChartSelection()
    legend = GroupLegend(
        selection, [(name, QColor(200, 200, 200), name) for name in ("a", "b")],
        all_label="All Locations",
    )
    layout = legend.layout()
    assert layout.itemAt(layout.count() - 1).widget() is legend.all_chip
    assert legend.all_chip.text() == "Clear"
    selection.set_muted("a", True)
    assert legend.all_chip.text() == "Show All"


def test_a_chosen_segment_is_grey_rather_than_lit(qt_app):
    """The pick-one row says which is chosen with a raised cell, not a colour."""
    from deltachrome_ui.charts import FilterChips

    chips = FilterChips([("a", "A"), ("b", "B")], current="a")
    assert "255,214,0" not in chips.chips["a"].styleSheet().replace(" ", "")


def test_a_single_choice_row_is_a_strip_too(qt_app):
    """`FilterChips` is exactly what a segmented control is -- one of several,
    always one. It shared `chip_style` with the legend before and still does."""
    from deltachrome_ui.charts import CHIP_RADIUS, FilterChips

    chips = FilterChips([("a", "A"), ("b", "B"), ("c", "C")], current="a")
    assert _radii(chips.chips["a"]) == (CHIP_RADIUS, 0)
    assert _radii(chips.chips["c"]) == (0, CHIP_RADIUS)


def test_a_wrapped_strip_does_not_overlap_the_row_above_it(qt_app):
    """The cells overlap horizontally by a pixel so two borders collapse into
    one seam. Carried into the vertical the same -1 would pull each row into the
    one above it, so the two spacings are separate numbers -- a 3x3 field is
    nine positions and an "all" chip, which wraps in any real pane.

    Asserted as the *difference* between two vertical spacings rather than as a
    height, because a widget that has never been shown gets its size from the
    style and the suite must not depend on what that is. Three rows means two
    gaps, so seven pixels of spacing is fourteen pixels of height either way.
    """
    from PySide6.QtWidgets import QWidget

    from deltachrome_ui.charts import CHIP_OVERLAP, FlowLayout

    def height(vertical: int) -> int:
        host = QWidget()
        layout = FlowLayout(host, CHIP_OVERLAP, vertical_spacing=vertical)
        for _ in range(9):
            chip = QPushButton("Upper Center")
            chip.setFixedWidth(100)
            layout.addWidget(chip)
        # 100 px cells overlapping by one: three to a 320 px row, so three rows.
        return layout.heightForWidth(320)

    gaps = 2
    assert height(6) - height(CHIP_OVERLAP) == gaps * (6 - CHIP_OVERLAP)
    assert height(6) > height(CHIP_OVERLAP)


def test_the_vertical_gap_defaults_to_the_horizontal_one(qt_app):
    """So a caller that does not care is unaffected by the strip's overlap."""
    from PySide6.QtWidgets import QWidget

    from deltachrome_ui.charts import FlowLayout

    def height(**kwargs) -> int:
        host = QWidget()
        layout = FlowLayout(host, 8, **kwargs)
        for _ in range(9):
            chip = QPushButton("Upper Center")
            chip.setFixedWidth(100)
            layout.addWidget(chip)
        return layout.heightForWidth(320)

    assert height() == height(vertical_spacing=8)


def test_stepping_to_another_chart_keeps_the_sfr_chips_as_they_were_left(qt_app):
    """Reported from the running application: comparing SFR50 across the targets
    of one capture meant re-pressing Orientation, Channel and Average on every
    chart. The view is carried on the report's selection, which outlives a chart
    switch; locations are not carried, each target's being its own."""
    from deltachrome_ui.charts import ChartSelection

    selection = ChartSelection()
    first = EdgeCurves(FakeRun(edges=field_of_edges()), selection)
    selection.set_muted("vertical", True)
    first._choose_average("orientation")

    selection.muted.clear()  # what the report does on a chart switch
    second = EdgeCurves(FakeRun(edges=field_of_edges()), selection)
    assert second._average == "orientation"
    assert "vertical" in selection.muted and "horizontal" not in selection.muted


def test_a_carried_view_never_leaves_the_plot_empty(qt_app):
    """A chart offering only the direction somebody turned off shows it anyway."""
    from deltachrome_ui.charts import ChartSelection

    selection = ChartSelection()
    selection.edge_view = {"muted": {"vertical"}, "average": "region"}
    only_vertical = [e for e in field_of_edges() if e.orientation == "vertical"]
    curves = EdgeCurves(FakeRun(edges=only_vertical), selection)
    assert "vertical" not in selection.muted
    assert curves.findChild(LinePlot).series


def test_only_a_metric_read_on_edges_gets_the_edges_in_the_capture_panel(qt_app):
    """The slanted edges, with their SFR50, were handed to every metric that
    placed nothing of its own: gain modulation's patch pairs, a metric the
    target cannot carry, the file's bit depth. And every SFR metric showed SFR50
    rather than its own reading of the edge."""
    from deltachrome_ui.detail_view import edge_readings_for

    run = FakeRun(edges=field_of_edges())
    for edge in run.edges:
        edge.bounds_px = [0, 0, 10, 10]
        edge.center_px = (5, 5)
    ids = [e.edge_id for e in run.edges]

    efficiency = {"per_patch": [{"patch_id": i, "value": 120.0 + n}
                                for n, i in enumerate(ids)]}
    regions, readings = edge_readings_for(run, efficiency, "% of Nyquist")
    assert set(regions) == set(ids)
    assert readings[ids[1]][0] == 121.0, "the metric's own reading, not SFR50"

    gain = {"per_patch": [{"patch_id": "10-13", "value": 0.92}]}
    assert edge_readings_for(run, gain, "") == ({}, {})
    assert edge_readings_for(run, {"per_patch": []}, "") == ({}, {})

    registration = {"per_patch": [{"patch_id": "module_tl:x", "value": 0.03}]}
    regions, readings = edge_readings_for(run, registration, "pixels")
    assert list(regions) == ["module_tl:x"] and readings["module_tl:x"][0] == 0.03


def test_a_gain_modulation_pair_is_shown_as_its_two_patches(qt_app):
    """Gain modulation compares two ramp steps, often with others between them,
    so its region is the two patches and not one box round the stretch."""
    from types import SimpleNamespace

    from deltachrome_ui.detail_view import pair_readings_for

    square = lambda x: [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]  # noqa: E731
    run = SimpleNamespace(patches=[
        SimpleNamespace(patch_id="11", roi_px=square(0)),
        SimpleNamespace(patch_id="14", roi_px=square(60)),
    ], edges=[])
    metric = {"per_patch": [{"patch_id": "11-14", "value": 0.95},
                            {"patch_id": "11-99", "value": 0.9}]}
    regions, readings = pair_readings_for(run, metric, "gain")
    assert list(regions) == ["11-14"], "a half with no patch behind it is not drawn"
    assert [s[0][0] for s in regions["11-14"].shapes] == [0.0, 60.0]
    assert regions["11-14"].bounds(0.0) == (0, 0, 71, 11)
    assert readings["11-14"][0] == 0.95


def test_an_unfinished_metric_with_nothing_read_is_its_reason_centred(qt_app):
    """The LaserSoft's highlight gain: its reason at the top, then a second line
    saying there was nothing to draw, left-aligned where the picture would be.
    The reason is the page now, centred, and stated once."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    reason = "No neutral pair averages L* 85–95 on this target; its ramp tops out at L* 88.0."
    page = one_metric({
        "metric_id": "gain_modulation_highlight", "outcome": "incomplete",
        "units": "gain", "reason": reason, "per_patch": [],
    }, key="tonal_response")
    labels = [label for label in page.findChildren(QLabel) if label.text() == reason]
    assert len(labels) == 1
    assert labels[0].alignment() & Qt.AlignmentFlag.AlignHCenter
    assert page.views.count() == 0
    assert "no per-reading values" not in text_of(page)
