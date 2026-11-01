"""The camera profile screen: patches clicked in and out, and what a build asks for.

Two rules are what these pin. **A deselection is the operator's whole answer** --
flagged patches start out and can be clicked back in, and Reset returns to the
sampler's own answer rather than to every patch. And **Save is offered only for
the panel it was built from**: any change after a build, a patch or an option,
puts the button back to Build Profile, so a saved profile cannot describe a
different panel from the one beside it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsSimpleTextItem  # noqa: E402

from deltachrome.imaging.loader import load_image  # noqa: E402
from deltachrome.profiling import ErrorSummary, gather_patches  # noqa: E402
from deltachrome.profiling.colprof import ColprofOptions  # noqa: E402
from deltachrome.profiling.encoding import detect_encoding  # noqa: E402
from deltachrome.reference import read_reference  # noqa: E402
from deltachrome.targets.registry import load_target  # noqa: E402
from deltachrome_ui import paths  # noqa: E402
from deltachrome_ui.imageproxy import ImageProxy  # noqa: E402
from deltachrome_ui.profile_view import ProfileView  # noqa: E402
from deltachrome_ui.workers import ProfileSetup  # noqa: E402
from fixtures import SyntheticChart  # noqa: E402

SG = "colorchecker_sg_v1"


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("profile")
    chart = SyntheticChart()
    reference = read_reference(chart.write_reference(folder / "ref.txt"))
    path, corners = chart.render_capture(folder / "capture.tif")
    image = load_image(path, profile_required=False)
    data = gather_patches(image, load_target(SG), reference, corners)
    return path, image, data


@pytest.fixture
def view(qt_app, capture, monkeypatch):
    path, image, data = capture
    monkeypatch.setattr(paths, "resolve_colprof", lambda: None)
    screen = ProfileView()
    screen.resize(1200, 800)
    setup = ProfileSetup(
        data=data, reference_name="ref.txt", keywords={"target": SG}, warnings=[],
        encoding=detect_encoding(data), merged_count=None,
    )
    screen.open(proxy=ImageProxy(image), image_path=path, target=load_target(SG),
                setup=setup)
    return screen


def test_the_screen_says_what_it_detected(view, capture):
    _, _, data = capture
    assert view.count_pill.text() == f"{len(data.all_patches)} Patches"
    assert view.used_label.text() == f"{len(data.all_patches)} of {len(data.all_patches)}"
    # No repeats were found to merge, so merging is not offered.
    assert not view.merge_check.isVisibleTo(view)


def test_a_patch_is_clicked_out_on_the_capture_and_back_in(view):
    view.show()
    view.canvas.fit()
    QApplication.processEvents()
    region = view.setup.data.regions["B2"]
    centre = QPointF(*np.mean(region, axis=0))
    assert view.canvas.patch_at(centre) == "B2"
    point = view.canvas.mapFromScene(centre)
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert view._excluded == {"B2"}
    QTest.mouseClick(view.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert view._excluded == set()


def test_reset_returns_to_the_samplers_own_answer(view):
    view._flagged = {"A1"}
    view._toggle_patch("C3")
    view._reset_patches()
    assert view._excluded == {"A1"}


def test_what_a_build_asks_for(view):
    view._toggle_patch("C3")
    request = view.request()
    assert "C3" in request["data"].excluded
    assert request["automatic"] and request["types"] is None
    assert request["scale_for"] == "white"
    view.type_combo.setCurrentIndex(view.type_combo.findData("matrix"))
    view.quality_combo.setCurrentIndex(view.quality_combo.findData("high"))
    request = view.request()
    assert not request["automatic"]
    assert request["options"].profile_type == "matrix"
    assert request["options"].quality == "high"
    # A scale typed in is used as given, and nothing is computed.
    view.scale_edit.setText("1.02")
    view._on_options_changed()
    request = view.request()
    assert request["options"].scale == 1.02 and request["scale_for"] is None


def test_smoothing_is_offered_for_a_table_only(view):
    view.advanced_toggle.setChecked(True)
    assert not view.smoothing_field.isVisibleTo(view)
    view.type_combo.setCurrentIndex(view.type_combo.findData("lab-table"))
    assert view.smoothing_field.isVisibleTo(view)


def test_without_colprof_nothing_can_be_built(view):
    assert not view.primary.isEnabled()
    assert "Settings" in view.primary.toolTip()


def test_any_change_after_a_build_takes_save_away(view, capture, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(paths, "resolve_colprof", lambda: "colprof")
    _, _, data = capture
    summary = ErrorSummary.of({"A1": 1.0})
    built = SimpleNamespace(
        icc=b"", fit=summary, held_out=summary, excluded=None, scale=1.0,
        options=ColprofOptions(profile_type="matrix"),
        run=SimpleNamespace(warnings=[]), data=data,
    )
    view.show_built(built)
    assert view.primary.text() == "Save Profile…"
    view._toggle_patch("D4")
    assert view.primary.text() == "Build Profile"
    view.show_built(built)
    view.white_combo.setCurrentIndex(view.white_combo.findData("chart"))
    assert view.primary.text() == "Build Profile"
    # Chart white cannot be scaled, so the aim is not offered with it.
    assert not view.scale_for_field.isEnabled()


def test_an_untagged_capture_opens_and_can_only_be_profiled(qt_app, tmp_path):
    import tifffile

    from deltachrome_ui.placement_view import PlacementView
    from deltachrome_ui.workers import ImageLoader

    path = tmp_path / "bare.tif"
    tifffile.imwrite(path, np.full((40, 60, 3), 32768, dtype=np.uint16),
                     photometric="rgb")
    loader = ImageLoader(path)
    got = []
    loader.loaded.connect(lambda proxy, _path: got.append(proxy))
    loader.run()
    assert got and got[0].image.profile is None
    placement = PlacementView()
    placement.set_untagged(True)
    assert not placement.continue_button.isEnabled()


def test_a_small_chart_chooses_automatically_on_build(view):
    assert view.type_combo.currentData() == "automatic"
    assert view.quality_combo.currentData() == "automatic"
    assert not view.find_best_button.isVisibleTo(view)


def test_a_large_chart_opens_on_a_shaper_and_searches_by_button(
        qt_app, capture, monkeypatch):
    """Where tables would be tried, Automatic costs minutes, so it is asked for."""
    from types import SimpleNamespace

    from deltachrome_ui import profile_view

    path, image, data = capture
    monkeypatch.setattr(profile_view, "TABLE_MIN_PATCHES", 10)
    monkeypatch.setattr(paths, "resolve_colprof", lambda: "colprof")
    screen = ProfileView()
    setup = ProfileSetup(
        data=data, reference_name="ref.txt", keywords={"target": SG}, warnings=[],
        encoding=detect_encoding(data), merged_count=None,
    )
    screen.open(proxy=ImageProxy(image), image_path=path, target=load_target(SG),
                setup=setup)
    assert (screen.type_combo.currentData(), screen.quality_combo.currentData()) \
        == profile_view.LARGE_CHART_DEFAULT
    assert screen.type_combo.findData("automatic") == -1
    assert screen.find_best_button.isVisibleTo(screen)
    assert not screen.request()["automatic"]

    asked = []
    screen.buildRequested.connect(asked.append)
    screen.find_best_button.click()
    assert asked[0]["automatic"] and asked[0]["types"] is None \
        and asked[0]["qualities"] is None

    # The winner is put into the menus and is what Save would write.
    summary = ErrorSummary.of({"A1": 1.0})
    screen.show_built(SimpleNamespace(
        icc=b"", fit=summary, held_out=summary, excluded=None, scale=1.0,
        options=ColprofOptions(profile_type="xyz-table", quality="low"),
        run=SimpleNamespace(warnings=[]), data=data,
    ))
    assert screen.type_combo.currentData() == "xyz-table"
    assert screen.quality_combo.currentData() == "low"
    assert screen.primary.text() == "Save Profile…"


def test_pointing_at_a_patch_lights_it_and_merging_lights_its_repeats(view):
    view.canvas._set_hover("B2")
    assert len(view.canvas._hover_items) == 1
    view.canvas._set_hover(None)
    assert not view.canvas._hover_items
    view.canvas.show_patches(view.setup.data.regions, set(), set(),
                             [("B2", "C3", "D4")])
    view.canvas._set_hover("C3")
    assert sorted(i.toolTip() for i in view.canvas._hover_items) == ["B2", "C3", "D4"]
    # Visible without pointing: each member carries its family's number.
    numbered = [child.text() for item in view.canvas._patch_items["C3"]
                for child in item.childItems()
                if isinstance(child, QGraphicsSimpleTextItem)]
    assert numbered == ["1"]


def test_the_display_gamma_is_offered_for_a_linear_untagged_capture_only(
        qt_app, capture):
    from dataclasses import replace
    from types import SimpleNamespace

    path, image, data = capture
    image = replace(image, profile=None)
    screen = ProfileView()
    linear = SimpleNamespace(linear=True, label="Linear", exponent=1.0, steps=12)
    for encoding, offered in ((detect_encoding(data), False), (linear, True)):
        screen.open(proxy=ImageProxy(image), image_path=path, target=load_target(SG),
                    setup=ProfileSetup(data=data, reference_name="ref.txt",
                                       keywords={"target": SG}, warnings=[],
                                       encoding=encoding, merged_count=None))
        assert screen.gamma_check.isVisibleTo(screen) == offered
    assert screen.gamma_check.isChecked() and screen.canvas.proxy.linear_display
    screen.gamma_check.setChecked(False)
    assert not screen.canvas.proxy.linear_display
