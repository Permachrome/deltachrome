"""Captures a batch meets that are wrong in ways the operator can act on.

A folder of real captures was made bad and a batch run over it. Each
test here is one refusal that came back unreadable, or not at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ImageError
from deltachrome.imaging import detect
from deltachrome.imaging.loader import load_image
from deltachrome.targets.registry import load_target

from fixtures import adobe_rgb_icc, write_tiff_with_profile


def _capture(tmp_path, width=600, height=400):
    pixels = np.full((height, width, 3), 30000, np.uint16)
    return load_image(write_tiff_with_profile(tmp_path / "c.tif", pixels, adobe_rgb_icc()))


def test_a_damaged_file_is_said_to_be_damaged(tmp_path):
    """A TIFF cut short in copying used to report tifffile's own words:
    "corrupted strip cannot be reshaped from (230444,) to (1, 16, 11608, 3)"."""
    good = write_tiff_with_profile(
        tmp_path / "whole.tif", np.full((400, 600, 3), 30000, np.uint16), adobe_rgb_icc()
    )
    cut = tmp_path / "cut.tif"
    cut.write_bytes(good.read_bytes()[:20000])
    with pytest.raises(ImageError, match="could not be read. The file may be damaged"):
        load_image(cut)


def test_a_seed_from_a_larger_frame_is_refused_in_words(tmp_path):
    """A seed checked on an 11608 x 9108 frame names a region wholly below an
    8686 x 5615 one. Clipped, it was a box of negative height, and it reached a
    finder as an empty image and raised from inside OpenCV."""
    image = _capture(tmp_path)
    target = load_target("isa_object_level_05x_v1")
    seed = [(100, 7466), (500, 7494), (500, 7887), (100, 7860)]
    found = detect.detect_placement_near(image, target, seed)
    assert not found.found
    assert "seed placement lies outside this 600 x 400 px frame" in found.reason


def test_a_chart_running_off_the_frame_says_so(tmp_path):
    """A chart cut by the frame's edge used to be refused with advice for somebody
    who typed four corners: check they are in TL TR BR BL order."""
    image = _capture(tmp_path)
    target = load_target("isa_object_level_05x_v1")
    detection = detect.Detection(
        corners=[(10, 10), (900, 10), (900, 100), (10, 100)], found=True,
        method="patch_array_corners",
    )
    detect._refine(image, target, detection)
    assert not detection.found
    assert detection.reason == "The chart runs off the edge of the frame."


def test_a_batch_cannot_be_seeded_from_a_capture_holding_several_charts(
    tmp_path, monkeypatch
):
    """Seeding from a board of four QA-62 cards took the single search's answer: a
    quad of one crosshair from each card, spanning the board. Every frame was then
    measured on it. Which card a batch follows is the operator's to say."""
    from pathlib import Path

    from deltachrome import cli
    from deltachrome.imaging import loader

    card = type("D", (), {"found": True, "corners": [(0, 0)] * 4, "reason": ""})()
    monkeypatch.setattr(loader, "load_image", lambda _p: object())
    monkeypatch.setattr(detect, "detect_placements", lambda *_a, **_k: [card] * 4)
    with pytest.raises(SystemExit, match="holds 4 charts of this model"):
        cli._seed_corners(Path("board.jp2"), object())

    monkeypatch.setattr(detect, "detect_placements", lambda *_a, **_k: [card])
    corners, _why = cli._seed_corners(Path("one.jp2"), object())
    assert corners == [(0, 0)] * 4
