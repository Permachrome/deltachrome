"""A small picture of a capture, without decoding the capture.

The frame map draws its outlines from the record and needs no pixels to say
where a target sits. The pixels only make the targets *recognisable*, and
getting them the obvious way -- the same decode the capture panel does -- costs
11.2 s and 303 MB on a JPEG 2000 master to fill a rectangle five hundred pixels
across.

Three measurements decided the shape of this, and two of them are traps that
look like the answer:

* OpenCV's ``IMREAD_REDUCED_*`` flags buy **nothing** on JPEG 2000 -- 11.2 s at
  /2, /4 and /8 alike -- because that path decodes the whole codestream and then
  downsamples.
* Pillow's documented ``draft`` is a **no-op** for its JPEG2000 plugin: 31.9 s
  at every size asked for.
* The plugin's own ``reduce`` attribute does read the codestream's resolution
  levels: 0.61 s and 4.7 MB for 1450 px, against 11.2 s and 303 MB for the lot.

That last one is an odd API -- ``reduce`` is also the name of an ``Image``
method on modern Pillow, so setting it shadows the method on the instance -- and
nothing about it announces itself, which is why it is pinned here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui.imageproxy import THUMBNAIL_WIDTH, read_thumbnail  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def write_tiff(path: Path, width: int, height: int, dtype=np.uint8) -> Path:
    import cv2

    top = 255 if dtype == np.uint8 else 65535
    block = np.zeros((height, width, 3), dtype=dtype)
    block[: height // 2] = top          # a seam, so a flip would be visible
    cv2.imwrite(str(path), block)
    return path


def test_a_thumbnail_is_small_and_is_the_right_way_up(qt_app, tmp_path):
    path = write_tiff(tmp_path / "big.tif", 4000, 3000)
    image = read_thumbnail(path, max_width=500)

    assert isinstance(image, QImage)
    assert image.width() < 4000, "the whole file was decoded"
    assert image.width() >= 500, "reduced past what was asked for"
    # Aspect is preserved, so the outlines drawn over it land where they should.
    assert abs(image.width() / image.height() - 4000 / 3000) < 0.01
    # Light on top, as written. A channel or row flip would not show in a size.
    assert QImage.pixelColor(image, image.width() // 2, 2).lightness() > 200
    assert QImage.pixelColor(
        image, image.width() // 2, image.height() - 3).lightness() < 60


def test_a_sixteen_bit_master_comes_back_as_eight(qt_app, tmp_path):
    """Which is what a thumbnail wants, and what the caller would otherwise have
    to do itself. A 16-bit file handed to Qt as 8-bit unconverted is noise."""
    path = write_tiff(tmp_path / "deep.tif", 2000, 1500, dtype=np.uint16)
    image = read_thumbnail(path, max_width=400)

    assert image is not None
    assert image.format() == QImage.Format.Format_RGB888
    assert QImage.pixelColor(image, image.width() // 2, 2).lightness() > 200


def test_a_small_capture_is_not_enlarged(qt_app, tmp_path):
    """Nothing is upscaled to reach the target width: the target is a ceiling."""
    path = write_tiff(tmp_path / "small.tif", 300, 200)
    image = read_thumbnail(path, max_width=THUMBNAIL_WIDTH)
    assert image is not None
    assert (image.width(), image.height()) == (300, 200)


def test_an_unreadable_capture_is_not_an_error(qt_app, tmp_path):
    """The map goes on drawing its outlines, which is what it does before a
    thumbnail arrives at all. A missing master is not a failure of the screen."""
    assert read_thumbnail(tmp_path / "nothing-here.tif") is None

    broken = tmp_path / "broken.jp2"
    broken.write_bytes(b"not a codestream")
    assert read_thumbnail(broken) is None

    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"")
    assert read_thumbnail(empty) is None


def test_a_master_is_not_refused_as_a_decompression_bomb(qt_app, tmp_path):
    """Pillow's default limit is 89 megapixels and a preservation master is
    routinely over it -- the ISA 4x5 is 101. Without disarming the check, the
    probe raises on ``open``, before a single pixel is read."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 89_478_485  # put the default back for this test
    path = write_tiff(tmp_path / "huge.tif", 11600, 8700)
    try:
        image = read_thumbnail(path, max_width=600)
    finally:
        Image.MAX_IMAGE_PIXELS = None
    assert image is not None, "a 101-megapixel master was refused"
    assert image.width() < 2000


# -- the JPEG 2000 path, which is the whole reason this exists ----------------

JP2 = Path(__file__).resolve().parents[1] / "data" / "QA62" / "QA62_2.jp2"
needs_jp2 = pytest.mark.skipif(not JP2.exists(), reason="sample JP2 not present")


@needs_jp2
def test_a_jpeg_2000_is_read_at_its_own_reduced_resolution(qt_app):
    """The measurement this whole module rests on: 0.61 s against 11.2 s.

    Timed rather than asserted on a threshold -- machines differ -- but the
    *size* is the proof that only some resolution levels were decoded, and it is
    the size that makes it cheap.
    """
    image = read_thumbnail(JP2, max_width=1200)
    assert image is not None
    assert (image.width(), image.height()) == (1450, 1088), (
        "not the codestream's own level: a full decode would give 11600x8700"
    )
    # Four light cards on a dark board: the picture is real, not a grey field.
    array = np.frombuffer(image.constBits(), dtype=np.uint8)
    assert array.max() > 200 and array.min() < 60


@needs_jp2
def test_pillow_still_honours_the_reduce_attribute(qt_app):
    """Pinned because nothing about this API announces itself.

    ``reduce`` is a *method* on ``Image`` in modern Pillow, and the JPEG2000
    plugin reads an *attribute* of that name -- so this works by shadowing the
    method on the instance. If a future Pillow stops honouring it, every JPEG
    2000 thumbnail silently becomes a full decode, which is eleven seconds
    rather than a failure, and nothing else here would notice.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(JP2) as probe:
        full = probe.size
    with Image.open(JP2) as image:
        image.reduce = 3
        image.load()
        assert image.size[0] == full[0] >> 3, (
            "Pillow ignored `reduce`: JPEG 2000 thumbnails are now full decodes"
        )


# -- the worker --------------------------------------------------------------


def test_the_worker_ends_its_thread_even_when_there_is_no_picture(qt_app, tmp_path):
    """``run_in_thread`` ends a thread on ``loaded``, ``finished`` or ``failed``.

    A worker that stays silent on the path it takes most often -- a master that
    has moved -- leaks the thread, the worker and the owner's record of both,
    for the life of the window. This is the whole reason ``finished`` exists on
    it, and nothing else in the application would have noticed.
    """
    import time

    from PySide6.QtCore import QObject
    from deltachrome_ui.workers import ThumbnailLoader, run_in_thread

    owner = QObject()
    for path in (tmp_path / "gone.tif", write_tiff(tmp_path / "there.tif", 80, 60)):
        worker = ThumbnailLoader(path, "key")
        run_in_thread(worker, owner)
        # Pumped rather than waited on. ``thread.quit`` is a *queued* delivery to
        # the thread object, which lives on this one, so ``QThread.wait`` here
        # blocks the very loop that has to deliver it -- a deadlock in the test
        # and not in the application, whose loop is always turning.
        deadline = time.monotonic() + 10
        while owner._threads and time.monotonic() < deadline:
            QApplication.processEvents()
        assert not owner._threads, (
            f"the thread and its record leaked for {path.name}"
        )


def test_the_worker_carries_the_key_it_was_asked_for(qt_app, tmp_path):
    """So a read that finishes after the reader has moved on can be dropped."""
    from deltachrome_ui.workers import ThumbnailLoader

    got: list = []
    worker = ThumbnailLoader(write_tiff(tmp_path / "a.tif", 80, 60), "abc123")
    worker.loaded.connect(lambda image, key: got.append((image, key)))
    worker.run()
    assert len(got) == 1
    assert got[0][1] == "abc123"
