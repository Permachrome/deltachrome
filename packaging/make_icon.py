"""Render the application icon to the files an installer needs.

The icon is drawn in code (``deltachrome_ui.mark.app_icon``) rather than kept
as an asset, so the bundle's icon is rendered from that same drawing and cannot
drift from the one the running application sets on its window. Writes
``build/icon/Deltachrome.icns`` (macOS, via ``iconutil``) and
``build/icon/Deltachrome.ico`` (Windows).

    python packaging/make_icon.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from deltachrome_ui.mark import app_icon  # noqa: E402
from deltachrome_ui.theme import BG_INPUT, MARK_INK, RAMP  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "build" / "icon"

#: The members of a macOS iconset: (file name, pixel size).
ICONSET = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]


def main() -> int:
    app = QGuiApplication(sys.argv)  # noqa: F841 -- painting needs an application
    icon = app_icon(MARK_INK, RAMP, BG_INPUT)
    OUT.mkdir(parents=True, exist_ok=True)

    iconset = OUT / "Deltachrome.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for name, size in ICONSET:
        icon.pixmap(QSize(size, size)).toImage().save(str(iconset / name))

    if sys.platform == "darwin":
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                        "-o", str(OUT / "Deltachrome.icns")], check=True)

    # Windows takes one .ico holding several sizes; Pillow writes it from the
    # largest rendering.
    from PIL import Image
    Image.open(iconset / "icon_512x512@2x.png").save(
        OUT / "Deltachrome.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                        (64, 64), (128, 128), (256, 256)])
    print(f"Icons written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
