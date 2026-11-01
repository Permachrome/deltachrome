"""About: what this is, whose it is, the terms it is offered on, and who to tell.

**The version comes first after the name**, because it is the first question
asked of anybody reporting a problem -- and every saved record already carries
``environment.deltachrome_version``, so a reader comparing an odd result with
the application in front of them has the two numbers to put side by side.

**The license is shown in full, not named.** MIT asks that its notice travel
with every copy, and an application installed from a bundle has no LICENSE file
anybody will find. The text is held here rather than read from the repository's
own file, which a packaged install does not carry; a test pins that the two
are identical, so the copy cannot drift from the one that is authoritative.

**What it is built on is listed with each component's own license.** These are
the libraries the application imports, not every package the installer pulls
in. Versions are read from what is installed, so the list says what *this* copy
runs on. Qt for Python is the one whose terms ask more than attribution -- the
LGPL -- and naming it is the least that requires.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from deltachrome import __version__
from deltachrome_ui.theme import TEXT
from deltachrome_ui.widgets import MarkBadge, heading, subtitle

CONTACT_EMAIL = "hello@permachro.me"
WEBSITE = "https://permachro.me/deltachrome"
COPYRIGHT = "© 2026 Permachrome"

#: The repository's LICENSE, verbatim. ``tests/test_about_panel.py`` compares.
LICENSE_TEXT = """\
MIT License

Copyright (c) 2026 Permachrome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#: ``(shown name, distribution name, license, project page)``. The libraries
#: the application imports; a new import belongs here too.
COMPONENTS: list[tuple[str, str, str, str]] = [
    ("Qt for Python", "PySide6", "LGPL-3.0", "https://www.qt.io/qt-for-python"),
    ("NumPy", "numpy", "BSD-3-Clause", "https://numpy.org"),
    ("SciPy", "scipy", "BSD-3-Clause", "https://scipy.org"),
    ("OpenCV", "opencv-python-headless", "Apache-2.0", "https://opencv.org"),
    ("Colour", "colour-science", "BSD-3-Clause", "https://www.colour-science.org"),
    ("Pillow", "pillow", "MIT-CMU", "https://python-pillow.org"),
    ("tifffile", "tifffile", "BSD-3-Clause", "https://github.com/cgohlke/tifffile"),
    ("imagecodecs", "imagecodecs", "BSD-3-Clause",
     "https://github.com/cgohlke/imagecodecs"),
    ("zxing-cpp", "zxing-cpp", "Apache-2.0", "https://github.com/zxing-cpp/zxing-cpp"),
]


def installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "—"


#: The full license texts of every bundled component, written into a packaged
#: build by packaging/collect_notices.py. Names and versions are on this screen;
#: a redistributed bundle owes the texts as well.
NOTICES_NAME = "THIRD_PARTY_NOTICES.txt"


def notices_path() -> Path | None:
    """The bundled notices file, or None when running from source."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is None:
        return None
    path = Path(bundle) / NOTICES_NAME
    return path if path.is_file() else None


def _link(url: str, text: str) -> QLabel:
    label = QLabel(f'<a href="{url}" style="color: {TEXT};">{text}</a>')
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(True)
    label.setToolTip(url)
    return label


class AboutPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        identity = QHBoxLayout()
        identity.setSpacing(16)
        identity.addWidget(MarkBadge(40), 0, Qt.AlignmentFlag.AlignVCenter)
        names = QVBoxLayout()
        names.setSpacing(2)
        name = QLabel("Deltachrome")
        name.setStyleSheet("font-size: 18px; font-weight: 600;")
        names.addWidget(name)
        self.version_label = subtitle(f"Version {__version__}")
        self.version_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        names.addWidget(self.version_label)
        identity.addLayout(names)
        identity.addStretch(1)
        column.addLayout(identity)
        column.addSpacing(14)

        facts = QGridLayout()
        facts.setHorizontalSpacing(14)
        facts.setVerticalSpacing(6)
        rows = [
            ("Copyright", QLabel(COPYRIGHT)),
            ("License", QLabel("MIT License")),
            ("Contact", _link(f"mailto:{CONTACT_EMAIL}", CONTACT_EMAIL)),
            ("Website", _link(WEBSITE, WEBSITE.removeprefix("https://"))),
        ]
        for i, (key, value) in enumerate(rows):
            label = QLabel(key)
            label.setFixedWidth(120)
            label.setObjectName("Subtitle")
            facts.addWidget(label, i, 0)
            facts.addWidget(value, i, 1)
        facts.setColumnStretch(2, 1)
        column.addLayout(facts)
        column.addSpacing(22)

        column.addWidget(heading("Open-Source Components"))
        components = QGridLayout()
        components.setHorizontalSpacing(14)
        components.setVerticalSpacing(6)
        for i, (shown, dist, license_name, url) in enumerate(COMPONENTS):
            name_label = _link(url, shown)
            name_label.setFixedWidth(120)
            components.addWidget(name_label, i, 0)
            version = subtitle(installed_version(dist))
            version.setFixedWidth(96)
            components.addWidget(version, i, 1)
            components.addWidget(subtitle(license_name), i, 2)
        components.setColumnStretch(3, 1)
        column.addLayout(components)
        notices = notices_path()
        if notices is not None:
            column.addSpacing(8)
            column.addWidget(_link(QUrl.fromLocalFile(str(notices)).toString(),
                                   "Third-Party Notices"))
        column.addSpacing(22)

        column.addWidget(heading("License"))
        self.license_text = QPlainTextEdit(LICENSE_TEXT)
        self.license_text.setReadOnly(True)
        self.license_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        column.addWidget(self.license_text, 1)
