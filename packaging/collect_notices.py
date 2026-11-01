"""Write THIRD_PARTY_NOTICES.txt for a packaged build.

A bundle redistributes every library it carries, so it must carry their
licenses too -- not their names, which the About screen already shows, but the
texts. Run in the environment the bundle is built from, after it is set up, so
each version and each license file is the one actually shipped:

    python packaging/collect_notices.py build/THIRD_PARTY_NOTICES.txt

Each library's text comes from the files its own distribution installs; nothing
is retyped. Two sections are written by hand because the facts are not in any
installed file: the LGPL components and how a recipient exercises the LGPL's
rights over them, and the Python runtime.
"""

from __future__ import annotations

import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Every distribution the bundle carries, as (distribution, shown name, home page,
#: license). The license is stated rather than read from metadata, which is vague
#: for some (one reads only "OSI Approved"), and matches the About screen's.
#: Qt for Python is absent: it has the LGPL section, with the source offer.
BUNDLED = [
    ("numpy", "NumPy", "https://numpy.org", "BSD-3-Clause, with bundled components below"),
    ("scipy", "SciPy", "https://scipy.org", "BSD-3-Clause, with bundled components below"),
    ("opencv-python-headless", "OpenCV", "https://opencv.org", "Apache-2.0"),
    ("colour-science", "Colour", "https://www.colour-science.org", "BSD-3-Clause"),
    ("pillow", "Pillow", "https://python-pillow.org", "MIT-CMU"),
    ("tifffile", "tifffile", "https://github.com/cgohlke/tifffile", "BSD-3-Clause"),
    ("imagecodecs", "imagecodecs", "https://github.com/cgohlke/imagecodecs",
     "BSD-3-Clause, with bundled codec libraries below"),
    ("zxing-cpp", "zxing-cpp", "https://github.com/zxing-cpp/zxing-cpp", "Apache-2.0"),
    ("pyinstaller", "PyInstaller bootloader", "https://pyinstaller.org",
     "GPL-2.0-or-later WITH the PyInstaller Bootloader exception, which permits "
     "distributing it with programs under any license"),
]

#: Said about a distribution before its license texts, where the texts alone
#: would mislead.
REMARKS = {
    "opencv-python-headless": (
        "The official opencv-python-headless wheel with its FFmpeg plugin\n"
        "(opencv_videoio_ffmpeg) removed; no FFmpeg is distributed. The upstream\n"
        "package's third-party list below also names that component."
        if sys.platform == "win32" else
        "Built by Permachrome from opencv-python release 93 without FFmpeg or any\n"
        "video input/output backend. The upstream package's third-party list below\n"
        "also names components this build does not contain."),
}

_LICENSE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE)[^/]*$", re.IGNORECASE)

RULE = "=" * 78


def _license_files(dist: metadata.Distribution) -> list[Path]:
    found = []
    for f in dist.files or []:
        if _LICENSE_FILE.search(str(f)):
            path = Path(dist.locate_file(f))
            if path.is_file():
                found.append(path)
    return sorted(set(found), key=str)


def lgpl_section() -> str:
    qt = metadata.version("PySide6_Essentials")
    return f"""{RULE}
LGPL COMPONENTS: Qt and Qt for Python {qt}
{RULE}

Deltachrome uses Qt and Qt for Python (PySide6, Shiboken6) under the GNU Lesser
General Public License version 3. They are not modified, and they are
dynamically linked: the libraries are separate files inside the application
(Deltachrome.app/Contents/Frameworks on macOS, the _internal folder on Windows),
and a recipient may replace them with a compatible build of their own.

The corresponding source code is published by The Qt Company:
  Qt for Python {qt}:  https://download.qt.io/official_releases/QtForPython/
  Qt:                  https://download.qt.io/official_releases/qt/
On request to hello@permachro.me, we will provide a copy of that source for
three years from the date of this release.

Qt incorporates third-party components under their own licenses, listed at
https://doc.qt.io/qt-6/licenses-used-in-qt.html.

The texts of the GNU LGPL version 3, and of the GNU GPL version 3 which it
incorporates, follow.

{(HERE / "licenses" / "LGPL-3.0.txt").read_text(encoding="utf-8")}

{(HERE / "licenses" / "GPL-3.0.txt").read_text(encoding="utf-8")}
"""


def argyll_section() -> str:
    """colprof, from build/argyll where packaging/fetch_colprof.py put it."""
    sys.path.insert(0, str(HERE))
    from fetch_colprof import OUT, SOURCE, VERSION

    license_file = OUT / "License.txt"
    if not license_file.is_file():
        raise SystemExit("Run packaging/fetch_colprof.py first: build/argyll has no "
                         "License.txt.")
    return f"""{RULE}
colprof from ArgyllCMS {VERSION}
{SOURCE}
License: GNU Affero General Public License version 3
{RULE}

Deltachrome runs colprof, one program from ArgyllCMS by Graeme W. Gill, to build
camera profiles. It is the unmodified program from ArgyllCMS's own release, run
as a separate program; on macOS it carries the publisher's code signature, which
Apple requires and which does not change what it does. Its source code, with
the rest of ArgyllCMS, is published at {SOURCE}. Profiles made with it belong
to whoever made them.

ArgyllCMS's own license statement follows.

{license_file.read_text(encoding="utf-8", errors="replace").strip()}
"""


def python_section() -> str:
    version = sys.version.split()[0]
    candidates = [Path(sysconfig.get_paths()["stdlib"]) / "LICENSE.txt",
                  Path(sys.base_prefix) / "LICENSE.txt"]
    text = next((p.read_text(encoding="utf-8", errors="replace") for p in candidates if p.is_file()), None)
    if text is None:
        raise SystemExit("The Python license text was not found in this runtime.")
    return f"""{RULE}
Python {version}
https://www.python.org
{RULE}

{text}
"""


def package_section(dist_name: str, shown: str, url: str, license_name: str) -> str:
    dist = metadata.distribution(dist_name)
    files = _license_files(dist)
    if not files:
        raise SystemExit(f"{dist_name} installs no license file; add its text by hand.")
    parts = [f"{RULE}\n{shown} {dist.version}\n{url}\nLicense: {license_name}\n{RULE}\n"]
    if dist_name in REMARKS:
        parts.append(REMARKS[dist_name] + "\n")
    for path in files:
        parts.append(f"--- {path.name} ---\n\n{path.read_text(encoding='utf-8', errors='replace').strip()}\n")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    from deltachrome import __version__

    sections = [f"""Deltachrome {__version__} -- third-party notices

Deltachrome is © 2026 Permachrome and released under the MIT License (see
LICENSE). It includes the third-party software below, each under its own
license, whose full text follows its name.
"""]
    sections.append(lgpl_section())
    sections.append(argyll_section())
    sections.append(python_section())
    for dist_name, shown, url, license_name in BUNDLED:
        sections.append(package_section(dist_name, shown, url, license_name))
    out.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 always: several license texts carry characters Windows' default
    # code page cannot encode.
    out.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {len(sections) - 1} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
