"""Put ArgyllCMS's colprof where the bundle carries it, from Argyll's own build.

    python packaging/fetch_colprof.py

Downloads the official ArgyllCMS archive for this machine's platform, refuses it
unless its SHA-256 is the one recorded here, and extracts exactly two files into
``build/argyll``: ``bin/colprof`` (``colprof.exe`` on Windows) and Argyll's
``License.txt``. The spec takes them from there, and the application looks for
colprof at ``argyll/bin`` inside the bundle.

colprof is shipped unmodified, which is what Graeme Gill's written terms for
the AGPL rest on: an MIT application with public
source may bundle and call it, re-signing it for notarization leaves it
unmodified, and a link to the ArgyllCMS website meets the source obligation.
**No build carrying it may be published before Deltachrome's source is.**

Argyll publishes no checksums, so the ones below were taken from the files as
downloaded on 2026-10-31. The Apple silicon one matches the archive installed
for development on 2026-10-26. A different file under the same name is refused
rather than shipped.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "3.5.0"
SOURCE = "https://www.argyllcms.com/"

#: (sys.platform, machine) -> (archive, SHA-256).
ARCHIVES = {
    ("darwin", "arm64"): (
        "Argyll_V3.5.0_macOS11_arm64_bin.tgz",
        "b1f27d638ddca054db3f5ff866410d5b36f179770cee5591f116352b1853d957",
    ),
    ("darwin", "x86_64"): (
        "Argyll_V3.5.0_osx10.6_x86_64_bin.tgz",
        "69fcab29d555890434e90c91f7a2b86f0a21551f194dd816a8681d0f7e99a01d",
    ),
    ("win32", "amd64"): (
        "Argyll_V3.5.0_win64_exe.zip",
        "c80f78ba30e715079a00a14c2e7c9f533c58e6c8fdbc27f2d32b3d7813f9d132",
    ),
}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "argyll"


def archive_for_this_machine() -> tuple[str, str]:
    key = (sys.platform, platform.machine().lower())
    if key not in ARCHIVES:
        raise SystemExit(f"No ArgyllCMS archive is recorded for {key[0]} {key[1]}.")
    return ARCHIVES[key]


def fetch(name: str, sha256: str) -> Path:
    """The archive, downloaded once into build/ and verified every time."""
    path = ROOT / "build" / "downloads" / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(SOURCE + name, timeout=120) as response, \
                open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise SystemExit(f"{name} has SHA-256 {digest}, not the {sha256} recorded "
                         "for it. Refused, and removed.")
    return path


def extract(archive: Path) -> Path:
    """colprof and Argyll's license into build/argyll; returns colprof's path."""
    exe = "colprof.exe" if sys.platform == "win32" else "colprof"
    wanted = {f"Argyll_V{VERSION}/bin/{exe}": OUT / "bin" / exe,
              f"Argyll_V{VERSION}/License.txt": OUT / "License.txt"}
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "bin").mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for member, dest in wanted.items():
                dest.write_bytes(z.read(member))
    else:
        with tarfile.open(archive) as t:
            for member, dest in wanted.items():
                source = t.extractfile(member)
                if source is None:
                    raise SystemExit(f"{member} is not a file in {archive.name}.")
                dest.write_bytes(source.read())
    colprof = wanted[f"Argyll_V{VERSION}/bin/{exe}"]
    colprof.chmod(0o755)
    return colprof


def main() -> None:
    name, sha256 = archive_for_this_machine()
    colprof = extract(fetch(name, sha256))
    print(f"colprof from ArgyllCMS {VERSION} ({name}) at {colprof.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
