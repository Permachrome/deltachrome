# PyInstaller recipe for the Deltachrome desktop application.
#
#     pyinstaller --noconfirm packaging/deltachrome.spec
#
# Run from the repository root, after packaging/make_icon.py. Produces
# dist/Deltachrome.app on macOS and dist/Deltachrome/ on Windows.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs,
                                     collect_submodules, copy_metadata)

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "src"))
from deltachrome import __version__  # noqa: E402

# Target and guideline definitions, and the generated CIE tables, are files
# beside the code and found through Path(__file__).
datas = collect_data_files("deltachrome") + collect_data_files("colour")

# The About screen reads each library's version from its installed metadata,
# so the metadata has to travel with the bundle.
for dist in ("deltachrome", "PySide6", "numpy", "scipy", "opencv-python-headless",
             "colour-science", "pillow", "tifffile", "imagecodecs", "zxing-cpp"):
    datas += copy_metadata(dist)

# The application's own license, and every bundled component's, beside the
# code: About links to the notices, and a redistributed bundle owes the texts.
datas += [(str(ROOT / "LICENSE"), "."),
          (str(ROOT / "build" / "THIRD_PARTY_NOTICES.txt"), ".")]

# colprof from ArgyllCMS, which builds camera profiles, put in build/argyll by
# packaging/fetch_colprof.py. The application looks for it at argyll/bin in the
# bundle (deltachrome.profiling.colprof.bundled_colprof). Required: a build
# without it would ship a profile screen that cannot build anything.
COLPROF = ROOT / "build" / "argyll" / "bin" / ("colprof.exe" if sys.platform == "win32"
                                                 else "colprof")
if not COLPROF.is_file():
    raise SystemExit(f"{COLPROF} is missing: run packaging/fetch_colprof.py first.")

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    # zxingcpp is imported inside a function and is optional to the engine:
    # without it the application silently stops reading Data Matrix and most
    # barcodes. imagecodecs loads each codec's extension module by name when
    # tifffile asks for one, so none is visible to the bundler: without them an
    # LZW or Deflate TIFF -- most masters -- is reported as unreadable.
    hiddenimports=["zxingcpp"] + collect_submodules("imagecodecs"),
    binaries=collect_dynamic_libs("imagecodecs") + [(str(COLPROF), "argyll/bin")],
    excludes=["tkinter", "matplotlib", "pandas", "IPython", "pytest"],
    noarchive=False,
)
# OpenCV's Windows wheel keeps FFmpeg in a separate plugin it loads only for
# video, which this application never opens. Leaving it out ships no FFmpeg at
# all. (On macOS the wheel links FFmpeg directly, which is why that platform
# uses an OpenCV built without it: packaging/build_opencv.sh.)
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0].lower()]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Deltachrome",
    console=False,
    icon=str(ROOT / "build" / "icon" / ("Deltachrome.icns" if sys.platform == "darwin"
                                         else "Deltachrome.ico")),
)

coll = COLLECT(exe, a.binaries, a.datas, name="Deltachrome")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Deltachrome.app",
        icon=str(ROOT / "build" / "icon" / "Deltachrome.icns"),
        bundle_identifier="me.permachro.deltachrome",
        version=__version__,
        info_plist={
            "CFBundleName": "Deltachrome",
            "CFBundleDisplayName": "Deltachrome",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            # The oldest macOS every bundled binary is built for; the build
            # script checks it. Older systems refuse with a clear message.
            "LSMinimumSystemVersion": "13.0",
            "NSHumanReadableCopyright": "© 2026 Permachrome",
        },
    )
