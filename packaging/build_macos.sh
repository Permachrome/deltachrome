#!/bin/sh
# Build Deltachrome.app and a disk image of it, for the Mac it runs on.
#
#     packaging/build_macos.sh
#
# PYTHON must be a python.org (or other macOS-13-compatible) Python 3.13: the
# interpreter is copied into the bundle, and Homebrew's is built for the macOS
# it was installed on. The build checks this and refuses otherwise.
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.13}
MIN_MACOS=13.0
ARCH=$(uname -m)
OPENCV_WHEEL=${OPENCV_WHEEL:-$(ls build/opencv/wheels/opencv_python_headless-*-macosx_*_"$ARCH".whl | head -1)}

# A fresh environment, from wheels chosen for the oldest supported macOS: pip
# would otherwise pick each library's newest wheel for *this* Mac.
rm -rf build/release-venv build/wheelhouse
"$PY" -m venv build/release-venv
V=build/release-venv/bin
$V/pip install -q --upgrade pip
$V/pip download -q --only-binary :all: --platform "macosx_${MIN_MACOS%.*}_0_$ARCH" \
    --python-version 3.13 -d build/wheelhouse -r packaging/requirements-lock.txt
$V/pip install -q --no-index --find-links build/wheelhouse -r packaging/requirements-lock.txt
$V/pip install -q "$OPENCV_WHEEL" pyinstaller==6.22.3
$V/pip install -q --no-deps .

VERSION=$($V/python -c "import deltachrome; print(deltachrome.__version__)")
$V/python packaging/make_icon.py
$V/python packaging/fetch_colprof.py
$V/python packaging/collect_notices.py build/THIRD_PARTY_NOTICES.txt
$V/python -m PyInstaller --noconfirm --log-level WARN \
    --workpath build/pyinstaller --distpath dist packaging/deltachrome.spec

# Every binary in the bundle must run on the oldest macOS we claim. A library
# built for a newer one is not an error anywhere else -- it loads on this Mac
# and fails to open on a tester's.
NEWER=$(find dist/Deltachrome.app -type f \( -name '*.dylib' -o -name '*.so' -o -perm -u+x \) \
  -exec sh -c 'v=$(otool -l "$1" 2>/dev/null | awk "/LC_BUILD_VERSION/{b=1} b&&/minos/{print \$2; exit}"); \
  [ -n "$v" ] && printf "%s %s\n" "$v" "$1"' _ {} \; | awk -v min="$MIN_MACOS" '
  { split($1, v, "."); split(min, m, "."); if (v[1] > m[1] || (v[1] == m[1] && v[2] > m[2])) print }')
if [ -n "$NEWER" ]; then
  echo "These need a newer macOS than $MIN_MACOS:"; echo "$NEWER" | head -20; exit 1
fi

# The bundle must answer as the source does.
dist/Deltachrome.app/Contents/MacOS/Deltachrome --cli --version

# colprof must be where the application looks, and must run. Asked for nothing,
# it prints its usage and exits non-zero.
COLPROF=dist/Deltachrome.app/Contents/Frameworks/argyll/bin/colprof
"$COLPROF" 2>&1 | grep -q "Create ICC profile" || { echo "Bundled colprof is missing or does not run: $COLPROF"; exit 1; }

rm -rf build/dmg && mkdir -p build/dmg
cp -R dist/Deltachrome.app build/dmg/
cp LICENSE build/dmg/LICENSE.txt
cp build/THIRD_PARTY_NOTICES.txt build/dmg/
ln -s /Applications build/dmg/Applications
DMG="dist/Deltachrome-$VERSION-macos-$ARCH.dmg"
rm -f "$DMG"
hdiutil create -quiet -volname Deltachrome -srcfolder build/dmg -fs HFS+ -format ULMO "$DMG"
echo "Built $DMG ($(du -h "$DMG" | cut -f1))"
