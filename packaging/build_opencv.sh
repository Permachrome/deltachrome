#!/bin/sh
# Build OpenCV without FFmpeg or any video backend, as a wheel.
#
#     packaging/build_opencv.sh            # writes build/opencv/wheels/*.whl
#
# The official opencv-python macOS wheels carry an FFmpeg configured with
# --enable-gpl (x264, x265), so shipping them distributes GPLv3 code -- for a
# video stack Deltachrome never uses. This is the same release (opencv-python
# 93, OpenCV 5.0.0) with video switched off and every image codec built from
# OpenCV's own sources, so nothing from the build machine is linked in.
# About 25 minutes on an M-series Mac.
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-packaging/.venv/bin/python}
MIN_MACOS=${MIN_MACOS:-13.0}
mkdir -p build/opencv && cd build/opencv
[ -d opencv-python ] || git clone -q --depth 1 --branch 93 https://github.com/opencv/opencv-python.git
cd opencv-python && git submodule update -q --init --depth 1 opencv
export ENABLE_HEADLESS=1 MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS
export CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
export CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF \
 -DWITH_TESSERACT=OFF -DWITH_OPENEXR=OFF -DBUILD_OPENEXR=OFF -DWITH_1394=OFF \
 -DOPENCV_FORCE_3RDPARTY_BUILD=ON -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local \
 -DCMAKE_OSX_DEPLOYMENT_TARGET=$MIN_MACOS"
"$PY" -m pip wheel . -w ../wheels --no-deps
