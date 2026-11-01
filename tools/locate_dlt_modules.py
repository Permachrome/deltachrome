"""Measure the Golden Thread Device Level Target's five modules off a capture.

Each module prints two nearly concentric squares: a **dark mount**, tilted about 4.6
degrees, which is where its four slanted edges come from; and a white **"dots per
inch (optical)" field** inside it, printed square to the chart, which is what the
four corner placements are made on. The two tilts are not a discrepancy -- the mount
is slanted because a slanted-edge ROI needs it to be, and the field is not because
the bar patterns printed in it are read against the chart's own axes.

Both are declared in the target definition, and both were approximate until this
tool was written: the mount's declared size implied 800 px where every one measures
760.7, its declared tilt was 4.85 against a measured 4.601, and the declared centres
were 1 to 6 px out.

That mattered in exactly one place and not at all in the others. Edge extraction
snaps to the real square whatever size is declared, so SFR was unaffected; a check
run before and after the correction found **every metric identical to the last
digit** at a fixed placement. Placement *refinement* is the consumer: it fits a
transform through the declared-to-measured correspondences, so the declaration's own
error is a floor under the correction. Correcting it took refinement's residual on
this chart from 2.19 px to 0.23, and a placement 23 px out from recovering to 6.07 px
to recovering to 0.10.

**Both squares are emitted, because ten features is a different fit from five.** A
model may be fitted only when the features over-determine it, so five modules get a
similarity and ten get a homography -- and a homography is the only one of the two
that can undo a perspective error, which is what four corners placed independently
wrong actually is. See :mod:`deltachrome.imaging.refine`.

Run from the repository root:

    python tools/locate_dlt_modules.py ISA_Device_Level_Target.tif

``--save-corners`` writes the fiducial quad where ``locate_dlt_rulers.py`` expects
it, which is the order the two tools have to be run in.

**One trap, and it is the reason for the two window sizes below.** The dark mount is
found by contour rather than by threshold-and-largest-component, because the white
field punches a hole through the middle of it and a filled component is not the same
shape. And the search window has to be tight: at half the chart's module pitch, the
windows around the two upper modules reach past the card's own edge into the dark
table behind it, the mount merges with the background, and ``minAreaRect`` returns a
1320 px square at 89 degrees -- plausible-looking, wrong, and silently averaged in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

#: Half-width of the window the white field is sought in, in pixels of the capture.
#: Generous: the field is about 420 px and nothing else on the chart is that bright.
FIELD_WINDOW = 300
#: Half-width for the dark mount. Tight on purpose -- see the module docstring.
MOUNT_WINDOW = 470
#: Stride for the first, coarse pass. The fields are 420 px, so an eighth of that is
#: still 52 px of blob and there is no reason to threshold 100 megapixels.
STRIDE = 8

POSITIONS = ("TL", "TR", "BR", "BL", "CENTER")


def load_luma(path: Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as handle:
        array = handle.pages[0].asarray()
    scale = float(np.iinfo(array.dtype).max) if array.dtype.kind in "ui" else 1.0
    array = array.astype(np.float32) / scale
    if array.ndim == 2:
        return array
    return array[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def find_fields(gray: np.ndarray) -> dict[str, tuple[float, float]]:
    """The five white resolution fields, coarsely, then refined at full resolution."""
    small = gray[::STRIDE, ::STRIDE]
    seeds: list[tuple[float, float]] = []
    for level in np.linspace(0.35, 0.60, 11):
        mask = cv2.morphologyEx(
            (small > level).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found = [
            centroids[i] * STRIDE
            for i in range(1, count)
            if stats[i][4] > small.size * 0.0005
            and 0.7 < stats[i][2] / max(stats[i][3], 1) < 1.4
            and stats[i][4] / (stats[i][2] * stats[i][3]) > 0.75
        ]
        if len(found) == 5:
            seeds = found
            break
    if len(seeds) != 5:
        raise SystemExit(
            f"expected five white resolution fields, found {len(seeds)}. The chart "
            "may be framed differently, or the capture may be much darker or lighter "
            "than the one this was measured on."
        )

    refined = {}
    for name, (cx, cy) in zip(_ordered(seeds), sorted(seeds, key=_order_key(seeds))):
        refined[name] = _square(gray, cx, cy, FIELD_WINDOW, bright=True)
    return refined


def _ordered(points) -> list[str]:
    return list(POSITIONS)


def _order_key(points):
    """TL, TR, BR, BL, CENTER -- the order the four placed corners are matched in."""
    centre = np.mean(points, axis=0)
    middle = int(np.argmin(np.linalg.norm(np.asarray(points) - centre, axis=1)))
    ranks = {}
    corners = [p for i, p in enumerate(points) if i != middle]
    by_row = sorted(corners, key=lambda p: p[1])
    top = sorted(by_row[:2], key=lambda p: p[0])
    bottom = sorted(by_row[2:], key=lambda p: p[0])
    for rank, point in enumerate([top[0], top[1], bottom[1], bottom[0],
                                  points[middle]]):
        ranks[tuple(point)] = rank
    return lambda p: ranks[tuple(p)]


def _square(gray, cx, cy, window, *, bright):
    """One square, as centre, side and tilt.

    ``bright`` picks the white field out of its mount by Otsu; otherwise the dark
    mount is taken as the largest *external contour* below the card level, because
    the field punches a hole through the middle of it.
    """
    x0, y0 = int(cx - window), int(cy - window)
    patch = gray[y0:int(cy) + window, x0:int(cx) + window]
    if bright:
        level = np.uint8(np.clip(patch * 255.0, 0, 255))
        _, mask = cv2.threshold(level, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count < 2:
            raise SystemExit("no bright field in the window")
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        points = np.column_stack(np.nonzero(labels == chosen))[:, ::-1].astype(np.float32)
    else:
        card = float(np.median(patch))
        mask = (patch < (card + float(patch.min())) / 2.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise SystemExit("no dark mount in the window")
        points = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    # A square's tilt is only defined modulo 90, so it is folded into (-45, 45] --
    # the representative a reader expects. Reported as 89.6 rather than -0.4, the
    # axis-aligned resolution field looks like a feature slanted the other way.
    tilt = (float(rect[2]) + 45.0) % 90.0 - 45.0
    return {
        "centre": (x0 + rect[0][0], y0 + rect[0][1]),
        "side": (float(width), float(height)),
        "tilt_deg": tilt,
        "aspect": float(width) / max(float(height), 1e-6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--save-corners", type=Path, default=None,
                        help="write the fiducial quad as .npy, for locate_dlt_rulers.py")
    args = parser.parse_args()

    gray = load_luma(args.capture)
    print(f"\n{args.capture.name}: {gray.shape[1]} x {gray.shape[0]}")

    fields = find_fields(gray)
    mounts = {
        name: _square(gray, *field["centre"], MOUNT_WINDOW, bright=False)
        for name, field in fields.items()
    }

    quad = np.array([fields[n]["centre"] for n in ("TL", "TR", "BR", "BL")])
    width = float(np.mean([np.linalg.norm(quad[1] - quad[0]),
                           np.linalg.norm(quad[2] - quad[3])]))
    height = float(np.mean([np.linalg.norm(quad[3] - quad[0]),
                            np.linalg.norm(quad[2] - quad[1])]))
    print("\nFIDUCIAL QUAD (paste as --corners), the four corner fields' centres:")
    print("  " + " ".join(f"{x:.1f},{y:.1f}" for x, y in quad))
    print(f"  {width:.2f} x {height:.2f} px, aspect {width / height:.4f}")
    if args.save_corners:
        np.save(args.save_corners, quad)
        print(f"  wrote {args.save_corners}")

    matrix = cv2.getPerspectiveTransform(
        quad.astype(np.float32), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    ).astype(np.float64)

    for label, found, region in (("dark mount", mounts, "module"),
                                 ("white field", fields, "dpi_field")):
        sides = np.array([s for entry in found.values() for s in entry["side"]])
        tilts = np.array([entry["tilt_deg"] for entry in found.values()])
        print(f"\n{label.upper()}: side {sides.mean():.2f} px "
              f"(spread {np.ptp(sides):.2f}, sd {sides.std():.2f}), "
              f"tilt {tilts.mean():.3f} deg (spread {np.ptp(tilts):.3f})")
        print(f"  half_size relative: "
              f"[{sides.mean() / 2 / width:.5f}, {sides.mean() / 2 / height:.5f}]")
        for name in POSITIONS:
            point = cv2.perspectiveTransform(
                np.array([[found[name]["centre"]]], np.float64), matrix
            )[0][0]
            print(f"    {region}_{name.lower():<7} center [{point[0]:+.5f}, "
                  f"{point[1]:+.5f}]   aspect {found[name]['aspect']:.4f}")

    offsets = np.array([np.asarray(mounts[n]["centre"]) - np.asarray(fields[n]["centre"])
                        for n in POSITIONS])
    print(f"\nThe field is NOT concentric with its mount: it sits "
          f"({offsets[:, 0].mean():+.2f}, {offsets[:, 1].mean():+.2f}) px from the "
          f"mount's centre, consistent to {np.abs(offsets - offsets.mean(axis=0)).max():.2f} "
          "px across the five modules -- so that is the print, not measurement noise.")


if __name__ == "__main__":
    main()
