"""Measure the IT8.7/1 and IT8.7/2 lattice off a capture.

The IT8 charts cannot be measured the way the NGT and the DT NGT were. Those
carry patches with a mount between them, so each patch is a region a detector can
find on its own; on an IT8 the patches are **butted**, and where two neighbours
happen to be the same color there is no boundary between them at all -- rows A13
to A19 are one continuous white. A region finder run on that returns a hundred
patches out of two hundred and sixty-four, and worse, the ones it merges are not
randomly placed: they cluster in the pale columns, so the lattice it fits is
pulled toward the saturated end of the chart.

What the array does have is **pitch**. Twenty-two columns and twelve rows of
identical cells put a boundary at a regular interval whether or not any given one
is visible, so the right instrument is the one `imaging/rulers.py` uses on a
printed rule: collect every boundary that *is* visible, fit one comb through all
of them at once, and let the invisible ones be interpolated by the fit rather
than guessed at. Sixty or seventy visible boundaries constraining two parameters
is a very over-determined fit, and its residual is the check.

Run from the repository root:

    python tools/locate_it8_lattice.py IT8-7.2.tif
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from deltachrome.imaging.rulers import FUNDAMENTAL_THRESHOLD

Image.MAX_IMAGE_PIXELS = None

#: The IT8 color array, by the standard: rows A-L, columns 1-22.
ROWS, COLUMNS = 12, 22

#: How many bands are scanned across the array in each direction. One band sees
#: only the boundaries that happen to be visible along it; a dozen spread over
#: the array between them see nearly all of them, and the ones seen twice are
#: agreement rather than duplication.
BANDS = 12

#: A band's thickness, as a fraction of the array's extent across it. Thin enough
#: to stay within one row of patches, thick enough to average out print grain.
BAND_FRACTION = 0.02

#: A detected boundary is accepted into the comb if it sits within this fraction
#: of a pitch of a comb line. Wide enough to admit a boundary softened by the
#: lens, far narrower than the half pitch at which it would be ambiguous which
#: line it belonged to.
COMB_TOLERANCE = 0.25


def load(path: str) -> np.ndarray:
    image = Image.open(path)
    rgb = np.asarray(image.convert("RGB"))
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    return rgb.astype(np.float32)


def frame_rect(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """The printed black rule that boxes the labelled area, as x0, y0, x1, y1.

    Not the fiducial -- the array's own corners are that -- but the search
    region. Everything this tool must not mistake for a patch boundary is
    outside it: the grayscale strip, the barcode, the vendor's logotype and the
    chart's own cut edge.

    The rule is not found by being dark; the array holds patches far darker than
    it. It is found by being a *narrow* dark line, which nothing else spanning
    the chart is. The chart's own cut edge is narrow and dark too, so the card is
    located first and anything sitting on its boundary discounted.
    """
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    card = _card_rect(gray)
    x0, y0, x1, y1 = card
    card_w, card_h = x1 - x0, y1 - y0

    rows = 255.0 - gray[y0:y1, x0 + int(card_w * 0.35) : x0 + int(card_w * 0.65)].mean(axis=1)
    cols = 255.0 - gray[y0 + int(card_h * 0.35) : y0 + int(card_h * 0.65), x0:x1].mean(axis=0)
    top, bottom = _outermost_line(rows, card_h, y0, (y0, y1))
    left, right = _outermost_line(cols, card_w, x0, (x0, x1))
    return left, top, right, bottom


def _card_rect(gray: np.ndarray) -> tuple[int, int, int, int]:
    """The chart itself, against whatever it was photographed on."""
    _, mask = cv2.threshold(
        gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if len(stats) < 2:
        return 0, 0, gray.shape[1], gray.shape[0]
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[biggest]
    return int(x), int(y), int(x + w), int(y + h)


def _outermost_line(
    darkness: np.ndarray, span: int, offset: int, card: tuple[int, int]
) -> tuple[int, int]:
    """First and last narrow dark line in a projection, ignoring the card edge."""
    smooth = cv2.GaussianBlur(darkness.reshape(-1, 1).astype(np.float32), (0, 0), 1.0).ravel()
    width = max(int(span * 0.012), 3)
    baseline = cv2.GaussianBlur(smooth.reshape(-1, 1), (0, 0), width * 4.0).ravel()
    prominence = smooth - baseline
    threshold = 0.35 * prominence.max()

    found: list[int] = []
    index = 0
    while index < len(prominence):
        if prominence[index] <= threshold:
            index += 1
            continue
        end = index
        while end < len(prominence) and prominence[end] > threshold:
            end += 1
        if end - index <= width:
            found.append(offset + index + int(np.argmax(prominence[index:end])))
        index = end

    # A line within a hair of the card's own boundary is the cut edge, not print.
    # A few pixels, not a fraction of the chart: on a capture cropped to the chart
    # the card box *is* the image and a generous margin discards the top rule,
    # which on the 7/1 sample sits 53 px in.
    margin = 5
    inside = [p for p in found if card[0] + margin < p < card[1] - margin]
    if len(inside) < 2:
        return card
    return min(inside), max(inside)


def boundary_profile(
    rgb: np.ndarray, rect: tuple[int, int, int, int], axis: int
) -> tuple[np.ndarray, int]:
    """How strongly a patch boundary shows at each position along one axis.

    Colour difference, not luminance difference: neighbouring IT8 patches are
    frequently the same lightness and a different hue, and a boundary read from
    gray alone misses every one of those.

    Summed over bands spread across the array. A boundary invisible along one
    band -- two neighbours of the same colour -- is usually visible along
    another, because the pair above or below it differ; adding the bands is what
    turns "most boundaries are visible somewhere" into one profile in which
    nearly all of them are.
    """
    x0, y0, x1, y1 = rect
    along = (x0, x1) if axis == 0 else (y0, y1)
    across = (y0, y1) if axis == 0 else (x0, x1)
    thickness = max(int((across[1] - across[0]) * BAND_FRACTION), 2)

    total = np.zeros(along[1] - along[0] - 1, dtype=np.float64)
    for centre in np.linspace(across[0], across[1], BANDS + 2)[1:-1]:
        lo = int(centre) - thickness // 2
        hi = lo + thickness
        block = rgb[lo:hi, along[0] : along[1]] if axis == 0 else rgb[along[0] : along[1], lo:hi]
        profile = block.mean(axis=0 if axis == 0 else 1)
        difference = np.abs(np.diff(profile, axis=0)).sum(axis=1)
        # Normalize each band before adding. One band crossing a row of
        # saturated patches would otherwise outweigh ten crossing pale ones, and
        # the boundaries only that band sees would dominate the comb.
        peak = float(difference.max())
        if peak > 0:
            total += difference / peak
    return cv2.GaussianBlur(total.reshape(-1, 1).astype(np.float32), (0, 0), 1.5).ravel(), along[0]


def comb_period(profile: np.ndarray, lines: int) -> float:
    """The pitch the boundaries repeat at, by autocorrelation.

    The same reasoning `imaging/rulers.py` applies to a printed scale, and for
    the same reason: taking the tallest autocorrelation peak reads a comb at a
    multiple of its own pitch. An IT8's columns invite exactly that -- the chart
    is built in blocks of colour families, so the boundary between blocks is
    stronger than the boundaries within one, and the block period is a multiple
    of the patch period. So the fundamental is searched for beneath the
    strongest lag, and it has to divide it.
    """
    centred = profile - profile.mean()
    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    # A pitch cannot be longer than the array divided by the number of intervals,
    # nor much shorter -- the array is known to hold `lines - 1` of them.
    expected = len(profile) / (lines - 1)
    lo = max(int(expected * 0.45), 4)
    hi = min(int(expected * 2.5), len(correlation) - 1)
    if hi <= lo + 2:
        raise SystemExit("The scanned span is too short to hold the array.")
    peaks, _ = find_peaks(correlation[lo:hi])
    if peaks.size == 0:
        raise SystemExit("No repeating boundary pattern was found.")
    peaks = peaks + lo
    strongest = int(peaks[int(np.argmax(correlation[peaks]))])
    period = float(strongest)
    for divisor in range(2, 5):
        candidate = strongest / divisor
        if candidate < lo:
            break
        window = int(round(candidate * 0.15)) + 1
        near = [int(q) for q in peaks if abs(q - candidate) <= window]
        if near and max(correlation[q] for q in near) > FUNDAMENTAL_THRESHOLD * float(
            correlation[strongest]
        ):
            period = float(near[int(np.argmax([correlation[q] for q in near]))])
    return period


def fit_comb(
    profile: np.ndarray, offset: int, lines: int
) -> tuple[float, float, float, int, int]:
    """Origin, pitch, residual, boundaries used and boundaries found.

    Boundaries that do not sit on the comb are dropped rather than accommodated:
    on a real chart they are the printed rule, a row label or a speck of dust,
    and a fit that bent to include them would be reporting the dust.
    """
    period = comb_period(profile, lines)
    peaks, _ = find_peaks(profile, distance=max(3, int(round(period * 0.6))))
    if len(peaks) < 4:
        raise SystemExit(f"Only {len(peaks)} boundaries found; cannot fit a comb.")

    # Sub-pixel, by centroid of the ridge; +0.5 because a difference between
    # samples i and i+1 is a boundary between them, not on either.
    located = []
    half = max(2, int(round(period * 0.2)))
    for peak in peaks:
        a, b = max(peak - half, 0), min(peak + half + 1, len(profile))
        weight = profile[a:b] - profile[a:b].min()
        if weight.sum() > 0:
            centroid = float((np.arange(a, b) * weight).sum() / weight.sum()) + 0.5
            # Carry the ridge's own height, read at the peak rather than at the
            # centroid. The array's outermost boundaries have a patch on one side
            # and the surround on the other, so their ridges are asymmetric and
            # the centroid slides several pixels off the top of one -- which
            # reported the strongest boundary on the chart as one of the weakest.
            located.append((centroid, float(profile[peak])))
    located.sort()
    positions = np.array([p for p, _ in located])
    strengths = np.array([s for _, s in located])

    origin, pitch = float(positions[0]), period
    used, strength = positions, strengths
    for _ in range(6):
        index = np.round((positions - origin) / pitch)
        inliers = np.abs(positions - (origin + index * pitch)) < pitch * COMB_TOLERANCE
        if inliers.sum() < 4:
            break
        design = np.column_stack([np.ones(int(inliers.sum())), index[inliers]])
        solution, *_ = np.linalg.lstsq(design, positions[inliers], rcond=None)
        origin, pitch = float(solution[0]), float(solution[1])
        used, strength = positions[inliers], strengths[inliers]

    index = np.round((used - origin) / pitch)
    residual = float(np.sqrt(np.mean((used - (origin + index * pitch)) ** 2)))

    # Where the array starts is a separate question from what its pitch is, and
    # answering it as "the first boundary on the comb" is wrong. The IT8 layout
    # puts its row and column labels in a margin exactly one patch wide, so the
    # printed rule that boxes them sits exactly one pitch outside the array and
    # lands on a comb line every time. Anchoring on the first inlier therefore
    # shifts the whole lattice by one patch -- on both sample charts, in both
    # axes, which is a systematic trap rather than bad luck.
    #
    # Two things separate the array's window of `lines` comb lines from that one:
    # it holds the most detected boundaries, and *both* its ends are strong,
    # because both are the array meeting the surround. The decoy window has one
    # strong end and one weak interior boundary at the other.
    present = {int(i): float(strength[j]) for j, i in enumerate(index)}
    lowest, highest = int(index.min()), int(index.max())
    best = (-1, -1.0, lowest)
    for start in range(lowest, highest - lines + 2):
        window = range(start, start + lines)
        count = sum(1 for i in window if i in present)
        ends = min(present.get(start, 0.0), present.get(start + lines - 1, 0.0))
        if (count, ends) > best[:2]:
            best = (count, ends, start)
    first = origin + pitch * best[2]
    return offset + first, pitch, residual, best[0], len(positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument(
        "--frame", help="x0,y0,x1,y1 of the printed rule, where the search for it "
        "lands on patch boundaries instead -- it looks for the outermost narrow dark "
        "line, and on the Wolf Faust IT8.7/2 the boundaries between its darkest and "
        "palest columns are narrower and darker than its own keyline",
    )
    args = parser.parse_args()

    rgb = load(args.capture)
    rect = (tuple(int(float(v)) for v in args.frame.split(",")) if args.frame
            else frame_rect(rgb))
    print(f"capture      {args.capture}  {rgb.shape[1]}x{rgb.shape[0]} px")
    print(f"printed rule x {rect[0]}..{rect[2]}, y {rect[1]}..{rect[3]}")

    bounds = {}
    for axis, lines, name in ((0, args.columns + 1, "columns"), (1, args.rows + 1, "rows")):
        profile, offset = boundary_profile(rgb, rect, axis)
        first, pitch, residual, used, found = fit_comb(profile, offset, lines)
        bounds[name] = (first, pitch)
        print(
            f"{name:8s}     pitch {pitch:8.3f} px, {used} of {found} boundaries used, "
            f"residual {residual:.3f} px ({100 * residual / pitch:.2f}% of a pitch)"
        )
        print(
            f"             array runs {first:.2f} .. {first + pitch * (lines - 1):.2f} px "
            f"({pitch * (lines - 1):.1f} px for {lines - 1} patches)"
        )

    x0, pitch_x = bounds["columns"]
    y0, pitch_y = bounds["rows"]
    x1 = x0 + pitch_x * args.columns
    y1 = y0 + pitch_y * args.rows
    print("\nquad (array corners, full px)")
    for label, point in (
        ("TL", (x0, y0)), ("TR", (x1, y0)), ("BR", (x1, y1)), ("BL", (x0, y1))
    ):
        print(f"  {label}  {point[0]:9.2f} {point[1]:9.2f}")

    geometry = {
        "patch_size_relative": {
            "width": round(1.0 / args.columns, 5),
            "height": round(1.0 / args.rows, 5),
        },
        "patch_spacing_relative": {
            "horizontal": round(1.0 / args.columns, 5),
            "vertical": round(1.0 / args.rows, 5),
        },
        "grid_origin_relative": [0.0, 0.0],
        "provisional": False,
    }
    print("\npatch_grid geometry, in the frame those four corners define:")
    print(json.dumps(geometry, indent=2))
    print(
        "\nPatch size equals the spacing because IT8 patches are butted. The 0.55 "
        "sampling inset is what keeps an ROI off the boundary."
    )


if __name__ == "__main__":
    main()
