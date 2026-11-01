"""Lay two or more real captures into one frame, with the ground truth beside them.

There is no capture here of a frame holding two *different* charts, and it is an
ordinary thing to shoot: a device-level target beside a colour chart on a copy stand,
a scale beside a chart. Reading such a frame cannot be built or checked against nothing.

**A composite is not a capture, and this tool is careful to be honest about which
claims it can support.** What it produces is real chart pixels, photographed by real
equipment, in an arrangement nobody photographed. That is enough to test the
*search*: whether the extents separate, whether each is identified as the right
model, whether one chart's finder is fooled by another chart's marks. It is not
enough to test anything about the optics, the illumination or the colour: the two
halves were shot at different times under different lighting at different scales, so
a measurement made across the seam means nothing.

So it refuses to compose captures that do not share an ICC profile and a bit depth.
A file whose left half is Adobe RGB and whose right half is eciRGB v2 is not a
plausible master, and a fixture that quietly is one would put a colour-management bug
somewhere nobody would look for it.

Run from the repository root::

    python tools/compose_multi_chart.py isa_golden_thread_dlt_v1 colorchecker_sg_v1
    python tools/compose_multi_chart.py --out data/composites isa_golden_thread_dlt_v1 colorchecker_sg_v1

The charts are named by target id and their captures and committed fiducial quads
come from ``tools/render_target_overlays.py``, which is what makes the ground truth
exact rather than approximate: each quad is translated by the paste offset and
nothing else, so a placement found in the composite can be compared against a number
that was measured rather than eyeballed.

Written beside the composite is a ``.json`` naming, for each chart, its target id and
its four corners in the composite's own pixels. That file is the answer key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from render_target_overlays import PLACEMENTS  # noqa: E402

#: How much room to leave around each chart's own extent, as a fraction of it.
#:
#: Not cosmetic. Every finder works inside an extent and the extent search finds a
#: chart as the region unlike its surround, so a chart cropped tight to its fiducials
#: has no surround to be told from -- the composite would test a case the operator
#: never produces. A fifth is roughly what a chart occupies on a copy stand.
MARGIN = 0.20

#: The gap between charts, as a fraction of the larger one's short side.
#:
#: **This is the constant the whole exercise is about.** Two charts closer together
#: than the extent search's morphological close -- a fiftieth of the frame -- merge
#: into one region, and a quad placed across a merged extent is the failure mode that
#: multi-chart detection exists to avoid. A quarter is comfortably clear of it; pass
#: --gap to squeeze them and see where it breaks.
GAP = 0.25


def _icc_bytes(path: Path) -> bytes:
    """The profile's own bytes, which ``ICCProfile`` does not keep -- it parses and
    discards them. The composite has to carry a real profile, not a re-encoded one."""
    if path.suffix.lower() in {".jp2", ".jpf", ".j2k"}:
        from deltachrome.imaging.loader import _jp2_color_spec, _read_head

        icc, _notes = _jp2_color_spec(_read_head(path), path)
        if icc is None:
            raise SystemExit(f"{path.name} carries no ICC profile to copy.")
        return icc

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        icc = img.info.get("icc_profile")
    if not icc:
        raise SystemExit(f"{path.name} carries no ICC profile to copy.")
    return icc


def _entry(target_id: str):
    for placement in PLACEMENTS:
        if placement.target_id == target_id:
            return placement
    raise SystemExit(
        f"no committed placement for {target_id!r}. Known: "
        + ", ".join(sorted({p.target_id for p in PLACEMENTS}))
    )


def _crop_around(image, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The chart's neighbourhood, and its quad in the crop's own coordinates."""
    low, high = quad.min(axis=0), quad.max(axis=0)
    margin = MARGIN * float(np.max(high - low))
    x0 = max(int(low[0] - margin), 0)
    y0 = max(int(low[1] - margin), 0)
    x1 = min(int(high[0] + margin), image.width)
    y1 = min(int(high[1] + margin), image.height)
    return image.pixels[y0:y1, x0:x1], quad - np.array([x0, y0], float)


def compose(target_ids: list[str], *, gap: float, down: bool):
    """Every named chart in one frame, with each quad in the composite's pixels."""
    from deltachrome.imaging.loader import load_image

    charts = []
    for target_id in target_ids:
        entry = _entry(target_id)
        path = entry.path
        if not path.exists():
            raise SystemExit(
                f"{entry.capture} is not present. The captures are gitignored -- they "
                "belong to the Library of Congress and to the operator."
            )
        image = load_image(path)
        crop, quad = _crop_around(image, np.array(entry.corners, float))
        charts.append((target_id, image, crop, quad, _icc_bytes(path)))

    profiles = {c[1].profile.description for c in charts}
    depths = {c[1].pixels.dtype for c in charts}
    if len(profiles) > 1 or len(depths) > 1:
        raise SystemExit(
            "these captures do not share a colour space and a bit depth, so a "
            "composite of them would not be a plausible master and any colour read "
            "from it would be meaningless:\n  "
            + "\n  ".join(
                f"{c[0]}: {c[1].profile.description}, {c[1].pixels.dtype}" for c in charts
            )
            + "\nCompose captures that agree, or test the search on a pair that does."
        )

    sizes = [c[2].shape for c in charts]
    pitch = int(gap * max(min(h, w) for h, w, _ in sizes))
    if down:
        height = sum(h for h, _w, _c in sizes) + pitch * (len(sizes) - 1)
        width = max(w for _h, w, _c in sizes)
    else:
        height = max(h for h, _w, _c in sizes)
        width = sum(w for _h, w, _c in sizes) + pitch * (len(sizes) - 1)

    # The ground is the darkest chart's own surround level, so the composite reads as
    # one sitting rather than as two crops on a colour nothing was shot against.
    ground = min(int(np.median(c[2][:8, :8])) for c in charts)
    canvas = np.full((height, width, 3), ground, charts[0][2].dtype)

    truth, cursor = [], 0
    for target_id, _image, crop, quad, _icc in charts:
        h, w, _ = crop.shape
        y = cursor if down else (height - h) // 2
        x = (width - w) // 2 if down else cursor
        canvas[y:y + h, x:x + w] = crop
        truth.append({
            "target_id": target_id,
            "corners": [[round(float(px + x), 2), round(float(py + y), 2)]
                        for px, py in quad],
        })
        cursor += (h if down else w) + pitch

    return canvas, truth, charts[0][1], charts[0][4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("targets", nargs="+", help="target ids to lay out")
    parser.add_argument("--out", type=Path, default=ROOT / "data/composites")
    parser.add_argument("--gap", type=float, default=GAP,
                        help="space between charts, as a fraction of the larger "
                             "chart's short side. Squeeze it to find where two "
                             "extents merge into one")
    parser.add_argument("--down", action="store_true",
                        help="stack vertically instead of side by side")
    args = parser.parse_args()

    canvas, truth, first, icc = compose(args.targets, gap=args.gap, down=args.down)

    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures import write_tiff_with_profile

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "__".join(args.targets)
    path = args.out / f"{stem}.tif"
    write_tiff_with_profile(path, canvas, icc)
    (args.out / f"{stem}.json").write_text(
        json.dumps({"capture": path.name, "charts": truth}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{path}  {canvas.shape[1]} x {canvas.shape[0]}, {canvas.dtype}, "
          f"{first.profile.description}")
    for chart in truth:
        corners = " ".join(f"{x:.1f},{y:.1f}" for x, y in chart["corners"])
        print(f"  {chart['target_id']:34s} {corners}")
    print(f"\nground truth written to {args.out / f'{stem}.json'}")


if __name__ == "__main__":
    main()
