"""Re-derive every committed placement automatically, and say what changed.

Adding a target adds a candidate to identification for every capture anybody
has, and changing a finder changes every chart that finder places. Both are
measured against the committed placements in
``tools/render_target_overlays.PLACEMENTS`` -- the corners each capture was
measured at by a named tool -- and until this script that measurement was made
by hand each time and quoted in prose ("twenty-three of twenty-four re-derived",
"sixteen of nineteen named, none wrongly"). This makes it one command, so it is
made every time.

Two questions per committed placement:

* **Placement.** ``detect_placements`` for the committed target, and the worst
  corner distance to the committed corners, in pixels and as a fraction of the
  chart's shorter side. Under 1% is a pass: refinement's capture range is 5%.
* **Identification.** ``identify_targets`` on the capture, once per capture, and
  whether the committed target is among the charts it names. Naming a target
  that is not committed for that capture is reported as WRONG -- the one outcome
  this application must not produce. Declining is not wrong.

Run from the repository root:

    python tools/check_detection.py                  # every committed placement
    python tools/check_detection.py --target ID      # those of one target
    python tools/check_detection.py --capture PATH [--region x0,y0,x1,y1] [--target ID ...]

The last form is for a capture not yet committed -- step 1 of
``docs/adding-a-target.md``: what the current build makes of it, and, for each
``--target``, its overlay and inset sensitivity at whatever placement detection
found. ``--region`` is *Search a Region*: a rectangle round the whole card.

Captures are gitignored, so whatever is absent is skipped and counted.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

#: A re-derived placement this close to the committed one, as a fraction of the
#: chart's shorter side, is a pass. Refinement's own capture range is five times
#: this, so a pass leaves a wide margin before an operator would notice.
PLACEMENT_TOLERANCE = 0.01


def _short_side(corners) -> float:
    import numpy as np

    c = np.asarray(corners, float)
    return float(min(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[3] - c[0])))


def _distance(a, b) -> float:
    import numpy as np

    return float(np.max(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=1)))


def _named(identifications) -> list[str]:
    return [i.target_id for i in identifications if i.found and i.target_id]


def check_committed(target_filter: str | None) -> int:
    from render_target_overlays import PLACEMENTS, capture_path

    from deltachrome.imaging.detect import detect_placements, identify_targets
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target

    by_capture = defaultdict(list)
    for entry in PLACEMENTS:
        if target_filter and entry.target_id != target_filter:
            continue
        by_capture[entry.capture].append(entry)

    rows, skipped, wrong, failed = [], 0, 0, 0
    for capture, entries in by_capture.items():
        path = capture_path(capture)
        if not path.exists():
            skipped += len(entries)
            continue
        started = time.time()
        image = load_image(path)
        identifications = identify_targets(image)
        named = _named(identifications)
        committed = {e.target_id for e in entries}
        intruders = sorted(set(named) - committed)
        wrong += len(intruders)
        for entry in entries:
            detections = [d for d in detect_placements(image, load_target(entry.target_id))
                          if d.found]
            if detections:
                best = min(_distance(d.corners, entry.corners) for d in detections)
                fraction = best / _short_side(entry.corners)
                placed = f"{best:8.2f} px {fraction:7.2%}"
                ok = fraction < PLACEMENT_TOLERANCE
            else:
                placed, ok = "   refused        ", False
            failed += not ok
            ident = ("named" if entry.target_id in named else
                     "declined" if not named else "missed")
            rows.append((entry.name, placed, "ok" if ok else "FAIL", ident,
                         ", ".join(intruders) or "-", f"{time.time() - started:5.1f}s"))
            print(*rows[-1], sep="  |  ", flush=True)

    print()
    print(f"{len(rows)} placements checked, {skipped} skipped (capture absent).")
    print(f"{len(rows) - failed} re-derived within {PLACEMENT_TOLERANCE:.0%}; "
          f"{sum(r[3] == 'named' for r in rows)} identified; "
          f"{wrong} WRONG name(s).")
    return 1 if wrong or failed else 0


def check_capture(capture: Path, region: str | None, targets: list[str]) -> int:
    import render_target_overlays as overlays

    from deltachrome.imaging.detect import (
        detect_placement, detect_placement_near, extent_from_corners,
        identify_target, identify_targets,
    )
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target

    image = load_image(capture)
    print(f"{capture.name}: {image.width} x {image.height}")
    quad = None
    if region:
        x0, y0, x1, y1 = (float(v) for v in region.split(","))
        quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        identifications = [identify_target(image, extent=extent_from_corners(image, quad))]
    else:
        identifications = identify_targets(image)
    for ident in identifications:
        print(f"\nidentified: {ident.target_id or '(nothing named)'} -- {ident.reason}")
        for candidate in ident.candidates[:5]:
            print(f"  candidate {candidate.target_id}: aspect {candidate.aspect_error:.3f}, "
                  f"ramp {candidate.ramp}, colour {candidate.color}, "
                  f"flagged {candidate.flagged:.0%}"
                  + (f" -- {candidate.reason}" if candidate.reason else ""))

    out = ROOT / "data" / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    for target_id in targets:
        target = load_target(target_id)
        detection = (detect_placement_near(image, target, quad) if quad
                     else detect_placement(image, target))
        print(f"\n{target_id}: {detection.reason}")
        if not detection.found:
            continue
        corners = tuple(tuple(round(v, 1) for v in p) for p in detection.corners)
        print(f"  corners {corners}")
        entry = overlays.Placement(target_id, str(capture), corners, "check_detection.py")
        print(overlays.render(entry, out, refine=False, labels=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--region", help="x0,y0,x1,y1 round the whole card")
    args = parser.parse_args()
    if args.capture:
        sys.exit(check_capture(args.capture, args.region, args.target))
    if len(args.target) > 1:
        parser.error("give one --target, or none, when checking committed placements")
    sys.exit(check_committed(args.target[0] if args.target else None))


if __name__ == "__main__":
    main()
