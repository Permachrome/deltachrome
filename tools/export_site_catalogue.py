"""Write what Deltachrome supports as JSON, for the website's support grid.

The website states which targets are supported, which guidelines ship, and
which metrics each target can carry. Copied by hand, that table would be out of
date the first time a target is added, so it is generated from the same
definitions the application reads: every shipped target, every shipped
guideline, and each metric's own ``requires_capability``.

A target carries a metric when it declares the capability that metric reads.
Three metrics read the file rather than the chart and apply to every target.
Sampling frequency reads no capability but needs a dimension it can measure the
chart by, and a target declaring ``dimensional_reference.available: false``
cannot supply one -- the same N/A the run itself reports.

Local guidelines in the operator's own folder are left out; only what ships is
described.

Run from the repository root:

    python tools/export_site_catalogue.py ../permachro.me/data/catalogue.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deltachrome.profiles.catalogue import catalogue, metric_order, metric_templates
from deltachrome.profiles.schema import is_local, load_profile
from deltachrome.reporting.labels import metric_title
from deltachrome.targets.registry import available_targets, load_target
from deltachrome_ui.families import CAPTURE_FAMILY, FAMILY_NAMES

#: Read from the file, so every target carries them.
FILE_METRICS = {"master_file_format", "color_space", "bit_depth"}

#: Bodies whose guidelines ship. ``measurement_only`` is a convenience profile
#: judging nothing, not a guideline anybody would look up.
SHIPPED_BODIES = {"FADGI", "Metamorfoze"}


def _family(capability: str | None) -> tuple[str, int]:
    if capability is None:
        return CAPTURE_FAMILY
    return FAMILY_NAMES.get(capability, (capability.replace("_", " ").title(), 99))


def _carries(target, metric_id: str, capability: str | None) -> bool:
    if metric_id in FILE_METRICS:
        return True
    if capability is None:
        # Sampling frequency: needs a dimension to measure the chart by.
        reference = target.dimensional_reference or {}
        return reference.get("available") is not False and reference.get("method") != "none"
    return target.supports(capability)


def _manufacturer(name: str | None) -> str:
    if not name:
        return ""
    return name.split(" -- ")[0]


def build() -> dict:
    templates = metric_templates()
    metrics = []
    for metric_id in metric_order():
        spec = templates.get(metric_id)
        capability = spec.requires_capability if spec else None
        family, order = _family(capability)
        metrics.append({
            "id": metric_id,
            "name": metric_title(metric_id),
            "capability": capability,
            "family": family,
            "family_order": order,
        })
    metrics.sort(key=lambda m: (m["family_order"], metric_order().index(m["id"])))

    targets = []
    for target_id in available_targets():
        target = load_target(target_id)
        kind = target.target_type or ""
        targets.append({
            "id": target_id,
            "name": target.display_name,
            "manufacturer": _manufacturer(target.manufacturer),
            "kind": ("flat field" if kind == "flat_field"
                     else "transmissive" if kind.startswith("transmissive")
                     else "reflective"),
            "metrics": [m["id"] for m in metrics
                        if _carries(target, m["id"], m["capability"])],
        })
    targets.sort(key=lambda t: (t["kind"] == "flat field", t["kind"], t["name"]))

    guidelines = []
    for summary in catalogue():
        if (is_local(summary.profile_id) or summary.local_criteria
                or summary.body not in SHIPPED_BODIES):
            continue
        profile = load_profile(summary.profile_id)
        guidelines.append({
            "id": summary.profile_id,
            "body": summary.body,
            "edition": summary.edition,
            "material": summary.material or "",
            "level": summary.level_label,
            "rank": summary.rank,
            "metrics": [m.metric_id for m in profile.metrics],
        })
    guidelines.sort(key=lambda g: (g["body"], g["material"], g["rank"]))

    return {"metrics": metrics, "targets": targets, "guidelines": guidelines}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(data['targets'])} targets, {len(data['guidelines'])} guidelines, "
          f"{len(data['metrics'])} metrics -> {args.out}")


if __name__ == "__main__":
    main()
