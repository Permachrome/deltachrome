"""Discovery and lookup of installed target definitions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from deltachrome.errors import TargetError
from deltachrome.targets.schema import TargetDefinition, load_target_file

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


@lru_cache(maxsize=1)
def _index() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(DEFINITIONS_DIR.glob("*.json")):
        try:
            target_id = json.loads(path.read_text(encoding="utf-8"))["target_id"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise TargetError(f"{path.name}: not a usable target definition -- {exc}") from exc
        found[target_id] = path
    return found


def available_targets() -> list[str]:
    return sorted(_index())


def load_target(target_id_or_path: str | Path) -> TargetDefinition:
    """Load a target by installed id, or from an explicit file path."""
    candidate = Path(target_id_or_path)
    if candidate.suffix == ".json" or candidate.exists():
        return load_target_file(candidate)

    index = _index()
    key = str(target_id_or_path)
    if key not in index:
        raise TargetError(
            f"Unknown target {key!r}. Installed targets: {available_targets()}. "
            "Pass a path to a .json file to use a custom definition."
        )
    return load_target_file(index[key])
