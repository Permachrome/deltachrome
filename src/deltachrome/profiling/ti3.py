"""The ``.ti3`` file ``colprof`` reads: device RGB beside the colour it should be.

Written the way Argyll's own ``scanin`` writes one for an input device, because
that is the file ``colprof`` was built to read: ``CTI3``, ``DEVICE_CLASS
"INPUT"``, ``COLOR_REP "XYZ_RGB"``, device values on 0-100 and XYZ on 0-100.
``colprof`` reads nothing else from it -- ``SAMPLE_ID`` is carried and unused,
and there is **no per-patch weight**, so leaving a patch out is the only way to
discount it.

The file is internal. ``colprof`` embeds it in the profile it writes (unless
told not to, which Deltachrome never does), so whatever is declared here --
which chart, which reference, which capture, which patches were left out -- is
in the finished profile for good.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from deltachrome import __version__
from deltachrome.profiling.patches import ProfilingSet

#: Keywords Deltachrome adds. CGATS requires a non-standard keyword to be
#: declared with ``KEYWORD`` before it is used.
KEYWORD_PREFIX = "DELTACHROME_"


def format_ti3(data: ProfilingSet, keywords: Mapping[str, str] | None = None) -> str:
    """The ``.ti3`` text for the used patches, with provenance as keywords.

    ``keywords`` are written as ``DELTACHROME_<NAME>``; the excluded patches are
    added to them here, since a profile built on a subset should say which.
    """
    declared = {f"{KEYWORD_PREFIX}{name.upper()}": value
                for name, value in (keywords or {}).items()}
    if data.excluded:
        declared[f"{KEYWORD_PREFIX}EXCLUDED"] = " ".join(data.excluded)

    lines = [
        "CTI3",
        "",
        'DESCRIPTOR "Argyll Calibration Target chart information 3"',
        f'ORIGINATOR "Deltachrome {__version__}"',
        f'CREATED "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"',
        'DEVICE_CLASS "INPUT"',
        'COLOR_REP "XYZ_RGB"',
        "",
    ]
    for name, value in declared.items():
        lines.append(f'KEYWORD "{name}"')
        lines.append(f'{name} "{_quoted(value)}"')
    if declared:
        lines.append("")

    lines += [
        "NUMBER_OF_FIELDS 7",
        "BEGIN_DATA_FORMAT",
        "SAMPLE_ID RGB_R RGB_G RGB_B XYZ_X XYZ_Y XYZ_Z",
        "END_DATA_FORMAT",
        "",
        f"NUMBER_OF_SETS {len(data.used)}",
        "BEGIN_DATA",
    ]
    for patch in data.used:
        rgb = patch.rgb * 100.0
        xyz = patch.xyz * 100.0
        lines.append(
            f"{patch.patch_id} "
            + " ".join(f"{value:.6f}" for value in rgb)
            + " "
            + " ".join(f"{value:.6f}" for value in xyz)
        )
    lines += ["END_DATA", ""]
    return "\n".join(lines)


def _quoted(value: str) -> str:
    """A value safe inside CGATS double quotes, which have no escape."""
    return str(value).replace('"', "'").replace("\n", " ")
