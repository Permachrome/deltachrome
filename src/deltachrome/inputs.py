"""Reading run inputs a person typed.

Four corner coordinates and a physical extent are not command-line concepts --
they are run inputs that any front end has to accept, validate and reject with
the same words. These parsers lived in ``cli.py`` until the desktop application
started importing one of them privately, which is the point at which "it is only
used by the CLI" stops being true.

Keeping them shared is also what lets a placement move between the two front
ends: the canvas writes ``--corners`` syntax and this reads it back, so one
geometry cannot acquire two descriptions.
"""

from __future__ import annotations

from deltachrome.errors import GeometryError, TargetError


def parse_extent(text: str | None) -> tuple[float, float] | None:
    """``"WIDTHxHEIGHT"`` in millimetres. ``None`` and empty mean "not given"."""
    if not text:
        return None
    parts = text.lower().replace("*", "x").replace(",", "x").split("x")
    if len(parts) != 2:
        raise TargetError(f'A fiducial extent is "WIDTHxHEIGHT" in mm; got {text!r}')
    try:
        width, height = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise TargetError(f"Could not read fiducial extent {text!r} as two numbers.") from exc
    if width <= 0 or height <= 0:
        raise TargetError("Fiducial extent values must be positive.")
    return (width, height)


def parse_corners(text: str) -> list[tuple[float, float]]:
    """Four ``"x,y"`` pairs in the target's own order (TL TR BR BL).

    The order is the chart's, not the image's: geometry is relative to the
    fiducial quadrilateral, so a chart photographed rotated needs no special
    handling and the homography absorbs it.
    """
    parts = text.replace(";", " ").split()
    if len(parts) != 4:
        raise GeometryError(
            f'Corners are four "x,y" pairs (TL TR BR BL); got {len(parts)}: {text!r}'
        )
    corners: list[tuple[float, float]] = []
    for part in parts:
        try:
            x, y = part.split(",")
            corners.append((float(x), float(y)))
        except ValueError as exc:
            raise GeometryError(f"Could not read corner {part!r} as 'x,y'.") from exc
    return corners
