"""Thresholds that depend on something the capture cannot tell you.

Metamorfoze's illumination tolerance is 3 / 4 / 5 / 6 delta-L\\* for originals up to
A3 / A2 / A1 / A0. That number is not a property of the capture being analyzed --
the illumination measurement is made on a blank white sheet, which has no original
size -- and it is not really a property of the standard either: a lab says it works
to Metamorfoze Full, not to Metamorfoze Full A2. It is a property of *the job the
system is being qualified for*, which is a third thing.

So it is declared per run, against a table the profile carries, rather than by
minting a profile per band. The alternative was nine profile ids differing in one
threshold, which would also mean a lab digitising mixed-size material needed a
different profile per item with everything else identical.

The cost, stated plainly because it is real: a profile no longer determines the
verdict on its own. The band therefore travels with the result the same way the ROI
inset and the percentile method do -- two runs under different bands are different
measurements.

**Without a declared size the metric is measured but not judged.** It reports the
value and the largest band the system clears, which is the question the measurement
actually answers -- "your illumination qualifies this system for originals up to A2"
is more useful than a pass against a band someone had to choose in advance, and it
cannot mislead in either direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from deltachrome.errors import ProfileError

#: ISO 216 A-series long edges, in millimetres. Used to turn a declared size into a
#: band when the operator gives a paper name rather than a measurement.
A_SERIES = {
    "A0": 1189.0, "A1": 841.0, "A2": 594.0, "A3": 420.0,
    "A4": 297.0, "A5": 210.0, "A6": 148.0,
}

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class Band:
    """One row of a banded threshold."""

    band: str
    pass_condition: str
    max_mm: float | None = None
    label: str | None = None

    def describe(self) -> str:
        return self.label or self.band


@dataclass(frozen=True)
class BandedCondition:
    """A profile's declaration that one metric's threshold depends on a run input."""

    parameter: str
    bands: tuple[Band, ...]
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BandedCondition":
        rows = data.get("bands") or []
        if not rows:
            raise ProfileError("A banded pass condition must declare at least one band.")
        bands = []
        for row in rows:
            try:
                bands.append(Band(
                    band=str(row["band"]),
                    pass_condition=str(row["pass_condition"]),
                    max_mm=None if row.get("max_mm") is None else float(row["max_mm"]),
                    label=row.get("label"),
                ))
            except KeyError as exc:
                raise ProfileError(f"Band entry is missing required key {exc}") from exc
        if not data.get("parameter"):
            raise ProfileError("A banded pass condition must name its 'parameter'.")
        return cls(
            parameter=str(data["parameter"]),
            bands=tuple(sorted(bands, key=lambda b: (b.max_mm is None, b.max_mm or 0.0))),
            description=data.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "description": self.description,
            "bands": [
                {k: v for k, v in
                 {"band": b.band, "label": b.label, "max_mm": b.max_mm,
                  "pass_condition": b.pass_condition}.items() if v is not None}
                for b in self.bands
            ],
        }

    def named(self, value: str) -> Band:
        """The band a declared size falls in.

        Accepts a band name as written (``A2``), an A-series paper name whose size
        the table brackets (``A4`` falls in the ``A3`` band), a measurement in
        millimetres (``500x700``), or a single largest dimension (``450``).
        """
        text = str(value).strip()
        by_name = {b.band.upper(): b for b in self.bands}
        if text.upper() in by_name:
            return by_name[text.upper()]

        longest: float | None = None
        match = _SIZE.match(text)
        if match:
            longest = max(float(match.group(1)), float(match.group(2)))
        elif text.upper() in A_SERIES:
            longest = A_SERIES[text.upper()]
        else:
            try:
                longest = float(text)
            except ValueError:
                longest = None

        if longest is None:
            raise ProfileError(
                f"Cannot read {value!r} as a size for {self.parameter!r}. Give a band "
                f"name ({', '.join(b.band for b in self.bands)}), an A-series name, a "
                "measurement in mm as WxH, or the longest dimension in mm."
            )
        for band in self.bands:
            if band.max_mm is not None and longest <= band.max_mm:
                return band
        largest = self.bands[-1]
        raise ProfileError(
            f"{value!r} is {longest:g} mm on its longest side, beyond the largest "
            f"band this profile declares ({largest.describe()}, up to "
            f"{largest.max_mm:g} mm). The standard states no requirement above that, "
            "so there is nothing to judge against rather than a threshold to guess."
        )

    def clears(self, value: float) -> list[Band]:
        """Every band this measured value satisfies, tightest first."""
        from deltachrome.metrics.base import evaluate_condition

        return [b for b in self.bands if evaluate_condition(value, b.pass_condition)[0]]
