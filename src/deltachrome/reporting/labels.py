"""How a measurement is written down for a person -- on screen and in an export.

Kept in the engine rather than the frontend because two things have to say the
same words: the report on screen and the files the export writes. None of it ever
needed a toolkit, and ``deltachrome_ui.labels`` re-exports all of it.

Three small things, together because they are the same thing: turning what the
record stores into what a reader should see, without touching the record.

**Names.** Metric ids are `snake_case` and were being shown that way with the
underscores swapped for spaces -- `sfr10 sampling efficiency`. A measurement is
a proper noun in this domain and gets Title Case, with the acronyms left as
acronyms rather than title-cased into `Sfr10`.

**Units.** The profiles store `dE2000`, which is what a keyboard produces and
not what the quantity is called. On screen it is ΔE*2000. This is presentation
only: the record keeps the ASCII spelling it was written with, because the
record is the archive and a run written this year has to still parse next year.

**Where a reading was taken.** A region id is `module_tl`, and the field
positions a chart prints run to a full three-by-three: `tl`, `tc`, `tr`, `cl`,
`cc`, `cr`, `bl`, `bc`, `br` on the UTT. Only the four corners and the centre
were ever named, so five of the UTT's nine came out as `Module Cc` and `Module
Tc` -- and on a registration axis with nine of them side by side that is a row
of overlapping words. Two forms are given: the long name for a chip, and a
one-or-two letter **field code** for an axis tick.

**Which way an edge was read.** ISO 12233 names the *direction of the
measurement*, not the geometry of the edge that produced it: its definition of
horizontal resolution is the value "measured in the longer image dimension ...
typically using a vertical or near vertical oriented test-[feature]", and
vertical resolution the converse. The engine's `orientation` is the edge's own
geometry, which is the right fact for it to hold -- it is what decides which way
the ROI is long -- and the wrong word to show a reader in this domain, who reads
"vertical" as vertical resolution. So the report shows the direction, and the
registration grid already did.

**Conditions.** `value < 3.5` and `45 <= value <= 65` are the two shapes a
profile's `pass_condition` takes, and until now they were only ever *printed*.
Parsed, they are two things. They are the aim band a plot can draw -- which is
the thing OpenDICE does that makes its plots readable at a glance: not "is this
number under the limit" but "how much room is left". And they are a criterion
that can be written the way a person would write it: `>= 190` rather than
`value >= 190`, since the row already says what the value is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Which resolution an edge measures, taken from the engine rather than copied
#: here. It is a fact about ISO 12233 rather than about a widget, and the console
#: report has to say the same word these screens do.
from deltachrome.imaging.edges import sfr_direction

#: Tokens that are acronyms or already correctly cased, spelled as they should
#: appear. Anything not here is title-cased.
_ACRONYMS = {
    "sfr": "SFR", "sfr10": "SFR10", "sfr50": "SFR50", "oecf": "OECF",
    "mtf": "MTF", "mtf50": "MTF50", "rgb": "RGB", "icc": "ICC",
    "ppi": "ppi", "dpi": "dpi", "jpeg": "JPEG", "tiff": "TIFF",
    "iso": "ISO", "fadgi": "FADGI", "roi": "ROI", "lab": "Lab",
    "90th": "90th", "50th": "50th",
}

#: Small words that stay lowercase unless they start or end the name.
_MINOR = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or",
          "the", "to", "vs", "with"}


def metric_title(metric_id: str) -> str:
    """``sfr10_sampling_efficiency`` -> ``SFR10 Sampling Efficiency``."""
    words = str(metric_id or "").replace("_", " ").split()
    if not words:
        return "—"
    out: list[str] = []
    for index, word in enumerate(words):
        lower = word.lower()
        if lower in _ACRONYMS:
            out.append(_ACRONYMS[lower])
        elif lower in _MINOR and 0 < index < len(words) - 1:
            out.append(lower)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


#: What the profiles store, and what the quantity is actually called. The delta
#: is the point: it is the symbol every one of these measurements is written
#: with in the standards, and the one this project is named after.
_UNITS = {
    "dE2000": "ΔE*2000",
    "Delta E2000": "ΔE*2000",
    "dE(a*b*)": "ΔE*ab",
    "dE2000(a*b*)": "ΔE*2000,ab",
    "dL*2000": "ΔL*2000",
    "dL*": "ΔL*",
    "delta L*": "ΔL*",
    "std dev of L*": "σ L*",
    "% (std dev / mean L*)": "% (σ / mean L*)",
    "dL* from family mean": "ΔL* from family mean",
    # Written out. The column is wide enough for the word, and an abbreviation
    # a reader has to expand is a worse unit than a long one they can read.
    "% of half-sampling freq": "% of half-sampling frequency",
    # Lower case, against the usual rule for a proper noun. Every other entry in
    # this column is a lower-case description of a quantity -- ppi, bits, pixels,
    # modulation, % of half-sampling frequency -- and one capitalised name in the
    # middle of them reads as a different kind of thing rather than as the same
    # kind measured against a different reference.
    "% of Nyquist": "% of nyquist",
}


def unit_label(units: str | None) -> str:
    """The unit as it should read. Presentation only -- the record is untouched."""
    text = str(units or "")
    return _UNITS.get(text, text)


def reading_unit(metric: dict) -> str:
    """What one row of a readings table is measured in.

    The record says, in ``reading_units``, wherever a reading is not in the
    aggregate's unit -- lightness uniformity's readings are L* differences under
    a coefficient of variation, geometric distortion's are each rule's ppi under
    a percent. Records written before it could say so are filled in when read.
    """
    own = metric.get("reading_units")
    return unit_label(own if own else metric.get("units"))


#: ``value < 3.5``, ``value >= 16``, ``45 <= value <= 65``.
_ONE_SIDED = re.compile(r"^\s*value\s*(<=|>=|<|>)\s*(-?[\d.]+)\s*$")
_TWO_SIDED = re.compile(
    r"^\s*(-?[\d.]+)\s*(<=|<)\s*value\s*(<=|<)\s*(-?[\d.]+)\s*$"
)


@dataclass(frozen=True)
class Bounds:
    """The aim band a condition describes, for drawing rather than for judging.

    The engine evaluates the condition itself; this only says where to put the
    lines. Nothing here can change a verdict, which is why parsing loosely is
    safe: a condition this cannot read draws no band, and the metric is judged
    exactly as it was.
    """

    low: float | None = None
    high: float | None = None
    #: True when the reading is meant to sit *between* the two, rather than
    #: under a single ceiling. Changes how the band is drawn: an interval is
    #: shaded, a ceiling gets one line with everything above it out of aim.
    interval: bool = False

    def __bool__(self) -> bool:
        return self.low is not None or self.high is not None

    @property
    def lines(self) -> list[tuple[float, str]]:
        out: list[tuple[float, str]] = []
        if self.low is not None:
            out.append((self.low, "min" if self.interval else "limit"))
        if self.high is not None:
            out.append((self.high, "max" if self.interval else "limit"))
        return out


#: ``format in ['TIFF', 'JPEG 2000']`` -- how a permitted-values condition was
#: written before the record stored the list itself (``TIFF, JPEG 2000``).
_LISTED = re.compile(r"^\s*[\w ]+?\s+in\s+\[(.*)\]\s*$")


def condition_label(condition: str | None) -> str:
    """The criterion as it should read beside a number.

    ``value >= 190`` becomes ``>= 190``. The word is the subject of a sentence
    nobody is reading: the row already says which metric this is and the column
    beside it holds the value, so ``value`` is the one word on the row that
    carries nothing.

    A band keeps it -- ``0.8 <= value <= 1.1`` -- because there the word is what
    says the reading sits *between* the two, and ``0.8 <= <= 1.1`` is not
    something a person can read. Presentation only: the string the run was judged
    against is untouched in the record, and this is why the parsing is exact
    rather than a substitution.

    Anything that is not a comparison is a list of permitted values, and goes
    through :func:`permitted_label` so a space named under two spellings is not
    shown as two spaces.
    """
    text = str(condition or "").strip()
    match = _ONE_SIDED.match(text)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    listed = _LISTED.match(text)
    if listed:
        # An older record's spelling. Read as the list it names, which is what a
        # current record stores -- otherwise the comma split below cut it
        # mid-list and lost the closing bracket.
        text = ", ".join(
            item.strip().strip("'\"") for item in listed.group(1).split(",")
        )
    return permitted_label(text)


def parse_condition(condition: str | None) -> Bounds:
    """The band a ``pass_condition`` describes, or an empty one.

    ``pass_condition_by`` -- a banded threshold whose value depends on the
    original's size -- is resolved by the engine before it reaches the record,
    so what arrives here is always a plain comparison or something this returns
    nothing for.
    """
    text = str(condition or "")
    match = _TWO_SIDED.match(text)
    if match:
        return Bounds(float(match.group(1)), float(match.group(4)), interval=True)
    match = _ONE_SIDED.match(text)
    if match:
        operator, number = match.group(1), float(match.group(2))
        if operator in ("<", "<="):
            return Bounds(high=number)
        return Bounds(low=number)
    return Bounds()


#: How an aggregation reads on a chart's own key. The record spells these as the
#: profile does -- ``percentile_90`` -- and a plot drawing that beside a line was
#: showing a variable name where it meant to name a statistic.
_AGGREGATIONS = {
    "mean": "mean",
    "median": "median",
    "percentile_90": "90th percentile",
    "max": "maximum",
    "min": "minimum",
    "max_per_patch": "worst patch",
    "worst": "worst case",
    "worst_case": "worst case",
    "worst_edge": "worst edge",
    "none": "reading",
}


def aggregation_label(aggregation: str | None) -> str:
    """What the metric's own line across a field of readings is called."""
    text = str(aggregation or "").strip()
    return _AGGREGATIONS.get(text, text.replace("_", " ") or "aggregate")


#: Long names for the field positions a chart prints. Keyed on the last token of
#: a region id, so ``module_tl`` and ``tl`` are one position.
_REGION_NAMES = {
    "center": "Center", "c": "Center", "cc": "Center",
    "tl": "Upper Left", "tc": "Upper Center", "tr": "Upper Right",
    "ul": "Upper Left", "uc": "Upper Center", "ur": "Upper Right",
    "cl": "Center Left", "cr": "Center Right",
    "bl": "Lower Left", "bc": "Lower Center", "br": "Lower Right",
    "ll": "Lower Left", "lc": "Lower Center", "lr": "Lower Right",
}

#: The same positions as an axis tick. Built from the long name rather than
#: listed again, so a position cannot be given a code that disagrees with what
#: it is called: the initials of the words, with Center alone as ``C``.
_FIELD_CODES = {
    key: ("C" if name == "Center"
          else "".join(word[0] for word in name.split()))
    for key, name in _REGION_NAMES.items()
}


#: Reading order across the field: top row left to right, then middle, then
#: bottom. Declared once because two screens sort by it -- the edge chips and
#: the registration grid -- and they listed only the corners and the centre, so
#: a chart with a full three-by-three came out UL, UR, LR, LL and then whatever
#: order the detector happened to find the other five in.
_FIELD_ORDER = (
    "tl", "ul", "tc", "uc", "tr", "ur",
    "cl", "center", "c", "cc", "cr",
    "bl", "ll", "bc", "lc", "br", "lr",
)


def field_order(region: str) -> tuple[int, str]:
    """A sort key putting field positions in the order a person reads them."""
    tail = str(region or "").rsplit("_", 1)[-1].lower()
    index = (_FIELD_ORDER.index(tail) if tail in _FIELD_ORDER
             else len(_FIELD_ORDER))
    return (index, str(region or ""))


def region_name(region: str) -> str:
    """``module_tl`` -> ``Upper Left``. Unknown tails are title-cased."""
    tail = str(region or "").rsplit("_", 1)[-1].lower()
    return _REGION_NAMES.get(tail, str(region or "").replace("_", " ").title())


def field_code(region: str) -> str:
    """``module_tl`` -> ``UL``. One or two letters, for an axis tick.

    A chart with nine field positions puts nine of these side by side under a
    plot 200 px wide, so anything longer overlaps its neighbours -- which is
    what ``Module Cc`` did. A position this does not know falls back to the
    first two characters of its own tail rather than to a long name, because an
    unknown position on a crowded axis is still a crowded axis.
    """
    tail = str(region or "").rsplit("_", 1)[-1].lower()
    if tail in _FIELD_CODES:
        return _FIELD_CODES[tail]
    return tail[:2].upper() or "—"


def sfr_direction_label(orientation: str | None) -> str:
    """The same, Title Case, for a chip or a table cell."""
    return sfr_direction(orientation).title()


def sfr_direction_hint(orientation: str | None) -> str:
    """Why the word on screen is not the word in the record."""
    direction = sfr_direction(orientation)
    if not direction:
        return ""
    return f"Measured across a near-{str(orientation).lower()} edge."


def permitted_label(condition: str | None) -> str:
    """A list of permitted values with its alternative spellings collapsed.

    A conformance table lists every spelling a file's ICC description might
    carry -- FADGI names both ``ProPhoto`` and ``ProPhoto RGB``, both
    ``ECIRGB_v2`` and ``eciRGB v2`` -- because the metric matches a description
    against the list and one profile in the wild is named each way. That is the
    right list to *match* against and the wrong list to *show*: a criterion
    naming the same space twice reads as two permitted spaces.

    Collapsed by the metric's own rule, which is
    :func:`deltachrome.metrics.conformance.permitted_key` plus containment
    either way -- so two entries are one value here exactly when a capture
    matching one would match the other. The spelling kept is the first, which in
    the shipped profiles is the standard's own; the alternatives are still in
    the record and on the row's hover.
    """
    from deltachrome.metrics.conformance import permitted_key

    text = str(condition or "").strip()
    if not text or "," not in text or any(op in text for op in ("<", ">", "=")):
        return text
    kept: list[str] = []
    keys: list[str] = []
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        key = permitted_key(entry)
        if any(key == seen or key in seen or seen in key for seen in keys):
            continue
        kept.append(entry)
        keys.append(key)
    return ", ".join(kept)


#: How a record spells the reference bases the engine can use, and what a person
#: calls them. Anything else a future engine writes is passed through as it
#: stands rather than flattened into one of these -- a basis nobody here has
#: heard of is still information, and guessing at it would be worse.
BASIS_LABELS = {
    "per_unit_measurement": "Measured",
    #: A mean of many measured units, shown to sit inside its model's own
    #: population. Between the other two and worth separating from both: it
    #: earns no conformance claim, and it is not a printed aim either.
    "population_mean": "Batch Mean",
    "manufacturer_nominal": "Nominal",
}


def basis_label(basis: str | None) -> str:
    """``manufacturer_nominal`` -> ``Nominal``; empty where none is recorded."""
    text = str(basis or "")
    return BASIS_LABELS.get(text, text.replace("_", " ").capitalize())


_CHANNELS = {"R": "red", "G": "green", "B": "blue", "K": "gray"}

#: A sampling flag as a reader should see it. The record keeps the engine's
#: spelling -- ``gamut_clipped(R)`` -- and a table of them read as code.
_FLAG = re.compile(r"^(?P<name>[a-z_]+)(?:\((?P<arg>[^)]*)\))?$")


def flag_labels(flags) -> list[str]:
    """``["clipped_low(4.1%)", "gamut_clipped(RB)"]`` -> ``["Out of gamut (red and blue clipped)"]``.

    A shadow flag is dropped where a channel flag is raised beside it: the
    engine raises ``clipped_low`` whenever a channel sits at the floor, and the
    channel flag is the more exact account of the same pixels, so showing both
    would report a shadow problem that is not there. An unknown flag is shown
    with its underscores as spaces rather than dropped.
    """
    parsed = [(m.group("name"), m.group("arg") or "") if (m := _FLAG.match(str(f)))
              else (str(f), "") for f in flags or []]
    names = {name for name, _ in parsed}
    out = []
    for name, arg in parsed:
        if name == "clipped_low" and names & {"gamut_clipped", "channel_floor"}:
            continue
        channels = [_CHANNELS.get(c, c) for c in arg]
        which = (" and ".join(channels) if len(channels) < 3
                 else ", ".join(channels[:-1]) + " and " + channels[-1])
        if name == "gamut_clipped":
            out.append(f"Out of gamut ({which} clipped)")
        elif name == "channel_floor":
            out.append(f"No signal in {which}")
        elif name == "clipped_high":
            out.append(f"Highlights clipped ({arg})" if arg else "Highlights clipped")
        elif name == "clipped_low":
            out.append(f"Shadows clipped ({arg})" if arg else "Shadows clipped")
        elif name == "mean_median_divergence":
            out.append("Possible dust or glare")
        elif name == "roi_clipped_to_image":
            out.append("Partly outside the frame")
        else:
            out.append(name.replace("_", " ").capitalize() + (f" ({arg})" if arg else ""))
    return out


#: ``r3c4`` -- a flat-field spot, by its zero-based place in the sampling grid.
_SPOT = re.compile(r"^r(\d+)c(\d+)$")


def reading_names(readings, edges=None) -> list[str]:
    """What each of a metric's readings is called, in the order given.

    A patch keeps the label the chart prints on it, and a pair of patches its
    two labels -- ``E5``, ``E5-F5`` -- those being what a person can find on the
    card. Everything else is an id a definition or a plugin made up, and is
    named by where it was taken: an edge by its field position and the
    direction it measured, a registration reading by its position and axis
    (with the channel pair it is the worst of), a flat-field spot by its row and
    column, a printed rule by its own region's words.

    ``edges`` maps an edge id to its record. Several edges at one position
    measuring one direction are numbered, since a name two readings share
    identifies neither.
    """
    edges = edges or {}
    names = [_reading_name(reading, edges) for reading in readings]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            name = f"{name} {seen[name]}"
        out.append(name)
    return out


def _reading_name(reading, edges) -> str:
    reading_id = str(reading.get("patch_id") or "")
    role = str(reading.get("role") or "")
    edge = edges.get(reading_id)
    if edge is not None:
        region = getattr(edge, "region", None) or role
        return f"{region_name(region)} {sfr_direction_label(getattr(edge, 'orientation', None))}"
    if ":" in reading_id:
        region, axis = reading_id.rsplit(":", 1)
        direction = {"x": "Horizontal", "y": "Vertical"}.get(axis.lower(), axis)
        name = f"{region_name(region)} {direction}"
        pair = role.rsplit(" ", 1)[-1] if role.startswith("worst pair") else ""
        return f"{name} ({pair})" if pair else name
    spot = _SPOT.match(reading_id)
    if spot:
        return f"Row {int(spot.group(1)) + 1}, Column {int(spot.group(2)) + 1}"
    if "_" in reading_id:
        return metric_title(reading_id)
    return reading_id
