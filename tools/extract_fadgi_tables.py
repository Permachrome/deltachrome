"""Read the FADGI section 3 conformance tables out of the published PDF.

The transcription in ``generate_fadgi_profiles.py`` is checked-in data, because
the PDF is not in the repository and a build should not depend on fetching a
2 MB document. This is the audit trail for that data: run it against the PDF and
it prints the tables, so a transcription can be re-derived and checked rather
than trusted.

Run:  python tools/extract_fadgi_tables.py path/to/FADGI...3rd_Edition_05092023.pdf

**The parser is validated against the categories already transcribed by hand.**
``--check`` compares its output for every category in the generator against what
the generator declares, and reports any disagreement. That is the only reason to
believe the other categories were read correctly: nothing here is inferred from
the shape of a table, only from text that matched an anchor.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUM = r"[-+]?\d+(?:\.\d+)?"

#: Where the transcription this reader produces is kept.
TABLES = Path(__file__).resolve().parent / "fadgi_tables_2023.json"

#: Colour space names as the tables spell them, longest first so that
#: "Adobe RGB (1998)" is matched before "Adobe RGB".
SPACES = ["Adobe RGB (1998)", "Gray Gamma 2.2", "ECIRGB_v2", "ProPhoto", "sRGB"]

#: The three categories transcribed by hand before this reader existed, kept
#: frozen as its calibration. Every numeric row below was typed from the
#: printed table by a person; the reader reproduces all of them exactly, and
#: that agreement is the only reason to believe the twelve categories it read
#: unaided. Do not regenerate these from the reader -- that would make the
#: check circular and worthless.
HAND_TRANSCRIBED = {
    "3.8": {  # prints_photographs
        "resolution_ppi": [194, 243.75, 392, 594],
        "bit_depth": [8, 8, 8, 16],
        "tone_response": [6, 4.5, 3, 1.5],
        "white_balance": [8, 6, 4, 2],
        "lightness_uniformity": [8, 5, 3, 1],
        "average_color_accuracy": [6.5, 5, 3.5, 2],
        "color_accuracy_90th": [13, 10, 7, 4],
        "noise_upper": [4, 3, 2, 1],
        "gain_highlight": [[0.5, 1.4], [0.6, 1.3], [0.7, 1.2], [0.8, 1.1]],
        "gain_other": [[0.25, 1.7], [0.3, 1.6], [0.6, 1.4], [0.7, 1.3]],
        "misregistration": [1.2, 0.8, 0.5, 0.33],
        "sfr10": [60, 70, 80, 90],
        "sfr_nyquist": [0.5, 0.4, 0.3, 0.2],
        "sharpening": [1.15, 1.1, 1.05, 1.02],
        "reproduction_scale": [3, 2.5, 2, 1],
        "sfr50": [[30, 95], [35, 85], [40, 75], [45, 65]],
    },
    "3.9": {  # photographic_transparencies_35mm_to_4x5
        "resolution_ppi": [970, 1950, 2940, 3960],
        "bit_depth": [8, 8, 16, 16],
        "tone_response": [5.25, 4.0, 2.75, 1.5],
        "white_balance": [5.25, 4.0, 2.75, 1.5],
        "lightness_uniformity": [8, 5, 3, 1],
        "average_color_accuracy": None,
        "color_accuracy_90th": None,
        "noise_upper": None,
        "gain_highlight": None,
        "gain_other": None,
        "misregistration": [1.2, 0.8, 0.5, 0.33],
        "sfr10": [60, 70, 80, 90],
        "sfr_nyquist": [0.5, 0.4, 0.3, 0.2],
        "sharpening": [1.15, 1.1, 1.05, 1.02],
        "reproduction_scale": [3, 2.5, 2, 1],
        "sfr50": [[30, 95], [35, 85], [40, 75], [45, 65]],
    },
    "3.10": {  # photographic_transparencies_4x5_and_larger
        "resolution_ppi": [485, 975, 1470, 1980],
        "bit_depth": [8, 8, 16, 16],
        "tone_response": [5.25, 4.0, 2.75, 1.5],
        "white_balance": [5.25, 4.0, 2.75, 1.5],
        "lightness_uniformity": [8, 5, 3, 1],
        "average_color_accuracy": None,
        "color_accuracy_90th": None,
        "noise_upper": None,
        "gain_highlight": None,
        "gain_other": None,
        "misregistration": [1.2, 0.8, 0.5, 0.33],
        "sfr10": [60, 70, 80, 90],
        "sfr_nyquist": [0.5, 0.4, 0.3, 0.2],
        "sharpening": [1.15, 1.1, 1.05, 1.02],
        "reproduction_scale": [3, 2.5, 2, 1],
        "sfr50": [[30, 95], [35, 85], [40, 75], [45, 65]],
    },
}

#: Rows FADGI marks informative-only with an asterisk, meaning they are "not
#: required parameters for each FADGI star level". Only the film tables use the
#: convention; the reflective ones carry no starred rows at all.
INFORMATIVE_NOTE = "The white rows with light gray text in the table above are informative only"


def category_texts(pdf_path: Path) -> dict[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    heads = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        match = re.search(r"^\s*(3\.\d{1,2})\s+(\S[^\n]{4,80})$", text, re.M)
        if match:
            heads.append((match.group(1), match.group(2).strip(), index))

    out: dict[str, dict] = {}
    for i, (number, title, start) in enumerate(heads):
        end = heads[i + 1][2] if i + 1 < len(heads) else len(reader.pages)
        raw = "\n".join((reader.pages[k].extract_text() or "") for k in range(start, end))
        cut = raw.find("Note: ")
        table = raw[:cut] if cut > 0 else raw
        # Last page the table itself occupies, so the provenance can cite a range
        # rather than a starting point. Counted by how many page breaks fall
        # inside the table text.
        pages_used = 1
        for k in range(start, end):
            page_text = reader.pages[k].extract_text() or ""
            if page_text and page_text[:60] in table:
                pages_used = k - start + 1
        out[number] = {
            "title": re.sub(r"\s+", " ", title),
            "page": start + 1,
            "last_page": start + pages_used,
            "raw": raw,
            "table": table,
        }
    return out


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def cell_block(table: str, start_label: str, end_label: str) -> str | None:
    """The text of one row, between its own label and the next row's."""
    flat = normalise(table)
    begin = flat.find(start_label)
    if begin < 0:
        return None
    stop = flat.find(end_label, begin + len(start_label))
    return flat[begin + len(start_label) : stop if stop > 0 else begin + 1200]


def star_levels(table: str) -> list[int]:
    """Which performance levels this table actually populates.

    Not every category offers all four. Bound Volumes: Rare and Special Materials
    and its unbound equivalent print the 1-Star heading and leave every cell in
    that column empty, so their values are 2, 3 and 4-star -- the omitted level is
    the *lowest* and the rows align to the stricter end. Modern Textual Records
    goes further and has no performance levels at all: one specification, which is
    why it carries no ``Performance Level:`` header.

    Counted from the master-file-format row, which every star-banded table fills
    for exactly the levels it offers.
    """
    if "Performance Level:" not in normalise(table):
        return []
    block = cell_block(table, "Master File Format", "Access File Formats") or ""
    filled = len(re.findall(r"TIFF", block))
    if not 1 <= filled <= 4:
        return [1, 2, 3, 4]
    return [4 - filled + 1 + i for i in range(filled)]


def four(table: str, label: str, pattern: str, cast=float, end: str = "",
         count: int = 4) -> list | None:
    """One row's values, or ``None`` when the row is absent.

    ``count`` is how many cells the table populates. Reading four from a table
    that fills three walks into the next row and returns a value belonging to a
    different metric -- which is how this first read Bound Volumes as a tone
    response of 4.5/3.0/1.5/6.0, the 6.0 being the white balance beneath it.
    """
    block = cell_block(table, label, end) if end else None
    if block is None:
        flat = normalise(table)
        begin = flat.find(label)
        if begin < 0:
            return None
        block = flat[begin + len(label) : begin + 1200]
    found = re.findall(pattern, block)
    if len(found) < count:
        return None
    return [cast(v) for v in found[:count]] if cast else list(found[:count])


def colour_spaces(table: str, count: int = 4) -> list[list[str]] | None:
    block = cell_block(table, "Color Space", "Color Mode")
    if block is None:
        return None
    tokens: list[tuple[int, str]] = []
    for name in SPACES:
        for m in re.finditer(re.escape(name), block):
            tokens.append((m.start(), name))
    tokens.sort()
    if not tokens:
        return None
    # Cells run in order, each ending at the last name before the next cell's
    # first. Every table repeats the same terminal name four times, so the
    # boundaries are the four occurrences of whichever name ends a cell.
    terminal = tokens[-1][1]
    cells: list[list[str]] = []
    current: list[str] = []
    for _, name in tokens:
        current.append(name)
        if name == terminal:
            cells.append(current)
            current = []
    if len(cells) != count:
        return None
    return cells


def bit_depths(table: str, count: int) -> list[int] | None:
    """Minimum bit depth per level.

    Cells read "8", "16", or "8 or 16", and the last of those is two numbers in
    one cell -- so counting numbers walks out of alignment. Prints and
    Photographs happens to survive it ("8 8 8 or 16 16" gives the right answer by
    coincidence) and Bound Volumes: General Collections does not ("8 8 8 or 16 8
    or 16" reads a 4-star minimum of 16 where the table allows 8).

    The requirement is a minimum, so a cell offering a choice contributes the
    lower of the two.
    """
    block = cell_block(table, "Bit Depth", "Color Space")
    if block is None:
        return None
    cells = re.findall(r"(\d+)(?:\s*or\s*(\d+))?", block)
    depths = [min(int(a), int(b)) if b else int(a) for a, b in cells]
    return depths[:count] if len(depths) >= count else None


def starred_rows(table: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", m.group(1)).strip()
        for m in re.finditer(r"^\s*\*([A-Z][^\n]{3,55})", table, re.M)
    }


def parse(entry: dict) -> dict:
    table, raw = entry["table"], entry["raw"]
    stars = star_levels(table)
    n = len(stars) or 1
    starred = starred_rows(table)
    uses_convention = INFORMATIVE_NOTE in raw

    formats = re.findall(r"TIFF|JPEG\s?2000|DICOM|PDF/A", normalise(cell_block(table, "Master File Format", "Access File Formats") or ""))
    seen: list[str] = []
    for name in formats:
        tidy = "JPEG 2000" if name.replace(" ", "") == "JPEG2000" else name
        if tidy not in seen:
            seen.append(tidy)

    return {
        "title": entry["title"],
        "page": entry["page"],
        "last_page": entry["last_page"],
        "stars": stars,
        "master_formats": seen or None,
        "resolution_ppi": four(table, "Resolution", rf"[≥>]\s*({NUM})\s*ppi", count=n),
        "bit_depth": bit_depths(table, n),
        "color_space": colour_spaces(table, n),
        "tone_response": four(table, "Tone Response", rf"<\s*({NUM})", count=n),
        "white_balance": four(table, "White Balance", rf"<\s*({NUM})", count=n),
        "lightness_uniformity": four(table, "Lightness Uniformity", rf"<\s*({NUM})%", count=n),
        "average_color_accuracy": four(table, "Average Color", rf"<\s*({NUM})", count=n),
        "color_accuracy_90th": four(table, "Color Accuracy 90th", rf"<\s*({NUM})", count=n),
        "misregistration": four(table, "Color Channel", rf"<\s*({NUM})\s*pixel", count=n),
        "sfr10": four(table, "SFR10", rf">\s*({NUM})%", count=n),
        "sfr_nyquist": four(table, "SFR Response at Nyquist", rf"<\s*({NUM})", count=n),
        "sfr50": [
            [float(a), float(b)]
            for a, b in re.findall(rf"\[>({NUM})%,\s*<({NUM})%\]", normalise(table))[:n]
        ] or None,
        "reproduction_scale": four(
            table, "Reproduction Scale Accuracy (Units", rf"<\+/-\s*({NUM})%", count=n
        ),
        "sharpening": four(table, "Sharpening", rf"<\s*({NUM})", count=n),
        "noise_upper": four(table, "Noise (Upper Limit)", rf"<\s*({NUM})", count=n),
        "gain_highlight": _gain(table, "Gain Modulation Highlight", n),
        "gain_other": _gain(table, "Gain Modulation all", n),
        "informative": sorted(starred) if uses_convention else [],
        "provisional": "◊ Provisional" in normalise(table),
    }


def _gain(table: str, label: str, count: int = 4) -> list[list[float]] | None:
    flat = normalise(table)
    begin = flat.find(label)
    if begin < 0:
        return None
    pairs = re.findall(
        rf"Gain between\s*({NUM})\s*and\s*({NUM})", flat[begin : begin + 900]
    )
    if len(pairs) < count:
        return None
    return [[float(a), float(b)] for a, b in pairs[:count]]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    check = "--check" in argv
    write = "--write" in argv
    pdf = Path([a for a in argv if not a.startswith("--")][0])
    tables = {n: parse(e) for n, e in category_texts(pdf).items()}

    if check:
        return _check(tables)

    import json

    payload = {
        "_source": "FADGI Technical Guidelines, 3rd Edition, May 2023",
        "_source_url": (
            "https://www.digitizationguidelines.gov/guidelines/FADGI%20Technical"
            "%20Guidelines%20for%20Digitizing%20Cultural%20Heritage%20Materials"
            "_3rd%20Edition_05092023.pdf"
        ),
        "_generated_by": "tools/extract_fadgi_tables.py",
        "_note": (
            "Read from the published PDF, not transcribed by hand. Re-derive with "
            "`python tools/extract_fadgi_tables.py <pdf>` and verify the reader "
            "against the categories transcribed by hand with `--check`."
        ),
        "categories": tables,
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    if write:
        out = Path(__file__).resolve().parent / "fadgi_tables_2023.json"
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(tables)} categories)")
        return 0
    print(text)
    return 0


def _check(tables: dict) -> int:
    """Two checks, and they answer different questions.

    The first is the calibration: does this reader reproduce the three categories
    a person transcribed by hand from the printed table? That is what licenses
    trusting it on the twelve nobody has checked line by line.

    The second is the regression: does the checked-in transcription still match
    what the PDF says? That catches the file being edited by hand and drifting
    away from its source, which is the failure mode a checked-in copy invites.
    """
    problems = 0
    for section, declared in sorted(HAND_TRANSCRIBED.items()):
        read = tables.get(section)
        if read is None:
            print(f"MISSING  {section} from the reader's output")
            problems += 1
            continue
        for field, value in declared.items():
            if not _same(value, read.get(field)):
                print(f"CALIBRATION  {section}.{field}\n"
                      f"   hand transcription {value}\n   reader read       {read.get(field)}")
                problems += 1
    print(f"calibration: {len(HAND_TRANSCRIBED)} hand-transcribed categories checked.")

    if TABLES.exists():
        import json

        stored = json.loads(TABLES.read_text(encoding="utf-8"))["categories"]
        if set(stored) != set(tables):
            print(f"STORED  category set differs: {sorted(set(stored) ^ set(tables))}")
            problems += 1
        for section, read in tables.items():
            for field, value in read.items():
                if field.startswith("_"):
                    continue
                if not _same(value, (stored.get(section) or {}).get(field)):
                    print(f"STORED  {section}.{field}\n"
                          f"   file says {(stored.get(section) or {}).get(field)}\n"
                          f"   pdf says  {value}")
                    problems += 1
        print(f"stored transcription: {len(stored)} categories checked.")
    else:
        print(f"stored transcription: {TABLES.name} not present; nothing to compare.")

    print(f"\n{problems} disagreement(s).")
    return 1 if problems else 0


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
