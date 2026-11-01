"""Every string the application shows a person, as a spreadsheet to edit.

The prose in this project is load-bearing rather than decorative -- a metric's
explanation, a refusal that has to read as a design rather than a crash, a
warning that says why a number cannot be quoted -- and none of it has ever been
edited as prose. It is scattered across sixty modules as f-strings, and the only
way anyone has read it is one file at a time, in the order the code happens to
run.

So this pulls it out. The point is the **round trip**: export, edit the wording
somewhere wording can be edited, hand it back, and apply it. That is why every
row carries an id, its ``file:line``, and its exact current text -- an edit is
matched against the current text, so a row that has drifted since the export is
refused rather than applied to whatever now sits at that line.

Run from the repository root::

    python tools/export_copy.py                     # writes deltachrome-copy.xlsx
    python tools/export_copy.py --out /tmp/copy.xlsx
    python tools/export_copy.py --format csv        # no openpyxl needed
    python tools/export_copy.py --propose new.json  # pre-fill the wording column
    python tools/export_copy.py --apply sheet.xlsx --dry-run
    python tools/export_copy.py --apply sheet.xlsx

**Applying is guarded in four places, and each guard is a failure that would
otherwise be silent.** A row whose current text is no longer in the source is
refused rather than written to whatever now sits at that line. A replacement
that *invents* a placeholder is refused, because the expression is evaluated in
whatever scope the message sits in and one that is not in scope there is a
NameError raised on somebody's run, from a cell in a spreadsheet -- where a
replacement that *drops* one is applied and reported, since removing a value is
most of what shortening a message is. A rewrite that would not parse is refused
per file. And a string written in two places is rewritten in both, because rows
are de-duplicated by text and changing one of them would leave the application
saying two things.

**Extraction is by AST, not by grep.** A regular expression over source finds
docstrings, log lines and half of an f-string; walking the tree finds the
argument of the call that actually shows the string, reconstructs implicit
concatenation, and renders an f-string's holes as ``{expression}`` so an editor
can see what is interpolated and keep it. Roughly two thirds of these strings
have at least one.

**What counts as user-facing is a judgement and it is written down here** rather
than left to the reader of the output: ``SHOWN`` is the errors a person is
handed directly, ``INNER`` the ones that only ever appear quoted inside a larger
message and therefore have to read as a clause. Everything else raised in the
engine is a programming error and is not copy.

The workbook is **not committed**. It is generated, it is binary, and the whole
point is that it is regenerated whenever the copy has moved.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"

#: Errors a person is shown directly, as a dialog or on the console.
SHOWN = {
    "ProfileError", "CapabilityError", "TargetError", "CGATSParseError",
    "MissingReferenceError", "GeometryError", "ImageError",
    "MissingICCProfileError", "UnsupportedICCProfileError",
    "DuplicateSampleError", "ReferenceError", "DeltachromeError",
}
#: Errors that are caught and quoted inside a warning. Their text has to read as
#: a clause rather than as a sentence, which is why they are kept apart.
INNER = {"SFRError", "RulerError", "DetectionRefused"}

#: Which screen a module is. Written out rather than derived, because "where it
#: appears" is the column this exists for and a file path is not an answer to it.
SURFACE = {
    "deltachrome_ui/explanations.py": "Documentation source (not shown in the app)",
    "deltachrome_ui/app.py": "Application shell",
    "deltachrome_ui/workers.py": "Failure dialogs",
    "deltachrome_ui/result_view.py": "Report",
    "deltachrome_ui/capture_dock.py": "Report — capture panel",
    "deltachrome_ui/detail_view.py": "Report — readings and provenance",
    "deltachrome_ui/charts.py": "Report — charts",
    "deltachrome_ui/metric_charts.py": "Report — charts",
    "deltachrome_ui/nav.py": "Report — metric list",
    "deltachrome_ui/session_view.py": "Session report",
    "deltachrome_ui/library_view.py": "Run library",
    "deltachrome_ui/run_table.py": "Run library — table",
    "deltachrome_ui/placement_view.py": "Placement screen",
    "deltachrome_ui/choices.py": "Placement screen — setup panel",
    "deltachrome_ui/canvas.py": "Placement screen — canvas",
    "deltachrome_ui/options_view.py": "Placement screen — run options",
    "deltachrome_ui/profile_picker.py": "Placement and report — profile",
    "deltachrome_ui/status.py": "Verdicts (everywhere)",
    "deltachrome_ui/tables.py": "Tables (everywhere)",
    "deltachrome_ui/widgets.py": "Shared widgets",
    "deltachrome_ui/library.py": "Run library",
    "deltachrome_ui/command.py": "Copy Command",
}

ENGINE_SURFACE = {
    "pipeline": "Report — warnings (analysis)",
    "metrics": "Report — warnings (metrics)",
    "imaging": "Report — warnings (imaging)",
    "reference": "Report — warnings (reference data)",
    "targets": "Report — warnings (target definition)",
    "profiles": "Report — warnings (profile)",
    "results": "Report — warnings (record)",
    "session": "Session report — warnings",
    "reporting": "Console report",
    "inputs": "Command line",
    "cli": "Command line",
    "errors": "Refusals",
}

#: Reading order: what a person meets deliberately first, what the engine says
#: when something has gone wrong last.
ORDER = [
    "Documentation", "Status hint", "Tooltip", "Chip", "Warning block",
    "Dialog", "Failure dialog", "Run warning", "Refusal",
    "Refusal (quoted in a warning)", "Command line",
]
CODE = {
    "Documentation": "DOC", "Status hint": "STAT", "Tooltip": "TIP", "Chip": "CHIP",
    "Warning block": "BLOCK", "Dialog": "DLG", "Failure dialog": "FAIL",
    "Run warning": "WARN", "Refusal": "REF",
    "Refusal (quoted in a warning)": "REFQ", "Command line": "CLI",
}

#: Shorter than this and it is a label, a key or a unit rather than prose.
MIN_LENGTH = 12


def literal(node: ast.AST) -> str | None:
    """A string constant, an f-string, or a concatenation of those.

    An f-string's holes come back as ``{expression}``. That is the whole reason
    this is an AST walk: the placeholder is the part an editor must not break,
    and it has to be visible in the exported text to be kept.

    **A hole carries its format spec and its conversion**, so ``{value:.3f}``
    comes back as ``{value:.3f}`` rather than as ``{value}``. That is not
    presentation: the export exists to be applied back, and a hole rendered
    without its spec would be written back without it -- turning every ``0.123``
    in the application into ``0.12300000000000001``. 309 of the 1,555 holes in
    this source carry one, over a fifth of them, and 106 are in the console
    report alone.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = ast.unparse(value.value)
                if value.conversion is not None and value.conversion != -1:
                    inner += "!" + chr(value.conversion)
                if value.format_spec is not None:
                    # The spec is itself an f-string, because it may hold a hole
                    # of its own -- {value:.{places}f} is legal and this project
                    # does not use it, which is exactly why it must not be
                    # silently dropped if somebody writes one.
                    spec = literal(value.format_spec)
                    if spec is None:
                        return None
                    inner += ":" + spec
                parts.append("{" + inner + "}")
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left), literal(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def is_prose(text: str) -> bool:
    """Whether a module-level constant is copy rather than code.

    A style sheet and a file dialog's filter are both long ALL_CAPS strings and
    neither is anything a person reads. Checked by shape rather than by name, so
    the next one is caught too.
    """
    if len(text) <= 16 or " " not in text:
        return False
    return not any(mark in text for mark in ("{", ";;", "*.", "  "))


def owners_of(tree: ast.AST) -> dict[int, str]:
    """Map each line to the innermost function or class containing it."""
    owner: dict[int, tuple[int, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                current = owner.get(line)
                if current is None or node.lineno > current[0]:
                    owner[line] = (node.lineno, node.name)
    return {line: name for line, (_, name) in owner.items()}


class Collector:
    """Rows, de-duplicated by (kind, text).

    The same sentence written twice is one thing to edit. Which of the two
    ``file:line`` values is recorded does not matter, because applying an edit
    matches on the text.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self._seen: set[tuple[str, str]] = set()

    def add(self, kind: str, rel: str, line: int, where: str,
            context: str, text: str | None) -> None:
        text = (text or "").strip()
        if len(text) < MIN_LENGTH:
            return
        if (kind, text) in self._seen:
            return
        self._seen.add((kind, text))
        self.rows.append({
            "kind": kind, "file": rel, "line": line, "where": where,
            "context": context or "", "text": text,
        })


def engine_where(rel: str) -> str:
    return ENGINE_SURFACE.get(rel.split("/")[1].removesuffix(".py"),
                              "Report — warnings")


def collect() -> list[dict[str, object]]:
    found = Collector()
    for path in sorted(SOURCE.rglob("*.py")):
        rel = path.relative_to(SOURCE).as_posix()
        if "__pycache__" in rel:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners = owners_of(tree)
        ui = rel.startswith("deltachrome_ui/")
        where = SURFACE.get(rel, "Frontend" if ui else engine_where(rel))

        # A file of nothing but copy, in (title, body) pairs, read structurally
        # rather than by walking calls. Nothing in the application shows it any
        # more -- it is the source for documentation written apart from the
        # application -- and it is still exported, because it is still prose
        # somebody has to be able to edit.
        if rel.endswith("explanations.py"):
            for node in tree.body:
                if not (isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        and isinstance(node.value, ast.Tuple)
                        and len(node.value.elts) == 2):
                    continue
                for part, element in zip(("heading", "body"), node.value.elts):
                    found.add("Documentation", rel, node.lineno, where,
                              f"{node.target.id} · {part}", literal(element))
            continue

        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        text = literal(node.value)
                        if text and is_prose(text):
                            found.add("Chip" if ui else "Run warning", rel,
                                      node.lineno, where, target.id, text)

        for node in ast.walk(tree):
            owner = owners.get(getattr(node, "lineno", 0), "")

            if rel.endswith("status.py"):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "Status"):
                    args = [literal(a) for a in node.args]
                    if len(args) > 2 and args[2]:
                        found.add("Status hint", rel, node.lineno, where,
                                  f"{args[0]} · tooltip", args[2])
                continue

            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")

            if name == "setToolTip" and node.args:
                found.add("Tooltip", rel, node.lineno, where, owner,
                          literal(node.args[0]))

            elif name == "IconButton" and len(node.args) > 1:
                found.add("Tooltip", rel, node.lineno, where,
                          f"{owner} · {literal(node.args[0])} icon",
                          literal(node.args[1]))

            elif name == "set_state" and node.args:
                found.add("Chip", rel, node.lineno, where, owner,
                          literal(node.args[0]))
                for keyword in node.keywords:
                    if keyword.arg == "detail":
                        found.add("Tooltip", rel, node.lineno, where,
                                  f"{owner} · chip detail", literal(keyword.value))

            elif name in {"Callout", "caveat"} and node.args:
                found.add("Warning block", rel, node.lineno, where, owner,
                          literal(node.args[0]))

            elif name in {"setText", "setInformativeText", "setWindowTitle"} \
                    and node.args and isinstance(func, ast.Attribute) \
                    and "box" in ast.unparse(func.value).lower():
                found.add("Dialog", rel, node.lineno, where,
                          f"{owner} · {name}", literal(node.args[0]))

            elif name in {"information", "warning", "critical", "question"} \
                    and isinstance(func, ast.Attribute) \
                    and ast.unparse(func.value).endswith("QMessageBox"):
                for index, arg in enumerate(node.args[1:], start=1):
                    part = "title" if index == 1 else "body"
                    found.add("Dialog", rel, node.lineno, where,
                              f"{owner} · {part}", literal(arg))

            elif name == "LoadFailure":
                for index, arg in enumerate(node.args):
                    part = "headline" if index == 0 else "detail"
                    found.add("Failure dialog", rel, node.lineno, where,
                              f"{owner} · {part}", literal(arg))

            elif name in {"append", "extend"} and node.args and not ui \
                    and isinstance(func, ast.Attribute) \
                    and re.search(r"warn|note|reason|caveat",
                                  ast.unparse(func.value), re.I):
                found.add("Run warning", rel, node.lineno, engine_where(rel),
                          owner, literal(node.args[0]))

            elif name in SHOWN and node.args and not ui:
                found.add("Refusal", rel, node.lineno, "Refusal dialog / console",
                          f"{owner} · {name}", literal(node.args[0]))

            elif name in INNER and node.args and not ui:
                found.add("Refusal (quoted in a warning)", rel, node.lineno,
                          engine_where(rel), f"{owner} · {name}",
                          literal(node.args[0]))

            elif name == "SystemExit" and node.args and rel.endswith("cli.py"):
                found.add("Command line", rel, node.lineno, "Command line",
                          owner, literal(node.args[0]))

    rows = found.rows
    rows.sort(key=lambda r: (ORDER.index(str(r["kind"])), r["where"],
                             r["file"], r["line"]))
    counters: dict[str, int] = {}
    for row in rows:
        kind = str(row["kind"])
        counters[kind] = counters.get(kind, 0) + 1
        row["id"] = f"{CODE[kind]}-{counters[kind]:03d}"
    return rows


HEADERS = [
    ("ID", 11), ("Kind", 22), ("Where it appears", 30), ("What it belongs to", 30),
    ("Current text", 78), ("New text", 78), ("Notes", 26),
    ("Body (proposed)", 52), ("Argument", 40),
    ("Placeholders", 13), ("Characters", 11), ("Source", 34),
]

LEGEND = [
    ("Deltachrome — user-facing copy", 16, True),
    ("Every piece of prose the application shows, extracted from the source by "
     "tools/export_copy.py.", 11, False),
    ("", 11, False),
    ("How to use this", 13, True),
    ("Edit the 'New text' column only — the shaded one. Leave 'Current text' "
     "exactly as it is: an edit is matched against it, and a row whose current "
     "text no longer matches the source is refused rather than guessed at.",
     11, False),
    ("Leave 'New text' blank to keep the current wording. Only filled-in rows "
     "change anything.", 11, False),
    ("'Notes' is yours — anything about a row that is not the replacement "
     "wording itself.", 11, False),
    ("", 11, False),
    ("Placeholders — the one thing that will break", 13, True),
    ("Text containing {braces} has a value substituted into it when the message "
     "is shown: {target.display_name} becomes the chart's name, {path.name} a "
     "filename. Keep every placeholder, spelled exactly. Moving one within the "
     "sentence is fine.", 11, False),
    ("The 'Placeholders' column marks the rows that have them.", 11, False),
    ("", 11, False),
    ("Example of a filled-in row", 13, True),
    ("", 11, False),
]

EXAMPLE = [
    ("ID", "WARN-012"),
    ("Kind", "Run warning"),
    ("Where it appears", "Report — warnings (analysis)"),
    ("Current text", "The placed quadrilateral's aspect ratio differs from the "
                     "chart's physical proportions by {deviation}."),
    ("New text", "This chart is placed {deviation} off its true shape."),
    ("Notes", "Shorter; kept the placeholder."),
]


def write_csv(rows, out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([title for title, _ in HEADERS])
        for row in rows:
            writer.writerow([
                row["id"], row["kind"], row["where"], row["context"],
                row["text"], row.get("proposed", ""), "",
                row.get("body", ""), row.get("argument", ""),
                "Yes" if "{" in str(row["text"]) else "",
                len(str(row["text"])), f"{row['file']}:{row['line']}",
            ])


def write_xlsx(rows, out: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    arial = "Arial"
    head_fill = PatternFill("solid", fgColor="1F3864")
    edit_fill = PatternFill("solid", fgColor="FFF2CC")
    alt_fill = PatternFill("solid", fgColor="F2F2F2")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    book = Workbook()

    readme = book.active
    readme.title = "Read me"
    readme.sheet_view.showGridLines = False
    for index, (text, size, bold) in enumerate(LEGEND, start=1):
        cell = readme.cell(row=index, column=1, value=text)
        cell.font = Font(name=arial, size=size, bold=bold,
                         color="1F3864" if bold else "000000")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    readme.column_dimensions["A"].width = 110

    start = len(LEGEND) + 1
    for offset, (label, value) in enumerate(EXAMPLE):
        cell = readme.cell(row=start + offset, column=1, value=f"{label}:  {value}")
        cell.font = Font(name=arial, size=11,
                         bold=label in {"Current text", "New text"},
                         color="7F6000" if label == "New text" else "000000")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    tail = readme.cell(row=start + len(EXAMPLE) + 2, column=1,
                       value="Counts by kind are on the Summary sheet; every "
                             "string is on 'All copy'.")
    tail.font = Font(name=arial, size=11, italic=True)

    sheet = book.create_sheet("All copy")
    for column, (title, width) in enumerate(HEADERS, start=1):
        cell = sheet.cell(row=1, column=column, value=title)
        cell.font = Font(name=arial, size=11, bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = border
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.row_dimensions[1].height = 26

    for index, row in enumerate(rows, start=2):
        values = [
            row["id"], row["kind"], row["where"], row["context"], row["text"],
            row.get("proposed") or None, None,
            row.get("body") or None, row.get("argument") or None,
            f'=IF(ISNUMBER(SEARCH("{{",E{index})),"Yes","")',
            f"=LEN(E{index})",
            f"{row['file']}:{row['line']}",
        ]
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=index, column=column, value=value)
            cell.font = Font(name=arial, size=10)
            cell.border = border
            cell.alignment = Alignment(
                wrap_text=column in {3, 4, 5, 6, 7, 8, 9}, vertical="top",
                horizontal="center" if column in {10, 11} else "left",
            )
            if column in {6, 8}:
                cell.fill = edit_fill
            elif index % 2 == 0:
                cell.fill = alt_fill
            if column == 1:
                cell.font = Font(name=arial, size=10, bold=True)
            elif column == 9:
                cell.font = Font(name=arial, size=9, italic=True, color="808080")
            elif column == 12:
                cell.font = Font(name=arial, size=9, color="808080")

    last = len(rows) + 1
    sheet.auto_filter.ref = f"A1:L{last}"
    sheet.freeze_panes = "B2"

    summary = book.create_sheet("Summary")
    summary.sheet_view.showGridLines = False
    title = summary.cell(row=1, column=1, value="How much copy there is, and where")
    title.font = Font(name=arial, size=14, bold=True, color="1F3864")
    summary.cell(row=2, column=1,
                 value="Counted from 'All copy', so these follow the rows.").font = (
        Font(name=arial, size=10, italic=True, color="808080"))

    for column, (label, width) in enumerate(
        [("Kind", 32), ("Rows", 10), ("Share", 10), ("Edited so far", 14)], start=1
    ):
        cell = summary.cell(row=4, column=column, value=label)
        cell.font = Font(name=arial, size=11, bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.border = border
        summary.column_dimensions[get_column_letter(column)].width = width

    total_row = 5 + len(ORDER)
    for offset, kind in enumerate(ORDER):
        row_index = 5 + offset
        summary.cell(row=row_index, column=1, value=kind)
        summary.cell(row=row_index, column=2,
                     value=f"=COUNTIF('All copy'!$B$2:$B${last},$A{row_index})")
        summary.cell(row=row_index, column=3,
                     value=f"=IF($B${total_row}=0,0,B{row_index}/$B${total_row})")
        summary.cell(row=row_index, column=4,
                     value=f"=COUNTIFS('All copy'!$B$2:$B${last},$A{row_index},"
                           f"'All copy'!$F$2:$F${last},\"<>\")")
        for column in range(1, 5):
            cell = summary.cell(row=row_index, column=column)
            cell.font = Font(name=arial, size=10)
            cell.border = border
            if column == 3:
                cell.number_format = "0.0%"
            if column > 1:
                cell.alignment = Alignment(horizontal="center")

    summary.cell(row=total_row, column=1, value="Total")
    summary.cell(row=total_row, column=2, value=f"=SUM(B5:B{total_row - 1})")
    summary.cell(row=total_row, column=3,
                 value=f"=IF(B{total_row}=0,0,B{total_row}/B{total_row})")
    summary.cell(row=total_row, column=4, value=f"=SUM(D5:D{total_row - 1})")
    for column in range(1, 5):
        cell = summary.cell(row=total_row, column=column)
        cell.font = Font(name=arial, size=11, bold=True)
        cell.border = Border(top=Side(style="double"), left=thin, right=thin,
                             bottom=thin)
        if column == 3:
            cell.number_format = "0.0%"
        if column > 1:
            cell.alignment = Alignment(horizontal="center")

    where_row = total_row + 3
    heading = summary.cell(row=where_row - 1, column=1, value="By screen")
    heading.font = Font(name=arial, size=12, bold=True, color="1F3864")
    for column, label in enumerate(["Where it appears", "Rows"], start=1):
        cell = summary.cell(row=where_row, column=column, value=label)
        cell.font = Font(name=arial, size=11, bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.border = border

    for offset, place in enumerate(sorted({str(r["where"]) for r in rows})):
        row_index = where_row + 1 + offset
        summary.cell(row=row_index, column=1, value=place)
        summary.cell(row=row_index, column=2,
                     value=f"=COUNTIF('All copy'!$C$2:$C${last},$A{row_index})")
        for column in (1, 2):
            cell = summary.cell(row=row_index, column=column)
            cell.font = Font(name=arial, size=10)
            cell.border = border
            if column == 2:
                cell.alignment = Alignment(horizontal="center")

    book.save(out)



# -- applying an edited sheet back --------------------------------------------
#
# The other half of the round trip. Every rule here exists because the failure it
# guards is silent: a replacement written to the wrong string, a placeholder
# dropped, a number that stops being rounded. None of them raises at the time --
# they raise, if at all, on somebody's report months later.


def string_nodes(tree: ast.AST) -> list[ast.AST]:
    """Every outermost string expression, docstrings excluded.

    Outermost matters: a concatenation yields a literal and so does each of its
    parts, so walking the whole tree would find one string three times and the
    rewrite would nest inside itself. This stops at the first node that yields a
    literal and does not descend.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    out: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        if (isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp))
                and id(node) not in docstrings and literal(node) is not None):
            out.append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


def holes(text: str) -> list[str] | None:
    """The ``{...}`` placeholders, or None if a brace is unbalanced.

    Depth-counted rather than matched by regular expression, because a format
    spec may hold a hole of its own: ``{value:.{places}f}`` is one placeholder
    and not two. Unbalanced anywhere means the row is refused -- a stray brace in
    an f-string is a syntax error, and guessing which side it belongs on would
    write one.
    """
    found, depth, start = [], 0, -1
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            if depth == 0:
                return None
            depth -= 1
            if depth == 0:
                found.append(text[start:index + 1])
    return None if depth else found


def quote(chunk: str, is_f: bool) -> str:
    body = chunk.replace("\\", "\\\\").replace('"', '\\"')
    return ('f"' if is_f else '"') + body + '"'


def emit(text: str, indent: int, width: int = 79) -> str:
    """Source for a string literal, wrapped as this project writes them.

    Implicit concatenation across lines, each chunk quoted, ``f`` only on the
    chunks that actually carry a hole -- which is how the surrounding code reads
    and means a plain chunk needs no brace doubling.
    """
    pieces = holes(text) or []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for hole in pieces:
        at = text.index(hole, cursor)
        spans.append((at, at + len(hole)))
        cursor = at + len(hole)

    def in_hole(position: int) -> bool:
        return any(a <= position < b for a, b in spans)

    # Break on spaces that are not inside a placeholder, so a hole is never split
    # across two chunks.
    room = max(width - indent - 3, 24)
    lines: list[str] = []
    line = ""
    index = 0
    while index < len(text):
        space = -1
        for look in range(index, min(index + room, len(text))):
            if text[look] == " " and not in_hole(look):
                space = look
        if len(text) - index <= room or space <= index:
            line = text[index:]
            index = len(text)
        else:
            line = text[index:space + 1]
            index = space + 1
        lines.append(line)

    pad = " " * indent
    out = []
    for offset, chunk in enumerate(lines):
        at = sum(len(part) for part in lines[:offset])
        has_hole = any(at <= a < at + len(chunk) for a, _ in spans)
        out.append(("" if offset == 0 else pad) + quote(chunk, has_hole))
    return "\n".join(out)


def byte_index(text: str) -> list[int]:
    """Character offset of the start of each line, for a file read as text.

    ``col_offset`` counts UTF-8 bytes, and this source is full of em dashes and a
    capital delta, so a column cannot be used as a character index directly.
    """
    starts, position = [0], 0
    for line in text.splitlines(keepends=True):
        position += len(line)
        starts.append(position)
    return starts


def span_of(source: str, starts: list[int], node: ast.AST) -> tuple[int, int]:
    def at(lineno: int, col: int) -> int:
        line_start = starts[lineno - 1]
        line = source[line_start:starts[lineno]]
        return line_start + len(line.encode("utf-8")[:col].decode("utf-8"))
    return at(node.lineno, node.col_offset), at(node.end_lineno, node.end_col_offset)


def read_sheet(path: Path) -> list[dict[str, str]]:
    """Rows of {ID, Current text, New text} from a workbook or a CSV."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    from openpyxl import load_workbook

    book = load_workbook(path, data_only=True)
    sheet = book["All copy"] if "All copy" in book.sheetnames else book.active
    rows = sheet.iter_rows(values_only=True)
    header = [str(cell or "") for cell in next(rows)]
    return [dict(zip(header, ["" if cell is None else str(cell) for cell in row]))
            for row in rows]


def apply_sheet(path: Path, dry_run: bool = False) -> int:
    edits = []
    for row in read_sheet(path):
        current = (row.get("Current text") or "").strip()
        replacement = (row.get("New text") or "").strip()
        if current and replacement and replacement != current:
            edits.append((row.get("ID") or "?", current, replacement))
    if not edits:
        print("No filled-in 'New text' cells. Nothing to apply.")
        return 0

    # Every occurrence, because rows are de-duplicated by text: one row can be
    # the same sentence written in two places, and changing one of them would
    # leave the application saying two things.
    sites: dict[Path, list[tuple[ast.AST, str, str, str]]] = {}
    unmatched: list[tuple[str, str]] = []
    lost_values: list[tuple[str, list[str]]] = []
    for identifier, current, replacement in edits:
        want = holes(current)
        got = holes(replacement)
        if want is None or got is None:
            unmatched.append((identifier, "unbalanced { } braces"))
            continue
        # Dropping a placeholder and inventing one are not the same risk, and
        # treating them alike refused almost every real edit. Shortening a
        # warning is mostly *removing* the values it interpolates, and a value
        # left out cannot fail. A value put in can: the expression is evaluated
        # in whatever scope the string sits in, and one that is not in scope
        # there is a NameError raised on somebody's run, from a cell in a
        # spreadsheet. Nothing here can check that, so it is refused.
        added = sorted(set(got) - set(want))
        if added:
            unmatched.append((
                identifier,
                "invents " + ", ".join(added) + " -- a placeholder can only be "
                "moved or removed here, never added, because nothing in a sheet "
                "can tell whether the value exists where the message is written",
            ))
            continue
        dropped = sorted(set(want) - set(got))
        if dropped:
            lost_values.append((identifier, dropped))
        hits = 0
        for source_path in sorted(SOURCE.rglob("*.py")):
            if "__pycache__" in source_path.as_posix():
                continue
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in string_nodes(tree):
                raw = literal(node)
                if raw is not None and raw.strip() == current:
                    sites.setdefault(source_path, []).append(
                        (node, raw, replacement, identifier))
                    hits += 1
        if not hits:
            unmatched.append((identifier, "no longer in the source as written"))

    changed = 0
    for source_path, found in sorted(sites.items()):
        text = source_path.read_text(encoding="utf-8")
        starts = byte_index(text)
        spans = []
        for node, raw, replacement, identifier in found:
            begin, end = span_of(text, starts, node)
            # Whatever space the original carried outside the prose is part of
            # how it joins to its neighbours, not part of the wording.
            lead = raw[:len(raw) - len(raw.lstrip())]
            tail = raw[len(raw.rstrip()):]
            spans.append((begin, end, emit(lead + replacement + tail,
                                           node.col_offset)))
        # Last first, so an earlier replacement cannot move a later one's span.
        for begin, end, replacement_source in sorted(spans, reverse=True):
            text = text[:begin] + replacement_source + text[end:]
        try:
            ast.parse(text)
        except SyntaxError as exc:
            print(f"  REFUSED {source_path.relative_to(ROOT)}: "
                  f"rewrite would not parse ({exc.msg} line {exc.lineno})",
                  file=sys.stderr)
            continue
        if not dry_run:
            source_path.write_text(text, encoding="utf-8")
        changed += len(found)
        print(f"  {'would change' if dry_run else 'changed'} "
              f"{len(found):3d} in {source_path.relative_to(ROOT)}")

    print(f"{changed} string(s) {'would be ' if dry_run else ''}rewritten "
          f"across {len(sites)} file(s)")
    if lost_values:
        # Reported rather than refused: it is almost always deliberate, and it is
        # the one change that silently removes a number somebody may be reading.
        print(f"\n{len(lost_values)} row(s) no longer show a value they used to:")
        for identifier, dropped in lost_values:
            print(f"  {identifier}: {', '.join(dropped)}")
    if unmatched:
        print(f"\n{len(unmatched)} row(s) refused, and left exactly as they were:",
              file=sys.stderr)
        for identifier, why in unmatched:
            print(f"  {identifier}: {why}", file=sys.stderr)
    return 1 if unmatched else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="where to write it")
    parser.add_argument("--format", choices=("xlsx", "csv"), default="xlsx")
    parser.add_argument("--apply", type=Path, metavar="SHEET",
                        help="write an edited sheet's 'New text' back to the "
                             "source, and report every row it would not")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --apply, say what would change and change "
                             "nothing")
    parser.add_argument("--propose", type=Path, metavar="JSON",
                        help="pre-fill 'New text' from a file of proposed "
                             "wording, keyed by the exact current text")
    args = parser.parse_args()

    if args.apply is not None:
        return apply_sheet(args.apply, dry_run=args.dry_run)

    rows = collect()
    if args.propose is not None:
        proposed = json.loads(args.propose.read_text(encoding="utf-8"))
        filled = 0
        unusable: list[tuple[str, list[str]]] = []
        for row in rows:
            entry = proposed.get(str(row["text"]))
            if entry is None:
                continue
            # A proposal is a headline and, sometimes, a body. Only the headline
            # can be applied today, because a warning is one string in the record
            # and a two-part warning is a schema change nobody has agreed to yet.
            # So the body rides in its own column and --apply does not read it.
            row["proposed"] = entry.get("head", "")
            row["body"] = entry.get("body", "")
            row["disposition"] = entry.get("disp", "")
            row["argument"] = entry.get("note", "")
            filled += 1
            # Checked here rather than left for --apply to refuse later. A
            # proposal is written by somebody reading the sentence, who names
            # the value {n} where the code calls it {small_rois} -- which reads
            # better and cannot be applied, because that name is not in scope
            # where the message is written. Saying so while the sheet is being
            # made is the difference between fixing forty rows now and finding
            # them refused one at a time afterwards.
            want = holes(str(row["text"])) or []
            got = holes(str(row["proposed"])) or []
            invented = sorted(set(got) - set(want))
            if invented:
                unusable.append((str(row["id"]), invented))
        print(f"{filled} row(s) pre-filled from {args.propose.name}")
        if unusable:
            print(f"  {len(unusable)} of them name a value that is not in scope "
                  f"where the message is written, so they are proposals to edit "
                  f"rather than text that can be applied:")
            for identifier, invented in unusable:
                print(f"    {identifier}: {', '.join(invented)}")
    out = args.out or (ROOT / f"deltachrome-copy.{args.format}")
    if args.format == "csv":
        write_csv(rows, out)
    else:
        try:
            write_xlsx(rows, out)
        except ImportError:
            # A maintainer tool, so the dependency is in the dev extra rather
            # than the package's. Saying which is more use than a traceback.
            print("openpyxl is needed for xlsx: pip install -e '.[dev]'",
                  file=sys.stderr)
            print("or run again with --format csv", file=sys.stderr)
            return 1

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["kind"])] = counts.get(str(row["kind"]), 0) + 1
    print(f"{len(rows)} strings -> {out}")
    for kind in ORDER:
        if counts.get(kind):
            print(f"  {kind:32s} {counts[kind]}")
    placeholders = sum("{" in str(r["text"]) for r in rows)
    print(f"  {'with placeholders':32s} {placeholders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
