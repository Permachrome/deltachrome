"""The engine stays free of UI code, and it is a test that says so.

``deltachrome/__init__.py`` states that no UI code lives below that package, and
``deltachrome_ui`` is a separate top-level package rather than a subpackage
precisely so that the claim is checkable. Without this test it is an intention,
and intentions drift: one convenient ``QColor`` import inside a metric plugin and
the engine can no longer be used headless, in CI, or from a notebook.

The scan is on the source rather than on imports, so it holds whether or not
PySide6 happens to be installed in the environment running the tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent / "src" / "deltachrome"
UI = Path(__file__).resolve().parent.parent / "src" / "deltachrome_ui"

#: Toolkits that would make the engine un-runnable headless.
GUI_MODULES = {"PySide6", "PySide2", "PyQt5", "PyQt6", "tkinter", "wx", "kivy"}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


ENGINE_FILES = sorted(ENGINE.rglob("*.py"))


def test_the_engine_package_exists_and_was_found():
    """Guards the guard: a path typo would make every check below vacuous."""
    assert len(ENGINE_FILES) > 20


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda p: p.name)
def test_no_engine_module_imports_a_gui_toolkit(path):
    offenders = imported_roots(path) & GUI_MODULES
    assert not offenders, (
        f"{path.relative_to(ENGINE.parent)} imports {sorted(offenders)}. The engine "
        "is a library and must stay usable with no GUI toolkit installed; UI code "
        "belongs in the deltachrome_ui package."
    )


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda p: p.name)
def test_no_engine_module_imports_the_frontend(path):
    assert "deltachrome_ui" not in imported_roots(path), (
        f"{path.relative_to(ENGINE.parent)} imports the frontend. The dependency "
        "runs one way only: the UI knows about the engine, never the reverse."
    )


def test_the_frontend_does_not_reimplement_measurement():
    """The UI displays numbers; it never produces one.

    A frontend that quietly computed its own Delta E, or its own aggregate, would
    be able to disagree with the CLI about the same capture -- and the whole point
    of the engine being a library is that there is one answer, not one per
    front end.
    """
    forbidden = {"delta_e_2000", "delta_e_76", "delta_e_94", "delta_e_ab"}
    for path in sorted(UI.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"def {name}" not in source, (
                f"{path.name} defines {name}. Measurement belongs to the engine."
            )
