"""The public tree: what it leaves out, and what it refuses to publish."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_public", ROOT / "packaging" / "export_public.py")
export_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_public)


def test_the_private_notes_and_the_release_workflow_are_excluded():
    for path in ("CLAUDE.md", "docs/decisions.md", ".github/workflows/release.yml",
                 "docs/site/placement.md", "docs/site/images/targets/a.svg",
                 "packaging/public/README.md"):
        assert export_public.excluded(path), path
    for path in ("README.md", "docs/understanding-your-results.md", "src/deltachrome/cli.py",
                 "packaging/export_public.py"):
        assert not export_public.excluded(path), path


def test_the_developer_readme_keeps_its_links_working_from_docs():
    text = ("See [the guide](docs/understanding-your-results.md), the [LICENSE](LICENSE), "
            "[a section](#install), the [site](https://permachro.me/) and "
            "[a module](src/deltachrome/cli.py).")
    moved = export_public.developer_readme(text)
    assert "(understanding-your-results.md)" in moved
    assert "(../LICENSE)" in moved
    assert "(#install)" in moved
    assert "(https://permachro.me/)" in moved
    assert "(../src/deltachrome/cli.py)" in moved


def test_a_tree_carrying_anything_private_is_refused(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "decisions.md").write_text("log")
    (tmp_path / "data").mkdir()
    (tmp_path / "notes.py").write_text("# see open item 12\n")
    (tmp_path / "capture.tif").write_bytes(b"\0" * (export_public.MAX_FILE_BYTES + 1))
    problems = " | ".join(export_public.check(tmp_path))
    assert "docs/decisions.md is in the tree" in problems
    assert "data/ is in the tree" in problems
    assert "notes.py:1 contains 'open item 1'" in problems
    assert "capture.tif is" in problems


def test_this_commit_exports_cleanly(tmp_path):
    """The whole export of HEAD, checks included -- what a publish would write."""
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                      capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")
    if not (ROOT / "CLAUDE.md").is_file():
        pytest.skip("already a public checkout")
    assert export_public.export("HEAD", tmp_path) == []
    assert (tmp_path / "README.md").read_text().startswith("# Deltachrome")
    assert (tmp_path / "docs" / "developer.md").is_file()
    assert not (tmp_path / "CLAUDE.md").exists()
