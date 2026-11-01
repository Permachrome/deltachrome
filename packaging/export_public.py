"""Write the public repository's tree from a commit of this one.

    python packaging/export_public.py ~/src/deltachrome          # from main
    python packaging/export_public.py ~/src/deltachrome --ref v0.9.0b2

The public repository is not a remote of this one. Pushing this repository's
history there would publish every past version of the files that stay private,
so each publish is a copy of one commit: exported, altered in the ways listed
below, checked, and written over a clone of the public repository. Nothing is
committed or pushed here; the script ends by printing the commands that do.

What the copy differs in:

* :data:`EXCLUDED` paths are left out -- the project primer, the decision log,
  the release workflow, which holds the signing setup and runs here, and the
  user guide's draft pages and illustrations.
* This repository's README becomes ``docs/developer.md``, its relative links
  rewritten for the new place, and the public landing page in
  ``packaging/public/`` becomes ``README.md``.

It reads a committed ref through ``git archive``, so neither uncommitted edits
nor anything gitignored -- ``data/`` above all -- can reach the copy. The checks
afterwards are the second line, and the copy is refused if any fails.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = "packaging/public"

#: Paths left out of the public tree. A trailing slash is a whole folder.
#: ``docs/site/`` is the user guide's working copy -- draft pages and target
#: illustrations -- which is published on the website when finished, not here.
EXCLUDED = ("CLAUDE.md", "docs/decisions.md", ".github/", "docs/site/", OVERLAY + "/")

#: Text that must not appear in any published file: pointers into the private
#: notes and a private folder's name. The files allowed to name them do so as
#: the rule itself, or to test it.
FORBIDDEN = (
    re.compile(r"\bopen items? \d"),
    re.compile(r"docs/decisions\.md"),
    re.compile(r"DT_Developer_Implementation"),
)
ALLOWED_TO_NAME = ("packaging/export_public.py", "tests/test_documentation.py",
                   "tests/test_export_public.py")

#: Above this a file is almost certainly a capture rather than source.
MAX_FILE_BYTES = 5 * 1024 * 1024


def excluded(relative: str) -> bool:
    return any(relative == e or (e.endswith("/") and relative.startswith(e))
               for e in EXCLUDED)


def developer_readme(text: str) -> str:
    """The README as it reads from ``docs/``: relative links one level up."""
    def relink(match: re.Match) -> str:
        target = match.group(2)
        if re.match(r"[a-z]+:|#|/", target):
            return match.group(0)
        moved = target[len("docs/"):] if target.startswith("docs/") else "../" + target
        return f"{match.group(1)}({moved})"
    return re.sub(r"(\]\s*)\(([^)\s]+)\)", relink, text)


def export(ref: str, into: Path) -> list[str]:
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref],
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(into, filter="data")
    readme = into / "README.md"
    (into / "docs" / "developer.md").write_text(
        developer_readme(readme.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copyfile(into / OVERLAY / "README.md", readme)
    for path in sorted(into.rglob("*"), reverse=True):
        relative = path.relative_to(into).as_posix()
        if excluded(relative) or (path.is_dir() and excluded(relative + "/")):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return check(into)


def check(tree: Path) -> list[str]:
    """Every reason the tree must not be published; empty if there is none."""
    problems = []
    for name in EXCLUDED:
        if (tree / name.rstrip("/")).exists():
            problems.append(f"{name} is in the tree")
    if (tree / "data").exists():
        problems.append("data/ is in the tree")
    for path in tree.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(tree).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"{relative} is {path.stat().st_size // 1024} KB")
        if relative in ALLOWED_TO_NAME:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN:
            if found := pattern.search(text):
                line = text.count("\n", 0, found.start()) + 1
                problems.append(f"{relative}:{line} contains {found.group(0)!r}")
    return problems


def replace_tree(dest: Path, tree: Path) -> None:
    """Make ``dest``'s working tree exactly ``tree``, leaving its ``.git`` alone."""
    for child in dest.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in tree.iterdir():
        target = dest / child.name
        shutil.copytree(child, target) if child.is_dir() else shutil.copy2(child, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dest", type=Path,
                        help="a clone of the public repository, or an empty folder")
    parser.add_argument("--ref", default="main", help="the commit to publish (default main)")
    args = parser.parse_args()

    dest = args.dest.expanduser().resolve()
    if dest == ROOT or ROOT in dest.parents:
        print("The destination is inside this repository.", file=sys.stderr)
        return 1
    if dest.exists() and any(dest.iterdir()) and not (dest / ".git").is_dir():
        print(f"{dest} is neither empty nor a git clone.", file=sys.stderr)
        return 1

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", args.ref],
                            check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch)
        problems = export(args.ref, tree)
        if problems:
            print("Refused. The exported tree would publish:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        replace_tree(dest, tree)

    print(f"Wrote {args.ref} ({commit}) to {dest}. Review, then:")
    print(f"  git -C {dest} add -A")
    print(f"  git -C {dest} commit -m 'Deltachrome source at {commit}'")
    print(f"  git -C {dest} push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
