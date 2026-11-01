"""Pin the user documentation to the code.

The ΔL*2000 weighting table in `docs/understanding-your-results.md` is the thing
users will reach for when a tone response number surprises them. If the code
changes and the table does not, the documentation becomes confidently wrong — which
is worse than having no table. So the table is parsed and checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from deltachrome.imaging.color import delta_L_2000

DOCS = Path(__file__).resolve().parents[1] / "docs"
GUIDE = DOCS / "understanding-your-results.md"
#: The developer README. The public repository's front page is a short landing
#: README and this ships beside the guide as docs/developer.md, so the checks on
#: it read whichever this checkout has.
README = (DOCS / "developer.md") if (DOCS / "developer.md").is_file() else DOCS.parent / "README.md"


def _private(relative: str) -> str:
    """A note kept in the private repository only, or a skip where it is absent.

    The project primer and the decision log are not published with the source,
    so the checks that keep them honest run where they exist and nowhere else.
    """
    path = DOCS.parent / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not part of this checkout")
    return path.read_text(encoding="utf-8")

# | 96 (highlight) | 1.687 | 1.78 | 5.06 |
ROW = re.compile(
    r"^\|\s*(?P<L>\d+)[^|]*\|\s*(?P<sl>[\d.]+)\s*\|\s*(?P<scored>[\d.]+)\s*\|\s*(?P<needed>[\d.]+)\s*\|"
)


def documented_rows():
    rows = []
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line.strip())
        if match:
            rows.append(
                (
                    int(match.group("L")),
                    float(match.group("sl")),
                    float(match.group("scored")),
                    float(match.group("needed")),
                )
            )
    return rows


def test_the_guide_documents_the_full_twelve_step_ramp():
    assert len(documented_rows()) == 12


@pytest.mark.parametrize("row", documented_rows(), ids=lambda r: f"L{r[0]}")
def test_documented_weighting_matches_the_implementation(row):
    L, s_l, scored, needed = row

    expected_s_l = 1.0 + 0.015 * (L - 50) ** 2 / np.sqrt(20.0 + (L - 50) ** 2)
    assert s_l == pytest.approx(expected_s_l, abs=0.001)

    # "A 3.0 L* error scores ..." — a ±1.5 excursion about the ramp value.
    actual = float(delta_L_2000(np.array([L + 1.5, 0, 0]), np.array([L - 1.5, 0, 0])))
    assert scored == pytest.approx(actual, abs=0.01)

    # "Error needed to reach 3.0" — the raw L* difference that scores exactly 3.0.
    assert needed == pytest.approx(3.0 * expected_s_l, abs=0.01)


def _plain_text() -> str:
    """The guide with markdown emphasis and line wrapping removed.

    Assertions below are about what the guide *says*, so they must not break when a
    sentence is re-wrapped or a word is emphasised.
    """
    import re

    text = GUIDE.read_text(encoding="utf-8")
    text = text.replace("\\*", "*").replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def test_the_guide_covers_each_metric_users_misread():
    """The weighting table is only useful alongside the claims it supports."""
    text = _plain_text()

    # Tone response: the weighted quantity, not the raw one.
    assert "raw difference between the measured and reference L* values" in text
    # Color accuracy: the 90th percentile is not the worst patch.
    assert "This is not the maximum." in text
    # White balance: chroma only, and why that separation is diagnostic.
    assert "with L* left out entirely" in text
    assert "different causes and different fixes" in text
    # Uniformity: areal coverage, not the whole frame.
    assert "What it does not measure: uniformity across your full camera frame" in text
    # Color management: why a missing profile is a hard stop.
    assert "Every image must have an embedded ICC profile" in text
    # LUT profiles, now that they are supported rather than refused.
    assert "LUT-based profiles" in text and "A2B1" in text


def test_the_guide_does_not_still_claim_lut_profiles_are_refused():
    """A stale refusal in user-facing docs would send people converting files they
    do not need to convert."""
    text = _plain_text().lower()
    assert "lut-based scanner/camera profiles are refused" not in text
    assert "convert the master to a matrix/trc working space" not in text


def test_mid_ramp_threshold_behaves_almost_like_a_raw_l_star_limit():
    """The guide tells users the threshold is ~'3.0 L* of error' near mid-gray.
    That claim has to stay true."""
    mid = float(delta_L_2000(np.array([51.5, 0, 0]), np.array([48.5, 0, 0])))
    assert mid == pytest.approx(3.0, abs=0.02)


def test_ramp_ends_are_more_permissive_than_the_middle():
    """The guide's central point: the same physical error scores lower at the ends."""
    mid = float(delta_L_2000(np.array([51.5, 0, 0]), np.array([48.5, 0, 0])))
    highlight = float(delta_L_2000(np.array([97.5, 0, 0]), np.array([94.5, 0, 0])))
    shadow = float(delta_L_2000(np.array([11.5, 0, 0]), np.array([8.5, 0, 0])))
    assert highlight < mid and shadow < mid
    assert highlight < 2.0 and shadow < 2.0


def test_documented_cli_commands_exist():
    from deltachrome.cli import _build_parser

    text = GUIDE.read_text(encoding="utf-8")
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for command in re.findall(r"deltachrome ([a-z-]+)", text):
        assert command in subparsers.choices, f"guide references unknown command {command!r}"


def test_documented_percentile_method_is_accepted():
    from deltachrome.metrics.base import PERCENTILE_METHODS

    assert "nearest_rank" in PERCENTILE_METHODS
    assert "nearest_rank" in GUIDE.read_text(encoding="utf-8")


def test_the_guide_covers_the_sharpness_metrics():
    """SFR carries more interpretation traps than any other metric here: an
    efficiency above 100% is not an error, sharpening is judged from a curve that is
    1.0 by definition at DC, and a mean can hide one soft corner."""
    text = _plain_text()

    assert "Sampling efficiency above 100%" in text
    assert "aliasing" in text
    assert "an unsharpened system reads exactly" in text
    assert "spatial overshoot" in text.lower()
    # The diagnostic value of per-edge data, which the verdict throws away.
    assert "astigmatism" in text
    assert "Every edge is kept" in text
    # And the aggregation choice, which no standard specifies.
    assert "Neither FADGI nor ISO 19264 says" in text
    assert "a mean can pass a system with one badly soft corner" in text


def test_the_guide_does_not_claim_a_standard_specifies_edge_aggregation():
    """It does not, and saying so would be the overclaiming this project avoids."""
    text = _plain_text().lower()
    assert "fadgi requires the worst edge" not in text
    assert "iso requires the mean" not in text


def test_the_guide_explains_where_the_sampling_rate_comes_from():
    """It is measured off the chart, not read from the header -- and the difference
    between those two is a whole metric."""
    text = _plain_text()
    assert "PRINTED SCALE" in text
    assert "It does not read it from the file header" in text
    assert "Check the header before you check the camera" in text


def test_the_guide_explains_what_an_indicative_verdict_means():
    """The one verdict a reader could mistake for a conformance result. The guide has
    to say what it is good for as well as what it is not."""
    text = _plain_text()
    assert "ADVISORY" in text
    assert "print tolerance" in text
    assert "--no-nominal-fallback" in text
    assert "certifying conformance" in text


def test_the_guide_explains_the_term_most_easily_mistaken_for_misregistration():
    """The antisymmetric term is usually the larger of the two numbers, and reading
    it as misregistration would send someone after a lens when the fault is flare."""
    text = _plain_text()
    assert "antisymmetric term" in text
    assert "not of your camera's channel registration" in text
    assert "flare" in text.lower()


def test_the_guide_does_not_claim_a_standard_specifies_registration_reduction():
    """It does not, and the two defaults differ between metrics, so the guide has to
    say which is which rather than implying either is required."""
    text = _plain_text().lower()
    assert "neither fadgi nor iso 19264-1 says" in text
    assert "the opposite of the sharpness default" in text


def test_the_readme_status_matches_what_is_actually_built():
    readme = README.read_text(encoding="utf-8")
    assert "vertical slice" not in readme.lower()
    for capability in ("Slanted-edge SFR", "gain modulation", "JPEG 2000", "CxF3",
                       "Color-channel misregistration", "printed rule"):
        assert capability in readme, capability
    # The claim that nothing is unimplemented has to be one the code can back.
    assert "NOT IMPLEMENTED" in readme
    from deltachrome.metrics import available_metrics
    from deltachrome.profiles.schema import available_profiles, load_profile

    registered = set(available_metrics())
    for profile_id in available_profiles():
        assert {s.metric_id for s in load_profile(profile_id).metrics} <= registered


def test_the_readme_target_table_lists_every_shipped_target():
    """It went stale exactly once: two transmissive targets were shipped and never
    added, and the Golden Thread's count still said 30 after twenty registration
    squares took it to 50. The table is the first thing someone adding a target
    reads, so a row count is cheaper to maintain than the trust it buys."""
    from deltachrome.targets.registry import available_targets, load_target

    readme = README.read_text(encoding="utf-8")
    section = readme[readme.index("## Supported targets"):readme.index("## Documentation")]
    rows = [
        line for line in section.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ][1:]  # drop the header

    targets = available_targets()
    assert len(rows) == len(targets), (
        f"{len(rows)} rows for {len(targets)} shipped targets: "
        f"{sorted(targets)}"
    )
    for target_id in targets:
        target = load_target(target_id)
        count = str(len(target.patches)) if target.patches else "—"
        assert any(f"| {count} |" in row for row in rows), (
            f"{target_id} has {count} patches; no row states that"
        )


# The white balance comparison table in the guide, checked against the code that
# produced it. Same reasoning as the ΔL*2000 table above: a table that drifts from
# the implementation is worse than no table, and this one exists to stop someone
# quoting a Metamorfoze figure against a FADGI threshold.
WHITE_BALANCE_ROW = re.compile(
    r"^\|\s*(?P<label>near-neutral|saturated red|saturated blue)\s*\|"
    r"\s*(?P<euclidean>[\d.]+)\s*\|\s*\*{0,2}(?P<weighted>[\d.]+)\*{0,2}\s*\|"
)

WHITE_BALANCE_CASES = {
    "near-neutral": ([50.0, 1.0, -2.0], [50.0, 4.0, 2.0]),
    "saturated red": ([50.0, 60.0, 0.0], [50.0, 64.0, 3.0]),
    "saturated blue": ([50.0, 0.0, -60.0], [50.0, 3.0, -64.0]),
}


def white_balance_rows():
    rows = []
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        match = WHITE_BALANCE_ROW.match(line.strip())
        if match:
            rows.append((match.group("label"), float(match.group("euclidean")),
                         float(match.group("weighted"))))
    return rows


def test_the_guide_documents_all_three_white_balance_cases():
    assert len(white_balance_rows()) == 3


@pytest.mark.parametrize("row", white_balance_rows(), ids=lambda r: r[0])
def test_documented_white_balance_table_matches_the_implementation(row):
    from deltachrome.imaging.color import delta_e_2000_ab, delta_e_ab

    label, euclidean, weighted = row
    reference, sample = (np.array(v) for v in WHITE_BALANCE_CASES[label])
    assert float(delta_e_ab(reference, sample)) == pytest.approx(euclidean, abs=0.001)
    assert float(delta_e_2000_ab(sample, reference)) == pytest.approx(weighted, abs=0.001)


def test_the_guide_covers_what_each_mtf50_unit_means():
    """A user reading '66.6 % of half-sampling freq' against a requirement written
    in lp/mm needs to know they are the same measurement."""
    guide = GUIDE.read_text(encoding="utf-8")
    for phrase in ("lp/mm", "cycles per pixel", "--frequency-units",
                   "a line pair needs two samples", "unreachable"):
        assert phrase.lower() in guide.lower(), phrase


# The status section went stale the moment the frontend existed: README and
# CLAUDE.md both still said "engine only, no GUI". CLAUDE.md's own preamble says
# that where it and the code disagree, the code is right and the file is a bug --
# so the disagreement is worth catching mechanically rather than by memory.


def test_the_readme_no_longer_claims_there_is_no_gui():
    readme = README.read_text(encoding="utf-8")
    for stale in ("engine only", "No GUI yet"):
        assert stale.lower() not in readme.lower(), stale


def test_the_primer_no_longer_claims_there_is_no_gui():
    primer = _private("CLAUDE.md")
    assert "there is no GUI" not in primer


def test_the_documented_gui_entry_point_exists():
    """A named command in the docs that nothing installs is worse than none."""
    import tomllib

    pyproject = tomllib.loads(
        (DOCS.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"]["gui-scripts"]
    readme = README.read_text(encoding="utf-8")
    for name, entry in scripts.items():
        assert name in readme, f"{name} is installed but undocumented"
        module, _, function = entry.partition(":")
        imported = __import__(module, fromlist=[function])
        assert callable(getattr(imported, function))


def test_the_gui_toolkit_is_an_optional_extra():
    """The engine is a library. A library that drags a GUI toolkit into every CI
    runner and every headless import is not one."""
    import tomllib

    pyproject = tomllib.loads(
        (DOCS.parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "PySide6" not in " ".join(pyproject["project"]["dependencies"])
    assert any(
        "PySide6" in requirement
        for requirement in pyproject["project"]["optional-dependencies"]["gui"]
    )


def test_the_frontend_boundary_claim_is_backed_by_a_test():
    """Both files claim the boundary is enforced by source scan. If that test is
    ever deleted, the claim becomes decoration."""
    for name in ("README.md", "CLAUDE.md"):
        text = (README.read_text(encoding="utf-8") if name == "README.md"
                else _private(name))
        assert "test_ui_boundary" in text, name
    assert (DOCS.parent / "tests" / "test_ui_boundary.py").exists()


# -- the primer describes what is actually there -----------------------------


def test_the_primer_states_the_schema_version_the_code_uses():
    """A record format documented one version behind is worse than undocumented:
    it tells a reader the aliases are unnecessary."""
    from deltachrome.results.models import SCHEMA_VERSION

    primer = _private("CLAUDE.md")
    assert f"schema {SCHEMA_VERSION}" in primer


def test_the_primer_names_every_key_the_reader_still_aliases():
    """These are the spellings in records already written. If one is dropped from
    the alias map, the primer should stop promising it."""
    from deltachrome.results.models import RENAMED_KEYS

    decisions = _private("docs/decisions.md")
    for old_key in RENAMED_KEYS:
        assert old_key in decisions, old_key


def test_the_documentation_describes_the_report_that_exists():
    """The report was a scrolling column of metrics, then a split pane, then
    compartments, and is now one metric at a time. Each rewrite left the docs
    describing the previous one -- including this test, which went on pinning
    the words "tree" and "Field" for two rewrites after the tree and the field
    map were both removed. What it asserts has to be what is there now."""
    readme = README.read_text(encoding="utf-8")
    for claim in ("Overview", "Charts", "Values", "Load Capture"):
        assert claim in readme, claim
    # And no longer describes what it was: a tree of compartments, a strip of
    # tiles, a field map beside a panel already showing the chart.
    for gone in ("A tree on the left", "**Field**", "Copy as `--corners`"):
        assert gone not in readme, gone


def test_every_frontend_module_the_primer_names_exists():
    from pathlib import Path

    primer = _private("CLAUDE.md")
    for module in ("explanations.py", "nav_tree.py", "capture_dock.py"):
        if module in primer:
            assert (Path("src/deltachrome_ui") / module).exists(), module


# -- the open items list must not describe a version of the code that is gone ----
#
# The failure this guards is quiet and it has happened: an edit striking four items
# through was written by a script that raised before it wrote the file, so the log
# went on saying grayscale was refused and three targets were unmodelled for two
# commits after both stopped being true. A decision log that is behind is worse than
# one that is absent -- a reader trusts it, and it is the only place several of
# these judgements are recorded at all.


def _open_items() -> str:
    text = _private("docs/decisions.md")
    return text.split("## Still open", 1)[1].split("## Verification status", 1)[0]


def _live_items() -> list[str]:
    """Items not struck through. A struck item is history and may say anything."""
    import re

    items = re.split(r"\n(?=\d+[a-z]?\. )", _open_items())
    return [item for item in items if "~~" not in item.split("\n")[0]]


def test_no_live_open_item_says_grayscale_is_refused():
    """It is read: a GRAY ICC profile is evaluated through its own transform and the
    loader carries a channel count. Only the three metrics that need more than one
    colour record are held back, and those report incomplete rather than refusing
    the file."""
    from deltachrome.imaging.icc import DEVICE_CHANNELS

    assert "GRAY" in DEVICE_CHANNELS
    offenders = [item.split("\n")[0] for item in _live_items()
                 if "rayscale" in item and "refuse" in item]
    assert not offenders, offenders


def test_no_live_open_item_calls_an_installed_target_unmodelled():
    """The other half of the same lapse. Every target id the log names as missing
    has to actually be missing."""
    from deltachrome.targets.registry import available_targets

    installed = set(available_targets())
    offenders = []
    for item in _live_items():
        if "not modelled" not in item and "not modeled" not in item:
            continue
        for target_id in installed:
            if target_id in item:
                offenders.append((item.split("\n")[0][:60], target_id))
    assert not offenders, offenders


def test_every_item_marked_done_is_struck_through():
    """Both halves of the convention, so a reader can tell at a glance. An item that
    says DONE in bold and is not struck reads as work in progress."""
    import re

    offenders = []
    for item in re.split(r"\n(?=\d+[a-z]?\. )", _open_items()):
        head = item.split("\n")[0]
        if not head.strip():
            continue
        done = re.search(r"\*\*(DONE|SETTLED)", item)
        if done and "~~" not in item:
            offenders.append(head[:70])
    assert not offenders, offenders
