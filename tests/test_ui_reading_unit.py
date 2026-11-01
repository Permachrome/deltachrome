"""What a readings table heads its value column with.

The record says, in ``reading_units``, where a reading is not in the aggregate's
unit. The frontend used to keep its own exception keyed on a metric id -- one
entry, when four metrics needed one -- so geometric distortion's twenty rule
readings were headed ``%`` while each was a sampling rate.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from deltachrome.results.models import AnalysisRun  # noqa: E402
from deltachrome_ui.labels import reading_unit  # noqa: E402


def test_the_column_is_headed_with_the_readings_own_unit():
    assert reading_unit({"metric_id": "geometric_distortion", "units": "%",
                         "reading_units": "ppi"}) == "ppi"
    assert reading_unit({"metric_id": "lightness_uniformity",
                         "units": "% (std dev / mean L*)",
                         "reading_units": "dL* from family mean"}) == (
        "ΔL* from family mean")


def test_otherwise_it_is_the_aggregates_unit():
    assert reading_unit({"metric_id": "noise", "units": "std dev of L*",
                         "reading_units": None}) == "σ L*"


def test_nothing_is_keyed_on_the_metric_id_any_more():
    """A metric the frontend has never heard of, stating its own reading unit,
    is labelled by what it states."""
    assert reading_unit({"metric_id": "something_new", "units": "%",
                         "reading_units": "px"}) == "px"


def test_a_reopened_older_record_is_labelled_correctly():
    record = {
        "run_id": "r", "created_utc": "", "schema_version": 8, "overall": "PASS",
        "metrics": [{"metric_id": "geometric_distortion", "outcome": "evaluated",
                     "value": 0.4, "units": "%"}],
    }
    metric = AnalysisRun.from_dict(record).metrics[0]
    assert reading_unit(metric) == "ppi"


def test_a_sampling_flag_is_written_in_words():
    """The record keeps ``gamut_clipped(R)``; a reader is shown what it means.

    A shadow flag raised beside a channel flag is the same pixels described
    less exactly, so it is not shown twice."""
    from deltachrome.reporting.labels import flag_labels

    assert flag_labels(["clipped_low(4.1%)", "gamut_clipped(RB)"]) == [
        "Out of gamut (red and blue clipped)"
    ]
    assert flag_labels(["channel_floor(B)", "clipped_low(1.0%)"]) == ["No signal in blue"]
    assert flag_labels(["clipped_high(0.3%)", "mean_median_divergence"]) == [
        "Highlights clipped (0.3%)", "Possible dust or glare"
    ]
    assert flag_labels(["clipped_low(2.0%)"]) == ["Shadows clipped (2.0%)"]
    assert flag_labels(["something_new(x)"]) == ["Something new (x)"]
    assert flag_labels(None) == []
