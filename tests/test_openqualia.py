"""OpenQualia labels: reading one, resolving it, and what it must never rewrite.

The three things worth pinning here are the three that were measured against the
live host and would otherwise be re-derived by guesswork:

* a label is a URL and the code layer used to throw it away,
* the URL is passed on exactly as printed, because the host distinguishes
  spellings that look interchangeable, and
* the file that comes back is filed under the unit that was *asked for*, not the
  serial it states about itself -- there is a published file where those differ.

Nothing here touches the network. :func:`fetch_measurement` takes an opener so
the failure paths can be exercised, which matters more than the happy one: a
404, a redirect off https and an HTML error page are all things a real host does.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from deltachrome.errors import MissingReferenceError, ReferenceError
from deltachrome.imaging.codes import _is_plausible, normalize_identifier
from deltachrome.reference.library import ReferenceLibrary
from deltachrome.reference.openqualia import (
    fetch_measurement,
    is_label,
    parse_label,
)

#: The example printed in the OpenQualia Label Standard itself.
LABEL = (
    "https://targets.digitaltransitions.com/measurements"
    "?Manufacturer=DT&TargetType=DTNGT2&TargetID=DT-AR-2020041"
)

MEASUREMENT = b"""OQM
DESCRIPTOR\t"None"
CREATED\t"2021-04-06"
SERIAL\t"DT-AR-2020041"
ILLUMINANT\t"D50"
OBSERVER\t2
NUMBER_OF_FIELDS\t5
BEGIN_DATA_FORMAT
SAMPLE_ID\tSAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B
END_DATA_FORMAT
NUMBER_OF_SETS\t1
BEGIN_DATA
1\tA0\t96.08\t-0.91\t2.13
END_DATA
"""


class _Response:
    """The bits of an HTTP response this module actually reads."""

    def __init__(self, body: bytes, *, url: str = LABEL, disposition: str | None = None):
        self._body = io.BytesIO(body)
        self.url = url
        self.headers = {} if disposition is None else {"Content-Disposition": disposition}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, result):
        self.result = result
        self.requested: list[str] = []

    def open(self, request, timeout=None):
        self.requested.append(getattr(request, "full_url", request))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# -- reading a label ------------------------------------------------------


def test_a_label_survives_the_plausibility_filter():
    """The bug this feature was blocked on.

    Every OQ Label was decoded correctly and dropped one line later: the
    standard's own example is 108 characters against a 64-character cap, and
    '?', '&' and '=' are outside the character class a chart serial is held to.
    """
    assert len(LABEL) > 64
    assert _is_plausible(LABEL)
    assert _is_plausible("CCSG21")


def test_a_label_folds_to_the_unit_it_names():
    """Not to its own URL, so the bank pairs on the serial without knowing
    anything about labels."""
    assert normalize_identifier(LABEL) == "DTAR2020041"


@pytest.mark.parametrize(
    "text",
    [
        "CCSG21",
        "1930021",
        "",
        # http, where the standard requires https.
        "http://x.org/m?Manufacturer=DT&TargetType=T&TargetID=A1",
        # Missing two of the three required parameters.
        "https://x.org/m?TargetID=A1",
        # A serial with a space in it is not one the standard allows.
        "https://x.org/m?Manufacturer=DT&TargetType=T&TargetID=A%201",
    ],
)
def test_what_is_not_a_label(text):
    """Returning None is the ordinary outcome: most codes a chart carries are
    plain serials, and this must not widen into a general URL sniffer."""
    assert parse_label(text) is None
    assert not is_label(text)


def test_a_self_served_label_is_a_label():
    """The standard provides for one built by hand on a Dropbox link, and such a
    label carries the file's own parameters as well as OpenQualia's."""
    label = parse_label(
        "https://www.dropbox.com/s/abc/M.txt?dl=1"
        "&Manufacturer=DT&TargetType=DTNGT2&TargetID=DT-AR-2020041"
    )
    assert label is not None
    assert label.target_id == "DT-AR-2020041"
    assert "dl=1" in label.measurement_url


# -- what the URL must not become ----------------------------------------


def test_the_url_is_passed_on_exactly_as_printed():
    """Measured against the host that serves the standard's own example: it
    distinguishes 'OLT1x' from 'OLT-1x' and 'TV24017' from 'tv24017', so
    re-encoding a query -- '%20' to '+', parameters reordered -- turns a working
    label into a 404 that looks like a missing measurement."""
    label = parse_label(
        "https://x.org/m?Manufacturer=DT&TargetType=DT%20NTVT&TargetID=TV24017"
    )
    url = label.measurement_url
    assert "TargetType=DT%20NTVT" in url
    assert "+" not in url
    assert url.index("Manufacturer") < url.index("TargetType") < url.index("TargetID")
    assert url.endswith("&AccessMode=ActiveMeasurement")


def test_an_existing_access_mode_is_replaced_not_appended():
    """A label carrying Interactive would otherwise fetch a web page."""
    label = parse_label(LABEL + "&AccessMode=Interactive")
    url = label.measurement_url
    assert url.count("AccessMode") == 1
    assert "Interactive" not in url


def test_unrecognised_parameters_are_kept():
    """The standard asks for it outright: a label may carry vendor parameters
    that this application has never heard of and the host needs."""
    label = parse_label(LABEL + "&MeasurementGeometry=45x0")
    assert "MeasurementGeometry=45x0" in label.measurement_url


# -- fetching -------------------------------------------------------------


def test_a_measurement_is_fetched_and_named():
    opener = _Opener(
        _Response(
            MEASUREMENT,
            disposition="attachment; filename*=utf-8''DT%20NGT2%20DT-AR-2020041.oqm.txt",
        )
    )
    fetched = fetch_measurement(parse_label(LABEL), opener=opener)
    assert fetched.content == MEASUREMENT
    assert fetched.filename == "DT NGT2 DT-AR-2020041.oqm.txt"
    assert opener.requested[0].endswith("&AccessMode=ActiveMeasurement")


def test_a_server_filename_cannot_escape_the_bank():
    """The name is chosen by a remote host, so any directory part is discarded
    outright rather than sanitised -- the only handling that cannot be talked
    around."""
    opener = _Opener(
        _Response(MEASUREMENT, disposition='attachment; filename="../../etc/passwd"')
    )
    fetched = fetch_measurement(parse_label(LABEL), opener=opener)
    assert "/" not in fetched.filename
    assert ".." not in fetched.filename


def test_a_missing_unit_says_the_code_was_read_correctly():
    """A 404 here is not a pairing failure, and an operator told only 'not
    found' would go looking at the chart."""
    error = urllib.error.HTTPError(LABEL, 404, "Not Found", {}, None)
    with pytest.raises(ReferenceError) as excinfo:
        fetch_measurement(parse_label(LABEL), opener=_Opener(error))
    assert "read correctly" in str(excinfo.value)


def test_a_page_that_is_not_a_measurement_is_refused():
    """A host answering a missing unit with a friendly HTML page would otherwise
    put an error document in the bank under a serial number, to be found, offered
    for pairing, and to fail in the middle of a run."""
    opener = _Opener(_Response(b"<!doctype html><title>Not found</title>"))
    with pytest.raises(ReferenceError) as excinfo:
        fetch_measurement(parse_label(LABEL), opener=opener)
    assert "not a measurement file" in str(excinfo.value)


def test_a_redirect_off_https_is_refused():
    opener = _Opener(_Response(MEASUREMENT, url="http://elsewhere.example/x"))
    with pytest.raises(ReferenceError) as excinfo:
        fetch_measurement(parse_label(LABEL), opener=opener)
    assert "not https" in str(excinfo.value)


def test_an_unreachable_host_is_a_network_problem_not_a_pairing_one():
    opener = _Opener(urllib.error.URLError("no route to host"))
    with pytest.raises(ReferenceError) as excinfo:
        fetch_measurement(parse_label(LABEL), opener=opener)
    assert "network problem" in str(excinfo.value)


# -- what lands in the bank ----------------------------------------------


@pytest.fixture
def bank(tmp_path):
    directory = tmp_path / "references"
    directory.mkdir()
    return ReferenceLibrary(directory)


def test_a_fetched_file_is_filed_under_the_unit_that_was_asked_for(bank):
    """The published measurement for DT-AR-2020013 states SERIAL DT-AR-2020014,
    and is a different measurement from the real 014 file. Indexing on what a
    file says about itself would file it under another chart's serial and pair
    it with the wrong capture ever after."""
    asked_for = "DT-AR-2020013"
    entry = bank.add_measurement(
        MEASUREMENT, "DT NGT2 DT-AR-2020013.oqm.txt", bind=asked_for, origin=LABEL
    )
    # What the file says about itself is still read, and still answers.
    assert entry.serial == "DT-AR-2020041"
    resolved = bank.resolve([asked_for])
    assert resolved.path == entry.path
    assert bank.origin_of(entry.name) == LABEL


def test_a_label_pairs_a_fetched_file_without_the_bank_knowing_about_labels(bank):
    """The whole point of folding a label to its TargetID in one place."""
    bank.add_measurement(MEASUREMENT, "m.oqm.txt", bind="DT-AR-2020041", origin=LABEL)
    assert bank.resolve([LABEL]).path is not None


def test_a_fetched_file_never_overwrites_one_already_there(bank):
    first = bank.add_measurement(MEASUREMENT, "m.oqm.txt", bind="A1")
    second = bank.add_measurement(MEASUREMENT, "m.oqm.txt", bind="A2")
    assert first.path != second.path
    assert first.path.exists()


def test_removing_a_fetched_file_forgets_where_it_came_from(bank):
    entry = bank.add_measurement(MEASUREMENT, "m.oqm.txt", bind="A1", origin=LABEL)
    bank.remove_file(entry.name)
    assert bank.origin_of(entry.name) is None
    assert bank.resolve(["A1"]).path is None


def test_a_filename_with_no_usable_name_is_refused(bank):
    with pytest.raises(MissingReferenceError):
        bank.add_measurement(MEASUREMENT, ".", bind="A1")


# -- one chart, two vocabularies ------------------------------------------


def _measurement_named(names, *, lab=(50.0, 1.0, -1.0)):
    from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement

    return MeasurementSet(
        samples={n: PatchMeasurement(sample_name=n, lab=lab) for n in names},
        metadata={},
        fields=[],
    )


def test_isa_files_are_reconciled_onto_the_labels_the_chart_prints():
    """Published measurements for the Golden Thread targets name their patches
    'A1'-'A30' where the card prints '1'-'30' -- and the OpenQualia standard
    names these very targets as its example of the bare-number case, so the
    definition here is the conformant one and the file is what moves."""
    from deltachrome.reference.sample_ids import reconcile_sample_ids
    from deltachrome.targets.registry import load_target

    target = load_target("isa_golden_thread_dlt_v1")
    reference = _measurement_named(f"A{n}" for n in range(1, 31))

    reconciled, note = reconcile_sample_ids(reference, target)

    assert set(reconciled.samples) >= {str(n) for n in range(1, 31)}
    assert reconciled.samples["1"].sample_name == "1"
    assert note and "'A1'" in note and "'1'" in note


def test_a_file_that_already_joins_is_untouched_and_silent():
    """No note, and the same object: a correspondence that changes nothing must
    not appear in a run's warnings."""
    from deltachrome.reference.sample_ids import reconcile_sample_ids
    from deltachrome.targets.registry import load_target

    target = load_target("isa_golden_thread_dlt_v1")
    reference = _measurement_named(p.patch_id for p in target.patches)

    reconciled, note = reconcile_sample_ids(reference, target)

    assert note is None
    assert reconciled is reference


def test_a_correspondence_is_never_guessed():
    """A target that declares no alias keeps the file exactly as written, however
    badly it joins. A rule like 'strip a leading letter' applied to any file would
    eventually fold the A and B columns of a real chart onto one patch."""
    from deltachrome.reference.sample_ids import reconcile_sample_ids
    from deltachrome.targets.registry import load_target

    target = load_target("colorchecker_sg_v1")
    assert "sample_id_aliases" not in target.reference_data
    reference = _measurement_named(f"X{p.patch_id}" for p in target.patches)

    reconciled, note = reconcile_sample_ids(reference, target)

    assert note is None
    assert reconciled is reference


# -- a file can name every patch right and describe a different chart ------


def test_a_ragged_printed_layout_is_declared_per_patch_not_generated():
    """The ISA FADGI - ISO 19264 numbers its patches 1-168 in reading order and
    *prints* a ragged A-N by 1-16 grid around its central modules -- 14 columns
    holding 9, 12, 13 or 16 patches. No template generates that, so each patch
    declares its own printed label and the reconciler prefers it."""
    from deltachrome.reference.sample_ids import reconcile_sample_ids
    from deltachrome.targets.registry import load_target

    target = load_target("isa_fadgi_19264_commercial_v1")
    printed = {p.printed_id for p in target.patches}
    assert len(printed) == 168 and None not in printed

    reference = _measurement_named(printed)
    reconciled, note = reconcile_sample_ids(reference, target)

    assert {p.patch_id for p in target.patches} <= set(reconciled.samples)
    assert note and "the labels the chart prints" in note


def test_the_early_version_declares_no_printed_labels():
    """It numbers its patches on the card, so there is nothing to reconcile --
    and a label invented for it would be a claim about a chart nobody checked."""
    from deltachrome.targets.registry import load_target

    assert all(p.printed_id is None
               for p in load_target("isa_fadgi_19264_v1").patches)


def test_a_reference_describing_another_version_of_the_chart_is_flagged():
    """The case this check was written for, and it is not hypothetical.

    ISA revised the FADGI - ISO 19264 after the unit OpenDICE published, changing
    a block of patches below the central modules. Six published measurements of
    the revised chart agree with each other to 0.2 dE2000 and disagree with this
    definition's nominals on 33 patches by a mean of 37.8 -- and the sample
    capture, measured, sits 1.49 from the nominals and 37.8 from the published
    files, so both sets of numbers are right about different objects.

    Every earlier guard misses it: the patch ids all match, and the chart's code
    identifies the unit correctly.
    """
    import numpy as np

    from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement
    from deltachrome.reference.crosscheck import crosscheck_against_nominal
    from deltachrome.reference import read_reference
    from deltachrome.targets.registry import load_target

    target = load_target("isa_fadgi_19264_commercial_v1")
    nominal = read_reference(target.nominal_reference_path)

    # A file that agrees everywhere except a block, as the revised chart does.
    samples, changed = {}, 0
    for patch in target.patches:
        ref = nominal.samples.get(patch.patch_id)
        if ref is None or ref.lab is None:
            continue
        lab = list(ref.lab)
        if changed < 20:
            lab[0] = 100.0 - lab[0]           # a wholly different colour
            lab[1], lab[2] = -lab[1], -lab[2]
            changed += 1
        samples[patch.patch_id] = PatchMeasurement(
            sample_name=patch.patch_id, lab=tuple(lab)
        )
    supplied = MeasurementSet(samples=samples, metadata={}, fields=[])

    warning = crosscheck_against_nominal(supplied, target)
    assert warning is not None
    # It names all three causes and picks none, because it cannot tell them
    # apart -- another chart's file, another batch, or a revision the definition
    # predates. Two of the three were found in the published corpus.
    assert "another chart's measurement" in warning
    assert "BATCH" in warning
    assert "revision" in warning


def test_an_ordinary_per_unit_file_is_not_flagged():
    """The check must be silent on the case it will meet most often: a genuine
    measurement of one unit, differing from a batch mean by ordinary spread."""
    import numpy as np

    from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement
    from deltachrome.reference.crosscheck import crosscheck_against_nominal
    from deltachrome.reference import read_reference
    from deltachrome.targets.registry import load_target

    target = load_target("isa_golden_thread_dlt_v1")
    nominal = read_reference(target.nominal_reference_path)
    rng = np.random.default_rng(0)
    samples = {
        pid: PatchMeasurement(
            sample_name=pid,
            lab=tuple(np.array(s.lab) + rng.normal(0, 0.4, 3)),
        )
        for pid, s in nominal.samples.items() if s.lab
    }
    supplied = MeasurementSet(samples=samples, metadata={}, fields=[])

    assert crosscheck_against_nominal(supplied, target) is None


def test_a_refusal_does_not_claim_a_code_was_read_when_none_was():
    """Wording, and it matters: the 404 text was written for a label decoded off
    a capture, where "the code was read correctly" usefully says the pairing is
    not at fault. Reached by typing a serial by hand it sends somebody looking
    for a code that was never involved -- and a mistyped serial is the likeliest
    cause of that very 404."""
    import urllib.error

    from deltachrome.reference.openqualia import FROM_CHART, FROM_SERIAL, label_for_serial
    from deltachrome.targets.registry import load_target

    typed = label_for_serial(load_target("isa_golden_thread_dlt_v1"), "D99999")
    assert typed.source == FROM_SERIAL
    assert parse_label(LABEL).source == FROM_CHART

    def refuse(label):
        error = urllib.error.HTTPError(label.url, 404, "Not Found", {}, None)
        with pytest.raises(ReferenceError) as excinfo:
            fetch_measurement(label, opener=_Opener(error))
        return str(excinfo.value)

    from_serial = refuse(typed)
    assert "Check the serial" in from_serial
    assert "code was read" not in from_serial

    from_chart = refuse(parse_label(LABEL))
    assert "code was read correctly" in from_chart


# -- a file that names a different unit than the one asked for -------------


def test_a_file_naming_another_unit_is_reported():
    """The one genuine defect found in about 140 published measurements read from
    the host: asking for NGT2 unit DT-AR-2020013 returns a file whose SERIAL says
    DT-AR-2020014, and it is a different measurement from the real 014 file.

    This compares identifiers rather than colours, and that is the point. Two
    other files that looked anomalous by colour turned out to be perfectly good
    measurements of later production batches -- so a colour-based outlier test
    would have raised six false alarms in the same corpus while missing this.
    """
    from deltachrome.reference.openqualia import FetchedMeasurement

    label = parse_label(
        "https://targets.digitaltransitions.com/measurements"
        "?Manufacturer=DT&TargetType=DTNGT2&TargetID=DT-AR-2020013"
    )
    fetched = FetchedMeasurement(
        label=label,
        content=MEASUREMENT.replace(b"DT-AR-2020041", b"DT-AR-2020014"),
        filename="m.oqm.txt",
        final_url=label.url,
    )
    assert fetched.stated_serial == "DT-AR-2020014"
    warning = fetched.serial_disagreement
    assert warning is not None
    assert "DT-AR-2020013" in warning and "DT-AR-2020014" in warning


def test_a_file_naming_the_unit_asked_for_is_silent():
    from deltachrome.reference.openqualia import FetchedMeasurement

    label = parse_label(LABEL)
    fetched = FetchedMeasurement(label=label, content=MEASUREMENT,
                                 filename="m.oqm.txt", final_url=label.url)
    assert fetched.stated_serial == "DT-AR-2020041"
    assert fetched.serial_disagreement is None


def test_the_serial_comparison_ignores_punctuation_and_case():
    """A serial is written by people and by instruments. `CCSG-21` and `ccsg21`
    are the same unit, and warning about that would be noise."""
    from deltachrome.reference.openqualia import FetchedMeasurement

    label = parse_label(
        "https://x.org/m?Manufacturer=CB&TargetType=CCDSG&TargetID=CCSG-21"
    )
    fetched = FetchedMeasurement(
        label=label,
        content=MEASUREMENT.replace(b'SERIAL\t"DT-AR-2020041"', b'SERIAL\t"ccsg21"'),
        filename="m.oqm.txt",
        final_url=label.url,
    )
    assert fetched.serial_disagreement is None


def test_a_file_stating_no_serial_is_not_second_guessed():
    """i1Profiler writes no serial at all -- the SG measurement in this repository
    is exactly that. Absence is not disagreement."""
    from deltachrome.reference.openqualia import FetchedMeasurement

    label = parse_label(LABEL)
    fetched = FetchedMeasurement(
        label=label,
        content=MEASUREMENT.replace(b'SERIAL\t"DT-AR-2020041"\n', b""),
        filename="m.oqm.txt",
        final_url=label.url,
    )
    assert fetched.stated_serial is None
    assert fetched.serial_disagreement is None


def test_the_caveat_is_kept_with_the_file(bank):
    """A serial disagreement matters every time the file is paired, not only on
    the day it arrived, so it lives in the manifest rather than in a dialog."""
    entry = bank.add_measurement(
        MEASUREMENT, "m.oqm.txt", bind="DT-AR-2020013", note="names another unit"
    )
    assert bank.note_of(entry.name) == "names another unit"
    bank.remove_file(entry.name)
    assert bank.note_of(entry.name) is None


def test_the_lookup_goes_to_the_host_the_target_declares():
    """OpenQualia has no central index, so where a serial is looked up is part
    of the model's identity rather than an application default."""
    from types import SimpleNamespace

    from deltachrome.reference.openqualia import label_for_serial

    target = SimpleNamespace(openqualia={
        "host": "https://measurements.example.org/oq",
        "manufacturer": "XR", "target_type": "THING",
    })
    label = label_for_serial(target, "U123")
    assert label is not None and label.host == "measurements.example.org"
    assert label.url.startswith("https://measurements.example.org/oq?")


@pytest.mark.parametrize("host", [None, "", "http://insecure.example.org/oq",
                                  "https://x.example.org/oq?Manufacturer=XR"])
def test_an_identity_without_a_usable_host_offers_no_lookup(host):
    """A manufacturer and target type with nowhere declared to look is not a
    lookup at somebody else's server."""
    from types import SimpleNamespace

    from deltachrome.reference.openqualia import label_for_serial, publishes_measurements

    identity = {"manufacturer": "XR", "target_type": "THING"}
    if host is not None:
        identity["host"] = host
    target = SimpleNamespace(openqualia=identity)
    assert not publishes_measurements(target)
    assert label_for_serial(target, "U123") is None


def test_every_shipped_identity_names_its_host():
    """A definition that declares a manufacturer and type but forgot the host
    would silently stop being offered a lookup."""
    from deltachrome.reference.openqualia import publishes_measurements
    from deltachrome.targets.registry import available_targets, load_target

    for target_id in available_targets():
        target = load_target(target_id)
        if target.openqualia:
            assert publishes_measurements(target), target_id
