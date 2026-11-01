"""Nothing may sit between the edge profile and its transform.

The SFR curve is ``|X(f)| / X(0)``. Everything the DFT sees is the windowed LSF, and
every point is divided by that array's own DC value -- so any step that alters the
profile beforehand moves the divisor, and the divisor scales the entire curve. Two
such steps used to sit there:

* ``detrend``, which levelled a slope common to both plateaux of the ESF, and
* a subtraction of the median of the LSF tails, immediately before the window.

Neither is in sfrmat3, in Imatest's published step list, or in ISO 12233's normative
text, and together they lifted maximum modulation to 1.105 on a capture that both
iQ-Analyzer and OpenDICE read as 1.00.

**Why this module needs a real edge profile.** The suite already covered SFR heavily
and none of it caught this, because ``synthetic_edge`` has perfectly flat plateaux:
the median of its LSF tails is exactly zero, so the subtraction was a no-op and the
tests were bit-identical either way. The property under test only exists on a
profile with real plateau structure, so one is committed alongside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deltachrome.imaging.sfr import compute_sfr

FIXTURE = Path(__file__).parent / "fixtures" / "real_esf_utt_br.txt"


@pytest.fixture(scope="module")
def real_esf() -> np.ndarray:
    """One slanted edge of a UTT A3, linearised and binned. See the file's header."""
    return np.loadtxt(FIXTURE)


def mtf_of(esf: np.ndarray, bin_width: float = 0.25, *, subtract_lsf_baseline=False):
    """The tail of the method, with the removed step available to reinstate.

    Written out here rather than called through ``compute_sfr`` so the test can put
    the deleted line back and show what it did.
    """
    lsf = np.gradient(esf)
    if subtract_lsf_baseline:
        n = len(lsf)
        lsf = lsf - np.median(np.concatenate([lsf[: n // 8], lsf[-n // 8 :]]))
    peak = int(np.argmax(np.abs(lsf)))
    window = np.roll(np.hamming(len(lsf)), -(len(lsf) // 2 - peak))
    spectrum = np.abs(np.fft.rfft(lsf * window))
    frequency = np.fft.rfftfreq(len(lsf), d=bin_width)
    return frequency, spectrum / spectrum[0]


# -- the property ----------------------------------------------------------


def test_a_real_edge_does_not_exceed_one(real_esf):
    """An unsharpened edge peaks at DC, where the curve is normalised, and nowhere
    else. This is the number FADGI and Metamorfoze cap, and the one every other
    implementation returns as exactly 1.00."""
    frequency, mtf = mtf_of(real_esf)
    band = frequency <= 0.5
    assert mtf[band].max() == pytest.approx(1.0, abs=1e-9)
    assert frequency[band][int(np.argmax(mtf[band]))] == 0.0


def test_reinstating_the_subtraction_breaks_it(real_esf):
    """The guard rail. If someone puts the line back, this says what it costs."""
    _, honest = mtf_of(real_esf)
    _, with_line = mtf_of(real_esf, subtract_lsf_baseline=True)
    assert with_line.max() > 1.05, (
        "the fixture no longer reproduces the defect it exists to pin"
    )
    # ...and it is a scale factor, not a feature: the ratio is flat across the band.
    band = slice(20, 200)
    ratio = with_line[band] / honest[band]
    assert ratio.std() < 1e-4
    assert ratio.mean() > 1.10


def test_detrend_is_off_by_default(real_esf):
    """It is ISO 12233 edition 4's *informative* correction, in a cruder form than
    the standard's, and it is what keeps this edge off 1.000."""
    from inspect import signature

    assert signature(compute_sfr).parameters["detrend"].default is False


def test_detrend_moves_a_real_curve(real_esf):
    """Not inert, which is why its default matters.

    Whether it lifts a *particular* edge past 1.0 is a property of that edge -- on
    the whole UTT field it took the worst from 1.000 to 1.012, and this is not that
    edge. What holds for any real profile is that it changes the answer, so leaving
    it on is a choice with consequences rather than a free safety measure.
    """
    n = len(real_esf)
    x = np.arange(n, dtype=np.float64)
    left = float(np.polyfit(x[: n // 10], real_esf[: n // 10], 1)[0])
    right = float(np.polyfit(x[-(n // 10) :], real_esf[-(n // 10) :], 1)[0])
    assert left * right > 0, "this edge's plateaux slope the same way"
    common = min(abs(left), abs(right)) * np.sign(left)

    _, plain = mtf_of(real_esf)
    _, detrended = mtf_of(real_esf - common * (x - x.mean()))
    assert not np.allclose(plain, detrended, atol=1e-6)


# -- and the synthetic case still holds, which is why it hid ----------------


def test_a_flat_plateau_makes_the_removed_step_a_no_op():
    """Why the existing suite was silent: on a synthetic edge the two paths agree
    exactly, so no amount of coverage over synthetic fixtures could have found it."""
    from fixtures_sfr import synthetic_esf

    esf = synthetic_esf(sigma=0.9)
    _, without = mtf_of(esf)
    _, with_line = mtf_of(esf, subtract_lsf_baseline=True)
    assert np.allclose(without, with_line, atol=1e-12)


def test_the_committed_profile_is_a_real_one(real_esf):
    """A guard on the fixture itself: flat plateaux would silently defeat this whole
    module, exactly as they defeated the rest of the suite."""
    lsf = np.gradient(real_esf)
    n = len(lsf)
    tails = np.concatenate([lsf[: n // 8], lsf[-n // 8 :]])
    assert abs(float(np.median(tails))) > 1e-5
    assert float(tails.std()) > 0.0


def test_no_shipped_profile_pins_the_window():
    """FADGI states thresholds and points at ISO 12233 for the method; it does not
    specify a window function. Fifty-five profiles used to carry `hamming_window:
    true` anyway -- our own generator's placeholder, not anything read from the
    tables -- which would have frozen them on the 2017 edition's window silently.
    """
    from deltachrome.profiles.catalogue import available_profiles
    from deltachrome.profiles.schema import load_profile

    pinned = [
        (profile_id, spec.metric_id)
        for profile_id in available_profiles()
        for spec in load_profile(profile_id).metrics
        if "hamming_window" in spec.algorithm or "window" in spec.algorithm
    ]
    assert pinned == []
