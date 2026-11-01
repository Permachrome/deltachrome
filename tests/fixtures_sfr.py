"""A synthetic edge profile, for the tests that need one alongside the real one."""
from __future__ import annotations

import numpy as np
from scipy.special import erf


def synthetic_esf(sigma: float = 0.9, half_width: float = 50.0,
                  bin_width: float = 0.25, low: float = 0.05, high: float = 0.95):
    """A Gaussian-blurred step, sampled on the same grid a real ESF lands on.

    Perfectly flat plateaux, which is the whole point: it is what a synthetic
    fixture cannot express about a real capture.
    """
    x = np.arange(-half_width, half_width, bin_width)
    return low + (high - low) * 0.5 * (1 + erf(x / (sigma * np.sqrt(2))))
