"""Exception hierarchy for Deltachrome.

Every failure mode the engine can hit gets a named type. The rule from the
project primer is "fail loudly": a QC tool that silently substitutes a default
is worse than one that refuses to run, because the operator cannot tell a
measurement from a guess.
"""

from __future__ import annotations


class DeltachromeError(Exception):
    """Base class for every error raised by this package."""


# -- Reference data ---------------------------------------------------------


class ReferenceError(DeltachromeError):
    """Problem with per-unit reference (measurement) data."""


class CGATSParseError(ReferenceError):
    """The CGATS file is malformed or uses a structure we cannot interpret."""


class DuplicateSampleError(ReferenceError):
    """A SAMPLE_NAME appears more than once with conflicting values.

    Never resolved silently. The primer records a real instance of this in the
    project's own reference file (K8 listed twice, the second entry a
    mislabelled K9), which is exactly the class of error that would otherwise
    corrupt a pass/fail result invisibly.
    """


class MissingReferenceError(ReferenceError):
    """No per-unit reference file was supplied, or it does not cover the target."""


# -- Imaging ----------------------------------------------------------------


class ImageError(DeltachromeError):
    """Problem loading or interpreting the input image."""


class MissingICCProfileError(ImageError):
    """The image carries no embedded ICC profile.

    Hard refusal for v1. Master files in this domain are routinely Adobe RGB,
    ProPhoto or ECI RGB; assuming sRGB produces plausible-looking but wrong
    Delta E values, which is the single most dangerous failure this tool has.
    """


class UnsupportedICCProfileError(ImageError):
    """The embedded ICC profile is not one we can transform at full precision."""


# -- Geometry / sampling ----------------------------------------------------


class GeometryError(DeltachromeError):
    """Fiducial placement or homography could not be resolved."""


# -- Targets and profiles ---------------------------------------------------


class TargetError(DeltachromeError):
    """Problem with a target definition."""


class ProfileError(DeltachromeError):
    """Problem with an analysis profile definition."""


class CapabilityError(ProfileError):
    """A profile requires a capability the target does not declare."""


# -- Building an ICC profile -------------------------------------------------


class ProfilingError(DeltachromeError):
    """A camera profile could not be built from this capture.

    Not a :class:`ProfileError`, which is about the *analysis* profiles a run is
    judged against. The two share a word and nothing else.
    """
