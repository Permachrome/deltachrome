"""Deltachrome — cultural heritage imaging QC.

The engine is a pure library: no UI code lives below this package. Everything a
frontend needs is reachable through :mod:`deltachrome.pipeline`.
"""

from deltachrome.errors import DeltachromeError

#: PEP 440. 0.9.0b1 is the first beta of what will be 0.9.0; later betas count
#: up (b2, b3), a release candidate is 0.9.0rc1, and the release drops the
#: suffix. The record carries it as environment.deltachrome_version.
__version__ = "0.9.0b2"
__all__ = ["DeltachromeError", "__version__"]
