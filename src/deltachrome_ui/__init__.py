"""Deltachrome's desktop frontend.

A separate top-level package, not a subpackage of :mod:`deltachrome`, so that the
engine's "no UI code lives below this package" commitment is a fact about the
filesystem rather than an intention. ``tests/test_ui_boundary.py`` enforces it.

Nothing here computes a measurement. Every number this package displays came out
of the engine, and every verdict it renders came out of
:class:`deltachrome.profiles.evaluator.ProfileEvaluation`. The one thing the UI is
allowed to decide is *how* to show them -- and the rules for that live in
:mod:`deltachrome_ui.status`, because flattening a qualified verdict into a green
tick is the failure mode this frontend has to avoid.
"""

__all__ = ["__version__"]

from deltachrome import __version__
