"""Small text helpers shared by everything that writes a line for a person.

One function so far, and it earns a module because both the run and the session
compose the same sentences: ``12 of 17 metric(s) judged`` is what a program
writes when nobody decided whether there was one of them or twelve. The plural
is knowable -- the count is right there -- so it is decided here rather than
deferred onto the reader.

Presentation only. Nothing in a record goes through it.
"""

from __future__ import annotations


def plural(count: int, singular: str, many: str | None = None) -> str:
    """``1 metric``, ``12 metrics`` -- the number and its noun, agreeing.

    ``many`` is for the nouns English does not pluralize with an ``s``; every
    caller here uses the default.
    """
    word = singular if abs(count) == 1 else (many or f"{singular}s")
    return f"{count} {word}"
