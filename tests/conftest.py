"""What every test in this suite gets whether it asks or not.

There is one thing here and it is not a convenience. The frontend remembers
window preferences -- which folder the run library is, which columns the table
shows -- in a settings file under the application's own home, and until this
existed that home was *this machine's* ``~/.deltachrome`` for tests as much as
for the application. So a test exercising the settings screen wrote a pytest
temporary directory into the operator's own remembered run library, and by the
time they next opened the application that directory was gone. The report was
"the Run Library path differs each time it starts", and it was the test suite
saying so.

Redirecting the home is the fix rather than patching each writer, because the
failure is not that any one test was careless: it is that a suite exercising a
component which remembers things had nowhere to remember them except the place a
person's own answers live. One directory per session, thrown away afterwards.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def application_home(tmp_path_factory):
    """Point ``DELTACHROME_HOME`` at a disposable folder for the whole session.

    Session-scoped and automatic: a per-test home would be correct too and far
    slower, and nothing here is asserting on state carried between tests -- what
    matters is only that it is not the operator's.

    Set through the environment rather than by patching :mod:`deltachrome_ui.paths`
    because that is the resolution order the application documents, and because a
    monkeypatched module attribute would not reach a subprocess.
    """
    home = tmp_path_factory.mktemp("deltachrome-home")
    previous = os.environ.get("DELTACHROME_HOME")
    os.environ["DELTACHROME_HOME"] = str(home)
    yield home
    if previous is None:
        os.environ.pop("DELTACHROME_HOME", None)
    else:
        os.environ["DELTACHROME_HOME"] = previous
