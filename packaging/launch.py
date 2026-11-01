"""The bundled application's entry point.

Opens the desktop frontend. Given ``--cli`` as its first argument it runs the
command-line tool instead, with the remaining arguments -- which is how a built
bundle is tested: the same analysis run inside the bundle and in a development
environment must give the same record.

    Deltachrome.app/Contents/MacOS/Deltachrome --cli analyze CAPTURE ...
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    # A windowed Windows executable starts with no standard streams unless its
    # caller redirected them. Borrow the console it was started from, so the
    # command line prints where it was typed.
    if sys.platform == "win32" and sys.stdout is None:
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8")
    from deltachrome.cli import main as cli_main
    raise SystemExit(cli_main(sys.argv[2:]))

from deltachrome_ui.app import main

raise SystemExit(main())
