"""Paths as the Windows app's Cygwin programs read them (solve-field and
astrometry-engine, app/windows/build_solve_field.sh). Cygwin opens files by
a `C:/...` path, but not directories (its index folder failed to open), so
they get its own form: `/cygdrive/c/...`."""
import re
import sys
from pathlib import PureWindowsPath

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def is_windows_path(text: str) -> bool:
    return bool(_WINDOWS_PATH.match(text))


def cygwin_path(path) -> str:
    """`C:\\x\\y` as `/cygdrive/c/x/y` on Windows; any other path unchanged."""
    text = str(path)
    if sys.platform != "win32" or not is_windows_path(text):
        return text
    p = PureWindowsPath(text)
    return "/".join(["/cygdrive", p.drive[0].lower(), *p.parts[1:]])
