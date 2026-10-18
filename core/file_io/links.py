import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    """`dst` reads as `src` without copying when the platform allows: a
    symlink, else a hard link (Windows refuses symlinks outside Developer
    Mode; a hard link needs the same volume), else a copy."""
    src = Path(src).resolve()
    for link in (os.symlink, os.link):
        try:
            link(src, dst)
            return
        except OSError:
            continue
    shutil.copyfile(src, dst)
