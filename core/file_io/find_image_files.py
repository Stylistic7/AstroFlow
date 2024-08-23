import glob
from pathlib import Path


def find_image_files(path: str, pattern: str = '*') -> list:
  return sorted([
      str(p.absolute())
      for p in Path(path).rglob(pattern)
      if p.suffix.lower() in ['.fit', '.fits', '.xisf']
  ])
