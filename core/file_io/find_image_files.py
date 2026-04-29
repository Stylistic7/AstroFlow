from pathlib import Path


def find_image_files(path: str, pattern: str = "*") -> list:
    """Find raw FITS frames under `path` for ingestion.

    Returns only raw frames — AstroFlow does its own stacking and never
    consumes pre-stacked masters as if they were single subframes:

    - `.xisf` is excluded entirely. AstroFlow's own outputs are XISF;
      including them as inputs would re-ingest masters that the
      pipeline just produced.
    - Files whose basename starts with `master` (case-insensitive) are
      excluded. These are the convention for externally pre-stacked
      calibration masters (`MasterBias*.fit`, `MasterDark*.fit`, etc.)
      and shouldn't be auto-associated with raw light frames; that
      should be a manual decision (see saliency-aware calibration
      matching, future work).
    """
    return sorted(
        [
            str(p.absolute())
            for p in Path(path).rglob(pattern)
            if p.suffix.lower() in [".fit", ".fits"]
            and not p.name.lower().startswith("master")
            and not any(part.startswith(".") for part in p.parts)
        ]
    )
