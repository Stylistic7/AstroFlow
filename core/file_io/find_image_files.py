from pathlib import Path


def is_external_master_name(name: str) -> bool:
    """The naming convention for a user-supplied pre-stacked master
    (`MasterBias*.fit`). One definition, so every consumer draws the same
    line between a raw subframe and someone's finished master."""
    return name.lower().startswith("master")


def find_image_files(
    path: str, pattern: str = "*", *, include_masters: bool = False,
) -> list:
    """Find FITS frames under `path` for ingestion.

    Default (`include_masters=False`) returns only raw frames, excluding
    pre-stacked masters that AstroFlow never consumes as single subframes:

    - `.xisf` is always excluded. AstroFlow's own outputs are XISF;
      including them as inputs would re-ingest masters that the
      pipeline just produced.
    - With `include_masters=False`, files whose basename starts with
      `master` (case-insensitive) are also excluded — the convention
      for externally pre-stacked masters (`MasterBias*.fit`, etc.).
    - With `include_masters=True`, the master* exclusion is dropped.
      Used by the metadata sweep so external masters' `ImageMetadata`
      lands in the diskcache, which makes the trace cache key
      (which embeds `metadata_ref.content_hash`) invalidate when a
      user drops in a fresh master.
    """
    return sorted(
        [
            str(p.absolute())
            for p in Path(path).rglob(pattern)
            if p.suffix.lower() in [".fit", ".fits"]
            and (include_masters or not is_external_master_name(p.name))
            and not any(part.startswith(".") for part in p.parts)
        ]
    )
