"""The pipeline stamp: which pipeline made a master, independent of whose
data went in.

Two users' masters can be merged into one mosaic only when the same pipeline
made both (a mosaic's level matching and feathering assume one background
model, one pedestal, one resampling kernel, one flux scale — all pipeline
properties). A call hash cannot say so: it folds the inputs, so no two users
ever share one, and the deployed code version moves on every commit, UI
included. The stamp is what "same pipeline" means, and it is written into
every produced image's header as `PIPEHASH` (`products.registry._stamp_header`):

    PIPEHASH = hash(science code hash | producing op | primary input's PIPEHASH)

  * The SCIENCE CODE HASH is the AST-normalized hash of every module that can
    touch a master's pixels (`SCIENCE_PACKAGES`), so an edit to a helper the
    op calls moves it — a per-op source hash would not (the callee-edit rule),
    and across users an invisible helper edit is exactly the inconsistency the
    stamp exists to catch. Comments, docstrings and formatting do not move it;
    nor does anything downstream of the master (finishes, the viewer, the
    installers), so a UI release splits nobody.
  * The PRODUCING OP and the PRIMARY INPUT's stamp carry the op chain, so a
    Siril-integrated master and a WBPP-integrated one stamp differently
    though the post-stack calibration is one op. Only the input bound to a
    NAMED parameter counts: `*cal_masters` are data (whether a user has darks
    is not a pipeline property), and raw frames carry no stamp.

Nothing user-specific enters: no paths, no data hashes, no parameters that
differ by user (today's science parameters are code defaults; a user-editable
one would be folded in here, by value, the day it exists).

Python version: the AST serialization is the same across the versions this
code runs on (3.12 in the app bundles, 3.13 in the dev venv) for ordinary
code; `hashing/test_hash_stability.py` is the tripwire.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import xxhash

from hashing.structural_hash import structural_text_hash

PIPELINE_STAMP_KEY = "PIPEHASH"

# Relative to `core/`: the packages and modules that can touch a master's
# pixels, from the raw frames up to the post-stack-calibrated master. NOT
# mosaics/ or targets/ (downstream of the master — each member merges on
# its own machine), not clustering/ (membership, not pixels), not the UI.
SCIENCE_PACKAGES = (
    "attributes",
    "calibration",
    "decisions/settings.py",
    "file_io",
    "geometry",
    "metadata",
    "photometry",
    "pipelines",
    "pix",
    "plate_solving.py",
    "quarantining",
    "stacks",
)

_ROOT = Path(__file__).resolve().parents[1]


def _science_files(root: Path = _ROOT) -> list[Path]:
    out: list[Path] = []
    for rel in SCIENCE_PACKAGES:
        p = root / rel
        if p.is_file():
            out.append(p)
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix not in (".py", ".js", ".xml"):
                continue
            parts = f.relative_to(p).parts
            if any(d in ("__pycache__", "testdata") for d in parts[:-1]):
                continue
            if f.name.startswith("test_") or f.name.startswith("conftest"):
                continue
            out.append(f)
    return sorted(set(out))


def science_code_hash_of(root: Path) -> str:
    """The science code hash of the tree at `root` (tests pass a temp tree)."""
    h = xxhash.xxh64()
    for f in _science_files(root):
        text = f.read_text(encoding="utf-8", errors="replace")
        if f.suffix == ".py":
            try:
                digest = structural_text_hash(text)
            except SyntaxError:
                digest = xxhash.xxh64(text).hexdigest()
        else:
            digest = xxhash.xxh64(text).hexdigest()
        h.update(f"{f.relative_to(root).as_posix()}|{digest}\n")
    return h.hexdigest()


@lru_cache(maxsize=1)
def science_code_hash() -> str:
    """The science code hash of the running code, computed once per process."""
    return science_code_hash_of(_ROOT)


def pipeline_stamp(op_name: str, upstream: str | None, code: str | None = None) -> str:
    """The stamp an op writes on its output: the science code, the op, and
    the stamp of its primary input (None for an op whose primary input is raw
    data, or has no image input at all)."""
    code = science_code_hash() if code is None else code
    return xxhash.xxh64(f"{code}|{op_name}|{upstream or ''}").hexdigest()


def stamp_of(header: dict | None) -> str | None:
    """The pipeline stamp a header carries, or None."""
    value = (header or {}).get(PIPELINE_STAMP_KEY)
    return str(value) if value not in (None, "") else None
