from __future__ import annotations
from dataclasses import MISSING, dataclass, fields
from typing import NewType
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

from hashing import hash_list_int

if TYPE_CHECKING:
    from plate_solving import PlateSolution


FrameHash = NewType("FrameHash", str)


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    raise TypeError(
        f"Old ImageMetadata pickle is missing the required (no-default) "
        f"field {f.name!r}; cannot forward-migrate without manual rebuild."
    )


@dataclass(frozen=True, slots=True)
class ImageStatistics:
    """Brightness distribution + sky-noise stats over the full image.

    Statistics live in the data's NATIVE units (uint16 ADU for raw
    frames; whatever float scaling SFC / MGC / pedestal produces for
    post-processed masters). `histogram_range` carries the (min, max)
    used for the histogram bins so consumers can map data units to
    display units without re-reading the image.

    Feeds the web-UI auto-stretch + structure-emphasis optimizers,
    the panel comparison view (goal #2), and per-mosaic QA metrics.

    Frozen + slotted to match `ImageMetadata`'s
    pickle-via-sidecar semantics.
    """
    # Fixed-size 1024-bin histogram over the data's actual range.
    # Sufficient for max-entropy GHS optimization and any histogram
    # visualization the UI could want.
    histogram_bins: int                       # always 1024
    histogram_range: tuple[float, float]      # (min, max) of data
    histogram_counts: tuple[int, ...]         # 1024 ints

    # Percentile cuts at common operating points, in data units.
    p0_1: float
    p1: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    p99_9: float

    # Sky-only statistics: median + RMS of the lower 25 % of pixels.
    # Same sky-pixel selection rule the pedestal step uses
    # (`stacks/post_processing/background_pedestal.py`); this lets the
    # cross-panel seam metric and the pedestal-target consistency
    # check read the same numbers without recomputing.
    sky_median: float
    sky_rms: float

    # Clipping fractions: pixels at the top / bottom 0.5 % of the
    # data's actual range. Lets the UI warn "this image has 0.2 % of
    # pixels saturated; consider a shorter exposure."
    clipped_low_fraction: float
    clipped_high_fraction: float

    # Bulk descriptive stats in data units.
    min_value: float
    max_value: float
    mean_value: float

    # Median absolute deviation across all (non-sentinel) pixels.
    # `mad = median(|pixel − median(all_pixels)|)`. Surfaced here so
    # the in-browser auto-stretch port can compute the STF shadow-clip
    # term `-2.8 · 1.4826 · MAD` without needing to re-read pixels.
    # For stacked + pedestaled masters the sky noise sits below the
    # histogram's bin width, so an in-browser MAD derived from
    # `histogram_counts` collapses to 0; the value computed here on the
    # raw pixel array preserves sub-bin precision. Defaulted for
    # forward-migration of sidecars written before the field existed.
    mad: float = 0.0

    def __setstate__(self, state: list) -> None:
        # Forward-compatible unpickling for slotted frozen dataclasses.
        # Old pickles missing trailing slots (a new optional field added
        # later) get padded with that field's declared default. Without
        # this, the auto-generated `_dataclass_setstate` leaves the new
        # slot unset; a subsequent re-pickle then crashes inside
        # `_dataclass_getstate` on `getattr(self, "mad")`.
        field_list = fields(type(self))
        if len(state) < len(field_list):
            missing = field_list[len(state):]
            state = list(state) + [_field_default(f) for f in missing]
        for f, value in zip(field_list, state):
            object.__setattr__(self, f.name, value)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    content_hash: str
    stable_hash: str
    filename: Path
    header: dict[str, Any]

    width_pixels: int
    height_pixels: int
    channels: int
    dtype: np.dtype

    plate_solution: PlateSolution | None = None

    # Stacker-asserted effective integration time AFTER the stacker's
    # own rejection pass. Populated when a readable source is present;
    # None for raw subframes (where header EXPTIME is the truth) and
    # for masters from stackers we don't yet know how to read.
    #   * PixInsight masters: XISF Property `PCL:TotalExposureTime`
    #   * (future) Siril, OSS, etc.: per-stacker readers added as needed
    # Read via file_io.master_integration.read_master_integration_seconds.
    master_integration_s: float | None = None

    # Brightness distribution and sky-noise statistics; populated by
    # `compute_image_metadata` from a one-time pixel scan and cached in
    # the sidecar. None on old sidecars (forward-compatible via
    # `__setstate__`) and on files where the pixel read failed.
    statistics: ImageStatistics | None = None

    # MRS background noise floor of a stacked master, read inline from the
    # FITS `NOISE00` keyword ImageIntegration writes (bit-identical to
    # PixInsight's `finalNoiseEstimate`). The absolute, signal-agnostic
    # depth measure driving the mosaic uniformity overlay. None for raw
    # subframes and stackers that don't record it (e.g. WBPP
    # FastIntegration). MUST stay the last field so `__setstate__`'s
    # trailing-slot padding fills it as None on older sidecars.
    noise_estimate: float | None = None

    def __setstate__(self, state: list) -> None:
        # Backward- AND forward-compatible unpickling for slotted dataclasses.
        #
        # Forward: an old pickle missing a trailing slot (a new optional
        # field was added later) is padded with the declared default.
        #
        # Backward: when `defect_candidates` was lifted off ImageMetadata
        # into a separate ContextRef, that field — which used to sit
        # between `plate_solution` and `master_integration_s` — was
        # removed. Pickles of that older shape have one extra entry at
        # the legacy position; we strip it so the surviving fields land
        # in the right slots. Without this targeted fix, naïve zip
        # alignment would mis-assign `master_integration_s` to the
        # discarded defect_candidates dict.
        field_list = fields(type(self))
        names = [f.name for f in field_list]
        if len(state) == len(field_list) + 1 and "defect_candidates" not in names:
            legacy_idx = names.index("master_integration_s")
            state = list(state)
            del state[legacy_idx]  # drop the orphaned defect_candidates slot
        if len(state) < len(field_list):
            missing = field_list[len(state):]
            state = list(state) + [_field_default(f) for f in missing]
        for f, value in zip(field_list, state):
            object.__setattr__(self, f.name, value)

    @property
    def pixels(self) -> int:
        return self.width_pixels * self.height_pixels

    @property
    def bytes_per_pixel(self) -> int:
        return self.channels * self.dtype.itemsize

    @property
    def total_bytes(self) -> int:
        return self.pixels * self.bytes_per_pixel

    @property
    def megapixels(self) -> float:
        return self.pixels / 1e6


class MetadataDict(dict[FrameHash, ImageMetadata]):
    def __hash__(self):
        return hash_list_int(sorted(self.keys()))

    def __str__(self):
        return f"MetadataDict({len(self)})"


class DefectTable:
    """Per-frame hot-pixel defect candidates as three flat numpy arrays:
    `frames` (sorted frame hashes), `offsets` (int64, n + 1) and `pixels`
    (int32 flat indices; frame i's are `pixels[offsets[i]:offsets[i+1]]`,
    ascending). Only the pixel indices are kept: the clustering never reads
    a defect's strength.

    Why arrays (2026-10-22): as `dict[FrameHash, dict[int, float]]` the
    library's defects were ~16M Python ints and floats, ~900 MB of heap that
    every reader -- the ContextActor storing it, the worker clustering it --
    unpickled into a private copy. Plain-dtype numpy arrays are what Ray
    shares zero-copy: one copy in the object store per machine, mapped
    read-only by every reader (measured on Ray 2.47: a 400 MB array read by
    two workers added 0.02 MB each when nested in an object). The arrays are
    therefore read-only in a reader; nothing here mutates them.

    Frames absent from the table have no defect candidates. The identity is
    the bytes of the three arrays (`content_hash`), so `ContextRef.store_large`
    keys it without walking Python objects."""

    __slots__ = ("frames", "offsets", "pixels", "_content_hash")

    def __init__(self, frames: np.ndarray, offsets: np.ndarray, pixels: np.ndarray):
        self.frames = frames
        self.offsets = offsets
        self.pixels = pixels
        self._content_hash = None

    def __getstate__(self):
        return (self.frames, self.offsets, self.pixels)

    def __setstate__(self, state):
        self.frames, self.offsets, self.pixels = state
        self._content_hash = None

    @classmethod
    def of(cls, defects) -> "DefectTable":
        """A table as is, or one built from `{frame: pixels-or-None}` (a set
        or a `{pixel: strength}` dict per frame; None and empty entries are
        absent from the table)."""
        return defects if isinstance(defects, DefectTable) else cls.from_sets(defects)

    @classmethod
    def from_sets(cls, sets) -> "DefectTable":
        items = sorted((str(h), s) for h, s in sets.items() if s)
        counts = np.fromiter((len(s) for _, s in items), np.int64, count=len(items))
        offsets = np.zeros(len(items) + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        pixels = np.fromiter((p for _, s in items for p in sorted(s)), np.int32, count=int(offsets[-1]))
        frames = np.array([h for h, _ in items], dtype=str) if items else np.empty(0, dtype="<U16")
        return cls(frames, offsets, pixels)

    @classmethod
    def concat(cls, tables) -> "DefectTable":
        """One table of all rows of `tables`, sorted by frame. Callers keep
        the frames disjoint (see `drop`)."""
        tables = [t for t in tables if len(t)]
        if not tables:
            return cls.from_sets({})
        bases = np.cumsum([0] + [len(t.pixels) for t in tables[:-1]])
        starts = np.concatenate([t.offsets[:-1] + b for t, b in zip(tables, bases)])
        frames = np.concatenate([t.frames for t in tables])
        counts = np.concatenate([t.counts() for t in tables])
        order = np.argsort(frames, kind="stable")
        return cls._gather(frames[order], starts[order], counts[order], np.concatenate([t.pixels for t in tables]))

    @classmethod
    def _gather(cls, frames, starts, counts, pixels) -> "DefectTable":
        offsets = np.zeros(len(frames) + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        index = np.repeat(starts - offsets[:-1], counts) + np.arange(offsets[-1])
        return cls(np.ascontiguousarray(frames), offsets, np.ascontiguousarray(pixels[index]))

    def __len__(self) -> int:
        return len(self.frames)

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def _row(self, h) -> int:
        i = int(np.searchsorted(self.frames, h))
        return i if i < len(self.frames) and self.frames[i] == h else -1

    def __contains__(self, h) -> bool:
        return self._row(h) >= 0

    def pixels_of(self, h) -> np.ndarray:
        i = self._row(h)
        if i < 0:
            raise KeyError(h)
        return self.pixels[self.offsets[i]:self.offsets[i + 1]]

    def take(self, rows) -> "DefectTable":
        """The table of rows `rows` (ascending row indices)."""
        rows = np.asarray(rows, np.int64)
        return self._gather(self.frames[rows], self.offsets[rows], self.counts()[rows], self.pixels)

    def subset(self, hashes) -> "DefectTable":
        """The rows of the frames in `hashes` that the table holds."""
        wanted = np.asarray(list(hashes), dtype=str)
        return self.take(np.flatnonzero(np.isin(self.frames, wanted)))

    def drop(self, hashes) -> "DefectTable":
        """The table without the frames in `hashes`."""
        unwanted = np.asarray(list(hashes), dtype=str)
        return self.take(np.flatnonzero(~np.isin(self.frames, unwanted))) if len(unwanted) else self

    @property
    def content_hash(self) -> str:
        if self._content_hash is None:
            import xxhash
            h = xxhash.xxh64()
            for a in (self.frames, self.offsets, self.pixels):
                h.update(a.dtype.str.encode())
                h.update(np.ascontiguousarray(a).tobytes())
            self._content_hash = h.hexdigest()
        return self._content_hash

    def __hash__(self):
        return int(self.content_hash, 16)

    def __eq__(self, other):
        return isinstance(other, DefectTable) and other.content_hash == self.content_hash

    def __str__(self):
        return f"DefectTable({len(self)} frames, {len(self.pixels)} defects)"
