"""Packed tile planes: what the combiner writes and the sky viewer reads.

A HiPS tree is one file per tile; MDW DR1's three planes came to 815,000
files, which no copy, backup, scrub or listing survives gracefully
(2026-11-12). Here a plane is a Zarr v3 group: array `<k>` for tile order k,
shape (12·4^k, w, w), one chunk per tile, SHARD_TILES tiles per shard file --
a contiguous NESTED run, the descendants of one ancestor three orders up, so
a shard is a patch of sky -- zstd per chunk, the shard's index at its end.
A tile never written reads as the fill value and costs nothing; writing the
fill value drops it. `coverage_<k>` (a byte per tile) says what exists, so
the viewer samples without listing anything; `<name>_<k>` carries a per-tile
scalar the plane declares (the mean's coadd PSF width). The group's
attributes are the HiPS `properties` and are written LAST: `zarr.json`'s
mtime is then the map's version, which the viewer caches tiles under as
immutable (allsky/maps.py).

Writers own shards, not tiles. A shard is rewritten whole -- read, replace,
atomic rename -- so two processes must never hold the same one; the combiner
hands each worker whole shards. A reader beside a writer sees the old shard
or the new one, never a torn one.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SHARD_TILES = 64        # tiles per shard file: 8 × 8 of them; the unit a writer owns and rewrites
ZSTD_LEVEL = 3


def n_tiles(order: int) -> int:
    return 12 << (2 * order)


def shard_of(npix: int) -> int:
    return npix // SHARD_TILES


def shard_range(order: int, shard: int) -> tuple[int, int]:
    lo = shard * SHARD_TILES
    return lo, min(lo + SHARD_TILES, n_tiles(order))


def is_plane(path: Path) -> bool:
    return (Path(path) / "zarr.json").is_file()


def create(root: Path, planes: dict[str, tuple[np.dtype, int]], order: int,
           scalars: dict[str, tuple[str, ...]] | None = None) -> dict[str, "Plane"]:
    """The groups `root/<name>` with arrays for orders 0..order, created where
    missing: `planes` maps a name to (dtype, tile width), `scalars` a name to
    its per-tile scalars. Returns the planes open for writing."""
    import zarr
    from zarr.codecs import ZstdCodec
    root = Path(root)
    for name, (dtype, width) in planes.items():
        g = zarr.open_group(root / name, mode="a")
        fill = 0 if np.issubdtype(dtype, np.integer) else np.nan
        for k in range(order + 1):
            if str(k) in g:
                continue
            n = n_tiles(k)
            g.create_array(str(k), shape=(n, width, width), chunks=(1, width, width),
                           shards=(SHARD_TILES, width, width), dtype=dtype, fill_value=fill,
                           compressors=ZstdCodec(level=ZSTD_LEVEL))
            g.create_array(f"coverage_{k}", shape=(n,), chunks=(SHARD_TILES,), dtype="uint8", fill_value=0)
            for scalar in (scalars or {}).get(name, ()):
                g.create_array(f"{scalar}_{k}", shape=(n,), chunks=(SHARD_TILES,), dtype="float32", fill_value=np.nan)
    return {name: Plane(root / name, "r+") for name in planes}


class Plane:
    """One plane's group, open to read (`"r"`) or to write (`"r+"`)."""

    remote = False      # the viewer's question of a map source: a plane is always on disk

    def __init__(self, path: Path, mode: str = "r"):
        import zarr
        self.path = Path(path)
        self.group = zarr.open_group(self.path, mode=mode)
        self._arrays: dict = {}

    @property
    def properties(self) -> dict:
        return dict(self.group.attrs)

    @property
    def version(self) -> float:
        """When the properties were last written: the map's version."""
        return os.path.getmtime(self.path / "zarr.json")

    def _array(self, name: str):
        a = self._arrays.get(name)
        if a is None:
            a = self._arrays[name] = self.group[name]
        return a

    def tile(self, order: int, npix: int) -> np.ndarray | None:
        """The tile's raster in file units (NaN or 0 where a cell has no
        data), or None where no tile was written."""
        arr = self._array(str(order))
        if not 0 <= npix < arr.shape[0]:
            raise IndexError((order, npix))
        if not self._array(f"coverage_{order}")[npix]:
            return None
        return arr[npix]

    def scalar(self, name: str, order: int, npix: int) -> float:
        return float(self._array(f"{name}_{order}")[npix])

    def existing(self, order: int) -> list[int]:
        """The tiles present at `order`, in NESTED order."""
        return np.flatnonzero(self._array(f"coverage_{order}")[:]).tolist()

    def write_block(self, order: int, shard: int, tiles: dict[int, np.ndarray | None],
                    scalars: dict[str, dict[int, float]] | None = None) -> None:
        """Replace the named tiles of one shard (None removes a tile); the
        shard's other tiles stay. One read and one atomic write of the shard
        file, so the caller must be the shard's only writer."""
        arr = self._array(str(order))
        cov = self._array(f"coverage_{order}")
        lo, hi = shard_range(order, shard)
        block, cblock = arr[lo:hi], cov[lo:hi]
        for npix, raster in tiles.items():
            if not lo <= npix < hi:
                raise ValueError(f"tile {npix} is not in shard {shard} of order {order}")
            block[npix - lo] = arr.fill_value if raster is None else raster
            cblock[npix - lo] = raster is not None
        arr[lo:hi] = block
        cov[lo:hi] = cblock
        for name, values in (scalars or {}).items():
            sarr = self._array(f"{name}_{order}")
            sblock = sarr[lo:hi]
            for npix, v in values.items():
                sblock[npix - lo] = v
            sarr[lo:hi] = sblock

    def write_properties(self, props: dict) -> None:
        """The HiPS properties, as the plane's attributes; last, because it
        moves the version."""
        self.group.attrs.update({k: str(v) for k, v in props.items()})
