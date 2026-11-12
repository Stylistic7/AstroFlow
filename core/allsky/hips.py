"""HiPS trees for the sky viewer.

HiPS (IVOA's Hierarchical Progressive Survey) is a layout, not a format or
a service: a fixed tree of ordinary files -- `properties` at the root, then
one image per HEALPix cell per order at
`Norder<k>/Dir<10000·⌊n/10000⌋>/Npix<n>.fits` -- that any web server or
folder serves as is. A tile is a `hips_tile_width`² raster of the cells
log2(width) orders deeper, so the tree IS the pyramid the viewer streams.
The browser reads a remote tree's tiles itself (CDS answers over HTTP/2,
about two seconds a tile, sixteen at a time); this module reads what comes
first -- the properties and a sample of the finest tiles, for the gauge and
the seed statistics -- mirrored under `~/AstroFlow/healpix/<name>/` so a
map is measured once.

A HiPS tile is the face raster (healpix_faces.py) turned a quarter turn:
FITS pixel (column c, row r) is raster cell (x = W−1−r, y = c). Witnessed
on CDS's unWISE tiles by the bright stars of testdata/hips_tile_layout.json;
the browser's parser (sky_tiles.js) is `raster_from_tile`'s twin.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from allsky import maps

SAMPLE_TILES = 24       # finest tiles measured for the gauge and the seed
DIR_SPAN = 10000


def tile_path(order: int, npix: int, ext: str = "fits") -> str:
    return f"Norder{order}/Dir{npix // DIR_SPAN * DIR_SPAN}/Npix{npix}.{ext}"


def read_properties(text: str) -> dict:
    props = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        props[k.strip()] = v.strip()
    return props


def raster_from_tile(tile: np.ndarray) -> np.ndarray:
    """A FITS tile's pixels as the face raster: `out[y, x] = tile[W−1−x, y]`."""
    return np.rot90(tile, 3)


def sample_npix(order: int, count: int) -> list[int]:
    """`count` tile indices spread over the sky: NESTED order is spatially
    coherent, so an even stride through it is an even spread."""
    n = 12 << (2 * order)
    return sorted({int(i) for i in np.linspace(0, n - 1, min(count, n))})


class HipsTree:
    """A HiPS root -- a URL or a folder -- with remote reads mirrored on disk."""

    def __init__(self, root: str, mirror: Path):
        self.root = root.rstrip("/")
        self.mirror = mirror
        self.remote = "://" in root

    def existing(self, order: int) -> list[int] | None:
        """The tiles a local tree holds at `order`, in NESTED order; None for
        a remote tree, which cannot be listed."""
        if self.remote:
            return None
        return sorted(int(p.stem[4:]) for p in (Path(self.root) / f"Norder{order}").glob("Dir*/Npix*.fits"))

    def read(self, rel: str) -> bytes | None:
        """The file's bytes, or None where the tree has no such file."""
        local = (self.mirror if self.remote else Path(self.root)) / rel
        if local.is_file():
            return local.read_bytes()
        if not self.remote:
            return None
        import requests
        r = requests.get(f"{self.root}/{rel}", timeout=120)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".part")
        part.write_bytes(r.content)
        part.replace(local)
        return r.content

    def tile(self, order: int, npix: int) -> np.ndarray | None:
        """The tile's raster, or None where the tree has no such tile."""
        data = self.read(tile_path(order, npix))
        return None if data is None else read_tile(data)


def read_tile(data: bytes) -> np.ndarray:
    """A FITS tile as the face raster, file units, 0 where it has no data."""
    from astropy.io import fits
    with fits.open(io.BytesIO(data)) as h:
        raster = raster_from_tile(np.asarray(h[0].data, dtype=np.float32))
    return np.where(np.isfinite(raster), raster, 0.0).astype(np.float32)


def load_hips(name: str, path: Path, mtime: float) -> maps.SkyMap:
    """The map a `.hips` pointer names: a packed plane (allsky/tiles.py) or a
    HiPS tree, local or remote."""
    from allsky.tiles import Plane, is_plane
    root = path.read_text().strip()
    if "://" not in root and is_plane(Path(root)):
        plane = Plane(Path(root))
        return _measured(name, path, mtime, plane.properties, plane, root)
    tree = HipsTree(root, maps.MAPS_DIR / name)
    text = tree.read("properties")
    if text is None:
        raise FileNotFoundError(f"{path.name}: no properties at {tree.root}")
    props = read_properties(text.decode("utf-8", "replace"))
    if "fits" not in props.get("hips_tile_format", "").split():
        raise ValueError(f"{path.name}: {tree.root} has no FITS tiles "
                         f"(hips_tile_format={props.get('hips_tile_format')!r})")
    return _measured(name, path, mtime, props, tree, root)


def _measured(name: str, path: Path, mtime: float, props: dict, source, root: str) -> maps.SkyMap:
    """The map over `source` (a HipsTree or a Plane: `tile`, `existing`),
    its gauge and seed statistics measured on a sample of its finest tiles.
    A sparse local source (a survey of part of the sky) is sampled over the
    tiles it has; a remote tree over the whole sphere."""
    order = int(props["hips_order"])
    width = int(props.get("hips_tile_width", 512))
    frame = maps.frame_of(props.get("hips_frame", "equatorial"))
    have = source.existing(order)
    sample = sample_npix(order, SAMPLE_TILES) if have is None else \
        [have[i] for i in sorted({int(i) for i in np.linspace(0, len(have) - 1, min(SAMPLE_TILES, len(have)))})]
    with ThreadPoolExecutor(16) as pool:
        tiles = list(pool.map(lambda n: source.tile(order, n), sample))
    rasters = [np.where(np.isfinite(t), t, 0.0).astype(np.float32) for t in tiles if t is not None]
    if not rasters:
        raise ValueError(f"{path.name}: none of the sampled tiles exist at {root}")
    gauge, stats = maps.gauge_and_stats(np.concatenate([r.ravel() for r in rasters]))
    remote = getattr(source, "remote", False)
    return maps.SkyMap(name=name, path=path, mtime=mtime, frame=frame, unit="",
                       gauge=gauge, stats=stats, tile_width=width, max_order=order,
                       tiles=root if remote else "", tree=source)
