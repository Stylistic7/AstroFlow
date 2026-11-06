"""A folder of foreign masters made merge-ready the way our own are
(`stacks/post_processing/graxpert_stack.calibrate_master`): every panel's
background extracted by GraXpert and its sky pinned to the shared pedestal,
so the combiner meets flat panels instead of each survey's own gradients.

Masters in physical units (MDW's are Jy) are first brought onto our [0, 1]
master scale by ONE gauge for the whole folder, so their ratios survive;
the gauge is written to each output's header (FLATGAUG). Holes (NaN or
exact zero) stay holes: filled with the sky for GraXpert, NaN on output.

    python -m allsky.flatten <masters> <out> [--gauge G] [--workers N]
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
from stacks.post_processing.graxpert_stack import GRAXPERT_AI_VERSION, GRAXPERT_CORRECTION, GRAXPERT_SMOOTHING

SKY_TARGET = 0.01       # where the gauge puts the folder's median sky before extraction


def sky_of(image: np.ndarray) -> float:
    """The sky: the median of the lower quartile's finite pixels
    (`background_pedestal`'s estimator)."""
    v = image[np.isfinite(image)]
    return float(np.median(v[v <= np.quantile(v, 0.25)]))


def flatten_master(src: Path, dst: Path, gauge: float, pedestal: float = DEFAULT_PEDESTAL) -> None:
    from astropy.io import fits

    from graxpert_cli import run_background_extraction
    with fits.open(src) as h:
        header = h[0].header.copy()
        image = np.asarray(h[0].data, np.float32) * np.float32(gauge)
    hole = ~np.isfinite(image) | (image == 0)
    image[hole] = sky_of(np.where(hole, np.nan, image))
    with tempfile.TemporaryDirectory(prefix="flatten_") as td:
        fits.PrimaryHDU(image).writeto(Path(td) / "in.fits")
        run_background_extraction(Path(td) / "in.fits", Path(td) / "out.fits", smoothing=GRAXPERT_SMOOTHING,
                                  correction=GRAXPERT_CORRECTION, ai_version=GRAXPERT_AI_VERSION)
        flat = np.asarray(fits.getdata(Path(td) / "out.fits"), np.float32).squeeze()
    flat = flat - sky_of(np.where(hole, np.nan, flat)) + np.float32(pedestal)
    flat[hole] = np.nan
    header["FLATGAUG"] = (gauge, "pixels = source x FLATGAUG, then GraXpert + pedestal")
    header["PEDESTAL"] = pedestal
    header.remove("BUNIT", ignore_missing=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(flat, header).writeto(dst.with_suffix(".tmp.fits"), overwrite=True)
    dst.with_suffix(".tmp.fits").rename(dst)


def folder_gauge(paths: list[Path]) -> float:
    """SKY_TARGET over the median of the masters' skies (strided samples)."""
    from astropy.io import fits
    skies = []
    for p in paths:
        with fits.open(p, memmap=True) as h:
            a = np.asarray(h[0].section[::64, ::64], np.float64)
        skies.append(sky_of(np.where(a == 0, np.nan, a)))
    return SKY_TARGET / float(np.median(skies))


def _one(args) -> str:
    src, dst, gauge = args
    if not dst.exists():
        flatten_master(src, dst, gauge)
    return src.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masters", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pattern", default="*.fits")
    ap.add_argument("--gauge", type=float, help="source → [0, 1] scale (default: from the folder's median sky)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N masters (a trial)")
    a = ap.parse_args(argv)
    paths = sorted(p for p in a.masters.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, a.pattern))
    gauge = a.gauge or folder_gauge(paths)
    print(f"{len(paths)} masters, gauge {gauge:.6g}", flush=True)
    jobs = [(p, a.out / p.relative_to(a.masters), gauge) for p in paths[:a.limit]]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(_one, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
