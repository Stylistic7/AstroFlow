"""Bake a finish into a file for other software (2026-10-04): the finish's
math and the viewer's parameters applied to every pixel, written as an
UNCOMPRESSED 16-bit RGB TIFF with an embedded sRGB profile -- the finish's
output is display-referred [0, 1], so 16 bits (65,536 levels) is more than
any later edit can reveal, and Photoshop opens 16-bit TIFF with every tool
(32-bit float would land it in HDR mode).

Streaming: a finish is per-pixel, so baking it strip by strip is exact. Each
strip of every input is read from the pyramid level, baked, and written into
a memory-mapped TIFF -- nothing but one strip is ever in memory, whatever the
image size. Photoshop refuses TIFFs over 4 GB, so a very large image exports
from the finest pyramid level that fits (the result says which).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

_STRIP_ROWS = 256
_MAX_BYTES = 3_900_000_000          # Photoshop's 4 GB TIFF limit, with header room


def _srgb_icc() -> bytes:
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def export_level(width: int, height: int, levels: int) -> int:
    """The finest pyramid level whose 16-bit RGB TIFF fits Photoshop's limit."""
    for level in range(levels):
        w, h = -(-width // 2 ** level), -(-height // 2 ** level)
        if w * h * 6 <= _MAX_BYTES:
            return level
    return levels - 1


def bake_to_tiff(bound, values: dict, dest: Path) -> dict:
    """Bake bound finish `bound` (catalog.BoundFinish) with parameter
    `values` into `dest`. Returns {path, level, width, height}."""
    import tifffile
    from previews.pyramid_pages import _get_vips_page, _pyramid_levels

    primary = os.fspath(bound.inputs[0].path)
    base = _get_vips_page(primary, 0)
    level = export_level(base.width, base.height, _pyramid_levels(primary))
    page = _get_vips_page(primary, level)
    width, height = page.width, page.height
    pages = [page]
    for i in bound.inputs[1:]:
        p = None
        if i.path is not None:
            q = _get_vips_page(os.fspath(i.path), level)
            if (q.width, q.height) == (width, height):
                p = q
        pages.append(p)          # absent or a different canvas: binds black, as in the viewer

    # the export is the picture, never the clipping diagnostic
    values = {**values, "show_clipping": 0.0}
    finish = bound.finish
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = tifffile.memmap(dest, shape=(height, width, 3), dtype=np.uint16,
                          photometric="rgb", iccprofile=_srgb_icc(),
                          bigtiff=False)
    try:
        for y in range(0, height, _STRIP_ROWS):
            h = min(_STRIP_ROWS, height - y)
            srcs = []
            for p in pages:
                if p is None:
                    srcs.append(np.zeros((h, width, 3), np.float32))
                    continue
                strip = p.crop(0, y, width, h)
                if strip.format == "uchar":      # legacy display-ready pyramid, as tiles.py
                    a = np.ndarray(buffer=strip.write_to_memory(), dtype=np.uint8,
                                   shape=(h, width, strip.bands)).astype(np.float32) / 255.0
                else:
                    a = np.ndarray(buffer=strip.cast("float").write_to_memory(),
                                   dtype=np.float32, shape=(h, width, strip.bands))
                srcs.append(a[..., :3] if strip.bands >= 3 else a)
            rgb = finish.bake(srcs, values)
            out[y:y + h] = np.round(np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
        out.flush()
    finally:
        del out
    return {"path": str(dest), "level": level, "width": width, "height": height}


# ── the export job: a separate process, so a minutes-long bake never blocks
# a (gevent) web worker. Driven by a job file; reports through a status file
# any web worker can read. Needs neither the catalog nor Ray: the web route
# resolved the inputs to file paths already. ─────────────────────────────

def newest_photoshop() -> "str | None":
    """The newest installed Adobe Photoshop .app, or None."""
    try:
        out = subprocess.run(
            ["mdfind", "kMDItemCFBundleIdentifier == 'com.adobe.Photoshop'"],
            capture_output=True, text=True, timeout=10).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        return None
    apps = sorted(p for p in out if p.endswith(".app"))
    return apps[-1] if apps else None


def _write_status(path: Path, status: dict) -> None:
    import json
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    os.replace(tmp, path)


def run_job(job_file: Path) -> None:
    import json
    from types import SimpleNamespace
    from finishes.registry import get_finish
    job = json.loads(Path(job_file).read_text())
    status_file = Path(job["status"])
    status = {"state": "running", "path": job["dest"]}
    try:
        finish = get_finish(job["finish"], primary=job["primary"])
        bound = SimpleNamespace(finish=finish, inputs=[
            SimpleNamespace(path=p) for p in job["paths"]])
        result = bake_to_tiff(bound, job["params"], Path(job["dest"]))
        status.update(result)
        subprocess.run(["open", "-a", job["app"], job["dest"]], check=True, timeout=60)
        status["state"] = "done"
    except Exception as e:                       # reported to the button, not swallowed
        status.update(state="error", error=f"{type(e).__name__}: {e}")
    _write_status(status_file, status)


if __name__ == "__main__":
    import sys
    run_job(Path(sys.argv[1]))
