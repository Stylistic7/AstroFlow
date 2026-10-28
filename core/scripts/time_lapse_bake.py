"""Re-bake a time-lapse's frames and video from the layers the pipeline kept.

    uv run python -m scripts.time_lapse_bake <mosaic_id> [--look <save id>] [--folder ...]

Ray-free. Reads `<folder>/<mosaic_id>/layers/<night>/` (written by
`pipelines/time_lapse.py`'s `keep_layers` as each night lands), seeds the
Hubble finish's data anchors from the newest night's composite — the same
pin the pipeline uses — lays the sliders of the saved look `--look` (a save
of the mosaic's production Hubble finish, from the viewer's "Save this
look") over them, renders every night with `render_frame`, writes
`<night>.png` and cuts `time_lapse.mp4`. A minute a frame, no products
touched: changing one's mind about the look costs nothing upstream.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mosaic_id")
    parser.add_argument("--look", default=None, help="id of a saved Hubble look; default: the finish's defaults")
    parser.add_argument("--folder", default="/Volumes/AstroProcessed/TimeLapse")
    parser.add_argument("--only", default=None, help="one night (YYYY-MM-DD) to re-bake, for a quick check")
    args = parser.parse_args()

    from finishes.preview import _stf_values, read_pyramid_level
    from finishes.registry import get_finish
    from pipelines.hybrid_siril_graxpert_oss_mosaic import PIPELINE as SIRIL_PIPELINE
    from pipelines.time_lapse import FRAMES_PER_SECOND, HOLD_LAST_SECONDS, LAYERS, render_frame

    root = Path(args.folder) / args.mosaic_id
    nights = sorted(p for p in (root / "layers").iterdir() if (p / "sho.tiff").exists())
    assert nights, f"no kept layers under {root / 'layers'}"
    read = lambda p: read_pyramid_level(p, min_px=1 << 30)

    final = read(nights[-1] / "sho.tiff")
    values = {k: float(v) for k, v in _stf_values(final, ("S", "H", "O"), anchors_from=final).items()}
    if args.look:
        from context.cache import get_durable_cache
        saves = get_durable_cache().get(("finish_saves", "mosaic", args.mosaic_id, SIRIL_PIPELINE, "Hubble")) or []
        save = next((s for s in saves if s["id"] == args.look), None)
        assert save is not None, f"no saved Hubble look {args.look!r} for {args.mosaic_id}"
        data = {p.name for p in get_finish("Hubble").params if p.kind == "data"}
        values.update({k: float(v) for k, v in save["params"].items() if k not in data})
    print(f"{len(nights)} nights; anchors from {nights[-1].name}; look {args.look or 'defaults'}", flush=True)

    for folder in nights:
        night = folder.name
        if args.only and night != args.only:
            continue
        layers = {name: read(folder / f"{name}.tiff")[..., 0]
                  for name in LAYERS[1:] if (folder / f"{name}.tiff").exists()}
        frame = render_frame(read(folder / "sho.tiff"), layers, values, (folder / "caption.txt").read_text())
        h, w = frame.shape[:2]
        tmp = root / f".{night}.{os.getpid()}.png"
        pyvips.Image.new_from_memory(np.ascontiguousarray(frame).data, w, h, 3, "ushort") \
            .cast("uchar", shift=True).pngsave(str(tmp))
        os.replace(tmp, root / f"{night}.png")
        print(f"  {night}", flush=True)
    if args.only:
        return
    tmp = root / f".time_lapse.{os.getpid()}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FRAMES_PER_SECOND),
         "-pattern_type", "glob", "-i", str(root / "*.png"),
         "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration={HOLD_LAST_SECONDS}",
         "-c:v", "hevc_videotoolbox", "-q:v", "65", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp)], check=True)
    os.replace(tmp, root / "time_lapse.mp4")
    print("video:", root / "time_lapse.mp4")


if __name__ == "__main__":
    sys.exit(main())
