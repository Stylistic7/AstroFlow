"""A run's subframes one after another, calibrated and autostretched, as a
short video that loops in the run's card on the stack page's Runs tab — so
whether a run is good or bad data is visible at a glance.

This is the per-subframe video that existed until 2026-10-01, rendered ON
DEMAND this time: the old one was generated eagerly for every stack at cold
start (so it was switched off), and served under a key the generator never
wrote (so it was blank). Now `web_server/backend/videos.py` starts
`python -m previews.run_video <job.json>` as its own process when a page
asks, the way a finish export runs, so a minute of frame reads never blocks a
web worker. The process joins the cluster for what the thumbnail path reads —
the stack and its calibration stacks — like the sweep CLI does.

Frames are in time order, each calibrated with the run's resolved masters and
autostretched on its own, with the night, the star count and the quality
gate's verdict burnt in. The video is H.264, which every browser plays, and
lives under `VIDEOS` as a file named by what it shows (`video_key`).
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

VIDEOS = Path.home() / "Pictures" / "AstroFlow" / "videos"
JOBS = VIDEOS / ".jobs"
WIDTH = 1200
FRAMES_PER_SECOND = 6


def video_key(stack, run_id: str, overrides, width: int = WIDTH) -> str:
    """What the video shows: the run's frames, its resolved calibration,
    the quality gate's override, the width."""
    from decisions.decisions import DecisionKey
    from decisions.resolve import resolve_calibration
    from context import context
    from hashing import hash_list
    run = resolve_calibration(stack, context.calibration_stacks, overrides).contents[run_id]
    calibration = sorted((kind, getattr(dep, "stack_id", str(dep))) for kind, dep in run.dependencies.items())
    gate = overrides.get(DecisionKey.subframe_quality(stack.stable_hash))
    return hash_list([run.content_hash, str(calibration), str(getattr(gate, "choice", None)), width])


def chronological(run) -> list:
    """`(frame hash, metadata)` for every frame of `run`, oldest first."""
    rows = [(h, run.metadata[h]) for h in run.frames]
    return sorted(rows, key=lambda r: str(r[1].header.get("DATE-OBS", "")))


def annotate(im: np.ndarray, text: str, included: bool) -> np.ndarray:
    """Burn `text` top-left into a grayscale uint8 frame; an excluded frame
    gets a bar along the bottom and says so."""
    h, w = im.shape
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1200)
    thickness = max(1, int(scale * 2))
    pos = (int(w * 0.03), int(h * 0.07))
    cv2.putText(im, text, pos, font, scale, 0, thickness + 2, cv2.LINE_AA)
    cv2.putText(im, text, pos, font, scale, 255, thickness, cv2.LINE_AA)
    if not included:
        bar = max(2, int(h * 0.02))
        im[h - bar:, :] = 200
        at = (int(w * 0.03), h - int(h * 0.04))
        cv2.putText(im, "EXCLUDED", at, font, scale * 0.8, 0, thickness + 2, cv2.LINE_AA)
        cv2.putText(im, "EXCLUDED", at, font, scale * 0.8, 200, thickness, cv2.LINE_AA)
    return im


def _frame(meta, run, width: int) -> np.ndarray:
    """One subframe as autostretched grayscale uint8: calibrated with the
    run's masters when it is solved, raw when it is not."""
    from file_io.generic import read_image
    from previews.calibration import calibrate_subframe
    from previews.operations import autostretch, resize_array
    if meta.plate_solution is not None:
        im = calibrate_subframe(meta, run.dependencies, width)
    else:
        im = resize_array(read_image(meta.filename), width)
    return (autostretch(np.clip(im, 0.0, 1.0)) * 255.0).astype(np.uint8)


def render(stack, run_id: str, overrides, dest: Path, width: int = WIDTH) -> dict:
    """Write the run's subframe video to `dest`; returns {frames, width, height}."""
    from decisions.resolve import is_frame_included, resolve_calibration
    from context import context
    resolved = resolve_calibration(stack, context.calibration_stacks, overrides)
    run = resolved.contents[run_id]
    rows = chronological(run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".run-video-") as tmp:
        shape = None
        for i, (frame_hash, meta) in enumerate(rows):
            im = _frame(meta, run, width)
            shape = shape or im.shape
            if im.shape != shape:                       # one grid per video
                im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            when = str(meta.header.get("DATE-OBS", ""))[:16].replace("T", " ")
            text = (f"{when}   {meta.plate_solution.num_stars} stars" if meta.plate_solution
                    else f"{when}   unsolved")
            annotate(im, text, is_frame_included(frame_hash, resolved, overrides))
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), im)
        if shape is None:
            raise ValueError(f"run {run_id} of {stack.stable_hash} has no frames to show")
        out = dest.with_name(f".{dest.stem}.{os.getpid()}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FRAMES_PER_SECOND),
             "-i", os.path.join(tmp, "%05d.png"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            check=True)
        os.replace(out, dest)
    return {"frames": len(rows), "width": shape[1], "height": shape[0]}


def run_job(job_file: Path) -> None:
    """The process the route starts: `{stack_id, run_id, width, dest, status}`."""
    job = json.loads(Path(job_file).read_text())
    status_file = Path(job["status"])
    status = {"state": "running", "path": job["dest"],
              "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    try:
        from context import context
        stack = context.stacks[job["stack_id"]].get()
        status.update(render(stack, job["run_id"], context.settings.decision_overrides,
                             Path(job["dest"]), job["width"]))
        status["state"] = "done"
    except Exception as e:                            # reported to the card, not swallowed
        status.update(state="error", error=f"{type(e).__name__}: {e}")
    tmp = status_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    os.replace(tmp, status_file)


if __name__ == "__main__":
    sys.exit(run_job(Path(sys.argv[1])))
