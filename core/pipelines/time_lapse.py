"""Pipeline: the Siril mosaic as it accumulated, one frame per night of shooting.

Each frame is the mosaic built from EVERY light run from the project's first
night through that night — the live-stacking feeling, one night per frame:
noise falling, depth growing, panels filling in — rendered in the Hubble
finish and cut into a video. Everything else is pinned to today, so the
only thing moving is the data: today's calibration masters and decisions,
the canvas and the stretch of the final frame (Stylistic7, 2026-10-28: replaying
the pipeline of the day as well would move two things at once and tell a
tool story instead of a sky story).

How a night is built. Its objects are REAL as-of objects — each stack with
only its runs through that night (`Stack.with_runs`), each target from
those stacks, the mosaic from those targets (`mosaic_from_targets`) — so
every key downstream folds that night's membership and nothing else. A
night built once is built forever: data added later lands in later nights
and changes no earlier frame. The last night IS today's mosaic, object for
object, so its masters and layers are the production pipeline's own cache
hits.

Canvas: fixed, at the Video quality (6016 px wide, a 6K display at 1:1), from the declared
geometry of the mosaic as of the CANVAS night — the newest night, or the
one pinned by the `time_lapse_canvas` decision so the frame stays put when
the project grows. Per-panel star separation (SXT, no BXT) runs on the
native grid as in production; placement, merge and polish run on the small
canvas, so a night costs minutes rather than an hour.

Frames: the S, H and O layers are band-joined into one linear SHO composite
and baked through the Hubble finish with the values seeded from the final
night's composite. Stars: RGB where all three broadband layers cover a
pixel, G alone where only it does, the H stars where no broadband does
(RGB came to 75a1 a year after H), screened in at the finish's own star
weight. The 16-bit frames are persisted intermediates (not display taps:
a hundred of them would bury the finishes page); the keepers — a PNG per
night, written as each night lands, the night's frame inputs under
`layers/<night>/`, and the MP4 at the end — go to
`<output root>/TimeLapse/<mosaic>/`, named by night. The kept layers are
what `scripts/time_lapse_bake.py` re-bakes with another look, so a change
of taste costs a minute a frame, never a rebuild.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyvips

from astroflow.pipeline import pipeline
from context import context
from decisions.decisions import DecisionKey
from decisions.quarantine import Quarantine
from decorators import node
from file_io.types import Produced, TiffFilePath
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.color.post_process import FAST_COMPRESSION, _open_mono_vips, _save_color_vips
from mosaics.identification import mosaic_from_targets
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from pipelines.hybrid_graxpert_oss_mosaic import _merged_channels
from pipelines.hybrid_siril_graxpert_oss_mosaic import PIPELINE as SIRIL_PIPELINE, channel
from previews.types import VIDEO
from targets.target_channel import _resolved_by_filter, group_stacks_by_filter
from targets.types import Target

PIPELINE = "Siril + GraXpert + OSS Time-lapse"
FRAMES_PER_SECOND = 4
HOLD_LAST_SECONDS = 3


def night_of(when: datetime.datetime) -> datetime.date:
    """The night a run belongs to: the date of the evening it started on
    (a run past midnight belongs to the night before)."""
    return (when - datetime.timedelta(hours=12)).date()


def nights_of(stacks) -> list[datetime.date]:
    return sorted({night_of(r.oldest_date) for s in stacks for r in s.contents.values()})


def runs_through(stack, night: datetime.date, quarantined) -> tuple[str, ...]:
    """The runs of `stack` shot by the end of `night` and not quarantined."""
    return tuple(h for h, r in stack.contents.items()
                 if night_of(r.oldest_date) <= night and h not in quarantined)


def mosaic_as_of(mosaic, full_targets: dict, stacks_by_target: dict, night, quarantine,
                 min_frames: int):
    """`(Mosaic, {target id: Target}, {(target id, filter): stacks})` as of
    `night`, or None when nothing was shot yet. An object whose membership
    is complete is today's own, so its keys are the production pipeline's.
    An as-of object keeps the production identity of what it is a version
    of — `Mosaic.targets` and friends resolve ids out of the context, and a
    night whose oldest member is missing would otherwise mint ids nothing
    knows — while its content hash is its own membership's.

    A stack with fewer than `min_frames` frames by that night is left out:
    the integration would refuse it at the same gate, and a refused channel
    fails that filter's merge for the night and with it the frame. The
    panel simply joins once it has enough."""
    targets, channels = {}, {}
    for tid, full in full_targets.items():
        if quarantine.target(tid, full.stack_ids):
            continue
        kept = []
        for stack in stacks_by_target[tid]:
            runs = runs_through(stack, night, quarantine.runs_of(stack))
            if runs and sum(len(stack.contents[r].frames) for r in runs) >= min_frames:
                kept.append(stack if len(runs) == len(stack.contents) else stack.with_runs(runs))
        if not kept:
            continue
        target = full if len(kept) == len(stacks_by_target[tid]) and all(
            s is f for s, f in zip(kept, stacks_by_target[tid])) else replace(
            Target.from_stacks(kept), stable_hash=full.stable_hash)
        targets[tid] = target
        for filter, group in group_stacks_by_filter(kept).items():
            channels[(tid, filter)] = tuple(group)
    if not targets:
        return None
    complete = set(targets) == set(mosaic.target_ids) and all(
        targets[tid] is full_targets[tid] for tid in targets)
    as_of = mosaic if complete else replace(
        mosaic_from_targets(tuple(targets), targets), stable_hash=mosaic.stable_hash)
    return as_of, targets, channels


def canvas_night(mosaic, decisions, nights: list[datetime.date]) -> datetime.date:
    """The night whose extent frames every frame: pinned by decision, else
    the newest."""
    override = decisions.get(DecisionKey.time_lapse_canvas(mosaic.stable_hash))
    if override is None:
        return nights[-1]
    pinned = datetime.date.fromisoformat(str(override.choice))
    return max(n for n in nights if n <= pinned)


def nights_window(mosaic, decisions, nights: list[datetime.date]) -> list[datetime.date]:
    """The nights to build this pass: those inside the `time_lapse_nights`
    decision's `A..B`, else all of them.

    Why windows: every night that adds a run to a stack re-integrates that
    stack and re-separates its stars at the native grid, and those are
    products — about a gigabyte per run added, kept for as long as a trace
    references them, and only the orphan sweep (`scripts/sweep_orphans.py`)
    reclaims them. A trace references only its window, so a window at a
    time, swept between windows, bounds the disk to one window's worth."""
    override = decisions.get(DecisionKey.time_lapse_nights(mosaic.stable_hash))
    if override is None:
        return nights
    first, last = (datetime.date.fromisoformat(s) for s in str(override.choice).split(".."))
    return [n for n in nights if first <= n <= last]


def saved_look(mosaic, decisions) -> dict:
    """The slider values of the saved Hubble look the `time_lapse_look`
    decision names — a save of the mosaic's PRODUCTION Hubble finish, made
    with the viewer's "Save this look" — or {} for the finish's defaults.
    Only the sliders: the data anchors (sky, noise, bright level) are
    measured from the final frame's own composite, not the production
    mosaic's."""
    override = decisions.get(DecisionKey.time_lapse_look(mosaic.stable_hash))
    if override is None:
        return {}
    from finishes.registry import get_finish
    saves = context.get(("finish_saves", "mosaic", mosaic.stable_hash, SIRIL_PIPELINE, "Hubble")) or []
    save = next((s for s in saves if s["id"] == str(override.choice)), None)
    assert save is not None, f"time-lapse look: no saved Hubble look {override.choice!r} for {mosaic.stable_hash}"
    data = {p.name for p in get_finish("Hubble").params if p.kind == "data"}
    return {k: float(v) for k, v in save["params"].items() if k not in data}


def time_lapse_canvas(mosaic_as_of_canvas_night) -> Viewbox:
    """The one canvas every night is placed on, at the Video quality, from
    the declared geometry of the mosaic as of the canvas night. Stamped
    with a content hash of what it is a function of, so every key that
    folds it (placements, merges, frames) reduces to that and nothing else."""
    canvas = compute_viewbox(VIDEO, mosaic_as_of_canvas_night)
    return replace(canvas, content_hash=hash_list([VIDEO, mosaic_as_of_canvas_night.content_hash]))


def stars_fallback(stars: dict, covered: dict) -> np.ndarray:
    """The RGB stars layer with the broadband gaps filled: RGB where R, G
    and B all cover a pixel, G alone where only it does, the H stars where
    no broadband does (and black where nothing does)."""
    full = covered["R"] & covered["G"] & covered["B"]
    g_only = covered["G"] & ~full
    h_only = covered["H"] & ~full & ~g_only
    out = np.zeros((*stars["H"].shape, 3), np.float32)
    for i, f in enumerate("RGB"):
        out[..., i] = np.where(full, stars[f], np.where(g_only, stars["G"], np.where(h_only, stars["H"], 0.0)))
    return out


def fill_missing_bands(sho: np.ndarray) -> np.ndarray:
    """Where a pixel lacks one or two of S, H and O, fill them with the mean
    of the bands it has: an H-only region renders grey through the same
    stretch, and colour spreads across the mosaic as the other filters
    arrive. Left black, the Hubble finish's green-to-gold step caps H at
    the mean of two black bands, which is what made the first frames black."""
    covered = sho > 0
    n = covered.sum(axis=-1, keepdims=True)
    mean = (sho * covered).sum(axis=-1, keepdims=True) / np.maximum(n, 1)
    return np.where(covered, sho, mean).astype(np.float32)


def _read(meta) -> np.ndarray:
    """A layer as (H, W, bands) float32 at the canvas's own resolution."""
    from finishes.preview import read_pyramid_level
    return read_pyramid_level(meta.filename, min_px=1 << 30)


@node(auto_memory=True)
def sho_composite(canvas: Viewbox, s, h, o, *, out: TiffFilePath) -> Produced:
    """A night's linear SHO composite: band-join of the three starless
    layers; a filter not yet shot is black."""
    black = pyvips.Image.black(canvas.num_pixels_x, canvas.num_pixels_y).cast("float")
    bands = [_open_mono_vips(m) if m is not None else black for m in (s, h, o)]
    _save_color_vips(str(out), bands[0].bandjoin(bands[1:]), "Linear SHO", bands=("S", "H", "O"))
    return Produced(out)


@node
def hubble_values(sho: ImageMetadata, look: dict) -> dict:
    """The Hubble finish's values: the data anchors seeded from one
    composite — the final night's, so every frame is stretched the same and
    the noise of the early ones shows for what it is — under the sliders of
    the saved `look` (`saved_look`)."""
    from finishes.preview import _stf_values
    arr = _read(sho)
    seeded = {k: float(v) for k, v in _stf_values(arr, ("S", "H", "O"), anchors_from=arr).items()}
    return {**seeded, **look}


def _caption(frame: np.ndarray, text: str) -> None:
    """Burn `text` into the frame's lower-left corner, in place."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame.shape[:2]
    size = max(18, h // 36)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1: one bitmap size
        font = ImageFont.load_default()
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((size, h - 2 * size), text, fill=255, font=font)
    a = (np.asarray(mask, np.float32) / 255.0)[..., None]
    frame[:] = (frame * (1.0 - a) + 65535.0 * a + 0.5).astype(np.uint16)


LAYERS = ("sho", "starless_r", "starless_g", "starless_b", "stars_r", "stars_g", "stars_b", "stars_h")


def render_frame(sho: np.ndarray, layers: dict, values: dict, caption: str) -> np.ndarray:
    """One frame as a 16-bit RGB array: the SHO composite (missing bands
    filled) through the Hubble finish with `values`, its stars from
    `stars_fallback` over the layers' coverage, the caption burnt in.
    `layers`: mono (H, W) arrays, or None, under the `LAYERS` names.
    Shared by the frame node and `scripts/time_lapse_bake.py`, which
    re-bakes kept layers with another look without the pipeline."""
    from finishes.registry import get_finish
    sho = fill_missing_bands(sho)
    h, w = sho.shape[:2]

    def mono(name):
        a = layers.get(name)
        return a if a is not None else np.zeros((h, w), np.float32)
    covered = {f: mono(f"starless_{f.lower()}") > 0 for f in "RGB"}
    covered["H"] = sho[..., 1] > 0
    stars = {f: mono(f"stars_{f.lower()}") for f in "RGBH"}
    rgb = get_finish("Hubble").bake([sho, stars_fallback(stars, covered)], values)
    frame = np.ascontiguousarray((np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))
    _caption(frame, caption)
    return frame


@node(auto_memory=True)
def hubble_frame(mosaic_id: str, night: str, caption: str, values: dict, sho: ImageMetadata,
                 starless_r, starless_g, starless_b, stars_r, stars_g, stars_b, stars_h,
                 *, out: TiffFilePath) -> Produced:
    """One frame (`render_frame`) as a 16-bit RGB TIFF."""
    metas = dict(zip(LAYERS[1:], (starless_r, starless_g, starless_b, stars_r, stars_g, stars_b, stars_h)))
    frame = render_frame(_read(sho), {k: _read(m)[..., 0] for k, m in metas.items() if m is not None},
                         values, caption)
    h, w = frame.shape[:2]
    img = pyvips.Image.new_from_memory(frame.data, w, h, 3, "ushort").copy(interpretation="rgb16")
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"PALETTE": "Hubble", "PIPELINE": "AstroFlow-OSS", "NIGHT": night}))
    img.tiffsave(str(out), compression=FAST_COMPRESSION, tile=True, bigtiff=True)
    return Produced(out)


@node
def keep_layers(root: str, night: str, caption: str, *layers) -> str:
    """The night's frame inputs — its SHO composite and the starless and
    stars layers, in `LAYERS` order, None where a filter is absent — copied
    to `<root>/layers/<night>/`, with the caption, the moment they exist.
    They are products otherwise, and the orphan sweep reclaims them once
    their window is done; kept here, `scripts/time_lapse_bake.py` can
    re-bake every frame with another look in a minute each, without the
    pipeline. Returns the folder."""
    import shutil
    folder = Path(root) / "layers" / night
    folder.mkdir(parents=True, exist_ok=True)
    for name, meta in zip(LAYERS, layers):
        dest = folder / f"{name}.tiff"
        if meta is None:
            dest.unlink(missing_ok=True)
            continue
        tmp = folder / f".{name}.{os.getpid()}.tiff"
        shutil.copyfile(meta.filename, tmp)
        os.replace(tmp, dest)
    (folder / "caption.txt").write_text(caption)
    return str(folder)


@node
def keep_frame(root: str, night: str, frame) -> tuple:
    """The frame as `<root>/<night>.png`, written the moment it exists so the
    nights can be watched arriving; returns `(path, the frame's content hash)`
    so the video's key moves when a frame does, not only when a name does."""
    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{night}.png"
    tmp = folder / f".{night}.{os.getpid()}.png"
    pyvips.Image.new_from_file(str(frame.filename)).cast("uchar", shift=True).pngsave(str(tmp))
    os.replace(tmp, png)
    return (str(png), frame.content_hash)


# Head-pinned like the other file-less orchestration ops: ffmpeg lives on the head.
@node(resources={"diskcache": 0.001})
def time_lapse_video(root: str, *kept) -> str:
    """`<root>/time_lapse.mp4` from EVERY night's PNG in the folder, in name
    (= night) order — the windows built before this one included — HEVC (a
    36 MP frame is past H.264's largest level) through the Mac's hardware
    encoder, at `FRAMES_PER_SECOND`, the last frame held; returns its path.
    `kept` is this window's frames: what makes the video re-cut when they
    change."""
    folder = Path(root)
    video = folder / "time_lapse.mp4"
    tmp = folder / f".time_lapse.{os.getpid()}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FRAMES_PER_SECOND),
         "-pattern_type", "glob", "-i", str(folder / "*.png"),
         "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration={HOLD_LAST_SECONDS}",
         "-c:v", "hevc_videotoolbox", "-q:v", "65", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp)],
        check=True)
    os.replace(tmp, video)
    return str(video)


# supersede=False: every night's as-of master of a stack, and every night's
# layer of the mosaic, lands under the SAME version-log key (object, this
# pipeline, display, filter), so N-1 supersession would delete each earlier
# night's file as the next one lands — while other nights still read it
# (2026-10-28: FileNotFoundError mid-pass). The window sweep bounds them instead.
@pipeline(PIPELINE, processes=Mosaic, supersede=False)
def for_mosaic(mosaic, config):
    """One frame per night of shooting, cumulative; the video at the end."""
    full_targets = {t.stable_hash: t for t in mosaic.targets}
    stacks_by_target = {
        tid: tuple(s for group in _resolved_by_filter(t.stack_ids).values() for s in group)
        for tid, t in full_targets.items()
    }
    all_stacks = {s.stable_hash: s for stacks in stacks_by_target.values() for s in stacks}
    quarantine = Quarantine(config.decisions, all_stacks.get)
    nights = nights_of(all_stacks.values())
    if not nights:
        return {}
    window = nights_window(mosaic, config.decisions, nights)
    final, framing = nights[-1], canvas_night(mosaic, config.decisions, nights)
    built = {}
    for night in sorted({*window, final, framing}):   # the final night seeds the stretch
        as_of = mosaic_as_of(mosaic, full_targets, stacks_by_target, night, quarantine,
                             config.min_frames_in_stack)
        if as_of is not None:
            built[night] = as_of
    canvas = time_lapse_canvas(built[framing][0])

    layers = {}
    for night, (m, targets, channel_stacks) in built.items():
        if night != final and night not in window:
            continue
        channels = {(tid, f): channel(targets[tid], f, stacks, config)
                    for (tid, f), stacks in channel_stacks.items()}
        starless, stars = _merged_channels(m, targets, channels, config, canvas,
                                           polish=len(targets) > 1, bxt=False)
        layers[night] = (starless, stars)
    composites = {night: sho_composite(canvas, sl.get("S"), sl.get("H"), sl.get("O"))
                  for night, (sl, _) in layers.items()}
    values = hubble_values(composites[final], saved_look(mosaic, config.decisions))

    root = str(Path(context.output_dir).parent / "TimeLapse" / mosaic.stable_hash)
    kept = []
    for night in sorted(n for n in layers if n in window or n == final):
        i = nights.index(night)
        sl, st = layers[night]
        runs = [r for s in all_stacks.values() for h, r in s.contents.items()
                if h in runs_through(s, night, quarantine.runs_of(s))]
        hours = sum(len(r.frames) * float(r.exposure) for r in runs) / 3600.0
        caption = (f"{night}   night {i + 1} of {len(nights)}   "
                   f"{hours:.1f} h   {sum(len(r.frames) for r in runs)} frames")
        inputs = (composites[night], sl.get("R"), sl.get("G"), sl.get("B"),
                  st.get("R"), st.get("G"), st.get("B"), st.get("H"))
        keep_layers(root, str(night), caption, *inputs)   # the final night's too: the anchors' source
        if night in window:
            frame = hubble_frame(mosaic.stable_hash, str(night), caption, values, *inputs)
            kept.append(keep_frame(root, str(night), frame))
    if not kept:
        return {}
    return {"Time-lapse": time_lapse_video(root, *kept)}
