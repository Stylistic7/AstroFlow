"""Fringe frames — the SUBTRACTIVE calibration type.

Interference in a thinned, back-illuminated sensor modulates its quantum
efficiency at a given wavelength. The night sky's emission lines are
effectively monochromatic and fringe at full amplitude; a continuum source
integrates over the fringe's ~5 nm period and fringes less. So the pattern
rides on the SKY, in proportion to the monochromatic sky, and not on the
total flux — which is why it is removed by SUBTRACTING a scaled map rather
than by dividing a flat (Arne Henden, AAVSO, 2012-10-20: "this is just like
a sky background (actually, it is EXACTLY like sky, just a modulated
version), so you can easily subtract it out").

Dividing instead — which is what a flat carrying the fringe does — is wrong
three ways, all small but systematic: it modulates the STARS, which fringe
at only ~75% of the sky's amplitude in a 3 nm passband (measured here as
`scale_star_flat`'s k ~ 1.33 for H, 1.41 for S); it applies the correction
regardless of how much sky is present, so it INJECTS a ripple into a
low-sky frame that never had one; and the flat's own illumination weights
the passband differently, which damps the pattern unevenly across the field
rather than merely scaling it (the phase shift with wavelength goes as the
local fringe order, so thicker silicon is damped more).

WHAT IS IN A FRINGE FRAME, and why the flat it is divided by must be RIGHT.
With V the true optical response and f the interference:

    frames:            light - dark  =  sky . V . (1 + f)
    divided by flat F: sky . (V/F) . (1 + f)
    averaged over the dither, normalised, minus one:

        map  =  f  +  (V/F - 1)

A CORRECT flat (F = V) leaves the pure fringe, which is the textbook scheme
and what this code is built for. A WRONG flat puts its own error straight
into the map, and if the train's real response is small the error can be the
whole of it: on the 351 mm Ha shoot 2026-09-24, V is only 0.34% while the
star flat carried 1.39% of structure that is not on the sensor, so the map
came out correlating -0.981 with that flat and holding no sky at all.

That is a broken FLAT, not a broken method. Where no trustworthy flat exists
the honest choice is none at all — the flat decision resolves to `NO_FLAT`,
`flat` arrives None here, and the map becomes `f + (V - 1)`: fringe plus the
response. Subtracting it then corrects the sky for both. Note what that does
NOT do: the object never gets flat-fielded, so it is only defensible on a
train whose V is negligible. It is a per-train decision, deliberately, and
not a property of fringe frames — which is why `CALIBRATION_TOPOLOGY` still
gives a fringe frame a flat to resolve.

A stacker then removes it exactly as Arne Henden describes (AAVSO,
2012-10-20): "You then just scale that to match the sky background in your
image, subtract, and voila! The fringing goes away." The map is already per
unit sky, so the scale factor IS the frame's sky level — no fit, no filter.

Arne's own answer to the thing a fit would chase — "the night sky emission
can vary during a night, so you need a fringe map that corresponds with the
monochromatic illumination source present at the time of the science image"
— is to SHOOT a matching map, not to compute a correction for a mismatched
one. Match the conditions at the telescope and the scale factor is just the
sky. (He lists two more: the pattern depends on wavelength, so a map is
per-filter; and it is a weak function of temperature, so it is per setpoint.
Both fall out of matching on camera, filter and optical train.)

GOOD SIGNAL-TO-NOISE MATTERS. Arne's last gotcha: "since you are
subtracting this, you want good signal/noise in the master fringe image, not
always easy to obtain." Hence `MIN_FRAMES` — a map built from too few frames
is noise that gets added to every science frame it touches.

A DISTINCT CALIBRATION TYPE, all the way through. `bias`, `dark` and `flat`
have a fourth sibling in `CALIBRATION_TOPOLOGY`, it resolves per Run like
the others, and it reaches a stacker as one more entry in `*cal_masters`.
How a stacker APPLIES it is the stacker's business: the PixInsight backend
folds it into the dark it stages for WBPP because WBPP has no other
subtractive slot, and that quirk stays inside
`stacks/stacking/pixinsight.py`. An OSS stacker should just subtract it.

OPTIONAL, and nobody's default. Blank-sky frames arrive stamped `Light`
like any other, so nothing in a header can identify them — the user marks a
stack as the source (`DecisionKey.fringe_source`), which is the only place
that knowledge exists.
"""
# NO `from __future__ import annotations` here, deliberately. `@node` decides
# an op is an artifact producer by resolving its RETURN ANNOTATION at
# decoration time; under PEP 563 that annotation is still a string while the
# module is executing, resolution fails, and the op silently never gets a
# `filename_fn`. `build_fringe_frame` then dies on its first real trace with
# "'function' object has no attribute 'filename_fn'" — invisible to every
# test, because no test calls the producer body. `stacks/star_flat.py` omits
# it for the same reason.

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from beartype import beartype
from frozendict import frozendict

from attributes import ChronologyAttributes, ImagingAttributes
from calibration.calibration_frames import focal_length_of
from decisions.decisions import BinaryChoice, DecisionType
from decorators import node
from file_io.generic import read_header, read_image
from file_io.sidecar import set_section
from file_io.types import WithSidecar, XisfFilePath
from hashing import hash_list, hash_list_int
from metadata.classes import ImageMetadata, MetadataDict
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict, RunsDict
from stacks.run import CalibrationRun
from stacks.types import StackHash

log = structlog.get_logger()

SECTION = "fringe"          # the built frame's sidecar section: quality metrics
IMAGE_TYPE = "Master Fringe"
# Below this many member frames the map is noise: the fringe is ~0.4% of the
# sky and a single 300 s sub's per-pixel noise is ~45x larger, so it only
# emerges after averaging at the pattern's own scale over many frames. 15 is
# the floor at which a 32 px-smoothed map reaches ~35% precision.
MIN_FRAMES = 15
# How the stars come out (see `combine_fringe`). A block median at 8x rejects
# any star smaller than ~4 px outright; a 9-cell median filter on top of that
# reaches ~72 px at full resolution, which covers the haloes of all but the
# brightest. Measured: worst surviving star drops from 217 sigma to 7.7.
BLOCK = 8
MEDIAN_FOOTPRINT = 9


@beartype
@dataclass(frozen=True)
class FringeStack(CalibrationStack):
    """A fringe frame whose "frames" are the blank-sky lights of the Stack a
    user promoted. `contents` is empty — there is nothing to integrate in the
    calibration sense; `member_files` are the lights the combine reads."""
    source_stack_id: str = ""
    member_files: tuple[str, ...] = ()
    focal_length: float | None = None

    @property
    def kind(self) -> str:
        return "fringe"

    @property
    def n_frames(self) -> int:
        return len(self.member_files)

    def __hash__(self):
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CalibrationStack):
            return NotImplemented
        return (self.content_hash, self.stable_hash) == (other.content_hash, other.stable_hash)

    def __str__(self) -> str:
        return (f"FringeStack(Content hash {self.content_hash}, Camera {self.camera}, "
                f"Filter {self.filter}, Focal {self.focal_length}, Frames {self.n_frames})")


class FringeStacks(dict):
    """`{StackHash: FringeStack}` — joins the calibration stacks so light↔cal
    resolution finds fringe frames the same way it finds flats."""
    def __hash__(self):
        return hash_list_int([self[k].content_hash for k in sorted(self)])

    def __str__(self):
        return f"FringeStacks({len(self)})"


def fringe_stack_id_for(stack_id: str) -> StackHash:
    """The id `fringe_stack_for` will give the fringe frame of this stack —
    without needing the stack itself (the UI asks before one exists).

    DERIVED from the source's, never equal to it: a fringe frame and the
    stack it came from are different objects and must not collide on an id.
    """
    return StackHash(hash_list([str(stack_id), "fringe"]))


def fringe_stack_for(source, metadata) -> FringeStack:
    """The `FringeStack` a promoted Stack yields.

    Identity follows the house convention — the members' content hashes —
    so the frame re-builds when its source gains frames and not otherwise.
    `stable_hash` is derived from the SOURCE stack's, which is what makes the
    product addressable and stable as the source grows; it is derived rather
    than reused so a fringe frame and its source stack never collide on an id.

    `contents` is ONE synthetic run over every member frame, typed `fringe`.
    That run is what `CALIBRATION_TOPOLOGY["fringe"]` resolves a dark
    against — and one run rather than the source's own means one dark, which
    is all `combine_fringe` can use. A fringe source is a dedicated blank-sky
    shoot, so that is also what it really is.
    """
    frames = sorted(source.all_frames())
    sub = MetadataDict({f: metadata[f] for f in frames if f in metadata})
    assert sub, f"fringe source {source.stable_hash} has no metadata for its frames"
    files = tuple(sorted(str(m.filename) for m in sub.values()))
    chron = ChronologyAttributes.from_frames(list(sub), sub)
    imaging = replace(ImagingAttributes.from_frames(list(sub), sub), image_type="fringe")
    content_hash = hash_list(sorted(str(f) for f in sub))
    attributes: dict[str, Any] = asdict(chron) | asdict(imaging) | {
        "content_hash": content_hash,
        "frames": tuple(sub),
        "metadata": sub,
        "dependencies": frozendict(),
        "state": frozendict(),
    }
    run = CalibrationRun(**attributes)
    return FringeStack(
        content_hash=content_hash,
        image_type="fringe",
        camera=source.camera,
        filter=source.filter,
        mean_temperature=source.mean_temperature,
        exposure=source.exposure,
        integration=source.integration,
        width_pixels=source.width_pixels,
        height_pixels=source.height_pixels,
        stable_hash=fringe_stack_id_for(source.stable_hash),
        oldest_date=source.oldest_date,
        newest_date=source.newest_date,
        state=frozendict(),
        contents=RunsDict({str(run.stable_hash): run}),
        camera_id_hash=source.camera_id_hash,
        source_stack_id=str(source.stable_hash),
        member_files=files,
        # The same reader the matcher uses, so the frame is filed under the
        # train it will be matched against: the FIRST header that carries a
        # FOCALLEN, not the first frame's (which may simply lack one).
        focal_length=focal_length_of(run),
    )


def promoted_stack_ids(decision_overrides) -> set[str]:
    """The stacks the user marked as blank sky. Nothing in a header says so —
    blank-sky frames arrive stamped `Light` like any other — so this decision
    is the only place that knowledge exists."""
    return {
        key.subject_id
        for key, override in decision_overrides.items()
        if key.decision_type == DecisionType.FRINGE_SOURCE
        and override.choice == BinaryChoice.INCLUDE
    }


@node
def identify_fringe_stacks(decision_overrides, stacks: dict) -> FringeStacks:
    """One `FringeStack` per promoted Stack.

    Takes the light stacks because a fringe frame is derived from one: the
    calibration chain genuinely waits on identification here, which is the
    cost of a calibration type whose source is a science stack."""
    out = FringeStacks()
    for sid in sorted(promoted_stack_ids(decision_overrides)):
        ref = stacks.get(sid)
        source = ref.get() if hasattr(ref, "get") else ref
        if source is None:
            log.info("fringe source no longer exists", stack=sid)
            continue
        fringe = fringe_stack_for(source, source.all_metadata())
        out[fringe.stable_hash] = fringe
    log.info("fringe stacks", count=len(out),
             detail={k: str(v) for k, v in out.items()})
    return out


@node
def merge_fringe_stacks(
    fringe_stacks: FringeStacks, cal_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    """Fringe frames join the calibration stacks, so every light↔cal
    resolution finds them exactly the way it finds flats."""
    return CalibrationStacksDict({**cal_stacks, **fringe_stacks})


def fringe_master_path(stable_hash) -> Path | None:
    """The built map for this fringe stack, or None when it is not on disk.

    Found by scanning the producer's output directory for the file whose
    `STBLHASH` matches, rather than by replaying `build_fringe_frame`'s
    argument tuple through the cache: the preview path does not have the
    dark and flat `ImageMetadata` those arguments carry, and a preview must
    never trigger a build. `gate()` locates star-flat masters the same way.
    """
    from context import context
    out = Path(context.output_dir) / "build_fringe_frame"
    if not out.is_dir():
        return None
    for f in sorted(out.glob("*.xisf")):
        try:
            if str(read_header(f).get("STBLHASH")) == str(stable_hash):
                return f
        except (OSError, ValueError):
            continue
    return None


def combine_fringe(
    member_files: list[str], dark: np.ndarray | None, flat: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """The zero-mean modulation per unit sky — Arne Henden's recipe, whole.

    `flat` is whatever the fringe stack's own flat decision resolved to, and
    it had better be right: see the module docstring. None is a legitimate
    answer and the one this train uses.

    "Take a few long exposures of dithered blank sky, remove the stars, and
    form a master fringe image." That is three steps and this is all three:

      1. dark subtract, divide by the flat if there is one, normalise each
         frame by its own median so frames taken through different
         transparency contribute equally;
      2. BLOCK-MEDIAN down by `BLOCK`, then a `MEDIAN_FOOTPRINT` median
         filter — together these are "remove the stars". A block median
         rejects anything smaller than half a block outright, and the filter
         takes the survivors' haloes. The pattern is large-scale (its power
         is unchanged from 8x to 32x binning), so nothing real is lost;
      3. plain mean across frames, then subtract 1.

    Then resample back to full resolution. The map carries no detail finer
    than a block by construction, so the upsample invents nothing — it
    restores the grid the map has to be subtracted on. Working small is the
    point rather than a shortcut: at full resolution a single sub's per-pixel
    noise is several times the pattern, and averaging 20 of them does not get
    below it. Measured on the 351 mm Ha shoot, the full-resolution map was
    1.9% noise around 1.1% signal; the resampled one keeps the signal and
    drops the noise, correlating +0.968 with it.

    No sigma clipping and no object mask: both were in here, both left
    visible stars in the map, and neither survives contact with a median.

    Returns `(fringe, quality)`; the fringe is float32, median 0.
    """
    from scipy.ndimage import median_filter, zoom

    def calibrated(path: str) -> np.ndarray:
        f = np.asarray(read_image(Path(path)), np.float64)
        if f.ndim == 3:
            f = f[:, :, 0]
        if dark is not None:
            f = f - dark
        if flat is not None:
            f = f / np.maximum(flat / np.nanmedian(flat), 1e-6)
        m = float(np.nanmedian(f[::8, ::8]))
        return f / (m if abs(m) > 1e-6 else 1.0)

    def starless(f: np.ndarray) -> np.ndarray:
        h, w = (f.shape[0] // BLOCK) * BLOCK, (f.shape[1] // BLOCK) * BLOCK
        small = np.median(
            f[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK), axis=(1, 3))
        return median_filter(small, size=MEDIAN_FOOTPRINT)

    shape = calibrated(member_files[0]).shape
    acc = None
    for path in member_files:
        small = starless(calibrated(path))
        acc = small if acc is None else acc + small
    small = acc / len(member_files)
    small = small / float(np.nanmedian(small)) - 1.0

    # back to the grid the map is subtracted on; order 1 is enough for a map
    # that holds nothing finer than a block
    full = zoom(small, (shape[0] / small.shape[0], shape[1] / small.shape[1]), order=1)
    fringe = np.zeros(shape, np.float64)
    fringe[:full.shape[0], :full.shape[1]] = full
    fringe = fringe - float(np.nanmedian(fringe))

    quality = {
        "n_frames": len(member_files),
        "rms": float(np.std(fringe)),
        "p1": float(np.percentile(fringe, 1)),
        "p99": float(np.percentile(fringe, 99)),
        "block": BLOCK,
        "median_footprint": MEDIAN_FOOTPRINT,
        "dark_subtracted": dark is not None,
        "flat_divided": flat is not None,
    }
    return fringe.astype(np.float32), quality


def apply_fringe(frame: np.ndarray, fringe: np.ndarray,
                 sky: float | None = None) -> np.ndarray:
    """`frame - sky * fringe` — the correction, and the whole of it.

    `sky` defaults to the frame's own median, which is what "the sky
    background in your image" means. The map is per unit sky, so the product
    is in the frame's own units and the subtraction is dimensionally honest.
    """
    f = np.asarray(frame, np.float64)
    s = float(np.nanmedian(f)) if sky is None else float(sky)
    return (f - s * np.asarray(fringe, np.float64)).astype(frame.dtype)


def write_fringe_frame(fringe: np.ndarray, dest, keywords: dict) -> None:
    """Write the map as an XISF. Stored ZERO-MEAN with an explicit pedestal of
    0.5 so it survives readers that assume [0, 1]; `FRNGPED` records the
    pedestal so a consumer subtracts it back off rather than guessing."""
    from xisf import XISF
    r = (np.asarray(fringe, np.float32) + 0.5).astype(np.float32)
    kw = {**keywords, "FRNGPED": 0.5}
    fits_keywords = {k: [{"value": str(v), "comment": ""}] for k, v in kw.items() if v is not None}
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    XISF.write(str(dest), r[..., None], creator_app="AstroFlow fringe frame",
               image_metadata={"FITSKeywords": fits_keywords, "XISFProperties": {}})


def read_fringe_frame(path) -> np.ndarray:
    """The zero-mean map back, pedestal removed."""
    img = np.asarray(read_image(Path(path)), np.float32).squeeze()
    ped = read_header(Path(path)).get("FRNGPED")
    return img - (float(ped) if ped is not None else 0.0)


# `plate_solve="none"`: a fringe map is a zero-mean residual with the stars
# median-filtered out and no detail finer than a block — there is nothing to
# solve, and the solver raising `PlateSolveFailedException` fails the node and
# sends it round again forever. The OLD map solved only because its star
# rejection leaked 16k residuals; removing them properly exposed this.
@node(num_cpus=4, display="Fringe Frame", resources={"diskcache": 0.001},
      plate_solve="none")
def build_fringe_frame(
    fringe: FringeStack, dark: ImageMetadata | None, flat: ImageMetadata | None,
) -> WithSidecar[XisfFilePath]:
    """The blank-sky residual for one promoted stack.

    Head-pinned for the same reason `build_star_flat` is: it reads every
    member frame, and a freshly joined worker briefly reports existing files
    as missing. A missing member is an error, never "no data" — the task
    fails and is retried rather than minting a blank map.

    `dark` and `flat` are the source stack's own resolved masters. A flat
    that is WRONG lands in the map; see the module docstring. On a train with
    no trustworthy flat the decision resolves to none, and that is correct."""
    missing = [p for p in fringe.member_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {fringe.n_frames} member frames unreadable here, e.g. {missing[0]}")
    dest = build_fringe_frame.filename_fn(fringe, dark, flat)
    d = np.asarray(read_image(Path(dark.filename)), np.float64) if dark is not None else None
    f = np.asarray(read_image(Path(flat.filename)), np.float64) if flat is not None else None
    if d is not None and d.ndim == 3:
        d = d[:, :, 0]
    if f is not None and f.ndim == 3:
        f = f[:, :, 0]
    log.info("building fringe frame", stack=str(fringe), frames=fringe.n_frames)

    if fringe.n_frames < MIN_FRAMES:
        # Too few frames for the pattern to emerge from the noise. A map of
        # zeros subtracts nothing, which is the honest answer, and the reason
        # rides in the sidecar so the UI can say why.
        out = np.zeros((fringe.height_pixels, fringe.width_pixels), np.float32)
        quality = {"n_frames": fringe.n_frames, "good": False,
                   "reason": f"only {fringe.n_frames} frames; need {MIN_FRAMES}"}
    else:
        out, quality = combine_fringe(list(fringe.member_files), d, f)
        quality["good"] = True
    write_fringe_frame(out, dest, {
        "IMAGETYP": IMAGE_TYPE, "FILTER": fringe.filter or "", "INSTRUME": fringe.camera,
        "FOCALLEN": fringe.focal_length, "XBINNING": 1, "YBINNING": 1,
        "EXPOSURE": float(fringe.exposure), "EXPTIME": float(fringe.exposure),
        "STBLHASH": str(fringe.stable_hash), "FRNGSRC": str(fringe.source_stack_id),
        "FRNGNUM": fringe.n_frames,
    })
    set_section(Path(dest), SECTION, quality)
    log.info("fringe frame built", stack=str(fringe), **quality)
    return WithSidecar(XisfFilePath(dest))
