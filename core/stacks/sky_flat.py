"""Night-sky flats — a FLAT, distinguished by its ILLUMINANT.

A flat measures the system's normalised response to a uniform source. What
makes this one different is only what lit it: the night sky, rather than a
panel or twilight.

That matters because the sensor's interference fringing depends on the
illuminating SPECTRUM. A thinned, back-illuminated sensor modulates its
quantum efficiency with a period of a few nm, so a near-monochromatic source
fringes at full depth while a continuum source averages over the modulation
and fringes far less (by `sinc(W / dlambda)` for a top-hat passband). The
night sky is dominated by emission lines; a flat panel and the twilight sky
are continuum. So a night-sky flat captures the etalon at the SAME spectral
weighting the science frames see it, and a panel or twilight flat does not —
which is precisely the deficit `scale_star_flat`'s k (~1.33 H, ~1.41 S) was
invented to patch on the star flat, and which this needs no patch for.

WHY DIVIDE, when the literature says fringes are additive. Write a frame as
sky plus object, each with its own response to its own spectrum:

    L - dark = S.R(x) + O(x).R_obj(x)          R = V.(1 + f)

`V` is the spectrum-independent part (vignetting, dust, QE); `f` the fringe.
Dividing by this flat (`1 + M = R/<R>`):

    (L-dark)/(1+M) = S.<R> + O(x).<R>.[R_obj/R],  R_obj/R ~ 1 + (f_obj - f_sky)

`V` cancels EXACTLY — vignetting and dust come off the object too, not just
the background. The only residual is the DIFFERENCE in fringe depth between
the object's spectrum and the sky's. Subtracting a scaled map instead leaves
`S.<R> + O.R_obj`: the background flattens but the object keeps vignetting,
dust and etalon whole, so subtraction is not an alternative to a flat, only a
supplement to a fringe-free one.

The published recipes subtract because they are BROADBAND: there the objects
are continuum (`f_obj ~ 0`) while the sky lines fringe, so dividing by the
sky's response would print `-f_sky` — the full etalon, inverted — onto every
star. In NARROWBAND the object is line emission at essentially the filter's
centre, `f_obj ~ f_sky`, and that failure mode disappears. Stars remain
continuum and are over-corrected by ~0.25 x the etalon (~0.09%), which is
negligible and spread over a 256-1024 px pattern.

MEASURED, 2026-09-25, two panels of the Cygnus mosaic (Kite Cluster 57
frames / 7 runs; Sadr Region 66 / 9), five calibration arms each, frame-for-
frame identical accounting: dividing by this flat beat subtracting the same
map, and matched or beat both the star flat and a twilight flat. The etalon
was not visible in any arm on Sadr, so this is a SIMPLIFICATION — one flavour
of flat instead of a fourth calibration type — and not a quality claim.

STORED like every other master flat: `(1 + M) * 0.5`, MULTIPLICATIVE, median
0.5, exactly as `star_flat.py` writes its response. It must never be stored
as `M + 0.5` (a zero-mean map on a pedestal): anything that normalises by the
median then divides by `1 + 2M` and applies the pattern at DOUBLE amplitude.
That bug shipped briefly and was caught by comparing rms against the map's.

BUILT dark-only. Its members are lights, so it takes a dark and nothing else
(`CALIBRATION_TOPOLOGY["sky"]`). Dividing the members by a flat first would
make this the RESIDUAL that flat could not remove — a different object, and
the one that came out as the star flat inverted (corr -0.981) when the flat
it was paired with was wrong.

NOT a separate calibration type. `kind` is "sky"; it resolves through the
FLAT slot, ahead of the star flat, exactly as `StarFlatStack` (kind "star")
already does. One flat ladder, three flavours: sky, star, panel.
"""

# NO `from __future__ import annotations` here, deliberately. `@node` decides
# an op is an artifact producer by resolving its RETURN ANNOTATION at
# decoration time; under PEP 563 that annotation is still a string while the
# module is executing, resolution fails, and the op silently never gets a
# `filename_fn`. `build_sky_flat` then dies on its first real trace with
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

SECTION = "sky_flat"        # the built frame's sidecar section: quality metrics
# WBPP recognises a pre-built master only as "Master Flat" — and that is
# what this IS, so there is nothing to disguise.
IMAGE_TYPE = "Master Flat"
# Below this many member frames the map is noise: the fringe is ~0.4% of the
# sky and a single 300 s sub's per-pixel noise is ~45x larger, so it only
# emerges after averaging at the pattern's own scale over many frames. 15 is
# the floor at which a 32 px-smoothed map reaches ~35% precision.
MIN_FRAMES = 15
# How the stars come out (see `combine_sky_flat`). A block median at 8x rejects
# any star smaller than ~4 px outright; a 9-cell median filter on top of that
# reaches ~72 px at full resolution, which covers the haloes of all but the
# brightest. Measured: worst surviving star drops from 217 sigma to 7.7.
BLOCK = 8
MEDIAN_FOOTPRINT = 9


@beartype
@dataclass(frozen=True)
class SkyFlatStack(CalibrationStack):
    """A night-sky flat whose "frames" are the blank-sky lights of the Stack a
    user promoted. `contents` is empty — there is nothing to integrate in the
    calibration sense; `member_files` are the lights the combine reads."""
    source_stack_id: str = ""
    member_files: tuple[str, ...] = ()
    focal_length: float | None = None

    @property
    def kind(self) -> str:
        return "sky"

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
        return (f"SkyFlatStack(Content hash {self.content_hash}, Camera {self.camera}, "
                f"Filter {self.filter}, Focal {self.focal_length}, Frames {self.n_frames})")


class SkyFlatStacks(dict):
    """`{StackHash: SkyFlatStack}` — joins the calibration stacks so light↔cal
    resolution finds night-sky flats the same way it finds any other flat."""
    def __hash__(self):
        return hash_list_int([self[k].content_hash for k in sorted(self)])

    def __str__(self):
        return f"SkyFlatStacks({len(self)})"


def sky_flat_id_for(stack_id: str) -> StackHash:
    """The id `sky_flat_for` will give the night-sky flat of this stack —
    without needing the stack itself (the UI asks before one exists).

    DERIVED from the source's, never equal to it: a night-sky flat and the
    stack it came from are different objects and must not collide on an id.
    """
    return StackHash(hash_list([str(stack_id), "sky_flat"]))


def sky_flat_for(source, metadata) -> SkyFlatStack:
    """The `SkyFlatStack` a promoted Stack yields.

    Identity follows the house convention — the members' content hashes —
    so the frame re-builds when its source gains frames and not otherwise.
    `stable_hash` is derived from the SOURCE stack's, which is what makes the
    product addressable and stable as the source grows; it is derived rather
    than reused so a night-sky flat and its source stack never collide on an id.

    `contents` is ONE synthetic run over every member frame, typed `sky`.
    That run is what `CALIBRATION_TOPOLOGY["sky"]` resolves a dark
    against — and one run rather than the source's own means one dark, which
    is all `combine_sky_flat` can use. A sky-flat source is a dedicated blank-sky
    shoot, so that is also what it really is.
    """
    frames = sorted(source.all_frames())
    sub = MetadataDict({f: metadata[f] for f in frames if f in metadata})
    assert sub, f"sky-flat source {source.stable_hash} has no metadata for its frames"
    files = tuple(sorted(str(m.filename) for m in sub.values()))
    chron = ChronologyAttributes.from_frames(list(sub), sub)
    # The synthetic RUN is typed "sky" so CALIBRATION_TOPOLOGY gives it a
    # DARK (its members are lights). The STACK below is typed "flat", so it
    # appears in the flat picker and resolves through the flat slot — the
    # same split `StarFlatStack` uses (run-less, stack image_type "flat",
    # kind "star").
    imaging = replace(ImagingAttributes.from_frames(list(sub), sub), image_type="sky")
    content_hash = hash_list(sorted(str(f) for f in sub))
    attributes: dict[str, Any] = asdict(chron) | asdict(imaging) | {
        "content_hash": content_hash,
        "frames": tuple(sub),
        "metadata": sub,
        "dependencies": frozendict(),
        "state": frozendict(),
    }
    run = CalibrationRun(**attributes)
    return SkyFlatStack(
        content_hash=content_hash,
        image_type="flat",
        camera=source.camera,
        filter=source.filter,
        mean_temperature=source.mean_temperature,
        exposure=source.exposure,
        integration=source.integration,
        width_pixels=source.width_pixels,
        height_pixels=source.height_pixels,
        stable_hash=sky_flat_id_for(source.stable_hash),
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
        if key.decision_type == DecisionType.SKY_FLAT_SOURCE
        and override.choice == BinaryChoice.INCLUDE
    }


@node
def identify_sky_flats(decision_overrides, stacks: dict) -> SkyFlatStacks:
    """One `SkyFlatStack` per promoted Stack.

    Takes the light stacks because a night-sky flat is derived from one: the
    calibration chain genuinely waits on identification here, which is the
    cost of a flat whose source is a science stack."""
    out = SkyFlatStacks()
    for sid in sorted(promoted_stack_ids(decision_overrides)):
        ref = stacks.get(sid)
        source = ref.get() if hasattr(ref, "get") else ref
        if source is None:
            log.info("sky-flat source no longer exists", stack=sid)
            continue
        sky_flat = sky_flat_for(source, source.all_metadata())
        out[sky_flat.stable_hash] = sky_flat
    log.info("night-sky flats", count=len(out),
             detail={k: str(v) for k, v in out.items()})
    return out


@node
def merge_sky_flats(
    sky_flats: SkyFlatStacks, cal_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    """Night-sky flats join the calibration stacks, so every light↔cal
    resolution finds them exactly the way it finds any other flat."""
    return CalibrationStacksDict({**cal_stacks, **sky_flats})


def sky_flat_master_path(stable_hash) -> Path | None:
    """The built flat for this sky-flat stack, or None when it is not on disk.

    Found by scanning the producer's output directory for the file whose
    `STBLHASH` matches, rather than by replaying `build_sky_flat`'s
    argument tuple through the cache: the preview path does not have the
    dark and flat `ImageMetadata` those arguments carry, and a preview must
    never trigger a build. `gate()` locates star-flat masters the same way.
    """
    from context import context
    out = Path(context.output_dir) / "build_sky_flat"
    if not out.is_dir():
        return None
    for f in sorted(out.glob("*.xisf")):
        try:
            if str(read_header(f).get("STBLHASH")) == str(stable_hash):
                return f
        except (OSError, ValueError):
            continue
    return None


def combine_sky_flat(
    member_files: list[str], dark: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """The zero-mean modulation per unit sky — Arne Henden's recipe, whole.

    Dark-subtracted only. Dividing the members by a flat first would make
    this the RESIDUAL that flat could not remove — a different object, and
    the one that came out as the star flat inverted when the flat was wrong.

    "Take a few long exposures of dithered blank sky, remove the stars, and
    form a master flat." That is three steps and this is all three:

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
    }
    return fringe.astype(np.float32), quality


def write_sky_flat(response: np.ndarray, dest, keywords: dict) -> None:
    """Write the flat as a WBPP-ready Master Flat XISF.

    `response` is the ZERO-MEAN modulation `M`; what lands on disk is
    `(1 + M) * 0.5` — multiplicative, median 0.5, byte-identical in
    convention to `star_flat.write_master_flat`. Anything that normalises by
    the median then divides by `1 + M`, which is the amplitude the map
    actually measured.

    Storing `M + 0.5` instead (a zero-mean map on an additive pedestal) makes
    the median normalisation yield `1 + 2M` and applies the pattern at DOUBLE
    amplitude. That is not a hypothetical: it shipped, and was caught only by
    comparing the normalised rms (0.00681) against the map's own (0.00341).
    """
    from xisf import XISF
    r = ((1.0 + np.asarray(response, np.float64)) * 0.5).astype(np.float32)
    fits_keywords = {k: [{"value": str(v), "comment": ""}]
                     for k, v in keywords.items() if v is not None}
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    XISF.write(str(dest), r[..., None], creator_app="AstroFlow night-sky flat",
               image_metadata={"FITSKeywords": fits_keywords, "XISFProperties": {}})


def read_sky_flat(path) -> np.ndarray:
    """The zero-mean modulation `M` back, however the file was scaled.

    Normalising by the median rather than subtracting a recorded pedestal
    makes this convention-independent: it reads a master written here, a
    star flat, or anything else stored as a flat.
    """
    img = np.asarray(read_image(Path(path)), np.float64).squeeze()
    m = float(np.nanmedian(img))
    return (img / (m if abs(m) > 1e-12 else 1.0) - 1.0).astype(np.float32)


# `plate_solve="none"`: a fringe map is a zero-mean residual with the stars
# median-filtered out and no detail finer than a block — there is nothing to
# solve, and the solver raising `PlateSolveFailedException` fails the node and
# sends it round again forever. The OLD map solved only because its star
# rejection leaked 16k residuals; removing them properly exposed this.
@node(num_cpus=4, display="Fringe Frame", resources={"diskcache": 0.001},
      plate_solve="none")
def build_sky_flat(
    sky_flat: SkyFlatStack, dark: ImageMetadata | None,
) -> WithSidecar[XisfFilePath]:
    """The blank-sky residual for one promoted stack.

    Head-pinned for the same reason `build_star_flat` is: it reads every
    member frame, and a freshly joined worker briefly reports existing files
    as missing. A missing member is an error, never "no data" — the task
    fails and is retried rather than minting a blank map.

    `dark` and `flat` are the source stack's own resolved masters. A flat
    that is WRONG lands in the map; see the module docstring. On a train with
    no trustworthy flat the decision resolves to none, and that is correct."""
    missing = [p for p in sky_flat.member_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {sky_flat.n_frames} member frames unreadable here, e.g. {missing[0]}")
    dest = build_sky_flat.filename_fn(sky_flat, dark)
    d = np.asarray(read_image(Path(dark.filename)), np.float64) if dark is not None else None
    if d is not None and d.ndim == 3:
        d = d[:, :, 0]
    log.info("building fringe frame", stack=str(sky_flat), frames=sky_flat.n_frames)

    if sky_flat.n_frames < MIN_FRAMES:
        # Too few frames for the pattern to emerge from the noise. A map of
        # zeros subtracts nothing, which is the honest answer, and the reason
        # rides in the sidecar so the UI can say why.
        out = np.zeros((sky_flat.height_pixels, sky_flat.width_pixels), np.float32)
        quality = {"n_frames": sky_flat.n_frames, "good": False,
                   "reason": f"only {sky_flat.n_frames} frames; need {MIN_FRAMES}"}
    else:
        out, quality = combine_sky_flat(list(sky_flat.member_files), d)
        quality["good"] = True
    write_sky_flat(out, dest, {
        "IMAGETYP": IMAGE_TYPE, "FILTER": sky_flat.filter or "", "INSTRUME": sky_flat.camera,
        "FOCALLEN": sky_flat.focal_length, "XBINNING": 1, "YBINNING": 1,
        "EXPOSURE": float(sky_flat.exposure), "EXPTIME": float(sky_flat.exposure),
        "STBLHASH": str(sky_flat.stable_hash), "FRNGSRC": str(sky_flat.source_stack_id),
        "FRNGNUM": sky_flat.n_frames,
    })
    set_section(Path(dest), SECTION, quality)
    log.info("fringe frame built", stack=str(sky_flat), **quality)
    return WithSidecar(XisfFilePath(dest))
