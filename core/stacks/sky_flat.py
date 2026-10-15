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
which is precisely the deficit the star flat's fitted k (~1.33 H, ~1.41 S) was
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
# How the stars come out (see `combine_sky_flat`). The PRIMARY rejector is the
# median ACROSS frames: the field is dithered, so a star sits on a different
# pixel every frame while the sensor's pattern does not move at all. That is
# what a median over 20 dithered frames is for, and it costs the pattern
# nothing.
#
# A block median at 8x additionally rejects any star smaller than ~4 px, and
# doubles as the downsample that makes the combine cheap. 8x = 8 native px,
# far finer than the 64-400 px etalon cells, so it costs nothing either.
#
# MEDIAN_FOOTPRINT is now a light residual cleanup, NOT the star rejector. It
# was 9 (=72 native px), which sits at the bottom of the etalon's own cell
# range and measurably crushed the pattern: the S map came out with 0.00107
# of etalon-band rms against the star flat's 0.00399 (2026-09-25). Spatial
# smoothing big enough to remove a star is big enough to remove the signal —
# that is why the rejection has to happen across frames instead.
BLOCK = 8
MEDIAN_FOOTPRINT = 3
# Object masking, the canonical star rejector (IRAF `objmasks`): pixels above
# `OBJECT_SIGMA` robust sigma over the frame's own sky are excluded, then the
# mask is DILATED by `OBJECT_PAD` px to catch the low-level haloes a bare
# threshold leaves behind. Masking beats smoothing because it removes a star
# and leaves every other pixel untouched, where a kernel wide enough to cover
# a star also degrades the 64-400 px etalon cells everywhere.
OBJECT_SIGMA = 3.0
OBJECT_PAD = 6


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
    """The stacks the user explicitly marked as blank sky."""
    return _sky_flat_decisions(decision_overrides, BinaryChoice.INCLUDE)


def demoted_stack_ids(decision_overrides) -> set[str]:
    """The stacks the user explicitly said are NOT blank sky — the way to
    overrule a stack shot as SkyFlat panes (`sky_flat_plans`)."""
    return _sky_flat_decisions(decision_overrides, BinaryChoice.EXCLUDE)


def _sky_flat_decisions(decision_overrides, choice) -> set[str]:
    return {
        key.subject_id
        for key, override in decision_overrides.items()
        if key.decision_type == DecisionType.SKY_FLAT_SOURCE
        and override.choice == choice
    }


def is_sky_flat_source(stack_id: str, filenames, decision_overrides) -> bool:
    """A Stack is a night-sky flat source when the user promoted it, or when
    it was shot as SkyFlat panes and the user has not demoted it. Blank-sky
    frames arrive stamped `Light` like any other, so the pane NAME the
    ASIAir writes into each file name is the only mark they carry."""
    from stacks.sky_flat_plans import is_named_sky_flat
    if stack_id in promoted_stack_ids(decision_overrides):
        return True
    if stack_id in demoted_stack_ids(decision_overrides):
        return False
    return is_named_sky_flat(filenames)


@node
def identify_sky_flats(decision_overrides, stacks: dict) -> SkyFlatStacks:
    """One `SkyFlatStack` per promoted Stack.

    Takes the light stacks because a night-sky flat is derived from one: the
    calibration chain genuinely waits on identification here, which is the
    cost of a flat whose source is a science stack."""
    out = SkyFlatStacks()
    promoted = promoted_stack_ids(decision_overrides)
    for sid in sorted(promoted - set(stacks)):
        log.info("sky-flat source no longer exists", stack=sid)
    # Every stack is read (1.5 s for ~1000): a stack shot as SkyFlat panes is
    # a source without a decision, so the names have to be looked at.
    for sid in sorted(stacks):
        ref = stacks[sid]
        source = ref.get() if hasattr(ref, "get") else ref
        if source is None:
            continue
        frames = [str(m.filename) for m in source.all_metadata().values()]
        if not is_sky_flat_source(str(sid), frames, decision_overrides):
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


def sky_flat_master_path(stable_hash, output_dir=None) -> Path | None:
    """The built flat for this sky-flat stack, or None when it is not on disk.

    Found by scanning the producer's output directory for the file whose
    `STBLHASH` matches, rather than by replaying `build_sky_flat`'s
    argument tuple through the cache: the preview path does not have the
    dark and flat `ImageMetadata` those arguments carry, and a preview must
    never trigger a build. `gate()` locates star-flat masters the same way.
    `output_dir`: the web server passes its own (Ray-free) context's.
    """
    if output_dir is None:
        from context import context
        output_dir = context.output_dir
    out = Path(output_dir) / "build_sky_flat"
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
      2. MASK the objects — pixels over OBJECT_SIGMA robust sigma above the
         frame's own sky, dilated by OBJECT_PAD for their haloes (IRAF
         `objmasks`). Masking EXCLUDES a star and leaves every other pixel
         alone; a spatial kernel wide enough to cover a star also degrades
         the 64-400 px etalon cells everywhere, which is how an earlier
         72 px median left the S map at a quarter of the star flat's
         etalon-band amplitude;
      3. BLOCK-MEDIAN down by `BLOCK`, ignoring masked pixels, then MEDIAN
         across frames. The dither is what makes this work: a star lands on
         a different pixel every frame and is outvoted, while the sensor
         pattern is in the same place every time. Then subtract 1.

    NOT done, deliberately, though the literature does it: subtracting a
    low-order fit per frame. Published recipes build the fringe frame from
    ALREADY FLAT-FIELDED frames, so a low-order fit removes residual gradient
    and leaves only fringe. This IS the flat, so a low-order fit would delete
    the vignetting it exists to measure. Per-frame sky gradients are
    suppressed by dither and rotation diversity instead.

    4. Divide out the flat's own DUST MOTES (`stacks.flat_motes`): a few
         round, compact, significant dips. The flat is reused across epochs;
         its dust is not. Real panel/twilight flats are never cleaned — their
         dust is supposed to be there.

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
    from scipy.ndimage import binary_dilation, median_filter, zoom

    def calibrated(path: str) -> np.ndarray:
        f = np.asarray(read_image(Path(path)), np.float64)
        if f.ndim == 3:
            f = f[:, :, 0]
        if dark is not None:
            f = f - dark
        m = float(np.nanmedian(f[::8, ::8]))
        return f / (m if abs(m) > 1e-6 else 1.0)

    def masked(f: np.ndarray) -> np.ndarray:
        """Objects out, everything else untouched (IRAF `objmasks`)."""
        sky = float(np.nanmedian(f))
        sigma = 1.4826 * float(np.nanmedian(np.abs(f - sky))) or 1e-6
        bad = f > sky + OBJECT_SIGMA * sigma
        bad = binary_dilation(bad, iterations=OBJECT_PAD)
        out = f.copy()
        out[bad] = np.nan
        return out

    def starless(f: np.ndarray) -> np.ndarray:
        """Block-median down, ignoring masked pixels."""
        h, w = (f.shape[0] // BLOCK) * BLOCK, (f.shape[1] // BLOCK) * BLOCK
        blocks = f[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
        with np.errstate(invalid="ignore"):
            small = np.nanmedian(blocks, axis=(1, 3))
        return small

    shape = calibrated(member_files[0]).shape
    # MEDIAN across the dithers: a star lands on a different pixel every frame
    # and is outvoted; the sensor pattern is in the same place every time.
    # `nanmedian` so masked pixels simply do not vote.
    cube = np.stack([starless(masked(calibrated(p))) for p in member_files])
    with np.errstate(invalid="ignore"):
        small = np.nanmedian(cube, axis=0)
    # any cell masked in EVERY frame (there should be none with real dithering)
    small = np.where(np.isfinite(small), small, np.nanmedian(small))
    small = median_filter(small, size=MEDIAN_FOOTPRINT)   # light residual cleanup
    small = small / float(np.nanmedian(small)) - 1.0

    # The flat's own dust goes: a night-sky flat is shot once and reused
    # across epochs, and its motes would print into later lights as bright
    # inverse motes. The etalon and vignetting stay exact; each epoch's dust
    # is the lights' business (the mote map). Done here, at block
    # resolution, because the upsample below invents the lattice a model
    # fitted at full resolution would imprint.
    from stacks.flat_motes import divide_motes, find_motes
    flat_b = 1.0 + small
    motes = find_motes(flat_b, BLOCK)
    if motes:
        flat_b = divide_motes(flat_b, motes, BLOCK)
        small = flat_b / float(np.nanmedian(flat_b)) - 1.0

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
        "object_sigma": OBJECT_SIGMA,
        "object_pad": OBJECT_PAD,
        "combine": "masked-median-across-frames",
        "dark_subtracted": dark is not None,
        "motes": motes,      # the flat's own dust, divided out (full-res px)
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


# Never plate-solve a fringe map: it is a zero-mean residual with the stars
# median-filtered out and no detail finer than a block — there is nothing to
# solve, and the solver raising `PlateSolveFailedException` fails the node and
# sends it round again forever. The OLD map solved only because its star
# rejection leaked 16k residuals; removing them properly exposed this.
@node(num_cpus=4, display="Fringe Frame", resources={"diskcache": 0.001})
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
