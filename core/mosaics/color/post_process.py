"""OSS mosaic color post-processing: @node ops + the `color_post_process`
workflow_group that `for_mosaic` calls.

PixInsight-free. BXT/SXT run via the rc-astro CLI on LINEAR data; a per-channel
autostretch then makes the dynamic palette weights meaningful; the palettes,
RGB-star combination, and screen blend are numpy (`mosaics.color.palettes`).

The framework is one-artifact-per-`@node`, so the "several outputs" are several
ops: each named color image (`Hubble`, `Foraxx`, `Stars`) is its own `@node`
with its own `display=` name and `filter=None` (a color composite has no single
filter). The per-channel steps are `display`-less intermediates that PERSIST —
every image is worth keeping unless a subgraph explicitly declares otherwise
via `Graph(..., delete_intermediates=True)`, which today is only the panel
crops (a crop is a view of another image, materialized for a CLI boundary).
`mock_pixinsight` passes the heavy rc-astro ML calls straight through so the
suite runs without rc-astro.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyvips
import structlog

from decisions.settings import ProcessingConfig
from decorators import node
from dependency_graph.tracing import workflow_group
from file_io.generic import read_image
from file_io.tiff_file_io import (write_tiff_image, _atomic_tiffsave,
                                  FAST_COMPRESSION, COMPATIBLE_COMPRESSION)
from file_io.tiling import feather_blend_to_bigtiff, tile_size_for
from file_io.types import TiffFilePath, WithSidecar, XisfFilePath
from file_io.xisf_file_io import write_xisf_image
from metadata.classes import ImageMetadata
from mosaics.color import palettes
from mosaics.panels import (
    PanelMosaic, base_panel_mosaic, blend_panels, combine_panels, process_in_panels,
)
# MosaicLike bound as `Mosaic`: widens beartype to accept a Target acting as
# its own single-panel mosaic WITHOUT touching the `mosaic: Mosaic`
# annotations — those serialize into the op code hash, so editing them would
# re-key every op in this module (see mosaics/types.py).
from mosaics.types import MosaicLike as Mosaic
from pix.tasks.filters import normalize_filter
from targets.types import Target

log = structlog.get_logger()

# Canonical single-letter filter classes (via pix/tasks/filters.normalize_filter).
_NARROWBAND = frozenset({"H", "O", "S"})
_BROADBAND = frozenset({"R", "G", "B"})


def _mono(meta: ImageMetadata) -> np.ndarray:
    img = read_image(Path(meta.filename)).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def _write_mono_tiff(path: str, data: np.ndarray, filter: str) -> None:
    """Write a mono channel as TIFF (write_tiff_image forces BigTIFF past 2 GB).
    autostretch emits this so XISF dies at the rc-astro boundary — from here on
    the color path is TIFF + pyvips (see module docstring)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Gigapixel mono channel, read only by pyvips (the color combine) — opt into
    # the fast multi-core codec; nothing downstream feeds it to PixInsight.
    write_tiff_image(Path(path), data.astype(np.float32),
                     {"FILTER": filter, "PIPELINE": "AstroFlow-OSS"},
                     compression=FAST_COMPRESSION)


def _open_mono_vips(meta: ImageMetadata) -> pyvips.Image:
    """Open a channel as a LAZY vips image. TIFFs stream (`new_from_file` —
    tiles/strips decode on demand); any other format (the legacy XISF/FITS
    boundary, which libvips can't read) falls back to one full numpy read
    wrapped in a vips buffer. In the current topology every mosaic-scale
    channel is a TIFF, so the fallback only fires for legacy inputs."""
    p = str(meta.filename)
    if p.lower().endswith((".tif", ".tiff")):
        return pyvips.Image.new_from_file(p, unlimited=True)
    arr = np.ascontiguousarray(_mono(meta), dtype=np.float32)
    return pyvips.Image.new_from_memory(          # pyvips refs `arr`'s buffer
        arr.data, arr.shape[1], arr.shape[0], 1, "float",
    )


def _save_mono_vips(path: str, img: pyvips.Image, filter: str) -> None:
    """Streaming sink twin of `_write_mono_tiff`: same tags, same codec, but
    the writer pulls the lazy vips pipeline stripe-by-stripe instead of being
    handed a full numpy array."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img.copy()
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"FILTER": filter, "PIPELINE": "AstroFlow-OSS"}))
    _atomic_tiffsave(img, Path(path),
                     tile=tile_size_for(img.width, img.height),
                     compression=FAST_COMPRESSION)


def _write_mono_xisf(path: str, data: np.ndarray, filter: str) -> None:
    """Mock-path writer for the rc-astro (BXT/SXT) nodes, whose real outputs are
    XISF — so the mock must emit XISF too (the extension drives the reader).
    `XISF.write` needs an (H, W, C) array, so give mono data a channel axis."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data[:, :, None]
    write_xisf_image(
        Path(path), data,
        {"FILTER": filter, "PIPELINE": "AstroFlow-OSS"},
    )


def _open_gig_vips(path) -> pyvips.Image:
    """THE house open for gigascale sources in the colour path. TIFFs get
    `unlimited=True`: libvips caps a single TIFF strip decode at ~50MB (the
    anti-bomb default) and our wide gigapixel TIFFs exceed it — without the
    lift the read dies mid-stream ("cumulated memory allocation … beyond the
    52428800 byte limit"), which killed combine_star_rgb for three days and
    then the 75a1 star screens the moment they first ran (2026-08-10). The
    kwarg is TIFF-loader-only, so FITS sources skip it."""
    p = str(path)
    tiff = p.lower().endswith((".tif", ".tiff"))
    return pyvips.Image.new_from_file(p, **({"unlimited": True} if tiff else {}))


def _load_mono_vips(meta: ImageMetadata) -> pyvips.Image:
    """Load a mono channel (an autostretch FITS output) as a 1-band float vips
    image. Lazy: vips pulls pixels only as the `tiffsave` sink demands them, so
    the palette never materializes a whole 9 GB channel."""
    img = _open_gig_vips(meta.filename)
    return img[0] if img.bands > 1 else img


def _save_color_vips(path: str, img: pyvips.Image, name: str,
                     bands: "tuple | None" = None) -> None:
    """Stream a float RGB vips image to a tiled BigTIFF deliverable (atomic
    tmp+rename), tagging the palette name / pipeline in the TIFF description
    (the same JSON `read_tiff_header` expects). `bands` (scene formats only)
    records which REAL filters occupy the S,H,O band slots — zero-filled
    absences are indistinguishable from data without it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Force scRGB: the bandjoined float palette otherwise inherits a `b-w`
    # (grayscale) interpretation, so tiffsave writes a MINISBLACK photometric and
    # every interp-aware reader (pyvips thumbnails, the gallery preview) renders
    # the 3 real colour channels as luminance → grey. scRGB is the correct
    # float-RGB interpretation and matches the pre-regression deliverables.
    img = img.copy(interpretation="scrgb")
    desc = {"PALETTE": name, "PIPELINE": "AstroFlow-OSS"}
    if bands is not None:
        desc["BANDS"] = list(bands)
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps(desc))
    tmp = f"{path}.{os.getpid()}.tmp.tiff"
    # deflate, NOT zstd: these named colour images are the user-facing
    # deliverables, opened directly in PixInsight and other tools — and PI can't
    # read zstd ("ZSTD compression support is not configured"). Widest
    # compatibility wins over zstd's speed/size for a final deliverable.
    img.tiffsave(tmp, compression=COMPATIBLE_COMPRESSION, tile=True, bigtiff=True)
    os.replace(tmp, path)


# ---- per-channel intermediates (no display=; ephemeral) ----

def _bxt_correct_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.bxt) — the only things a
    BXT'd channel depends on. `channel` is the trailing data-dependency; `config`
    sits just before it (as in place_image's _place_image_narrow)."""
    mosaic, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    # Route on stable_hash, NOT the full mosaic (content_hash churns on ANY panel
    # re-stack — that would re-run every panel's BXT tile on one new sub). The
    # per-tile identity comes from `channel` (the crop), which is per-panel.
    return (mosaic.stable_hash, filter, (mock, config.bxt), channel), kwargs


def _sxt_narrow(args, kwargs):
    """Cache-key hook for sxt_starless: project `config` to
    (mock, config.sxt). Shared by both ops — their differing bodies (starless vs
    unscreened stars) already give them distinct call hashes via op identity."""
    mosaic, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic.stable_hash, filter, (mock, config.sxt), channel), kwargs   # stable_hash: no cross-panel churn


@node(cache_key_narrow=_bxt_correct_narrow, auto_memory=True, resources={"gpu_slots": 1}, plate_solve=False)  # derived from an already-solved channel
def bxt_correct(
    mosaic: Mosaic, filter: str, config: ProcessingConfig, channel: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """BlurXTerminator a per-filter mosaic channel (linear deconvolution).

    Runs the normal sharpening deconvolution (config.bxt; defaults to the
    PixInsight-GUI look), NOT --correct-only unless config.bxt.correct_only is set.
    The cache key is narrowed to (mock, config.bxt), so tuning the BXT settings
    re-derives BXT and its downstream, and nothing else.

    XISF output: full-res channels exceed rc-astro's cfitsio FITS-writer limit
    (2**31 px); XISF has none. See `rc_astro_cli`."""
    out = bxt_correct.filename_fn(mosaic, filter, config, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        _write_mono_tiff(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_bxt
        run_bxt(channel.filename, out,
                sharpen_stars=config.bxt.sharpen_stars,
                sharpen_nonstellar=config.bxt.sharpen_nonstellar,
                correct_only=config.bxt.correct_only,
                ml_version=config.bxt.ml_version,
                overlap=config.bxt.overlap)
    return WithSidecar(TiffFilePath(out))


def _bxt_correct_only_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.bxt). `correct_only` is
    forced True in the body, so the sharpen fields don't affect the output, but
    keying on config.bxt is harmless and matches bxt_correct."""
    target, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    return (target, filter, (mock, config.bxt), channel), kwargs


@node(cache_key_narrow=_bxt_correct_only_narrow, auto_memory=True, resources={"gpu_slots": 1}, plate_solve=True)  # RE-SOLVE the corrected panel (Croman)
def bxt_correct_only(
    target: Target, filter: str, config: ProcessingConfig, channel: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """BlurXTerminator Correct-Only on a mosaic PANEL, BEFORE assembly — the first
    half of Croman's recommended two-pass mosaic workflow.

    Correct-Only fixes each frame's optical aberrations (coma, tilt, chromatic,
    guiding-elongation) WITHOUT sharpening. Aberrations differ between panels — a
    corner of one frame is the OPPOSITE corner of the next, so they don't match —
    which is why correcting them per-panel, before merging, is more accurate than
    only correcting the assembled mosaic. `plate_solve=True` RE-solves the
    corrected panel (also per Croman: aberration correction shifts the geometry),
    which `place_image` requires (it asserts a plate solution and projects from
    that WCS). The SHARPENING BXT pass runs later, on the assembled mosaic
    (`bxt_correct` in `color_post_process`)."""
    out = bxt_correct_only.filename_fn(target, filter, config, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        _write_mono_tiff(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_bxt
        run_bxt(channel.filename, out, correct_only=True,
                ml_version=config.bxt.ml_version, overlap=config.bxt.overlap)
    return WithSidecar(TiffFilePath(out))


@node(cache_key_narrow=_sxt_narrow, auto_memory=True, resources={"gpu_slots": 1}, plate_solve=False)  # starless: no stars to solve against
def sxt_starless(
    mosaic: Mosaic, filter: str, config: ProcessingConfig, channel: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """StarXTerminator a narrowband channel; keep the starless body (linear).

    Cache key narrowed to (mock, config.sxt). XISF output (its FITS writer
    overflows on full-res channels; `channel` is the XISF BXT output, so rc-astro
    reads and writes XISF here)."""
    out = sxt_starless.filename_fn(mosaic, filter, config, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        _write_mono_tiff(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, starless_out=out,
                ml_version=config.sxt.ml_version, overlap=config.sxt.overlap)
    return WithSidecar(TiffFilePath(out))


def _sxt_rgb_narrow(args, kwargs):
    """Cache-key hook for sxt_stars_rgb: project `config` to (mock, config.sxt)."""
    mosaic, config, rgb = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic.stable_hash, (mock, config.sxt), rgb), kwargs   # stable_hash: no cross-panel churn


# display="Linear RGB" makes the ALREADY-MATERIALIZED linear combine a served
# scene tap (live-finish source format 3). Decorator kwargs are STRIPPED from
# the structural op hash, so stamping the display re-keys NOTHING — every
# cached combine keeps its key and gains a tap.
@node(display="Linear RGB", filter="RGB", auto_memory=True, plate_solve=False)  # pure bandjoin, no PixInsight
def combine_broadband_rgb(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Combine the three LINEAR BXT'd broadband channels into one linear RGB
    XISF so StarXTerminator can extract stars from all channels AT ONCE.

    Extracting per-channel (the old path) let SXT catch a different set of
    marginal stars in each of R/G/B — a faint star found in R but missed in
    G/B became a pure-red dot, giving 'confetti' stars of impossible primary
    colours. Seeing the combined image, SXT's star decision is identical across
    channels. XISF keeps rc-astro happy at gigapixel sizes."""
    out = combine_broadband_rgb.filename_fn(mosaic, r, g, b)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Streaming bandjoin (the stack_sho_channels pattern): three lazy mono
    # opens, joined and pulled through the tiled writer stripe-by-stripe.
    # The previous np.stack held 3 full gigapixel monos PLUS the 3-band
    # stack (~54 GB peak) — the worst full-array op in the fleet.
    monos = [_open_mono_vips(m) for m in (r, g, b)]
    rgb = monos[0].bandjoin([monos[1], monos[2]]).copy()
    rgb.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"PIPELINE": "AstroFlow-OSS"}))
    _atomic_tiffsave(rgb, Path(out),
                     tile=tile_size_for(rgb.width, rgb.height))
    return WithSidecar(TiffFilePath(out))


@node(cache_key_narrow=_sxt_rgb_narrow, auto_memory=True, resources={"gpu_slots": 1}, plate_solve=False)  # coherent RGB stars
def sxt_stars_rgb(
    mosaic: Mosaic, config: ProcessingConfig, rgb: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """StarXTerminator the combined linear RGB; keep the RGB stars (simple
    subtraction). One SXT pass over the RGB (not one per channel) makes the
    extracted stars coherent across R/G/B, so the star image has natural colours
    instead of per-channel primary-colour confetti.

    `unscreen=False` on purpose: rc-astro's `--unscreen` mode is for NONLINEAR
    (stretched) data — it inverts a screen blend to make "screen-ready" stars.
    Our input here is LINEAR (combine_broadband_rgb), where stars combine
    additively, so simple subtraction is the correct extraction and gives the
    best star colour (per RC-Astro's StarXTerminator usage notes).

    The mono stars this op once emitted were NOT an rc-astro artifact (a synthetic
    3-band → SXT → 3-band repro cleared it): under the OLD arbitrary-grid tiling
    the crop handed here was a mono nebula crop, because the crop cache key dropped
    the source's identity and every tiled source sharing a mosaic/rect collided
    onto one crop. Panels-as-tiles (`mosaics.panels`) keys each crop on its panel's
    manifest hash, so the RGB-star crop can never collide with a nebula crop;
    `combine_star_rgb` still fail-fasts (`_require_rgb_stars`) if it's ever mono."""
    out = sxt_stars_rgb.filename_fn(mosaic, config, rgb)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        write_tiff_image(Path(out), read_image(Path(rgb.filename)).astype(np.float32),
                         {"PIPELINE": "AstroFlow-OSS"})
    else:
        from rc_astro_cli import run_sxt
        run_sxt(rgb.filename, stars_out=out, unscreen=False,
                ml_version=config.sxt.ml_version, overlap=config.sxt.overlap)
    return WithSidecar(TiffFilePath(out))


@node(auto_memory=True, plate_solve=False)  # nonlinear stretch of an already-solved channel
def autostretch_channel(
    mosaic: Mosaic, filter: str, channel: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """STF-equivalent autostretch — palettes need non-linear input for their
    dynamic weights to be meaningful.

    Streaming since 2026-08-08: the input is the merged starless TIFF (the
    old "XISF blocks streaming" rationale went stale when the tail moved to
    per-panel TIFFs), so the two-pass histogram autostretch
    (`palettes.autostretch_vips`) applies directly — constant memory instead
    of ~2 full gigapixel copies. Non-TIFF legacy inputs fall back to a full
    read inside `_open_mono_vips`."""
    out = autostretch_channel.filename_fn(mosaic, filter, channel)
    stretched = palettes.autostretch_vips(lambda: _open_mono_vips(channel))
    _save_mono_vips(out, stretched, filter)
    return WithSidecar(TiffFilePath(out))


# ---- named color deliverables (display=name, filter=None, final=True) ----

# The display= of the linear-SHO viewer source. Kept as a constant because the
# tile server keys `colorspace="linear-sho"` on this exact display name (see
# web_server/backend/tiles.py LINEAR_FINISH_DISPLAYS) so the WebGPU shader knows
# to run the live colour finish instead of passing the tile through as baked RGB.
LINEAR_SHO_DISPLAY = "Linear SHO"


def _nxt_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.nxt) — the only config an
    NXT'd image depends on. `image` (whole OR one tile) is the trailing data-dep."""
    mosaic, config, image = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic.stable_hash, (mock, config.nxt), image), kwargs   # stable_hash: no cross-panel churn


def _blend_sho_narrow(args, kwargs):
    """Cache-key hook for the reassembly: beyond the tile handles + their rects,
    only the feather params (config.tiling) matter."""
    mosaic, config, rects, *tiles = args
    return (mosaic.stable_hash, config.tiling, rects, *tiles), kwargs   # stable_hash: no cross-panel churn


@node(auto_memory=True, plate_solve=False)  # streaming bandjoin of the 3 starless channels
def stack_sho_channels(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Stack the three LINEAR starless narrowband channels (S→R, H→G, O→B) into
    one 3-band linear BigTIFF — the pyvips-streamable source the NXT tiles crop
    from. Fully streaming since 2026-08-08: the inputs are the merged starless
    TIFFs, so the per-channel numpy round-trip (a temp mono BigTIFF per band,
    ~8 GB resident each) is gone — three lazy opens, one bandjoin, one tiled
    write. Legacy non-TIFF inputs fall back inside `_open_mono_vips`."""
    out = stack_sho_channels.filename_fn(mosaic, sii, ha, oiii)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {"PIPELINE": "AstroFlow-OSS", "COLORSPC": "linear-sho"}
    monos = [_open_mono_vips(m) for m in (sii, ha, oiii)]
    stacked = monos[0].bandjoin([monos[1], monos[2]]).copy()
    stacked.set_type(pyvips.GValue.gstr_type, "image-description", json.dumps(meta))
    _atomic_tiffsave(stacked, Path(out),
                     tile=tile_size_for(stacked.width, stacked.height))
    return WithSidecar(TiffFilePath(out))


@node(cache_key_narrow=_nxt_narrow, auto_memory=True, resources={"gpu_slots": 1}, plate_solve=False)  # NXT on an image (whole OR one tile)
def nxt(
    mosaic: Mosaic, config: ProcessingConfig, image: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """NoiseXTerminator on an image — one PANEL tile fed by process_in_panels;
    SAME code whether it sees a whole channel or a panel crop. A display-less
    transform: the deliverable identity lives on the blend passed to
    process_in_panels. rc-astro reads `image` directly (crop_panel writes it
    strip-based; rc-astro rejects tiled TIFF); its forced-XISF output is converted
    to the node's BigTIFF artifact."""
    out = nxt.filename_fn(mosaic, config, image)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        write_tiff_image(Path(out), read_image(Path(image.filename)), {"PIPELINE": "AstroFlow-OSS"})
    else:
        from rc_astro_cli import run_nxt
        with tempfile.TemporaryDirectory(prefix="nxt_") as td:
            tout = os.path.join(td, "out.xisf")
            run_nxt(image.filename, tout,
                    denoise_intensity_hf=config.nxt.denoise_intensity_hf,
                    denoise_intensity_lf=config.nxt.denoise_intensity_lf,
                    denoise_color_hf=config.nxt.denoise_color_hf,
                    denoise_color_lf=config.nxt.denoise_color_lf,
                    frequency_scale=config.nxt.frequency_scale,
                    iterations=config.nxt.iterations)
            write_tiff_image(Path(out), read_image(Path(tout)), {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


@node(cache_key_narrow=_blend_sho_narrow, display=LINEAR_SHO_DISPLAY, filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel LINEAR, NXT-denoised
def blend_linear_sho(
    mosaic: Mosaic, config: ProcessingConfig, rects: tuple, *tiles: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Feather the NXT'd tiles back into the single linear-SHO master — the
    source the WebGPU tile viewer finishes live. Fully streaming (see
    file_io/tiling.feather_blend_to_bigtiff): never materializes the canvas.
    `final=True` exempts this deliverable from any ephemeral scope (the NXT
    tiles themselves are GPU work — kept, never ephemeral). display="Linear
    SHO" makes the tap run the live SHO colour finish (see
    LINEAR_SHO_DISPLAY)."""
    out = blend_linear_sho.filename_fn(mosaic, config, rects, *tiles)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        taper_px=config.tiling.overlap_px, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS", "COLORSPC": "linear-sho"})
    return WithSidecar(TiffFilePath(out))


def linear_sho(
    mosaic: Mosaic, sii: PanelMosaic, ha: PanelMosaic, oiii: PanelMosaic,
    config: ProcessingConfig, viewbox,
):
    """Traced helper (NOT a @node): the linear-SHO deliverable = stack the 3
    starless channel `PanelMosaic`s into one SHO `PanelMosaic`, then
    NoiseXTerminator it PER PANEL. `nxt` is the display-less NXT op;
    `process_in_panels(nxt, …, blend=blend_linear_sho)` runs it over each panel
    tile, and `blend_linear_sho` — display="Linear SHO", final=True — feathers
    them into the deliverable. `combine_panels` threads the S/H/O manifests into
    the SHO source's manifest so NXT's crops key per-panel.

    `viewbox` supplies the trace-time canvas geometry (the panel grid is built
    before any node runs); it's the mosaic viewbox at the pipeline's quality
    (`mosaic.viewbox("Master")` / `"DownscaledMaster"`)."""
    # The SHO band-join is cheap × big — ephemeral. The NXT tiles are GPU
    # work — kept (outside the scope), so incremental runs cache-hit them.
    source = combine_panels(stack_sho_channels, mosaic, sii, ha, oiii)
    result = process_in_panels(
        nxt, mosaic, config, source, viewbox=viewbox, blend=blend_linear_sho)
    return result.merged


@node(display="Stars", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def combine_star_rgb(
    mosaic: Mosaic, rgb_stars: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The RGB-stars deliverable: a LINKED autostretch of the coherent SXT'd RGB
    stars. `palettes.autostretch` takes one median/MAD over all three bands, so
    every channel gets the same MTF — star colour is preserved (no per-channel
    cast). Saved as the scRGB TIFF deliverable."""
    out = combine_star_rgb.filename_fn(mosaic, rgb_stars)
    src = str(rgb_stars.filename)
    # Streaming: header-only band check, then the two-pass histogram
    # autostretch (palettes.autostretch_vips) with a lazy MTF apply that the
    # writer sink evaluates stripe-by-stripe. The previous body read the full
    # gigapixel RGB into numpy (~27 GB) plus a stretched copy — silently
    # SIGKILLed on the 64 GiB head whenever anything else was co-scheduled.
    probe = pyvips.Image.new_from_file(src)
    _require_rgb_bands(probe.bands, src)
    # unlimited=True: libvips caps a single TIFF strip decode at ~50MB (the
    # anti-bomb default), and this gigapixel source's strips exceed it — the
    # read dies deterministically at open ("cumulated memory allocation …
    # beyond the 52428800 byte limit"; "no space for data buffer at
    # scanline …"). Same lift the tile-pyramid builder documents
    # (previews/create_tile_pyramid.py). This ONE failure blocked the star
    # trio AND census soundness on every pass for two days. Default
    # (random) access; the vips cache keeps memory bounded — still
    # streaming, never a full materialization.
    stretched = palettes.autostretch_vips(
        lambda: pyvips.Image.new_from_file(src, unlimited=True)
    )
    _save_color_vips(out, stretched, "Stars")
    return WithSidecar(TiffFilePath(out))


def _require_rgb_bands(bands: int, src: str = "") -> None:
    """REQUIRE a 3-channel star layer (header-only check — no pixel read).

    The star layer is always extracted from the combined RGB
    (`combine_broadband_rgb` → `sxt_stars_rgb`), so it must be 3-channel. A mono
    (or otherwise non-RGB) star layer is never a valid state — it means the
    upstream SXT extraction is broken (historically: `--unscreen` on LINEAR data,
    whose screen-inversion is luminance-oriented, collapsing colour to a single
    band). Fail loudly rather than silently degrade to grey stars, so the
    regression surfaces instead of shipping wrong output."""
    if bands != 3:
        raise ValueError(
            f"combine_star_rgb: star layer {src} has {bands} band(s), "
            f"expected 3-channel RGB. Stars are always extracted from the "
            f"combined RGB — a non-RGB star layer means the upstream SXT "
            f"star extraction (sxt_stars_rgb) is broken."
        )


def _palette_image(out, combine_vips, sii, ha, oiii, stars, name):
    rgb = combine_vips(_load_mono_vips(sii), _load_mono_vips(ha), _load_mono_vips(oiii))
    if stars is not None:  # screen the RGB stars back over the starless palette
        rgb = palettes.screen_vips(rgb, _open_gig_vips(stars.filename))
    _save_color_vips(out, rgb, name)


@node(display="Hubble", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def hubble_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Hubble (SHO) palette of the stretched starless narrowband, RGB stars
    screened in. `stars` is the `combine_star_rgb` output, or None (no broadband)."""
    out = hubble_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.hubble_vips, sii, ha, oiii, stars, "Hubble")
    return WithSidecar(TiffFilePath(out))


@node(display="Foraxx", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def foraxx_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Foraxx dynamic palette of the stretched starless narrowband, RGB stars
    screened in."""
    out = foraxx_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.foraxx_vips, sii, ha, oiii, stars, "Foraxx")
    return WithSidecar(TiffFilePath(out))


@node(display="Hubble Starless", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def hubble_starless(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Hubble (SHO) palette of the stretched starless narrowband, NO stars
    screened in — the pure starless palette (the `Hubble` tap adds RGB stars)."""
    out = hubble_starless.filename_fn(mosaic, sii, ha, oiii)
    _palette_image(out, palettes.hubble_vips, sii, ha, oiii, None, "Hubble Starless")
    return WithSidecar(TiffFilePath(out))


@node(display="Foraxx Starless", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def foraxx_starless(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Foraxx palette of the stretched starless narrowband, NO stars screened in
    — the pure starless palette (the `Foraxx` tap adds RGB stars)."""
    out = foraxx_starless.filename_fn(mosaic, sii, ha, oiii)
    _palette_image(out, palettes.foraxx_vips, sii, ha, oiii, None, "Foraxx Starless")
    return WithSidecar(TiffFilePath(out))


@node(display="RGB", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def rgb_image(
    mosaic: Mosaic, rgb: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The broadband colour photo: LINKED autostretch of the linear BXT'd
    R/G/B combine, stars in place (the star branch extracts a copy, never
    subtracts here). Linked = `_hist_median_vips` pools the 3-band histogram
    into one distribution, so a single MTF applies to all bands and the
    stretch cannot shift hue; the channels are SPFC-calibrated per stack
    upstream, so the linear combine is already colour-calibrated."""
    out = rgb_image.filename_fn(mosaic, rgb)
    stretched = palettes.autostretch_vips(lambda: _open_gig_vips(rgb.filename))
    _save_color_vips(out, stretched, "RGB")
    return WithSidecar(TiffFilePath(out))


@node(display="LRGB", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def lrgb_image(
    mosaic: Mosaic, rgb: ImageMetadata, lum: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """PixInsight-style LRGBCombination: the stretched RGB's CIE lightness
    replaced by the stretched (BXT'd) luminance master in Lab space — detail
    and SNR ride the deep L, colour comes from RGB. Math in
    `palettes.lrgb_vips`; inputs are the `rgb_image` deliverable and the
    autostretched L channel (both nonlinear, as Lab requires)."""
    out = lrgb_image.filename_fn(mosaic, rgb, lum)
    combined = palettes.lrgb_vips(
        _open_gig_vips(rgb.filename), _load_mono_vips(lum))
    _save_color_vips(out, combined, "LRGB")
    return WithSidecar(TiffFilePath(out))


# ---- linear scene formats: the ≤3-band basis the live WebGPU finishes
# ---- sample. Formats 1/2 are always 3 bands in canonical S,H,O order with
# ---- absent bands ZERO-FILLED (uniform texture shape, zeros deflate to
# ---- nothing) and the REAL band list in the TIFF description; format 3 is
# ---- the display-stamped combine above; starless RGB is NOT baked (the
# ---- shader derives it exactly as combine − stars); nobody wants
# ---- narrowband-stars-only, so it does not exist.

_NB_ORDER = ("S", "H", "O")


def _nb_threeband(channels: dict) -> pyvips.Image:
    """Bandjoin `{filter: ImageMetadata}` into canonical 3-band S,H,O with
    zero-filled absences. Each band is its own lazy open (single consumer)."""
    first = next(iter(channels.values()))
    bands = [
        _open_mono_vips(channels[f]) if f in channels
        else _open_mono_vips(first) * 0.0
        for f in _NB_ORDER
    ]
    return bands[0].bandjoin(bands[1:])


@node(display="Linear Narrowband", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-band linear
def linear_narrowband(
    mosaic: Mosaic, filters: tuple, *channels: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Scene format 1: the WITH-STARS narrowband combination, linear —
    BXT'd but never SXT'd, so compact HII knots that star removal would
    eat survive intact. This is the raw-Hα source the galaxy finishes
    sample (live continuum subtraction: Hα − q·R), and the native-star
    source for narrowband-only images. `filters` lists the real bands in
    S,H,O order; missing bands are zero-filled."""
    out = linear_narrowband.filename_fn(mosaic, filters, *channels)
    img = _nb_threeband(dict(zip(filters, channels)))
    _save_color_vips(out, img, "Linear Narrowband", bands=filters)
    return WithSidecar(TiffFilePath(out))


@node(display="Linear Narrowband Starless", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-band linear
def linear_narrowband_starless(
    mosaic: Mosaic, filters: tuple, *channels: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Scene format 2 for PARTIAL narrowband sets (an Hα-only galaxy, an
    HOO dual-band OSC): the starless combination, zero-filled to S,H,O.
    Full-SHO images don't get this — "Linear SHO" (NXT-denoised) already
    serves that role."""
    out = linear_narrowband_starless.filename_fn(mosaic, filters, *channels)
    img = _nb_threeband(dict(zip(filters, channels)))
    _save_color_vips(out, img, "Linear Narrowband Starless", bands=filters)
    return WithSidecar(TiffFilePath(out))


@node(display="Linear RGB Stars", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-band linear
def linear_rgb_stars(
    mosaic: Mosaic, stars: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Scene format 5: the LINEAR coherent RGB stars (the SXT extraction,
    pre-stretch — the stretched version is the "Stars" deliverable). A
    star image is overwhelmingly zeros, so the deflate tap costs almost
    nothing on disk; live finishes use it for star intensity/reduction
    (`starless + s·stars`) and to derive starless RGB as combine − stars."""
    out = linear_rgb_stars.filename_fn(mosaic, stars)
    _save_color_vips(out, _open_gig_vips(stars.filename), "Linear RGB Stars")
    return WithSidecar(TiffFilePath(out))


@node(display="Linear L", filter="L", final=True, auto_memory=True, plate_solve=False)  # mono linear
def linear_lum(
    mosaic: Mosaic, lum: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Scene format 6: the BXT'd luminance master as a served mono tap —
    the lightness source for live LRGB/HaLRGB finishing."""
    out = linear_lum.filename_fn(mosaic, lum)
    _save_mono_vips(out, _open_mono_vips(lum), "L")
    return WithSidecar(TiffFilePath(out))


# Hα-blend tuning. Deliberately CONSTANTS, not ProcessingConfig fields:
# adding a config field moves the hash of every config-folding op in the
# system (a cache-wipe event), while editing a constant re-keys exactly the
# blend op that owns it plus its downstream — the right invalidation radius
# for an artistic knob.
_HA_CHROMA_STRENGTH = 0.4    # k in R′ = R + k·Ha_starless
_HA_LIGHTNESS_WEIGHT = 0.5   # w in L′ = max(L, w·Ha_starless)


@node(auto_memory=True, plate_solve=False)  # pure bandjoin, no PixInsight
def combine_ha_rgb(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The Hα-boosted linear combine feeding `hargb_image` — the same
    streaming bandjoin as `combine_broadband_rgb`, as its OWN display-less
    op: the plain combine is the served "Linear RGB" scene format, and
    tapping this interior variant too would put a second, misleading
    "Linear RGB" row on every dashboard."""
    out = combine_ha_rgb.filename_fn(mosaic, r, g, b)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    monos = [_open_mono_vips(m) for m in (r, g, b)]
    rgb = monos[0].bandjoin([monos[1], monos[2]]).copy()
    rgb.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"PIPELINE": "AstroFlow-OSS"}))
    _atomic_tiffsave(rgb, Path(out),
                     tile=tile_size_for(rgb.width, rgb.height))
    return WithSidecar(TiffFilePath(out))


@node(auto_memory=True, plate_solve=False)  # streaming linear add
def blend_ha_chroma(
    mosaic: Mosaic, red: ImageMetadata, ha: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """R′ = R + k·Ha_starless, LINEAR: inject the Hα line flux the ~100nm R
    band dilutes ~30:1 back into the red channel, pre-stretch. STARLESS Hα is
    the trick — the layer contains no stars, so nothing reddens the stellar
    continuum and no explicit continuum subtraction is needed; HII regions
    come out cherry pink while stars stay white."""
    out = blend_ha_chroma.filename_fn(mosaic, red, ha)
    img = _open_mono_vips(red) + _HA_CHROMA_STRENGTH * _open_mono_vips(ha)
    _save_mono_vips(out, img, "R")
    return WithSidecar(TiffFilePath(out))


@node(auto_memory=True, plate_solve=False)  # streaming lighten blend
def blend_ha_lightness(
    mosaic: Mosaic, lum: ImageMetadata, ha: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """L′ = max(L, w·Ha_starless), LINEAR lighten: HII structure pops in the
    detail channel while the colour stays whatever the chrominance source
    says (the HaLRGB tap pairs this with the PLAIN stretched RGB)."""
    out = blend_ha_lightness.filename_fn(mosaic, lum, ha)
    l = _open_mono_vips(lum)
    h = _HA_LIGHTNESS_WEIGHT * _open_mono_vips(ha)
    _save_mono_vips(out, (l > h).ifthenelse(l, h), "L")
    return WithSidecar(TiffFilePath(out))


@node(display="HaRGB", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def hargb_image(
    mosaic: Mosaic, rgb: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The HaRGB colour photo (Hα in CHROMINANCE): linked stretch of the
    combine whose R channel carries the starless-Hα line flux
    (`blend_ha_chroma`). Same stretch discipline as `rgb_image`."""
    out = hargb_image.filename_fn(mosaic, rgb)
    stretched = palettes.autostretch_vips(lambda: _open_gig_vips(rgb.filename))
    _save_color_vips(out, stretched, "HaRGB")
    return WithSidecar(TiffFilePath(out))


@node(display="HaLRGB", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def halrgb_image(
    mosaic: Mosaic, rgb: ImageMetadata, lum: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The Hα-detail LRGB (Hα in LIGHTNESS): colour from the PLAIN stretched
    RGB, lightness from the Hα-lightened L (`blend_ha_lightness`) — HII
    regions sharp and deep, hue untouched. The chrominance-side Hα look is
    the separate HaRGB tap; comparing the two side by side is the point."""
    out = halrgb_image.filename_fn(mosaic, rgb, lum)
    combined = palettes.lrgb_vips(
        _open_gig_vips(rgb.filename), _load_mono_vips(lum))
    _save_color_vips(out, combined, "HaLRGB")
    return WithSidecar(TiffFilePath(out))


@workflow_group
def color_post_process(mosaic: Mosaic, channels: dict, config, viewbox=None) -> dict:
    """Turn merged per-filter mosaic channels into named color images.

    `channels` is `{filter: merged-channel handle}` from `for_mosaic`. BXT every
    channel; SXT the narrowband to starless and the broadband to unscreened
    stars; autostretch; then build the Hubble + Foraxx palettes (screening the
    combined RGB stars in). Returns `{name: handle}` for the color deliverables.

    `viewbox` gives the trace-time canvas dims for the tiled ops (`linear_sho`);
    each pipeline passes its quality-specific viewbox. Defaults to the Master
    viewbox for callers (e.g. tests) that don't tile at another quality.
    """
    if viewbox is None:
        viewbox = mosaic.viewbox("Master")
    by_class = {normalize_filter(f): ch for f, ch in channels.items()}

    # Ephemeral = cheap-to-recompute × big-on-disk. The GPU stages (BXT/SXT
    # tiles + their blends) are deliberately OUTSIDE every scope: reclaiming
    # them made each incremental run repay hours of GPU work re-deriving
    # unchanged tiles (the 619-redo autopsy, 2026-08-05). They persist;
    # superseded hashes are collected by the orphan sweep. Only the genuinely
    # cheap intermediates (stage-0 wrap, stretches, band-joins) stay ephemeral.
    starless_nb: dict[str, object] = {}
    corrected_nb: dict[str, object] = {}
    stretched_nb: dict[str, object] = {}
    bxt_bb: dict[str, object] = {}
    bxt_l = None
    for cf, channel in by_class.items():
        # Wrap the merged channel as the stage-0 PanelMosaic (merged image +
        # per-panel base manifest); every rc-astro op then tiles by PANEL.
        pm = base_panel_mosaic(mosaic, cf, channel)
        if cf in _NARROWBAND:
            corrected = process_in_panels(bxt_correct, mosaic, cf, config, pm, viewbox=viewbox)
            starless = process_in_panels(sxt_starless, mosaic, cf, config, corrected, viewbox=viewbox)
            corrected_nb[cf] = corrected  # with stars — scene format 1
            starless_nb[cf] = starless  # PanelMosaic — linear source for the Linear SHO tap
            stretched_nb[cf] = autostretch_channel(mosaic, cf, starless.merged)
        elif cf in _BROADBAND:
            bxt_bb[cf] = process_in_panels(bxt_correct, mosaic, cf, config, pm, viewbox=viewbox)
        elif cf == "L" and _BROADBAND <= set(by_class):
            # Luminance: BXT'd like the broadband, stretched later as the
            # LRGB lightness. Never enters the RGB combine; skipped entirely
            # unless full R/G/B exists to give it chrominance.
            bxt_l = process_in_panels(bxt_correct, mosaic, cf, config, pm, viewbox=viewbox)

    # Broadband stars: combine the linear BXT'd channels into one RGB, then
    # SXT ONCE on that RGB so the extracted stars are coherent across R/G/B
    # (no per-channel primary-colour confetti). `combine_panels` threads the
    # R/G/B manifests into the RGB source so the SXT crops key per-panel.
    rgb_star_lin = None
    rgb_lin = None
    if _BROADBAND <= set(bxt_bb):
        rgb_lin = combine_panels(combine_broadband_rgb, mosaic, bxt_bb["R"], bxt_bb["G"], bxt_bb["B"])
        rgb_star_lin = process_in_panels(sxt_stars_rgb, mosaic, config, rgb_lin, viewbox=viewbox)

    out: dict[str, object] = {}
    rgb_stars = None
    if rgb_star_lin is not None:
        rgb_stars = combine_star_rgb(mosaic, rgb_star_lin.merged)
        out["Stars"] = rgb_stars
    if rgb_lin is not None:
        # The broadband colour photo: the same linear RGB the star branch
        # reads, linked-stretched whole. With a luminance channel, also the
        # LRGB variant (lightness replaced by the stretched L in Lab).
        out["RGB"] = rgb_image(mosaic, rgb_lin.merged)
        if bxt_l is not None:
            out["LRGB"] = lrgb_image(
                mosaic, out["RGB"],
                autostretch_channel(mosaic, "L", bxt_l.merged))
        # Hα blends — two SEPARATE looks from the starless Hα the narrowband
        # branch already computed: chrominance (Hα → R, pre-combine) and
        # lightness (Hα → L, colour from the plain RGB).
        ha_starless = starless_nb.get("H")
        if ha_starless is not None:
            r_ha = blend_ha_chroma(
                mosaic, bxt_bb["R"].merged, ha_starless.merged)
            hargb_lin = combine_ha_rgb(
                mosaic, r_ha, bxt_bb["G"].merged, bxt_bb["B"].merged)
            out["HaRGB"] = hargb_image(mosaic, hargb_lin)
            if bxt_l is not None:
                l_ha = blend_ha_lightness(
                    mosaic, bxt_l.merged, ha_starless.merged)
                out["HaLRGB"] = halrgb_image(
                    mosaic, out["RGB"],
                    autostretch_channel(mosaic, "L", l_ha))
    if _NARROWBAND <= set(starless_nb):
        # The linear S/H/O viewer source: stacked + NXT-denoised, finished live
        # in the WebGPU shader with the standard Hubble/SCNR palette.
        out["Linear SHO"] = linear_sho(
            mosaic, starless_nb["S"], starless_nb["H"], starless_nb["O"], config, viewbox)
    if _NARROWBAND <= set(stretched_nb):
        s, h, o = stretched_nb["S"], stretched_nb["H"], stretched_nb["O"]
        # Always emit the pure starless palettes; add the with-stars variants
        # only when broadband RGB stars exist to screen in.
        out["Hubble Starless"] = hubble_starless(mosaic, s, h, o)
        out["Foraxx Starless"] = foraxx_starless(mosaic, s, h, o)
        if rgb_stars is not None:
            out["Hubble"] = hubble_image(mosaic, s, h, o, rgb_stars)
            out["Foraxx"] = foraxx_image(mosaic, s, h, o, rgb_stars)

    # Linear scene formats — the ≤3-band basis the live WebGPU finishes
    # sample ("Linear RGB" is the display-stamped combine, already in the
    # graph above; starless RGB is derived in-shader as combine − stars).
    nb_present = tuple(f for f in _NB_ORDER if f in corrected_nb)
    if nb_present:
        out["Linear Narrowband"] = linear_narrowband(
            mosaic, nb_present, *[corrected_nb[f].merged for f in nb_present])
        if not (_NARROWBAND <= set(starless_nb)):
            # Partial sets only — full SHO's starless role is the
            # NXT-denoised "Linear SHO" tap.
            out["Linear Narrowband Starless"] = linear_narrowband_starless(
                mosaic, nb_present,
                *[starless_nb[f].merged for f in nb_present])
    if rgb_star_lin is not None:
        out["Linear RGB Stars"] = linear_rgb_stars(mosaic, rgb_star_lin.merged)
    if bxt_l is not None:
        out["Linear L"] = linear_lum(mosaic, bxt_l.merged)
    return out
