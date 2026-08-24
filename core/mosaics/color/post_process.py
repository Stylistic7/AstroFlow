"""OSS colour post-processing: @node ops + the `color_post_process`
workflow_group that `for_mosaic`/`for_target` call.

PixInsight-free. BXT/SXT run via the rc-astro CLI on LINEAR data. The group's
colour OUTPUT is the linear scene formats only — the orthogonal ≤3-band basis
(Linear SHO / Narrowband / Narrowband Starless / RGB / RGB Stars / L) that
the viewer's live WGSL finishes assemble into Hubble, Foraxx, LRGB, and every
other look on demand (finishes/stock.py). No palette is baked to disk
(2026-08-12) — a palette is a per-pixel function of the formats, and baking
one cost ~40-50 GB per look per mosaic.

The framework is one-artifact-per-`@node`. The per-channel steps are
`display`-less intermediates that PERSIST — every image is worth keeping
unless a subgraph explicitly declares otherwise via
`Graph(..., delete_intermediates=True)`, which today is only the panel crops
(a crop is a view of another image, materialized for a CLI boundary).
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
    downstream consumers (the Linear RGB Stars tap) would surface it as grey
    stars if it ever regressed to mono."""
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
    """Cache-key hook for the reassembly: beyond the tile handles + the rect
    inputs (panels + viewbox) and the feather params (config.tiling), nothing
    matters. Panels reduce to stable ids — their content already rides in the
    tiles; the viewbox reduces to its content_hash (handle or runtime-stamped
    value, identical by the pure-Merkle stamp)."""
    mosaic, config, viewbox, panels, *tiles = args
    return (mosaic.stable_hash, config.tiling, viewbox,
            tuple(p.stable_hash for p in panels), *tiles), kwargs   # stable_hash: no cross-panel churn


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
    mosaic: Mosaic, config: ProcessingConfig, viewbox, panels: tuple,
    *tiles: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Feather the NXT'd tiles back into the single linear-SHO master — the
    source the WebGPU tile viewer finishes live. Fully streaming (see
    file_io/tiling.feather_blend_to_bigtiff): never materializes the canvas.
    Rects re-derived from `(panels, viewbox, pad)` — the same arithmetic the
    crops used, so tile and placement can't disagree. `final=True` exempts
    this deliverable from any ephemeral scope (the NXT tiles themselves are
    GPU work — kept, never ephemeral). display="Linear SHO" makes the tap run
    the live SHO colour finish (see LINEAR_SHO_DISPLAY)."""
    from mosaics.panels import panel_rect
    pad = config.tiling.overlap_px
    rects = [panel_rect(panel, viewbox, pad) for panel in panels]
    assert all(r is not None for r in rects), "panel projects off-canvas"
    out = blend_linear_sho.filename_fn(mosaic, config, viewbox, panels, *tiles)
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


def panel_source_manifest(targets_by_id: dict, filter: str, config,
                          channels: dict) -> dict:
    """`{panel_id: pre-merge identity}` for `color_post_process(panel_sources=)`.

    The identity is `bxt_correct_only`'s output stem for that panel — the same
    `op.filename_fn(...).stem` trick `place_image` refs use, so it is the real
    product identity rather than a re-derivation that could drift from it.

    This is what makes two pipelines processing ONE target produce distinct
    colour tails: `channels[(tid, filter)]` is the per-filter merged channel,
    which already differs between pipelines, so the stem does too. Keyed per
    panel, so it keeps the O(change) re-run property (see mosaics/panels.py).

    `channels` is the `{(target_id, filter): channel}` map the pipelines build;
    entries for other filters and unknown targets are ignored."""
    return {
        tid: Path(bxt_correct_only.filename_fn(
            targets_by_id[tid], filter, config, channel)).stem
        for (tid, f), channel in channels.items()
        if f == filter and tid in targets_by_id
    }


@workflow_group
def color_post_process(mosaic: Mosaic, channels: dict, config, viewbox=None,
                       panel_sources: dict | None = None) -> dict:
    """Turn merged per-filter channels into the LINEAR SCENE FORMATS — the
    orthogonal ≤3-band basis every colour look is assembled from, LIVE, in
    the viewer's WGSL finishes (finishes/stock.py + ~/AstroFlow/finishes/).

    No palette is baked here (2026-08-12): a Hubble or LRGB image is a
    per-pixel function of these formats, and materializing it burned
    ~40-50 GB per look per mosaic re-storing data the formats already hold.
    This group's colour output is exactly: Linear SHO (NXT'd starless
    narrowband), Linear Narrowband (raw, with stars — HII detail SXT never
    touched), Linear Narrowband Starless (partial sets), Linear RGB (the
    with-stars combine, display-stamped on `combine_broadband_rgb`),
    Linear RGB Stars, and Linear L.

    `channels` is `{filter: merged-channel handle}` from `for_mosaic`. BXT
    every channel; SXT the narrowband to starless and the broadband to
    coherent RGB stars; band-join the formats.

    `viewbox` gives the trace-time canvas dims for the tiled ops
    (`linear_sho`); each pipeline passes its quality-specific viewbox.
    Defaults to the Master viewbox for callers (e.g. tests) that don't tile
    at another quality.
    """
    if viewbox is None:
        viewbox = mosaic.viewbox("Master")
    by_class = {normalize_filter(f): ch for f, ch in channels.items()}
    # Normalized the same way as `channels`, so the lookup below can't miss on
    # a filter-name spelling ("Ha" vs "H") and silently fall back to the
    # colliding content_hash manifest.
    panels_by_class = {normalize_filter(f): p
                       for f, p in (panel_sources or {}).items()}

    # Ephemeral = cheap-to-recompute × big-on-disk. The GPU stages (BXT/SXT
    # tiles + their blends) are deliberately OUTSIDE every scope: reclaiming
    # them made each incremental run repay hours of GPU work re-deriving
    # unchanged tiles (the 619-redo autopsy, 2026-08-05). They persist;
    # superseded hashes are collected by the orphan sweep. Only the genuinely
    # cheap intermediates (stage-0 wrap, stretches, band-joins) stay ephemeral.
    starless_nb: dict[str, object] = {}
    corrected_nb: dict[str, object] = {}
    bxt_bb: dict[str, object] = {}
    bxt_l = None
    for cf, channel in by_class.items():
        # Wrap the merged channel as the stage-0 PanelMosaic (merged image +
        # per-panel base manifest); every rc-astro op then tiles by PANEL.
        pm = base_panel_mosaic(mosaic, cf, channel, panels_by_class.get(cf))
        if cf in _NARROWBAND:
            corrected = process_in_panels(bxt_correct, mosaic, cf, config, pm, viewbox=viewbox)
            starless = process_in_panels(sxt_starless, mosaic, cf, config, corrected, viewbox=viewbox)
            corrected_nb[cf] = corrected  # with stars — scene format 1
            starless_nb[cf] = starless  # PanelMosaic — linear source for the Linear SHO tap
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
    if _NARROWBAND <= set(starless_nb):
        # The linear S/H/O viewer source: stacked + NXT-denoised, finished
        # live in the WebGPU shader (Hubble/Foraxx and friends).
        out["Linear SHO"] = linear_sho(
            mosaic, starless_nb["S"], starless_nb["H"], starless_nb["O"], config, viewbox)

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
