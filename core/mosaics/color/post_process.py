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
                                  FAST_COMPRESSION)
from file_io.tiling import feather_blend_to_bigtiff, tile_size_for
from file_io.types import Produced, TiffFilePath
from file_io.xisf_file_io import write_xisf_image
from metadata.classes import ImageMetadata
from mosaics.panels import (
    PanelMosaic, base_panel_mosaic, combine_panels, process_in_panels,
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


# The physical layer an image IS: "starless" or "stars". Layer identity used
# to ride only in the op's output directory name plus SKYVER ("overlap-2" vs
# "stars-none-1") — recoverable, but not self-describing, so a file lifted out
# of the tree could not be told apart from its twin. Colour deliverables carry
# it already in PALETTE ("Linear RGB Stars"); the mono writers below did not.
def _save_mono_vips(path: str, img: pyvips.Image, filter: str,
                    layer: str | None = None) -> None:
    """Streaming sink twin of `_write_mono_tiff`: same tags, same codec, but
    the writer pulls the lazy vips pipeline stripe-by-stripe instead of being
    handed a full numpy array. `layer` stamps the LAYER tag (see above)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img.copy()
    desc = {"FILTER": filter, "PIPELINE": "AstroFlow-OSS"}
    if layer is not None:
        desc["LAYER"] = layer
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps(desc))
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
                     bands: "tuple | None" = None, layer: str | None = None) -> None:
    """Stream a float RGB vips image to a tiled BigTIFF deliverable (atomic
    tmp+rename), tagging the palette name / pipeline in the TIFF description
    (the same JSON `read_tiff_header` expects). `bands` (scene formats only)
    records which REAL filters occupy the S,H,O band slots — zero-filled
    absences are indistinguishable from data without it. `layer` stamps the
    LAYER tag ("starless" | "stars"): a finish's stars inputs are recognised
    by it, never by name."""
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
    if layer is not None:
        desc["LAYER"] = layer
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps(desc))
    tmp = f"{path}.{os.getpid()}.tmp.tiff"
    # zstd like every AstroFlow TIFF; PixInsight gets a planar export copy.
    img.tiffsave(tmp, compression=FAST_COMPRESSION, tile=True, bigtiff=True)
    os.replace(tmp, path)


# ---- the native per-(target, filter) chain: BXT → SXT → stars ----
#
# Star removal runs at the NATIVE pixel grid, per mono channel, BEFORE
# placement. Measured 2026-09-02: SXT is spotless on native masters and
# leaves residual halo "donuts" at every star on the projection-resampled
# canvas — canvas stars are feather-blends of independently-resampled,
# Mercator-warped PSFs, outside the model's training distribution. This
# also puts star removal where RC Astro's docs prescribe it ("right after
# integration", linear, BXT first so its auto-PSF still sees stars). The
# canvas keeps only the grid-insensitive, coherence-sensitive work: the
# channel combines and NXT (which needs 3 co-registered channels —
# per-channel denoising mints chroma noise; placement co-registers free).

def _bxt_channel_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.bxt) — the only
    things a BXT'd channel depends on. `channel` is the trailing
    data-dependency; `config` sits just before it."""
    target, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    return (target, filter, (mock, config.bxt), channel), kwargs


def _sxt_channel_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.sxt)."""
    target, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    return (target, filter, (mock, config.sxt), channel), kwargs


@node(cache_key_narrow=_bxt_channel_narrow, auto_memory=True, resources={"gpu_slots": 1})
def bxt_channel(
    target: Target, filter: str, config: ProcessingConfig, channel: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """Full BlurXTerminator (correct + sharpen, config.bxt) on a per-filter
    channel master at its NATIVE grid — the one and only BXT pass.

    Native mono is where BXT belongs: each channel's PSF is physically its
    own (per-filter seeing/focus), the pixels are camera PSFs rather than
    resampled blends, and aberration correction per panel before assembly
    is Croman's own mosaic guidance (a corner of one frame is the OPPOSITE
    corner of the next, so canvas-level correction can't match both).
    Its output is unsolved: correction shifts the geometry, so
    `native_star_separation` re-solves it in its own node (outside this
    task's GPU slot), and `place_image` projects from that WCS."""
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
    return Produced(out)


@node(cache_key_narrow=_sxt_channel_narrow, auto_memory=True, resources={"gpu_slots": 1})
def sxt_channel(
    target: Target, filter: str, config: ProcessingConfig, channel: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """StarXTerminator the BXT'd native mono channel; keep the starless
    body (linear, 32F). The STARLESS layer's per-panel source.

    A starless image has nothing to solve against, but SXT moves no
    pixels: `native_star_separation` gives it `channel`'s solution
    (`inherit_plate_solution`), which `place_image` requires.

    The mock skims 10% of the flux above the pedestal (footprint zeros
    stay zero): dimensionally a star removal — background preserved, some
    signal moved to the stars layer — so layer algebra, placement
    footprints, and merge level statistics all exercise realistically
    without rc-astro."""
    if config.development_settings.mock_pixinsight:
        from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
        arr = _mono(channel)
        mock = np.where(arr > 0,
                        DEFAULT_PEDESTAL + 0.9 * (arr - DEFAULT_PEDESTAL),
                        np.float32(0)).astype(np.float32)
        _write_mono_tiff(out, mock, filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, starless_out=out,
                ml_version=config.sxt.ml_version, overlap=config.sxt.overlap)
    return Produced(out)


@node(auto_memory=True)
def stars_channel(
    target: Target, filter: str, channel: ImageMetadata, starless: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """The STARS layer: `channel − starless`, clamped at 0 — exactly
    rc-astro's own definition of the stars image on linear data (simple
    subtraction; `--unscreen` is for stretched data). Streaming vips
    subtract, no GPU, no config (pure arithmetic — in mock mode the
    layer algebra still holds because the mock starless is a scaled
    channel). Mostly zeros, so it compresses to almost nothing; the
    with-stars views are reassembled live in the viewer's finishes as
    `starless + star_s·stars`."""
    diff = _open_mono_vips(channel) - _open_mono_vips(starless)
    stars = (diff < 0).ifthenelse(0.0, diff).cast("float")
    _save_mono_vips(out, stars, filter, layer="stars")
    return Produced(out)


@workflow_group
def native_star_separation(target, filter, config, channel, bxt=True):
    """The per-(target, filter) native chain as one legible graph group:
    `bxt_channel → sxt_channel → stars_channel`, returning the
    `(starless, stars)` layer pair the pipelines place and merge.
    `bxt=False` (labs only) skips deconvolution — BXT is PSF cosmetics,
    orthogonal to what the labs gate on, and its rc-astro launches were
    the largest lab compute lane (Stylistic7, 2026-09-02).

    BXT shifts the geometry, so its output is re-solved; the starless and
    stars layers sit on the corrected channel's grid and inherit its
    solution (nothing to solve in a starless image)."""
    from metadata.solve_nodes import inherit_plate_solution, plate_solve_image
    corrected = (plate_solve_image(bxt_channel(target, filter, config, channel))
                 if bxt else channel)
    starless = inherit_plate_solution(sxt_channel(target, filter, config, corrected), corrected)
    stars = inherit_plate_solution(stars_channel(target, filter, corrected, starless), corrected)
    return starless, stars


def _rgb_join(r: ImageMetadata, g: ImageMetadata, b: ImageMetadata) -> pyvips.Image:
    """Streaming bandjoin of three mono layer mosaics: three lazy opens,
    pulled through the tiled writer stripe-by-stripe (never ~54 GB of
    numpy stacks)."""
    monos = [_open_mono_vips(m) for m in (r, g, b)]
    return monos[0].bandjoin([monos[1], monos[2]])


@node(display="Linear RGB Starless", filter="RGB", final=True, auto_memory=True)  # pure bandjoin
def linear_rgb_starless(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """Scene format: the linear STARLESS RGB — bandjoin of the three
    starless channel layer mosaics. Star coherence across R/G/B is a
    property of the LAYERS now (each channel's stars were removed at the
    native grid and the residue subtracted exactly), so no canvas SXT
    pass exists; the with-stars RGB is never baked — the shader adds
    `starless + star_s·stars` live."""
    _save_color_vips(out, _rgb_join(r, g, b), "Linear RGB Starless")
    return Produced(out)


@node(display="Linear RGB Stars", filter="RGB", final=True, auto_memory=True)  # pure bandjoin
def linear_rgb_stars(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """Scene format: the linear RGB STARS — bandjoin of the three stars
    layer mosaics (each exactly `channel − starless` per panel, merged
    without level matching). Coherent star colours come from the shared
    native SXT star decision per panel; a channel's marginal-star misses
    surface in the lab's confetti gate, not as canvas artifacts.
    Overwhelmingly zeros, so the deflate tap costs almost nothing; live
    finishes use it for star intensity (`starless + s·stars`)."""
    _save_color_vips(out, _rgb_join(r, g, b), "Linear RGB Stars", layer="stars")
    return Produced(out)


# ---- named color deliverables (display=name, filter=None, final=True) ----

# The display= of the linear-SHO scene format. The payload's COLORSPC tag
# ("linear-sho", stamped below) is what tells the viewer this 3-band tile is
# S,H,O for a live colour finish rather than baked RGB; the name is the
# tap the Hubble/Foraxx/H/S/O finishes declare as their primary.
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


@node(auto_memory=True)  # streaming bandjoin of the 3 starless channels
def stack_sho_channels(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """Stack the three LINEAR starless narrowband channels (S→R, H→G, O→B) into
    one 3-band linear BigTIFF — the pyvips-streamable source the NXT tiles crop
    from. Fully streaming since 2026-08-08: the inputs are the merged starless
    TIFFs, so the per-channel numpy round-trip (a temp mono BigTIFF per band,
    ~8 GB resident each) is gone — three lazy opens, one bandjoin, one tiled
    write. Legacy non-TIFF inputs fall back inside `_open_mono_vips`."""
    meta = {"PIPELINE": "AstroFlow-OSS", "COLORSPC": "linear-sho"}
    monos = [_open_mono_vips(m) for m in (sii, ha, oiii)]
    stacked = monos[0].bandjoin([monos[1], monos[2]]).copy()
    stacked.set_type(pyvips.GValue.gstr_type, "image-description", json.dumps(meta))
    _atomic_tiffsave(stacked, Path(out),   # zstd: read only by pyvips (the NXT crops)
                     tile=tile_size_for(stacked.width, stacked.height),
                     compression=FAST_COMPRESSION)
    return Produced(out)


@node(cache_key_narrow=_nxt_narrow, auto_memory=True, resources={"gpu_slots": 1}, ephemeral=True)  # NXT on an image (whole OR one tile)
def nxt(
    mosaic: Mosaic, config: ProcessingConfig, image: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """NoiseXTerminator on an image — one PANEL tile fed by process_in_panels;
    SAME code whether it sees a whole channel or a panel crop. A display-less
    transform: the deliverable identity lives on the blend passed to
    process_in_panels. rc-astro reads `image` directly (crop_panel writes it
    strip-based; rc-astro rejects tiled TIFF); its forced-XISF output is converted
    to the node's BigTIFF artifact.

    `ephemeral=True` (2026-09-03, with the merge-content manifest term):
    tile keys now fold the merged raster's identity, and the level solve
    is global — so ANY panel change invalidates EVERY tile. GPU outputs
    normally persist for incremental cache hits, but a cache no upstream
    change can ever hit again is pure disk; when nothing changed, the
    blend itself cache-hits and the tiles never re-run at all. Stylistic7's
    ruling: correctness over compute — a re-merge re-NXTs the mosaic.

    Zero pixels are NO DATA (padding past the canvas, the merge rim, and a
    band whose footprint is smaller than the others'). NXT stretches
    internally from image statistics, and zeros cripple it for the WHOLE
    image: Heart 334b9c01 (19% zeros) lost 13% of its sky noise instead of
    89% (2026-10-02). So each band's zeros are filled with that band's
    median before rc-astro sees them, and set back to zero afterwards."""
    if config.development_settings.mock_pixinsight:
        write_tiff_image(Path(out), read_image(Path(image.filename)), {"PIPELINE": "AstroFlow-OSS"})
    else:
        from rc_astro_cli import run_nxt
        filled = np.atleast_3d(read_image(Path(image.filename))).copy()
        empty = filled == 0
        for band, hole in zip(np.moveaxis(filled, -1, 0), np.moveaxis(empty, -1, 0)):
            if (~hole).any():
                band[hole] = np.median(band[~hole])
        with tempfile.TemporaryDirectory(prefix="nxt_") as td:
            tin, tout = os.path.join(td, "in.tiff"), os.path.join(td, "out.xisf")
            write_tiff_image(Path(tin), filled)
            run_nxt(tin, tout,
                    denoise_intensity_hf=config.nxt.denoise_intensity_hf,
                    denoise_intensity_lf=config.nxt.denoise_intensity_lf,
                    denoise_color_hf=config.nxt.denoise_color_hf,
                    denoise_color_lf=config.nxt.denoise_color_lf,
                    frequency_scale=config.nxt.frequency_scale,
                    iterations=config.nxt.iterations,
                    overlap=config.nxt.overlap)
            result = np.atleast_3d(read_image(Path(tout))).copy()
            result[empty] = 0
            write_tiff_image(Path(out), result, {"PIPELINE": "AstroFlow-OSS"})
    return Produced(out)


@node(cache_key_narrow=_blend_sho_narrow, display=LINEAR_SHO_DISPLAY, filter="RGB", final=True, auto_memory=True)  # 3-channel LINEAR, NXT-denoised
def blend_linear_sho(
    mosaic: Mosaic, config: ProcessingConfig, viewbox, panels: tuple,
    *tiles: ImageMetadata, out: TiffFilePath,
) -> Produced:
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
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        taper_px=config.tiling.overlap_px, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS", "COLORSPC": "linear-sho"},
        canvas=(viewbox.num_pixels_x, viewbox.num_pixels_y))
    return Produced(out)


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
    (`mosaic.viewbox("Master")`)."""
    # The SHO band-join is cheap × big — ephemeral. The NXT tiles are
    # ephemeral too (2026-09-03): their keys fold the merged raster's
    # identity, so any re-merge invalidates all of them — persistence
    # bought nothing, and a no-change run cache-hits the blend without
    # ever re-running a tile.
    source = combine_panels(stack_sho_channels, mosaic, sii, ha, oiii)
    result = process_in_panels(
        nxt, mosaic, config, source, viewbox=viewbox, blend=blend_linear_sho)
    return result.merged


# ---- linear scene formats: the ≤3-band basis the live WebGPU finishes
# ---- sample. Served as STARLESS + STARS layer pairs (with-stars views
# ---- are shader sums, never baked) — except narrowband, which gets no
# ---- stars combine: star colour rides "Linear RGB Stars", and a format
# ---- no finish reads is disk spent on nothing. Narrowband formats are
# ---- always 3 bands in canonical S,H,O order with absent bands
# ---- ZERO-FILLED (uniform texture shape, zeros deflate to nothing) and
# ---- the REAL band list in the TIFF description.

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


@node(display="Linear Narrowband Starless", filter="RGB", final=True, auto_memory=True)  # 3-band linear
def linear_narrowband_starless(
    mosaic: Mosaic, filters: tuple, *channels: ImageMetadata, out: TiffFilePath,
) -> Produced:
    """Scene format for PARTIAL narrowband sets (an Hα-only galaxy, an
    HOO dual-band OSC): the starless layer combination, zero-filled to
    S,H,O. Full-SHO images don't get this — "Linear SHO" (NXT-denoised)
    already serves that role. The galaxy finishes (continuum subtraction
    Hα − q·R) sample this directly — starless Hα is the cleaner line
    source (star continuum never enters the injection)."""
    img = _nb_threeband(dict(zip(filters, channels)))
    _save_color_vips(out, img, "Linear Narrowband Starless", bands=filters)
    return Produced(out)


@node(display="Linear L Starless", filter="L", final=True, auto_memory=True)  # mono linear
def linear_lum(
    mosaic: Mosaic, lum: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """The starless luminance layer mosaic as a served mono tap — the
    lightness source for live LRGB/HaLRGB finishing."""
    _save_mono_vips(out, _open_mono_vips(lum), "L", layer="starless")
    return Produced(out)


@node(display="Linear L Stars", filter="L", final=True, auto_memory=True)  # mono linear
def linear_lum_stars(
    mosaic: Mosaic, stars: ImageMetadata, *, out: TiffFilePath,
) -> Produced:
    """The luminance STARS layer as a served mono tap — added back into
    the LRGB lightness live by the shader."""
    _save_mono_vips(out, _open_mono_vips(stars), "L", layer="stars")
    return Produced(out)


def panel_source_manifest(targets_by_id: dict, filter: str, config,
                          channels: dict) -> dict:
    """`{panel_id: pre-merge identity}` for `color_post_process(panel_sources=)`.

    The identity is `sxt_channel`'s output stem for that panel — the terminal
    of the native per-panel chain (BXT folds in via its input), computed with
    the same `op.filename_fn(...)` + `IdentityRef` arithmetic `place_image`
    refs use, so it is the real product identity rather than a re-derivation
    that could drift from it.

    This is what makes two pipelines processing ONE target produce distinct
    colour tails: `channels[(tid, filter)]` is the per-filter merged channel,
    which already differs between pipelines, so the stem does too. Keyed per
    panel for tile IDENTITY (which panel's pixels a tile carries — see
    mosaics/panels.py; the merged raster's content term governs staleness).

    `channels` is the `{(target_id, filter): channel}` map the pipelines build;
    entries for other filters and unknown targets are ignored."""
    from dependency_graph.tracing import IdentityRef
    out = {}
    for (tid, f), channel in channels.items():
        if f != filter or tid not in targets_by_id:
            continue
        target = targets_by_id[tid]
        bxt_stem = Path(bxt_channel.filename_fn(
            target, filter, config, channel)).stem
        out[tid] = Path(sxt_channel.filename_fn(
            target, filter, config, IdentityRef(bxt_stem))).stem
    return out


@workflow_group
def color_post_process(mosaic: Mosaic, starless: dict, stars: dict, config,
                       viewbox=None,
                       panel_sources: dict | None = None) -> dict:
    """Turn the per-filter LAYER mosaics into the LINEAR SCENE FORMATS —
    the orthogonal ≤3-band basis every colour look is assembled from,
    LIVE, in the viewer's WGSL finishes (finishes/stock.py +
    ~/AstroFlow/finishes/).

    All rc-astro star work already happened at the NATIVE grid
    (`bxt_channel` → `sxt_channel` → `stars_channel`, before placement —
    see the chain comment above); this group is the CANVAS tail: pure
    layer combines plus the one op that genuinely wants co-registered
    3-channel input, NXT (per-channel denoising mints chroma noise;
    placement co-registers the channels for free). No palette is baked
    (2026-08-12), and neither is any with-stars view — both are per-pixel
    functions of these formats, assembled in-shader
    (`starless + star_s·stars`).

    Output formats: Linear SHO (NXT'd starless SHO composite), Linear
    Narrowband Starless (partial sets), Linear RGB Starless + Linear RGB
    Stars, Linear L Starless + Linear L Stars. No narrowband-stars
    combine exists — no finish samples one (see the note below).

    `starless` / `stars` are `{filter: merged layer handle}` from the
    pipelines' `_merged_channels`. `viewbox` gives the trace-time canvas
    dims for the tiled NXT round (`linear_sho`); defaults to the Master
    viewbox for callers (e.g. tests) that don't tile at another quality.
    """
    if viewbox is None:
        viewbox = mosaic.viewbox("Master")
    # Normalized the same way throughout, so no lookup can miss on a
    # filter-name spelling ("Ha" vs "H") and silently fall back to the
    # colliding content_hash manifest.
    sl = {normalize_filter(f): ch for f, ch in starless.items()}
    st = {normalize_filter(f): ch for f, ch in stars.items()}
    panels_by_class = {normalize_filter(f): p
                       for f, p in (panel_sources or {}).items()}

    out: dict[str, object] = {}
    if _NARROWBAND <= set(sl):
        # The linear S/H/O viewer source: starless layers stacked +
        # NXT-denoised per panel, finished live in the WebGPU shader
        # (Hubble/Foraxx and friends). base_panel_mosaic wraps each merged
        # layer with its per-panel manifest so the NXT crops key per-panel.
        pms = {
            f: base_panel_mosaic(mosaic, f, sl[f], panels_by_class.get(f))
            for f in _NARROWBAND
        }
        out["Linear SHO"] = linear_sho(
            mosaic, pms["S"], pms["H"], pms["O"], config, viewbox)

    nb_present = tuple(f for f in _NB_ORDER if f in sl)
    if nb_present and not (_NARROWBAND <= set(sl)):
        # Partial sets only — full SHO's starless role is the
        # NXT-denoised "Linear SHO" tap. NO narrowband-stars combine is
        # baked: no finish samples one (narrowband star colour comes
        # from "Linear RGB Stars"; the per-filter star layers are served
        # as the "Mosaic Stars" taps) — a scene format nothing reads is
        # disk spent on nothing. If a finish ever wants it, re-add the
        # combine over `st` and give that finish the input.
        out["Linear Narrowband Starless"] = linear_narrowband_starless(
            mosaic, nb_present, *[sl[f] for f in nb_present])

    if _BROADBAND <= set(sl):
        out["Linear RGB Starless"] = linear_rgb_starless(
            mosaic, sl["R"], sl["G"], sl["B"])
        out["Linear RGB Stars"] = linear_rgb_stars(
            mosaic, st["R"], st["G"], st["B"])
        if "L" in sl:
            # Luminance serves as the LRGB lightness; skipped entirely
            # unless full R/G/B exists to give it chrominance.
            out["Linear L Starless"] = linear_lum(mosaic, sl["L"])
            out["Linear L Stars"] = linear_lum_stars(mosaic, st["L"])
    return out
