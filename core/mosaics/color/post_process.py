"""OSS mosaic color post-processing: @node ops + the `color_post_process`
workflow_group that `for_mosaic` calls.

PixInsight-free. BXT/SXT run via the rc-astro CLI on LINEAR data; a per-channel
autostretch then makes the dynamic palette weights meaningful; the palettes,
RGB-star combination, and screen blend are numpy (`mosaics.color.palettes`).

The framework is one-artifact-per-`@node`, so the "several outputs" are several
ops: each named color image (`Hubble`, `Foraxx`, `Stars`) is its own `@node`
with its own `display=` name and `filter=None` (a color composite has no single
filter). The per-channel BXT/SXT/stretch steps are `display`-less intermediates
wrapped in `ephemeral_intermediates()`. `mock_pixinsight` passes the heavy
rc-astro ML calls straight through so the suite runs without rc-astro.
"""
import json
import os
from pathlib import Path

import numpy as np
import pyvips
import structlog

from astroflow.pipeline import ephemeral_intermediates
from decisions.settings import ProcessingConfig
from decorators import node
from dependency_graph.tracing import workflow_group
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image, FAST_COMPRESSION, COMPATIBLE_COMPRESSION
from file_io.types import TiffFilePath, WithSidecar, XisfFilePath
from file_io.xisf_file_io import write_xisf_image
from metadata.classes import ImageMetadata
from mosaics.color import palettes
from mosaics.types import Mosaic
from pix.tasks.filters import normalize_filter

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


def _load_mono_vips(meta: ImageMetadata) -> pyvips.Image:
    """Load a mono channel (an autostretch FITS output) as a 1-band float vips
    image. Lazy: vips pulls pixels only as the `tiffsave` sink demands them, so
    the palette never materializes a whole 9 GB channel."""
    img = pyvips.Image.new_from_file(str(meta.filename))
    return img[0] if img.bands > 1 else img


def _save_color_vips(path: str, img: pyvips.Image, name: str) -> None:
    """Stream a float RGB vips image to a tiled BigTIFF deliverable (atomic
    tmp+rename), tagging the palette name / pipeline in the TIFF description
    (the same JSON `read_tiff_header` expects)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Force scRGB: the bandjoined float palette otherwise inherits a `b-w`
    # (grayscale) interpretation, so tiffsave writes a MINISBLACK photometric and
    # every interp-aware reader (pyvips thumbnails, the gallery preview) renders
    # the 3 real colour channels as luminance → grey. scRGB is the correct
    # float-RGB interpretation and matches the pre-regression deliverables.
    img = img.copy(interpretation="scrgb")
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"PALETTE": name, "PIPELINE": "AstroFlow-OSS"}))
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
    return (mosaic, filter, (mock, config.bxt), channel), kwargs


def _sxt_narrow(args, kwargs):
    """Cache-key hook for sxt_starless: project `config` to
    (mock, config.sxt). Shared by both ops — their differing bodies (starless vs
    unscreened stars) already give them distinct call hashes via op identity."""
    mosaic, filter, config, channel = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic, filter, (mock, config.sxt), channel), kwargs


@node(cache_key_narrow=_bxt_correct_narrow, auto_memory=True, plate_solve=False)  # derived from an already-solved channel
def bxt_correct(
    mosaic: Mosaic, filter: str, config: ProcessingConfig, channel: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
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
        _write_mono_xisf(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_bxt
        run_bxt(channel.filename, out,
                sharpen_stars=config.bxt.sharpen_stars,
                sharpen_nonstellar=config.bxt.sharpen_nonstellar,
                correct_only=config.bxt.correct_only,
                ml_version=config.bxt.ml_version,
                overlap=config.bxt.overlap)
    return WithSidecar(XisfFilePath(out))


@node(cache_key_narrow=_sxt_narrow, auto_memory=True, plate_solve=False)  # starless: no stars to solve against
def sxt_starless(
    mosaic: Mosaic, filter: str, config: ProcessingConfig, channel: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """StarXTerminator a narrowband channel; keep the starless body (linear).

    Cache key narrowed to (mock, config.sxt). XISF output (its FITS writer
    overflows on full-res channels; `channel` is the XISF BXT output, so rc-astro
    reads and writes XISF here)."""
    out = sxt_starless.filename_fn(mosaic, filter, config, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        _write_mono_xisf(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, starless_out=out,
                ml_version=config.sxt.ml_version, overlap=config.sxt.overlap)
    return WithSidecar(XisfFilePath(out))


def _sxt_rgb_narrow(args, kwargs):
    """Cache-key hook for sxt_stars_rgb: project `config` to (mock, config.sxt)."""
    mosaic, config, rgb = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic, (mock, config.sxt), rgb), kwargs


@node(auto_memory=True, plate_solve=False)  # pure bandjoin, no PixInsight
def combine_broadband_rgb(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Combine the three LINEAR BXT'd broadband channels into one linear RGB
    XISF so StarXTerminator can extract stars from all channels AT ONCE.

    Extracting per-channel (the old path) let SXT catch a different set of
    marginal stars in each of R/G/B — a faint star found in R but missed in
    G/B became a pure-red dot, giving 'confetti' stars of impossible primary
    colours. Seeing the combined image, SXT's star decision is identical across
    channels. XISF keeps rc-astro happy at gigapixel sizes."""
    out = combine_broadband_rgb.filename_fn(mosaic, r, g, b)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rgb = np.stack([_mono(r), _mono(g), _mono(b)], axis=2).astype(np.float32)
    write_xisf_image(Path(out), rgb, {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(XisfFilePath(out))


@node(cache_key_narrow=_sxt_rgb_narrow, auto_memory=True, plate_solve=False)  # coherent RGB stars
def sxt_stars_rgb(
    mosaic: Mosaic, config: ProcessingConfig, rgb: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """StarXTerminator the combined linear RGB; keep the UNSCREENED RGB stars.
    One SXT pass over the RGB (not one per channel) makes the extracted stars
    coherent across R/G/B, so the star image has natural colours instead of
    per-channel primary-colour confetti."""
    out = sxt_stars_rgb.filename_fn(mosaic, config, rgb)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if config.development_settings.mock_pixinsight:
        write_xisf_image(Path(out), read_image(Path(rgb.filename)).astype(np.float32),
                         {"PIPELINE": "AstroFlow-OSS"})
    else:
        from rc_astro_cli import run_sxt
        run_sxt(rgb.filename, stars_out=out, unscreen=True,
                ml_version=config.sxt.ml_version, overlap=config.sxt.overlap)
    return WithSidecar(XisfFilePath(out))


@node(auto_memory=True, plate_solve=False)  # nonlinear stretch of an already-solved channel
def autostretch_channel(
    mosaic: Mosaic, filter: str, channel: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """STF-equivalent autostretch — palettes need non-linear input for their
    dynamic weights to be meaningful.

    The bridge op: reads the XISF BXT/SXT output (full-array numpy — libvips has
    no xisfload, so this is the one non-streaming step; memory-reserved in
    `profiling.memory_estimate`) and emits BigTIFF, so the palettes downstream
    stream via pyvips `tiffload`. When BXT/SXT are eventually tiled to emit TIFF,
    this can move to a streaming pyvips implementation (`palettes.autostretch`
    has a proven vips port; see the tiled-BXT note in `rc_astro_cli`)."""
    out = autostretch_channel.filename_fn(mosaic, filter, channel)
    _write_mono_tiff(out, palettes.autostretch(_mono(channel)), filter)
    return WithSidecar(TiffFilePath(out))


# ---- named color deliverables (display=name, filter=None, final=True) ----

# The display= of the linear-SHO viewer source. Kept as a constant because the
# tile server keys `colorspace="linear-sho"` on this exact display name (see
# web_server/backend/tiles.py LINEAR_FINISH_DISPLAYS) so the WebGPU shader knows
# to run the live colour finish instead of passing the tile through as baked RGB.
LINEAR_SHO_DISPLAY = "Linear SHO"


def _linear_sho_narrow(args, kwargs):
    """Cache-key hook: project `config` to (mock, config.nxt) — the only config a
    denoised linear-SHO master depends on. Mirrors _bxt_correct_narrow: the three
    channel data-deps sit before `config`."""
    mosaic, sii, ha, oiii, config = args
    mock = config.development_settings.mock_pixinsight
    return (mosaic, sii, ha, oiii, (mock, config.nxt)), kwargs


@node(cache_key_narrow=_linear_sho_narrow, display=LINEAR_SHO_DISPLAY, filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel LINEAR, NXT-denoised
def linear_sho(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata,
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    """Stack the three LINEAR starless narrowband channels (S→R, H→G, O→B) into
    one 3-band linear master, then denoise it with NoiseXTerminator — the source
    the WebGPU tile viewer finishes live.

    NXT placement follows the RC-Astro manual: AFTER deconvolution (`bxt_correct`)
    and AFTER channel combination (this stack), on the LINEAR combined colour
    image, so its intensity/colour + HF/LF separation reduces the weak-channel
    (SII/OIII) colour noise hard while keeping the Hα-dominated luminance detail.
    The shader then does only the STANDARD per-channel stretch + Hubble/SCNR
    palette on this clean data — deliberately NO stretch/palette here.

    Being `final=True`, this consumes the otherwise-ephemeral `sxt_starless`
    outputs and keeps them alive until it runs. The channels share the mosaic
    viewbox (co-registered), so they stack pixel-aligned. XISF keeps the gigapixel
    master within rc-astro/format limits (the served pyramid is the auto-built
    `_tiles.tiff` sibling)."""
    out = linear_sho.filename_fn(mosaic, sii, ha, oiii, config)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sho = np.stack([_mono(sii), _mono(ha), _mono(oiii)], axis=2).astype(np.float32)
    meta = {"PIPELINE": "AstroFlow-OSS", "COLORSPC": "linear-sho"}
    if config.development_settings.mock_pixinsight:
        write_xisf_image(Path(out), sho, meta)
    else:
        # Stack → temp XISF → NXT (rc-astro reads/writes XISF at gigapixel; its
        # FITS writer overflows). NXT runs on the 3-band colour image so its
        # intensity/colour separation applies (the whole point for narrowband).
        tmp = f"{out}.stack.xisf"
        write_xisf_image(Path(tmp), sho, meta)
        from rc_astro_cli import run_nxt
        run_nxt(tmp, out,
                denoise_intensity_hf=config.nxt.denoise_intensity_hf,
                denoise_intensity_lf=config.nxt.denoise_intensity_lf,
                denoise_color_hf=config.nxt.denoise_color_hf,
                denoise_color_lf=config.nxt.denoise_color_lf,
                frequency_scale=config.nxt.frequency_scale,
                iterations=config.nxt.iterations)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return WithSidecar(XisfFilePath(out))


@node(display="Stars", filter="RGB", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def combine_star_rgb(
    mosaic: Mosaic, rgb_stars: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The RGB-stars deliverable: a LINKED autostretch of the coherent SXT'd RGB
    stars. `palettes.autostretch` takes one median/MAD over all three bands, so
    every channel gets the same MTF — star colour is preserved (no per-channel
    cast). Saved as the scRGB TIFF deliverable."""
    out = combine_star_rgb.filename_fn(mosaic, rgb_stars)
    arr = read_image(Path(rgb_stars.filename)).astype(np.float32)  # (H, W, 3) linear
    stretched = np.ascontiguousarray(palettes.autostretch(arr), dtype=np.float32)
    h, w = stretched.shape[:2]
    vimg = pyvips.Image.new_from_memory(stretched.data, w, h, 3, "float")
    _save_color_vips(out, vimg, "Stars")
    return WithSidecar(TiffFilePath(out))


def _palette_image(out, combine_vips, sii, ha, oiii, stars, name):
    rgb = combine_vips(_load_mono_vips(sii), _load_mono_vips(ha), _load_mono_vips(oiii))
    if stars is not None:  # screen the RGB stars back over the starless palette
        rgb = palettes.screen_vips(rgb, pyvips.Image.new_from_file(str(stars.filename)))
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


@workflow_group
def color_post_process(mosaic: Mosaic, channels: dict, config) -> dict:
    """Turn merged per-filter mosaic channels into named color images.

    `channels` is `{filter: merged-channel handle}` from `for_mosaic`. BXT every
    channel; SXT the narrowband to starless and the broadband to unscreened
    stars; autostretch; then build the Hubble + Foraxx palettes (screening the
    combined RGB stars in). Returns `{name: handle}` for the color deliverables.
    """
    by_class = {normalize_filter(f): ch for f, ch in channels.items()}

    starless_nb: dict[str, object] = {}
    stretched_nb: dict[str, object] = {}
    bxt_bb: dict[str, object] = {}
    with ephemeral_intermediates():
        for cf, channel in by_class.items():
            if cf in _NARROWBAND:
                corrected = bxt_correct(mosaic, cf, config, channel)
                starless = sxt_starless(mosaic, cf, config, corrected)
                starless_nb[cf] = starless  # linear source for the Linear SHO tap
                stretched_nb[cf] = autostretch_channel(mosaic, cf, starless)
            elif cf in _BROADBAND:
                bxt_bb[cf] = bxt_correct(mosaic, cf, config, channel)

        # Broadband stars: combine the linear BXT'd channels into one RGB, then
        # SXT ONCE on that RGB so the extracted stars are coherent across R/G/B
        # (no per-channel primary-colour confetti).
        rgb_star_lin = None
        if _BROADBAND <= set(bxt_bb):
            rgb_lin = combine_broadband_rgb(mosaic, bxt_bb["R"], bxt_bb["G"], bxt_bb["B"])
            rgb_star_lin = sxt_stars_rgb(mosaic, config, rgb_lin)

    out: dict[str, object] = {}
    rgb_stars = None
    if rgb_star_lin is not None:
        rgb_stars = combine_star_rgb(mosaic, rgb_star_lin)
        out["Stars"] = rgb_stars
    if _NARROWBAND <= set(starless_nb):
        # The linear S/H/O viewer source: stacked + NXT-denoised, finished live
        # in the WebGPU shader with the standard Hubble/SCNR palette.
        out["Linear SHO"] = linear_sho(
            mosaic, starless_nb["S"], starless_nb["H"], starless_nb["O"], config)
    if _NARROWBAND <= set(stretched_nb):
        s, h, o = stretched_nb["S"], stretched_nb["H"], stretched_nb["O"]
        # Always emit the pure starless palettes; add the with-stars variants
        # only when broadband RGB stars exist to screen in.
        out["Hubble Starless"] = hubble_starless(mosaic, s, h, o)
        out["Foraxx Starless"] = foraxx_starless(mosaic, s, h, o)
        if rgb_stars is not None:
            out["Hubble"] = hubble_image(mosaic, s, h, o, rgb_stars)
            out["Foraxx"] = foraxx_image(mosaic, s, h, o, rgb_stars)
    return out
