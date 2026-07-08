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
from decorators import node
from dependency_graph.tracing import workflow_group
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image, TIFF_COMPRESSION
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
    write_tiff_image(Path(path), data.astype(np.float32),
                     {"FILTER": filter, "PIPELINE": "AstroFlow-OSS"})


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
    img = img.copy()
    img.set_type(pyvips.GValue.gstr_type, "image-description",
                 json.dumps({"PALETTE": name, "PIPELINE": "AstroFlow-OSS"}))
    tmp = f"{path}.{os.getpid()}.tmp.tiff"
    img.tiffsave(tmp, compression=TIFF_COMPRESSION, tile=True, bigtiff=True)
    os.replace(tmp, path)


# ---- per-channel intermediates (no display=; ephemeral) ----

@node(auto_memory=True, plate_solve=False)  # derived from an already-solved channel
def bxt_correct(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """BlurXTerminator a per-filter mosaic channel (linear deconvolution).

    XISF output: full-res channels exceed rc-astro's cfitsio FITS-writer limit
    (2**31 px); XISF has none. See `rc_astro_cli`."""
    out = bxt_correct.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_xisf(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_bxt
        run_bxt(channel.filename, out)
    return WithSidecar(XisfFilePath(out))


@node(auto_memory=True, plate_solve=False)  # starless: no stars to solve against
def sxt_starless(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """StarXTerminator a narrowband channel; keep the starless body (linear).

    XISF output (its FITS writer overflows on full-res channels; `channel` is
    the XISF BXT output, so rc-astro reads and writes XISF here)."""
    out = sxt_starless.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_xisf(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, starless_out=out)
    return WithSidecar(XisfFilePath(out))


@node(auto_memory=True, plate_solve=False)  # derived from an already-solved channel
def sxt_stars(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """StarXTerminator a broadband channel; keep the UNSCREENED stars (linear),
    ready to re-apply with a screen blend.

    XISF output (its FITS writer overflows on full-res channels; `channel` is
    the XISF BXT output, so rc-astro reads and writes XISF here)."""
    out = sxt_stars.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_xisf(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, stars_out=out, unscreen=True)
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

@node(display="Stars", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def combine_star_rgb(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Stack the three unscreened star channels into one RGB star image."""
    out = combine_star_rgb.filename_fn(mosaic, r, g, b)
    rgb = palettes.combine_star_rgb_vips(
        _load_mono_vips(r), _load_mono_vips(g), _load_mono_vips(b))
    _save_color_vips(out, rgb, "Stars")
    return WithSidecar(TiffFilePath(out))


def _palette_image(out, combine_vips, sii, ha, oiii, stars, name):
    rgb = combine_vips(_load_mono_vips(sii), _load_mono_vips(ha), _load_mono_vips(oiii))
    if stars is not None:  # screen the RGB stars back over the starless palette
        rgb = palettes.screen_vips(rgb, pyvips.Image.new_from_file(str(stars.filename)))
    _save_color_vips(out, rgb, name)


@node(display="Hubble", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def hubble_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Hubble (SHO) palette of the stretched starless narrowband, RGB stars
    screened in. `stars` is the `combine_star_rgb` output, or None (no broadband)."""
    out = hubble_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.hubble_vips, sii, ha, oiii, stars, "Hubble")
    return WithSidecar(TiffFilePath(out))


@node(display="Foraxx", final=True, auto_memory=True, plate_solve=False)  # 3-channel color
def foraxx_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Foraxx dynamic palette of the stretched starless narrowband, RGB stars
    screened in."""
    out = foraxx_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.foraxx_vips, sii, ha, oiii, stars, "Foraxx")
    return WithSidecar(TiffFilePath(out))


@workflow_group
def color_post_process(mosaic: Mosaic, channels: dict, config) -> dict:
    """Turn merged per-filter mosaic channels into named color images.

    `channels` is `{filter: merged-channel handle}` from `for_mosaic`. BXT every
    channel; SXT the narrowband to starless and the broadband to unscreened
    stars; autostretch; then build the Hubble + Foraxx palettes (screening the
    combined RGB stars in). Returns `{name: handle}` for the color deliverables.
    """
    mock = config.development_settings.mock_pixinsight
    by_class = {normalize_filter(f): ch for f, ch in channels.items()}

    stretched_nb: dict[str, object] = {}
    stretched_bb: dict[str, object] = {}
    with ephemeral_intermediates():
        for cf, channel in by_class.items():
            if cf in _NARROWBAND:
                corrected = bxt_correct(mosaic, cf, mock, channel)
                starless = sxt_starless(mosaic, cf, mock, corrected)
                stretched_nb[cf] = autostretch_channel(mosaic, cf, starless)
            elif cf in _BROADBAND:
                corrected = bxt_correct(mosaic, cf, mock, channel)
                stars = sxt_stars(mosaic, cf, mock, corrected)
                stretched_bb[cf] = autostretch_channel(mosaic, cf, stars)

    out: dict[str, object] = {}
    rgb_stars = None
    if _BROADBAND <= set(stretched_bb):
        rgb_stars = combine_star_rgb(
            mosaic, stretched_bb["R"], stretched_bb["G"], stretched_bb["B"],
        )
        out["Stars"] = rgb_stars
    if _NARROWBAND <= set(stretched_nb):
        s, h, o = stretched_nb["S"], stretched_nb["H"], stretched_nb["O"]
        out["Hubble"] = hubble_image(mosaic, s, h, o, rgb_stars)
        out["Foraxx"] = foraxx_image(mosaic, s, h, o, rgb_stars)
    return out
