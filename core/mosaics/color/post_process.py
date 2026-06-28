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
import os
from pathlib import Path

import numpy as np
import structlog

import astropy.io.fits as pyfits

from astroflow.pipeline import ephemeral_intermediates
from decorators import node
from dependency_graph.tracing import workflow_group
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from file_io.types import FitsFilePath, TiffFilePath, WithSidecar
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


def _write_mono_fits(path: str, data: np.ndarray, filter: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = pyfits.Header()
    header["FILTER"] = filter
    header["PIPELINE"] = "AstroFlow-OSS"  # no IMAGETYP=Light -> never plate-solved
    pyfits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(
        str(path), overwrite=True,
    )


def _write_color_tiff(path: str, rgb: np.ndarray, name: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_tiff_image(Path(path), rgb.astype(np.float32),
                     {"PALETTE": name, "PIPELINE": "AstroFlow-OSS"})


# ---- per-channel intermediates (no display=; ephemeral) ----

@node(auto_memory=True)
def bxt_correct(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """BlurXTerminator a per-filter mosaic channel (linear deconvolution)."""
    out = bxt_correct.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_fits(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_bxt
        run_bxt(channel.filename, out)
    return WithSidecar(FitsFilePath(out))


@node(auto_memory=True)
def sxt_starless(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """StarXTerminator a narrowband channel; keep the starless body (linear)."""
    out = sxt_starless.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_fits(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, starless_out=out)
    return WithSidecar(FitsFilePath(out))


@node(auto_memory=True)
def sxt_stars(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool, channel: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """StarXTerminator a broadband channel; keep the UNSCREENED stars (linear),
    ready to re-apply with a screen blend."""
    out = sxt_stars.filename_fn(mosaic, filter, mock_pixinsight, channel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if mock_pixinsight:
        _write_mono_fits(out, _mono(channel), filter)
    else:
        from rc_astro_cli import run_sxt
        run_sxt(channel.filename, stars_out=out, unscreen=True)
    return WithSidecar(FitsFilePath(out))


@node(auto_memory=True)
def autostretch_channel(
    mosaic: Mosaic, filter: str, channel: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """STF-equivalent autostretch — palettes need non-linear input for their
    dynamic weights to be meaningful."""
    out = autostretch_channel.filename_fn(mosaic, filter, channel)
    _write_mono_fits(out, palettes.autostretch(_mono(channel)), filter)
    return WithSidecar(FitsFilePath(out))


# ---- named color deliverables (display=name, filter=None, final=True) ----

@node(display="Stars", final=True, auto_memory=True)
def combine_star_rgb(
    mosaic: Mosaic, r: ImageMetadata, g: ImageMetadata, b: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Stack the three unscreened star channels into one RGB star image."""
    out = combine_star_rgb.filename_fn(mosaic, r, g, b)
    rgb = palettes.combine_star_rgb(_mono(r), _mono(g), _mono(b))
    _write_color_tiff(out, rgb, "Stars")
    return WithSidecar(TiffFilePath(out))


def _palette_image(out, combine, sii, ha, oiii, stars, name):
    rgb = combine(_mono(sii), _mono(ha), _mono(oiii))
    if stars is not None:  # screen the RGB stars back over the starless palette
        rgb = palettes.screen(rgb, read_image(Path(stars.filename)).astype(np.float32))
    _write_color_tiff(out, rgb, name)


@node(display="Hubble", final=True, auto_memory=True)
def hubble_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Hubble (SHO) palette of the stretched starless narrowband, RGB stars
    screened in. `stars` is the `combine_star_rgb` output, or None (no broadband)."""
    out = hubble_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.hubble, sii, ha, oiii, stars, "Hubble")
    return WithSidecar(TiffFilePath(out))


@node(display="Foraxx", final=True, auto_memory=True)
def foraxx_image(
    mosaic: Mosaic, sii: ImageMetadata, ha: ImageMetadata, oiii: ImageMetadata, stars,
) -> WithSidecar[TiffFilePath]:
    """Foraxx dynamic palette of the stretched starless narrowband, RGB stars
    screened in."""
    out = foraxx_image.filename_fn(mosaic, sii, ha, oiii, stars)
    _palette_image(out, palettes.foraxx, sii, ha, oiii, stars, "Foraxx")
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
