"""Synthesize bias/dark/flat/light frames + masters from a physics model.

Built for the calibration-validation harness (`core/calibration/
test_end_to_end_mosaic.py`). The model is the standard CCD/CMOS
electron-bookkeeping equation:

    frame_ADU[x,y] = B[x,y] + ( signal_e[x,y]
                                + Poisson(signal_e[x,y])
                                + N(0, σ_r) ) / gain

with `signal_e` depending on frame type:
    Bias  (0 s):       0
    Dark  (t_l s):     D[x,y] · t_l
    Flat  (t_f s):     F[x,y] · I_flat · t_f
    Light (t_l s):     F[x,y] · I_sky  · t_l   +   D[x,y] · t_l

The function `synthesize_calibration_suite` builds the ground-truth
fields once (`B`, `D`, `F`), draws N independent realizations per
frame type, writes uint16 FITS via `make_synthetic_frames`, and also
builds the three master frames in Python so the test stages them
directly into a WBPP working directory.

Two master-dark topologies are switchable via `pre_calibrate_dark`:
    False — Landmann §6.2 + §7.1 Case 2 compliant: master dark is a
            simple integration of raw darks. Contains the bias
            pedestal as a feature.
    True  — Today's AstroFlow behavior: bias is subtracted from the
            master dark before saving. Explicitly named as an anti-
            pattern by Landmann §6.2 ("NEVER pre-calibrate").

The test invokes both, runs WBPP on each, and asserts the calibrated
outputs differ — falsifying the assumption that the pre-calibration
is mathematically irrelevant.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np
import xxhash

from testing.test_frames import make_synthetic_frames


@dataclass(frozen=True)
class CalibrationGroundTruth:
    """Everything the calibration-validation test needs to compare the
    WBPP-calibrated output against a known signal.

    All electron-domain fields are in electrons (per-pixel for arrays,
    per-second for rates as noted). `B` is in ADU (bias pedestal is a
    digital readout offset, not a photon count). Frame paths point at
    files on disk written by `make_synthetic_frames`.
    """
    # Ground-truth instrument response and signal.
    B: np.ndarray            # bias pedestal, ADU
    D: np.ndarray            # dark current rate, e⁻/sec/pixel
    F: np.ndarray            # flat field, dimensionless [0,1]
    I_sky_e_per_sec: float   # uniform sky background, e⁻/sec/pixel
    I_flat_e_per_sec: float  # flat illumination level, e⁻/sec/pixel
    read_noise_sigma_e: float
    gain_e_per_adu: float
    light_exposure_s: float
    flat_exposure_s: float

    # On-disk frame files.
    bias_files: list[str]
    dark_files: list[str]
    flat_files: list[str]
    light_files: list[str]

    # On-disk master files (FITS, float32, with IMAGETYP="Master *").
    master_bias_file: Path
    master_dark_file: Path
    master_flat_file: Path

    # Flag the test asserts against — tells the topology under test.
    is_master_dark_pre_calibrated: bool


def _build_bias_pattern(
    width: int, height: int, mean_adu: float, amplitude_adu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bias pedestal: uniform DC + low-amplitude horizontal banding.

    Real CMOS bias has spatial structure (row noise, banding); the
    sinusoidal pattern is a stand-in for that. Amplitude is small
    enough that the pattern is invisible to the eye in a single
    frame but accumulates into a measurable seam signal under
    miscorrected calibration."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    # One sinusoid per axis; phases come from rng so different
    # callers get different (still deterministic) patterns.
    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)
    pattern = (
        np.sin(2 * np.pi * cols / width + phase_x) * 0.5
        + np.sin(2 * np.pi * rows / height + phase_y) * 0.5
    )
    return mean_adu + amplitude_adu * pattern.astype(np.float64)


def _build_dark_current(
    width: int, height: int, base_rate_e_per_sec: float,
    n_hot_pixels: int, hot_pixel_rate_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform low-rate dark current plus a handful of hot pixels.

    Real CMOS dark current is dominated by ~base_rate ≈ 0.003 e/s at
    typical -10 °C with sparse pixels orders of magnitude hotter."""
    D = np.full((height, width), base_rate_e_per_sec, dtype=np.float64)
    xs = rng.integers(0, width, size=n_hot_pixels)
    ys = rng.integers(0, height, size=n_hot_pixels)
    D[ys, xs] = base_rate_e_per_sec * hot_pixel_rate_multiplier
    return D


def _build_flat_field(
    width: int, height: int,
    corner_response: float, dust_spot_amplitude: float,
) -> np.ndarray:
    """Radial vignetting (1.0 at center → `corner_response` at corners)
    plus a deterministic Gaussian "dust spot" off-center.

    The dust spot makes the flat-correction signal stronger than pure
    vignetting and lets the test detect a residual flat error that a
    purely smooth vignetting wouldn't catch."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx, cy = width / 2, height / 2
    # Normalize so the corner has r=1.
    rmax = np.sqrt(cx ** 2 + cy ** 2)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rmax
    # Cosine vignetting: smooth falloff from 1 at center.
    vignette = 1.0 - (1.0 - corner_response) * (r ** 2)
    # Dust spot: 50 px Gaussian at (0.3 W, 0.7 H), shadowing.
    dx = xx - 0.3 * width
    dy = yy - 0.7 * height
    sigma_dust = 50.0
    spot = dust_spot_amplitude * np.exp(
        -(dx ** 2 + dy ** 2) / (2 * sigma_dust ** 2)
    )
    return np.clip(vignette - spot, 0.0, 1.0)


def _synthesize_pixels(
    image_type: str,
    *,
    exposure: float,
    B: np.ndarray,
    D: np.ndarray,
    F: np.ndarray,
    illumination_e_per_sec: float | np.ndarray,
    read_noise_sigma_e: float,
    gain_e_per_adu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize one frame's pixels per the physics model. Returns
    uint16 array of shape (h, w).

    `illumination_e_per_sec` is the photon-source intensity. A scalar
    gives a uniform field; an ndarray models a per-pixel sky map
    (e.g. uniform sky + light-pollution gradient). Interpreted as flat
    illumination for Flat and as sky background for Light. Bias/Dark
    ignore it.
    """
    shape = B.shape
    if image_type == "Bias":
        signal_e = np.zeros(shape, dtype=np.float64)
    elif image_type == "Dark":
        signal_e = D * exposure
    elif image_type == "Flat":
        signal_e = F * (illumination_e_per_sec * exposure)
    elif image_type == "Light":
        # Sky × flat-field × exposure + thermal × exposure. When
        # `illumination_e_per_sec` is an ndarray the sky is per-pixel
        # (the gradient case); a scalar is a uniform field.
        signal_e = F * (illumination_e_per_sec * exposure) + D * exposure
    else:
        raise ValueError(f"unknown image_type {image_type!r}")

    signal_e_nonneg = np.clip(signal_e, 0.0, None)
    realized_e = rng.poisson(signal_e_nonneg).astype(np.float64)
    if read_noise_sigma_e > 0:
        realized_e += rng.normal(0.0, read_noise_sigma_e, shape)

    frame_adu = B + realized_e / gain_e_per_adu
    return np.clip(np.round(frame_adu), 0, 65535).astype(np.uint16)


def _render_stars_for_light_wcs(
    *,
    template_pixels: np.ndarray,
    template_dir: Path,
    prefix: str,
    ra: float, dec: float,
    width: int, height: int,
    instrument: str,
    filter_name: str,
    exposure_s: float,
    peak_adu: float,
) -> np.ndarray:
    """Render Gaia DR3 stars at the light frames' WCS.

    Writes one synthetic FITS frame with `template_pixels` as a stand-
    in (so `make_synthetic_frames` produces a file with a complete WCS
    header), runs `CatalogStarGenerator` on it via `pix.recipe`, reads
    the stars-only output back as a numpy array, and rescales the
    brightest star to `peak_adu`. Returned array is float64 in ADU
    units; callers add it directly to their light-frame pixel
    arrays before quantizing to uint16.
    """
    from pix.image import Image
    from pix.tasks.catalog_star_generator import (
        CatalogStarGenerator as RenderCatalogStars,
    )
    from file_io.generic import read_image
    from testing.test_frames import make_synthetic_frames

    template_paths = make_synthetic_frames(
        prefix=f"{prefix}_csg_template", number=1, image_type="Light",
        exposure=int(exposure_s), filter=filter_name,
        instrument=instrument, width_pixels=width, height_pixels=height,
        dest_dir=str(template_dir), ra=ra, dec=dec,
        pixel_arrays=[template_pixels], add_noise=False,
    )
    template_path = Path(template_paths[0])
    # `Image.save_to`'s recipe internally writes XISF; `copy_output`
    # is a plain `shutil.copy2` (see `pix.recipe`), so the destination
    # contains XISF bytes regardless of the extension we ask for.
    # Use `.xisf` so `read_image` dispatches to the XISF reader.
    stars_path = template_dir / f"{prefix}_csg_stars.xisf"

    image = Image.open(template_path, filter=filter_name)
    stars = RenderCatalogStars(image)
    stars.save_to(stars_path)

    star_image = read_image(stars_path).astype(np.float64)
    # CSG output is normalized flux (~[0,1]). Rescale so the brightest
    # pixel lands at `peak_adu`, leaving headroom on top of the bias
    # pedestal + sky background in our light-frame composition.
    star_max = float(star_image.max())
    if star_max > 0:
        star_image = star_image * (peak_adu / star_max)
    return star_image


def _write_master_fits(
    path: Path, data: np.ndarray, imagetyp: str,
    *, exposure: float, filter: str, instrument: str,
) -> None:
    """Write a master frame as float32 FITS with WBPP-readable headers.

    Stamps a CONTHASH (so the file has a stable identity) and a
    STBLHASH (AstroFlow convention). WBPP keys off IMAGETYP. The
    EXPOSURE on a Master Dark must match the lights for matched-
    exposure calibration (Case 2) to be applied without dark
    optimization. NAXIS1/NAXIS2 are set automatically by astropy
    from the data shape — setting them by hand corrupts the header
    card ordering and astropy refuses the write.
    """
    data_f32 = data.astype(np.float32)
    content_hash = xxhash.xxh64(data_f32.tobytes()).hexdigest()
    hdu = pyfits.PrimaryHDU(data=data_f32)
    hdu.header["IMAGETYP"] = imagetyp
    hdu.header["EXPOSURE"] = exposure
    if filter:
        hdu.header["FILTER"] = filter
    hdu.header["INSTRUME"] = instrument
    hdu.header["CONTHASH"] = content_hash
    hdu.header["STBLHASH"] = content_hash
    hdu.writeto(str(path), overwrite=True)


def synthesize_calibration_suite(
    *,
    output_dir: Path,
    width: int = 540,
    height: int = 360,
    light_exposure_s: float = 300.0,
    flat_exposure_s: float = 3.0,
    n_bias: int = 16,
    n_dark: int = 16,
    n_flat: int = 16,
    n_light: int = 16,
    rng_seed: int = 0xAB101,
    # Physics knobs (defaults chosen for an ASI2600-like CMOS at
    # unity gain, -10 °C).
    bias_mean_adu: float = 1000.0,
    bias_pattern_amplitude_adu: float = 5.0,
    dark_current_e_per_sec: float = 0.003,
    n_hot_pixels: int = 50,
    hot_pixel_rate_multiplier: float = 100.0,
    vignetting_corner_response: float = 0.7,
    dust_spot_amplitude: float = 0.15,
    sky_e_per_sec: float = 0.7,
    # Light-pollution gradient. The composite sky map per pixel is
    #   sky(x,y) = sky_e_per_sec
    #             + sky_gradient_x_e_per_sec_per_px * (x − width/2)
    #             + sky_gradient_y_e_per_sec_per_px * (y − height/2)
    # Models a city light dome from a particular direction. Setting
    # `sky_gradient_x_e_per_sec_per_px = sky_e_per_sec / (width/2)`
    # makes the +x edge twice as bright as the centre and zeroes the
    # −x edge — a strong wedge typical of suburban skies.
    sky_gradient_x_e_per_sec_per_px: float = 0.0,
    sky_gradient_y_e_per_sec_per_px: float = 0.0,
    flat_illumination_e_per_sec: float = 5000.0,
    read_noise_sigma_e: float = 1.5,
    gain_e_per_adu: float = 0.25,
    ra: float = 100.0,
    dec: float = 45.0,
    instrument: str = "ZWO ASI6200MM Pro",
    filter_name: str = "H",
    # Master-dark topology switch (see module docstring).
    pre_calibrate_dark: bool = False,
    # Star rendering (required for WBPP — SubframeSelector rejects
    # any frame it can't measure stars in, blocking the rest of the
    # pipeline). Stars are rendered once via CatalogStarGenerator at
    # the light frame's WCS, scaled to a sensible peak intensity,
    # and added to every light's pixel array before quantization.
    add_stars: bool = True,
    star_peak_adu: float = 30_000.0,
    # Naming.
    prefix: str = "calphys",
) -> CalibrationGroundTruth:
    """Generate a complete calibration suite + masters per the physics
    model. Returns ground-truth arrays + on-disk file paths for the
    test to stage and validate against."""
    rng = np.random.default_rng(rng_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-pixel ground truth (built ONCE, shared across all frames).
    B = _build_bias_pattern(
        width, height, bias_mean_adu, bias_pattern_amplitude_adu, rng,
    )
    D = _build_dark_current(
        width, height, dark_current_e_per_sec,
        n_hot_pixels, hot_pixel_rate_multiplier, rng,
    )
    F = _build_flat_field(
        width, height, vignetting_corner_response, dust_spot_amplitude,
    )

    # Helper to build N independent draws of one frame type.
    def _draw(image_type: str, exposure: float,
              illumination: float | np.ndarray,
              n: int) -> list[np.ndarray]:
        return [
            _synthesize_pixels(
                image_type,
                exposure=exposure, B=B, D=D, F=F,
                illumination_e_per_sec=illumination,
                read_noise_sigma_e=read_noise_sigma_e,
                gain_e_per_adu=gain_e_per_adu, rng=rng,
            )
            for _ in range(n)
        ]

    # Composite sky map: uniform floor + optional light-dome gradient.
    # Float scalar when there's no gradient (cheap), full ndarray when
    # the gradient parameters are non-zero.
    if (sky_gradient_x_e_per_sec_per_px != 0.0
            or sky_gradient_y_e_per_sec_per_px != 0.0):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        sky_map: float | np.ndarray = (
            sky_e_per_sec
            + sky_gradient_x_e_per_sec_per_px * (xx - width / 2.0)
            + sky_gradient_y_e_per_sec_per_px * (yy - height / 2.0)
        )
        sky_map = np.clip(sky_map, 0.0, None)  # no negative sky
    else:
        sky_map = sky_e_per_sec

    bias_arrays = _draw("Bias", 0.0, 0.0, n_bias)
    dark_arrays = _draw("Dark", light_exposure_s, 0.0, n_dark)
    flat_arrays = _draw(
        "Flat", flat_exposure_s, flat_illumination_e_per_sec, n_flat,
    )
    light_arrays = _draw(
        "Light", light_exposure_s, sky_map, n_light,
    )

    # Render real Gaia DR3 stars via CatalogStarGenerator and add
    # them to every light frame. Without this, WBPP's SubframeSelector
    # rejects every frame as unmeasurable (no stars → no FWHM/SNR →
    # frame discarded) and the integration step writes no master.
    # Stars are rendered ONCE per `synthesize_calibration_suite` call
    # using the first light frame's WCS as the template, then added
    # to every frame's pixel array in ADU space prior to uint16
    # quantization.
    star_array_adu: np.ndarray | None = None
    if add_stars:
        star_array_adu = _render_stars_for_light_wcs(
            template_pixels=light_arrays[0],
            template_dir=output_dir,
            prefix=prefix,
            ra=ra, dec=dec,
            width=width, height=height,
            instrument=instrument,
            filter_name=filter_name,
            exposure_s=light_exposure_s,
            peak_adu=star_peak_adu,
        )
        # Add stars to every physics-modelled light (pre-quantization
        # gymnastics: round trip through float64 to avoid uint16
        # overflow on bright stars).
        light_arrays = [
            np.clip(
                np.round(a.astype(np.float64) + star_array_adu),
                0, 65535,
            ).astype(np.uint16)
            for a in light_arrays
        ]

    # Write per-frame FITS via the existing test-frame writer. We
    # bypass its random-noise + hot-pixel paths by handing it our
    # physics-modelled arrays directly (added in Phase 0a).
    common = dict(
        instrument=instrument, width_pixels=width, height_pixels=height,
        dest_dir=str(output_dir), ra=ra, dec=dec, add_noise=False,
    )
    bias_files = make_synthetic_frames(
        prefix=f"{prefix}_bias", number=n_bias, image_type="Bias",
        exposure=0, pixel_arrays=bias_arrays, **common,
    )
    dark_files = make_synthetic_frames(
        prefix=f"{prefix}_dark", number=n_dark, image_type="Dark",
        exposure=int(light_exposure_s), pixel_arrays=dark_arrays, **common,
    )
    flat_files = make_synthetic_frames(
        prefix=f"{prefix}_flat", number=n_flat, image_type="Flat",
        exposure=int(flat_exposure_s), filter=filter_name,
        pixel_arrays=flat_arrays, **common,
    )
    light_files = make_synthetic_frames(
        prefix=f"{prefix}_light", number=n_light, image_type="Light",
        exposure=int(light_exposure_s), filter=filter_name,
        pixel_arrays=light_arrays, **common,
    )

    # Build masters in float64 to avoid quantization on integration.
    def _stack_mean(arrays: list[np.ndarray]) -> np.ndarray:
        return np.mean(
            np.stack([a.astype(np.float64) for a in arrays]), axis=0,
        )

    master_bias = _stack_mean(bias_arrays)

    # Master Dark: simple integration (§6.2 compliant) vs the named
    # anti-pattern (bias-subtracted before saving).
    master_dark_raw = _stack_mean(dark_arrays)
    if pre_calibrate_dark:
        master_dark = master_dark_raw - master_bias
    else:
        master_dark = master_dark_raw

    # Master Flat: each individual flat IS bias-calibrated (§6.3.1
    # CMOS path), then integrated. Result is bias-free, ready for
    # WBPP to normalize by its mean during light calibration.
    bias_calibrated_flats = [
        a.astype(np.float64) - master_bias for a in flat_arrays
    ]
    master_flat = np.mean(np.stack(bias_calibrated_flats), axis=0)

    master_bias_path = output_dir / f"{prefix}_master_bias.fits"
    master_dark_path = output_dir / f"{prefix}_master_dark.fits"
    master_flat_path = output_dir / f"{prefix}_master_flat.fits"

    _write_master_fits(
        master_bias_path, master_bias, "Master Bias",
        exposure=0, filter="", instrument=instrument,
    )
    _write_master_fits(
        master_dark_path, master_dark, "Master Dark",
        exposure=light_exposure_s, filter="", instrument=instrument,
    )
    _write_master_fits(
        master_flat_path, master_flat, "Master Flat",
        exposure=flat_exposure_s, filter=filter_name, instrument=instrument,
    )

    return CalibrationGroundTruth(
        B=B, D=D, F=F,
        I_sky_e_per_sec=sky_e_per_sec,
        I_flat_e_per_sec=flat_illumination_e_per_sec,
        read_noise_sigma_e=read_noise_sigma_e,
        gain_e_per_adu=gain_e_per_adu,
        light_exposure_s=light_exposure_s,
        flat_exposure_s=flat_exposure_s,
        bias_files=bias_files,
        dark_files=dark_files,
        flat_files=flat_files,
        light_files=light_files,
        master_bias_file=master_bias_path,
        master_dark_file=master_dark_path,
        master_flat_file=master_flat_path,
        is_master_dark_pre_calibrated=pre_calibrate_dark,
    )


def expected_calibrated_light_adu(truth: CalibrationGroundTruth) -> float:
    """Closed-form expectation for the calibrated light pixel value.

    Per Landmann §7.1 Case 2:
        Cal[x,y] = (light[x,y] − MD[x,y]) · s0 / MF[x,y]

    Substituting noise-free expectations and using
    `s0 = mean(F)·I_flat·t_f/gain`:
        Cal[x,y] ≈ I_sky · t_l · mean(F) / gain

    i.e. a uniform image (the per-pixel F factor cancels with the
    flat correction). The test asserts that the calibrated master
    closely matches this scalar within propagated noise.
    """
    return (
        truth.I_sky_e_per_sec * truth.light_exposure_s
        * float(np.mean(truth.F)) / truth.gain_e_per_adu
    )
