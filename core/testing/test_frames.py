import os
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
import astropy.io.fits as pyfits  # type: ignore
import numpy as np
from scipy.signal.windows import gaussian  # type: ignore
import datetime
import uuid
import shutil
import xxhash
from typing import Any
from pathlib import Path

from metadata.classes import ImageMetadata
from context import context
import plate_solving
from file_io.normalize_header import normalize_header
from file_io.sidecar import sidecar_path_for, get_section
from units import Degrees, Radians, ArcsecondsPerPixel
from hashing import hash_source_code, hash_call
from metadata.image_metadata import compute_image_metadata

_TESTDATA_TARGETS = {
    "M42": [
        "Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit",
        "Light_M42_180.0s_Bin1_2600MM_20231130-234248_0024.fit",
        "Light_M42_180.0s_Bin1_2600MM_20231130-234549_0025.fit",
    ],
    "NGC7635": [
        "Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213019_0004.fit",
        "Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213521_0005.fit",
        "Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-214053_0006.fit",
    ],
    "Orion Pane 1": [
        "Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit",
        "Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211007_0002.fit",
        "Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211510_0003.fit",
    ],
    "Orion Pane 2": [
        "Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220301_0001.fit",
        "Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220803_0002.fit",
        "Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-221305_0003.fit",
    ],
    "Orion Pane 3": [
        "Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230116_0001.fit",
        "Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230618_0002.fit",
        "Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-231120_0003.fit",
    ],
    "Orion Pane 4": [
        "Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001210_0001.fit",
        "Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001712_0002.fit",
        "Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-002215_0003.fit",
    ],
    "Sagittarius Pane 1": [
        "Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-000215_0001.fit",
        "Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-000717_0002.fit",
        "Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-001219_0003.fit",
    ],
    "Sagittarius Pane 2": [
        "Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-010605_0001.fit",
        "Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-011107_0002.fit",
        "Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-011609_0003.fit",
    ],
    "Sagittarius Pane 3": [
        "Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-022800_0001.fit",
        "Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-023301_0002.fit",
        "Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-023803_0003.fit",
    ],
}


def _add_dither(
    rng: np.random._generator.Generator,
    center: tuple[float, float],
    max_distance_deg: Degrees,
) -> tuple[float, float]:
    ra, dec = center
    distance = rng.uniform(0.0, max_distance_deg)
    angle = rng.uniform(0.0, 2.0 * np.pi)

    return ra + distance * np.sin(angle), dec + distance * np.cos(angle)


def _make_flat_frame(
    height_pixels: int, width_pixels: int, std: float | None = None
) -> np.ndarray:
    if std is None:
        std = width_pixels / 2.0

    gkern1d = gaussian(width_pixels, std=std).reshape(width_pixels, 1)
    img = np.outer(gkern1d, gkern1d)

    bottom = int((width_pixels - height_pixels) / 2)
    top = bottom + height_pixels
    img = img[bottom:top, 0:width_pixels]

    img *= 32768

    return img.astype(np.uint16)


def copy_testdata_frames(
    target_name: str,
    dest_dir: str,
    prefix: str = "",
    add_noise: bool = True,
    random_seed: int | None = None,
    fits_header_overrides: dict[str, Any] | None = None,
    start_datetime: datetime.datetime | None = None,
) -> None:
    if not fits_header_overrides:
        fits_header_overrides = {}

    random_seed = abs(xxhash.xxh64(str([random_seed, prefix])).intdigest())
    rng = np.random.default_rng(seed=random_seed)

    date_obs = start_datetime
    for filename in _TESTDATA_TARGETS[target_name]:
        src_filename = os.path.join("testdata", "input", filename)
        dest_filename = os.path.join(dest_dir, prefix + filename)
        shutil.copy(src_filename, dest_filename)
        print(f"### Copied filename {dest_filename}")
        fits = pyfits.open(dest_filename, mode="update", mmap=False)
        header = fits[0].header
        image = fits[0].data

        # Add a small amount of noise so it will have a unique hash.
        assert image.dtype == np.uint16
        if add_noise:
            image += rng.integers(
                low=0, high=2, size=image.shape, dtype=image.dtype
            )

        content_hash = xxhash.xxh64(image.tobytes()).hexdigest()
        header["CONTHASH"] = content_hash

        if date_obs:
            header["DATE-OBS"] = date_obs.strftime("%Y-%m-%dT%H:%M:%S")
            date_obs += datetime.timedelta(seconds=int(header["EXPOSURE"]) + 1)

        fits.flush()
        fits.close()

        src_sidecar = sidecar_path_for(Path(src_filename))
        assert src_sidecar.exists(), (
            f"Testdata sidecar missing: {src_sidecar}. "
            f"Run scripts/plate_solve_testdata.py to generate it."
        )
        assert get_section(Path(src_filename), "plate_solution"), (
            f"Testdata sidecar has no plate solution: {src_sidecar}. "
            f"Run scripts/plate_solve_testdata.py to regenerate it."
        )
        dest_sidecar = sidecar_path_for(Path(dest_filename))
        shutil.copy(src_sidecar, dest_sidecar)

        # Update sidecar to reflect any FITS header overrides (DATE-OBS,
        # content hash) so the sidecar short-circuit returns correct metadata.
        from file_io.sidecar import set_section
        im_obj = get_section(Path(dest_filename), "image_metadata")
        if im_obj is not None:
            im_obj["content_hash"] = content_hash
            if date_obs is not None:
                im_obj["header"]["DATE-OBS"] = header["DATE-OBS"]
            set_section(Path(dest_filename), "image_metadata", im_obj)

        print(f"### Copied sidecar {dest_sidecar}")


def make_synthetic_frames(
    start_datetime: datetime.datetime = datetime.datetime(
        2023, 12, 6, 16, 29, 43
    ),
    number: int = 1,
    image_type: str = "Light",
    prefix: str | None = None,
    ids: list[str] | None = None,
    content_hashes: list[str] | None = None,
    dest_dir: str | None = None,
    ra: float = 100,
    dec: float = 45,
    centers: list[tuple[float, float]] | None = None,
    dither: float = 0.01,
    focal_length_m: float = 0.35,
    pixel_pitch_um: float = 3.76,
    instrument: str = "ZWO ASI6200MM Pro",
    width_pixels: int = 1080,
    height_pixels: int = 720,
    temperature: float = -20,
    exposure: int = 300,
    filter: str = "H",
    fill_value: Any | None = None,
    add_noise: bool = True,
    random_seed: int | None = None,
    num_stars: int = 100,
    image_scale: float = 1.0,
    rotation_angle_deg: float = 90.0,
    pixel_arrays: list[np.ndarray] | None = None,
    solvable: bool = False,
) -> list[str]:
    """Write `number` synthetic FITS frames.

    When `pixel_arrays` is provided, each frame's uint16 pixel data
    comes from `pixel_arrays[i]` and the internal fill_value / random
    / Flat-Gaussian / `add_noise` / hot-pixel paths are bypassed.
    Caller is responsible for everything in the array. This is the
    path the calibration-physics harness uses to supply
    `B + signal + Poisson + read_noise`-modelled pixels.

    When `pixel_arrays` is None (default), the historical fill_value /
    random / Flat-Gaussian generator + `add_noise` + hot-pixel
    injection runs unchanged, so all existing callers continue to
    work bit-identically.
    """
    if prefix is None:
        prefix = image_type + "_" + str(uuid.uuid4())

    number = len(ids) if ids else number

    ids = ids if ids else [f"{prefix}_{i:05}.fits" for i in range(number)]

    random_seed = abs(
        xxhash.xxh64(str([random_seed, image_type, prefix])).intdigest()
    )
    rng = np.random.default_rng(seed=random_seed)

    output_filenames = []
    for i in range(number):
        fits_filename = ids[i]
        if dest_dir is not None:
            fits_filename = os.path.join(dest_dir, fits_filename)

        if pixel_arrays is not None:
            image_data = pixel_arrays[i]
            assert image_data.dtype == np.uint16, (
                f"pixel_arrays[{i}].dtype must be uint16, got {image_data.dtype}"
            )
            assert image_data.shape == (height_pixels, width_pixels), (
                f"pixel_arrays[{i}].shape {image_data.shape} != "
                f"({height_pixels}, {width_pixels})"
            )
        elif image_type == "Flat":
            image_data = _make_flat_frame(height_pixels, width_pixels)
        elif fill_value:
            image_data = (
                np.ones((height_pixels, width_pixels), dtype=np.uint16)
                * fill_value
            )
        else:
            image_data = rng.integers(
                100, size=(height_pixels, width_pixels), dtype=np.uint16
            )

        # Add a small amount of noise so it will have a unique hash.
        # Skip when pixel_arrays is provided — caller's array IS the
        # final pixel data, and adding stochastic noise here would
        # break the physics-model determinism the harness depends on.
        if pixel_arrays is None and add_noise:
            assert image_data.dtype == np.uint16
            image_data += rng.integers(
                low=0, high=2, size=image_data.shape, dtype=image_data.dtype
            )

        # Inject deterministic hot pixels so the defect-fingerprint
        # extractor produces overlapping, stable defect sets across all
        # frames from this "camera" (= instrument + sensor shape). Without
        # this, fingerprint clustering would see only random noise outliers
        # and exclude every synthetic frame as no-fingerprint.
        # Skip when pixel_arrays is provided — caller bakes hot pixels
        # into the array via the dark-current ndarray if they want them
        # (and the physics harness does, via the same `_hot_seed` shape).
        if pixel_arrays is None and image_type != "Bias":
            _hot_seed = xxhash.xxh64(
                f"{instrument}|{width_pixels}x{height_pixels}"
            ).intdigest()
            _hot_rng = np.random.default_rng(_hot_seed)
            n_hot = 1200
            xs = _hot_rng.integers(0, width_pixels, size=n_hot)
            ys = _hot_rng.integers(0, height_pixels, size=n_hot)
            image_data[ys, xs] = np.iinfo(np.uint16).max

        if content_hashes is not None:
            content_hash = content_hashes[i]
        else:
            content_hash = xxhash.xxh64(image_data.tobytes()).hexdigest()

        centers_or = centers if centers else [(ra, dec)] * number
        center = _add_dither(rng, centers_or[i], dither)
        date_obs = start_datetime + i * datetime.timedelta(seconds=exposure + 1)
        deg_per_pix = image_scale / 3600.0

        fits_header = {
            "SIMPLE": True,
            "BITPIX": 16,
            "CONTHASH": content_hash,
            "DATE-OBS": date_obs.strftime("%Y-%m-%dT%H:%M:%S"),
            "RA": center[0],
            "DEC": center[1],
            "IMAGETYP": image_type,
            "INSTRUME": instrument,
            "CCD-TEMP": temperature,
            "EXPOSURE": exposure,
            "CTYPE1": "RA---TAN-SIP",
            "CTYPE2": "DEC--TAN-SIP",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": center[0],
            "CRVAL2": center[1],
            "CRPIX1": width_pixels / 2 + 0.5,
            "CRPIX2": height_pixels / 2 + 0.5,
            "CDELT1": deg_per_pix,
            "CDELT2": deg_per_pix,
            "CD1_1": deg_per_pix * np.cos(rotation_angle_deg),
            "CD1_2": -deg_per_pix * np.sin(rotation_angle_deg),
            "CD2_1": deg_per_pix * np.sin(rotation_angle_deg),
            "CD2_2": deg_per_pix * np.cos(rotation_angle_deg),
            "WCSAXES": 2,
            "NAXIS": 2,
            "NAXIS1": width_pixels,
            "NAXIS2": height_pixels,
            "IMAGEW": width_pixels,
            "IMAGEH": height_pixels,
            "FOCALLEN": focal_length_m,
            "XPIXSZ": pixel_pitch_um,
            "YPIXSZ": pixel_pitch_um,
            # Synthetic frames contain only noise (or a flat gradient) — mark
            # them so solve-field is skipped if anything ever tries to run it.
            "DONTSOLV": True,
        }

        if image_type.lower() in ["light", "flat"]:
            fits_header["FILTER"] = filter

        if solvable:
            # Treat this as a real input frame: drop the opt-out so the
            # pipeline runs solve-field on it. The WCS keywords stay as a
            # seed for solve-field (then it blind-solves from the stars).
            fits_header.pop("DONTSOLV", None)

        header = normalize_header(fits_header)

        if image_type == "Light" and not solvable:
            plate_solution = plate_solving.PlateSolution(
                image_filename=fits_filename,
                header=fits_header,
                num_stars=num_stars,
                center=(Degrees(center[0]), Degrees(center[1])),
                image_scale=ArcsecondsPerPixel(image_scale),
                rotation_angle=Radians(np.radians(rotation_angle_deg)),
            )
        else:
            plate_solution = None

        # Extract defects from the synthetic image so the cached metadata
        # mirrors what the production pipeline would have produced.
        # Without this, no defects would be recorded for the synthetic
        # frames and every synthetic light would be excluded by
        # `build_fingerprint_map`. defects are now lifted off
        # `ImageMetadata` into a separate `defects_ref`; we write them
        # to the sidecar's `defect_candidates` section below.
        from calibration.sensor_fingerprint import extract_defects
        defect_candidates = extract_defects(image_data) if image_type != "Bias" else None

        metadata = ImageMetadata(
            content_hash=content_hash,
            stable_hash=content_hash,
            filename=fits_filename,
            header=header,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            channels=1,
            dtype=np.dtype(np.uint16),
            plate_solution=plate_solution,
        )

        fn_hash = hash_source_code(compute_image_metadata)
        call_hash = hash_call(fn_hash, Path(fits_filename))
        key = ("compute_image_metadata", call_hash)
        if not solvable:
            # Pre-seed the metadata cache only for non-solvable frames.
            # A solvable frame must flow through the real
            # compute_image_metadata + solve-field path like a real image.
            context.set(key, metadata)

        if dest_dir is None:
            continue

        hdu = pyfits.PrimaryHDU(image_data)
        for k, v in fits_header.items():
            hdu.header[k] = v
        hdu.writeto(fits_filename)
        print(
            f"### Wrote FITS filename {fits_filename} with content hash {content_hash}"
        )

        # Write a sidecar so compute_image_metadata can short-circuit
        # without reading the FITS file or invoking solve-field. Skipped
        # for solvable frames — they must look exactly like a fresh real
        # input file (no sidecar), so the pipeline solves them itself.
        if solvable:
            output_filenames.append(fits_filename)
            continue
        from metadata.sidecar_io import (
            write_image_metadata, write_defect_candidates,
        )
        sidecar_metadata = ImageMetadata(
            content_hash=content_hash,
            stable_hash=content_hash,
            filename=Path(fits_filename),
            header=header,
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            channels=1,
            dtype=np.dtype(np.uint16),
            plate_solution=plate_solution,
        )
        write_image_metadata(sidecar_metadata)
        write_defect_candidates(Path(fits_filename), defect_candidates)
        if plate_solution is not None:
            from file_io.sidecar import set_section
            set_section(
                Path(fits_filename), "plate_solution",
                plate_solving._plate_solution_to_obj(plate_solution),
            )

        output_filenames.append(fits_filename)

    return output_filenames
