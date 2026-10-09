from pathlib import Path
import numpy as np
from astropy.io import fits


def downscale_fits(
    in_path: str | Path, out_path: str | Path, *, K: int, overwrite: bool = True
) -> None:
    """
    Power-of-two downscale that mimics *in-camera binning* for WBPP:

    - Reads the frame in the camera's physical uint16 ADU domain (Astropy applies BZERO/BSCALE).
    - Downscales by block-mean (so output stays in the same ADU range as the input).
    - Writes a FITS that looks like a native camera frame:
        * stored dtype: int16
        * BSCALE=1, BZERO=32768  (unsigned16 convention)
    - Drops SIP/WCS by construction (minimal safe header copy).
    """
    if K <= 1 or (K & (K - 1)) != 0:
        raise ValueError(f"K must be a power of two >= 1, got {K}")

    in_path = Path(in_path)
    out_path = Path(out_path)

    with fits.open(in_path) as hdul:
        src = hdul[0]
        h0 = src.header
        data = src.data

    if data is None or data.ndim != 2:
        raise ValueError("Expected 2D primary image in HDU[0]")

    h, w = data.shape

    # Trim edges so dimensions are divisible by K (at most K-1 pixels lost).
    h_trim = h - (h % K)
    w_trim = w - (w % K)
    data = data[:h_trim, :w_trim]
    h, w = data.shape

    # Astropy applies BZERO/BSCALE and unsigned16 convention automatically:
    # img is in "physical ADU" space, typically [0..65535] for these ZWO frames.
    img = np.asarray(data, dtype=np.float32)
    assert np.isfinite(img).all()
    assert img.min() >= -1e-3  # should be >= 0 in practice
    assert img.max() <= 65535 + 1e-3

    y = (
        img.reshape(h // K, K, w // K, K)
        .mean(axis=(1, 3))
        .astype(np.float32, copy=False)
    )

    # Range invariants for mean downscale in ADU space
    assert np.isfinite(y).all()
    assert y.min() >= img.min() - 1e-3
    assert y.max() <= img.max() + 1e-3

    # Clamp to legal uint16 physical range, then encode as FITS unsigned16 convention:
    # stored int16 with BZERO=32768, BSCALE=1
    y = np.clip(y, 0.0, 65535.0, out=y)
    q = np.rint(y - 32768.0).astype(np.int32, copy=False)
    q = np.clip(q, -32768, 32767).astype(np.int16, copy=False)

    # Fresh primary HDU to avoid header/data mismatches.
    dst = fits.PrimaryHDU(data=q)
    hdr = dst.header
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 32768.0

    # Copy only safe metadata (no structural/scaling/WCS/SIP families).
    keep = (
        "CREATOR",
        "INSTRUME",
        "GUIDECAM",
        "TELESCOP",
        "RIG",
        "IMAGETYP",
        "FILTER",
        "EXPOSURE",
        "EXPTIME",
        "DATE-OBS",
        "DATE-END",
        "TIMESYS",
        "GAIN",
        "OFFSET",
        "EGAIN",
        "CCD-TEMP",
        "SET-TEMP",
        "FOCALLEN",
        "FOCUSPOS",
        "DONTSOLV",
    )
    for k in keep:
        if k in h0:
            hdr[k] = h0[k]

    # Update camera-geometry metadata to match "binned" appearance.
    for k in ("XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN"):
        if k in h0:
            try:
                hdr[k] = int(round(float(h0[k]) * K))
            except (KeyError, ValueError, TypeError):
                pass

    for k in ("XPIXSZ", "YPIXSZ"):
        if k in h0:
            try:
                hdr[k] = float(h0[k]) * K
            except (KeyError, ValueError, TypeError):
                pass

    for k in ("XORGSUBF", "YORGSUBF"):
        if k in h0:
            try:
                hdr[k] = float(h0[k]) / K
            except (KeyError, ValueError, TypeError):
                pass

    dst.writeto(out_path, overwrite=overwrite, output_verify="exception")
