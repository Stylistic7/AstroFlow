from __future__ import annotations
from pathlib import Path
import numpy as np
from astropy.io import fits


def downscale_fits(in_path: str | Path,
                   out_path: str | Path,
                   *,
                   K: int,
                   overwrite: bool = True) -> None:
  """
    Downscale a 2D FITS primary image by a power-of-two factor K using block mean.

    K = 1  -> no-op
    K = 2  -> half resolution in each dimension
    K = 4  -> quarter resolution in each dimension
    etc.

    Policy:
      - output dtype: float32
      - always drops SIP + WCS-ish header keywords
      - updates basic geometry + binning/pixel-size keywords when present
    """
  if K < 1 or (K & (K - 1)) != 0:
    raise ValueError(f"K must be a power of two >= 1, got {K}")

  in_path = Path(in_path)
  out_path = Path(out_path)

  with fits.open(in_path, memmap=True) as hdul:
    hdu = hdul[0]
    hdr = hdu.header.copy()
    data = hdu.data
    if data is None or data.ndim != 2:
      raise ValueError("Expected 2D primary image in HDU[0]")

    h, w = data.shape
    if (h % K) or (w % K):
      raise ValueError(f"Image shape {data.shape} not divisible by K={K}")

    # Astropy applies BSCALE/BZERO on read; convert to float32 for compute + output.
    img = np.asarray(data, dtype=np.float32)

    if K == 1:
      y = img
    else:
      y = img.reshape(h // K, K, w // K, K).mean(axis=(1, 3)).astype(np.float32,
                                                                     copy=False)

    # --- header updates ---
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(w // K)
    hdr["NAXIS2"] = int(h // K)

    for k in ("XORGSUBF", "YORGSUBF"):
      if k in hdr:
        try:
          hdr[k] = float(hdr[k]) / K
        except Exception:
          pass

    for k in ("XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN"):
      if k in hdr:
        try:
          hdr[k] = int(round(float(hdr[k]) * K))
        except Exception:
          pass

    for k in ("XPIXSZ", "YPIXSZ"):
      if k in hdr:
        try:
          hdr[k] = float(hdr[k]) * K
        except Exception:
          pass

    if "IMAGEW" in hdr:
      hdr["IMAGEW"] = int(w // K)
    if "IMAGEH" in hdr:
      hdr["IMAGEH"] = int(h // K)

    # Always drop SIP keywords
    for k in list(hdr.keys()):
      if (k in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"} or
          k.startswith("A_") or k.startswith("B_") or k.startswith("AP_") or
          k.startswith("BP_")):
        del hdr[k]

    # Always drop WCS-ish keywords (and RA/DEC-like fields)
    wcs_prefixes = ("CRPIX", "CRVAL", "CD", "CDELT", "CTYPE", "CUNIT", "PC",
                    "PV", "LONPOLE", "LATPOLE")
    drop_exact = {
        "WCSAXES", "EQUINOX", "RADESYS", "RADECSYS", "RA", "DEC", "OBJCTRA",
        "OBJCTDEC"
    }
    for k in list(hdr.keys()):
      if k in drop_exact or any(k.startswith(p) for p in wcs_prefixes):
        del hdr[k]

    # Ensure no integer scaling keywords remain (we are writing float32)
    for k in ("BZERO", "BSCALE"):
      if k in hdr:
        del hdr[k]

    fits.PrimaryHDU(data=y, header=hdr).writeto(out_path, overwrite=overwrite)
