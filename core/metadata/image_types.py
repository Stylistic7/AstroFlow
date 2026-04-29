_KNOWN_STACKING_TOOLS = (
    "pixinsight",
    "siril",
    "deepskystacker",
    "astap",
    "astropixelprocessor",
    "autostakkert",
    "astroflow",        # AstroFlow's own outputs, if they ever leak in
)


def is_master(header: dict) -> bool:
    """Heuristic: does this FITS header describe a stacked / processed image
    rather than a raw camera sub?

    Recognized signals (any one is enough):

    1. `IMAGETYP` contains "master" (case-insensitive). The cleanly-tagged
       case — "Master Bias", "Master Dark", etc.
    2. `BITPIX <= -32` — the frame is stored as float32 or float64. Raw
       camera output is virtually always int16 / uint16 (BITPIX=16), so a
       float-typed FITS is essentially the calling card of a post-processed
       integration even when the user leaves IMAGETYP as a plain
       "Bias Frame" / "Dark Frame".
    3. `PROGRAM` mentions a known stacking tool. PixInsight, Siril,
       DeepSkyStacker, ASTAP, etc. all stamp their name into PROGRAM
       when their FITS writer emits a processed frame; capture software
       (NINA, SharpCap, ASIAir, INDI, ...) does not.

    These signals catch externally pre-stacked masters that slip past
    filename-based filtering (`find_image_files` excludes basenames
    starting with `master*`, but doesn't help when the user named the
    file `Bias_calibrated.fit` or similar).
    """
    if not header:
        return False

    image_type = header.get("IMAGETYP")
    if isinstance(image_type, str) and "master" in image_type.lower():
        return True

    bitpix = header.get("BITPIX")
    if bitpix is not None:
        try:
            if int(bitpix) <= -32:
                return True
        except (TypeError, ValueError):
            pass

    program = header.get("PROGRAM")
    if isinstance(program, str):
        prog_lower = program.lower()
        if any(tool in prog_lower for tool in _KNOWN_STACKING_TOOLS):
            return True

    return False


def is_light_frame(header: dict) -> bool:
    if not header:
        return False
    if "IMAGETYP" not in header:
        return False
    image_type = header["IMAGETYP"].lower()
    return "light" in image_type


def is_bias_frame(header: dict) -> bool:
    if not header:
        return False
    if "IMAGETYP" not in header:
        return False
    image_type = header["IMAGETYP"].lower()
    return "bias" in image_type


def is_dark_frame(header: dict) -> bool:
    if not header:
        return False
    if "IMAGETYP" not in header:
        return False
    image_type = header["IMAGETYP"].lower()
    return "dark" in image_type


def is_flat_frame(header: dict) -> bool:
    if not header:
        return False
    if "IMAGETYP" not in header:
        return False
    image_type = header["IMAGETYP"].lower()
    return "flat" in image_type


def is_frame_type(header: dict, type: str) -> bool:
    if type == "light":
        return is_light_frame(header)
    elif type == "dark":
        return is_dark_frame(header)
    elif type == "bias":
        return is_bias_frame(header)
    elif type == "flat":
        return is_flat_frame(header)

    raise ValueError(f"Unknown image type {type}")
