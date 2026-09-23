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
    3. `PROGRAM` mentions a known stacking tool AND the frame is not an
       explicit light. PixInsight/Siril/etc. stamp PROGRAM whenever their
       FITS *writer* emits a frame — but that includes merely re-saving a
       raw light sub during a culling / format-conversion pass. Remote-
       imaging services (e.g. Starbase) deliver PixInsight-culled int16
       light subs stamped `PROGRAM='PixInsight'` that are pixel-untouched
       raw ADU, not integrations. A genuine integrated master is float32
       and is therefore already caught by signal 2, so the PROGRAM signal
       is scoped away from lights — where it can only misfire — while
       still catching a bias/dark/flat master left with a plain IMAGETYP.

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
    if isinstance(program, str) and not is_light_frame(header):
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


def is_fringe_frame(header: dict) -> bool:
    """A fringe frame — AstroFlow's own `Master Fringe`, or one a user has
    exported and dropped back into a calibration folder. It is only ever a
    MASTER: nothing is shot as a fringe frame, it is combined from blank-sky
    lights (`stacks/fringe.py`)."""
    if not header:
        return False
    if "IMAGETYP" not in header:
        return False
    return "fringe" in header["IMAGETYP"].lower()


def is_frame_type(header: dict, type: str) -> bool:
    if type == "light":
        return is_light_frame(header)
    elif type == "dark":
        return is_dark_frame(header)
    elif type == "bias":
        return is_bias_frame(header)
    elif type == "flat":
        return is_flat_frame(header)
    elif type == "fringe":
        return is_fringe_frame(header)

    raise ValueError(f"Unknown image type {type}")
