from dateutil.parser import parse


def normalize_header(header: dict) -> dict:
    output = dict()
    for k, v in header.items():
        if k in ["COMMENT", "HISTORY"]:
            continue

        if k == "IMAGETYP":
            image_type = None
            # and may inherit their IMAGETYP wording, and the specific type is
            # the one that matters.
            for t in ["bias", "dark", "flat", "light"]:
                if t in v.lower():
                    image_type = t

            if image_type is None:
                raise ValueError(f"Unknown IMAGETYP {v}")

            if "master" in v.lower():
                output[k] = f"master {image_type}"
            else:
                output[k] = image_type

            continue

        # Try to coerce the type to a date.
        if "DATE" in k:
            try:
                output[k] = parse(v + " UTC")
                continue
            except Exception as e:
                output[k] = v

        # Leave it as a string.
        output[k] = v

    # FITS 4.0 spellings (survey products, e.g. MAST's HLSPs): XPOSURE is
    # the exposure, DATE-BEG the start of the observation.
    if "EXPTIME" not in output and "XPOSURE" in output:
        output["EXPTIME"] = float(output["XPOSURE"])
    if "DATE-OBS" not in output and "DATE-BEG" in output:
        output["DATE-OBS"] = output["DATE-BEG"]

    if "EXPTIME" in output and "EXPOSURE" not in output:
        output["EXPOSURE"] = float(output["EXPTIME"])

    if "FILTER" not in output:
        output["FILTER"] = "none"

    return output
