from dateutil.parser import parse


def normalize_header(header: dict) -> dict:
    output = dict()
    for k, v in header.items():
        if k in ["COMMENT", "HISTORY"]:
            continue

        if k == "IMAGETYP":
            image_type = None
            # "fringe" before "light": a fringe frame is BUILT from lights
            # and may inherit their IMAGETYP wording, and the specific type is
            # the one that matters (see stacks/fringe.py).
            for t in ["bias", "dark", "flat", "fringe", "light"]:
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

    if "EXPTIME" in output and "EXPOSURE" not in output:
        output["EXPOSURE"] = float(output["EXPTIME"])

    if "FILTER" not in output:
        output["FILTER"] = "none"

    return output
