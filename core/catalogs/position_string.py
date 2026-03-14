from collections.abc import Sequence

import catalogs.ngc2000_catalog as ngc2000_catalog
from beartype import beartype


@beartype
def ra_dec_string(ra: float, dec: float) -> str:
    def round_to_nearest_tenth(number: float) -> float:
        return int(round(number * 10.0)) / 10.0

    ra = round_to_nearest_tenth(ra)
    dec = round_to_nearest_tenth(dec)

    return f"RA_{ra:.2f}_DEC_{dec:.2f}"


@beartype
def canonical_target_string(position: Sequence[float]) -> str:
    ra, dec = position

    _, entry = ngc2000_catalog.find_best_object(ra, dec)
    if entry and entry["name"]:
        return entry["name"]
    else:
        return ra_dec_string(ra, dec)
