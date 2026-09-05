import re
import numpy as np
from functools import cache
from pathlib import Path
from typing import Any

from catalogs.messier import read_messier_aliases, read_messier_entries
from catalogs.vizier_catalogs import read_vizier_catalogs
from geometry.distance import distance
from units import Degrees
from beartype import beartype

_CATALOG_DIR = Path(__file__).parent


def _normalize_ngc2000_id(raw_id: str) -> str:
    """Normalize an ngc2000.dat ID to a canonical form like 'NGC40' or 'IC348'."""
    raw_id = raw_id.strip().replace(" ", "")
    if raw_id.startswith("I"):
        return "IC" + raw_id[1:]
    return "NGC" + raw_id


@beartype
def _read_names_catalog() -> dict[str, str]:
    """Parse the Stellarium names catalog.

    Returns a dict mapping normalized catalog ID (e.g. 'NGC40', 'IC348')
    to the preferred common name. The first name listed for each object is
    used as the preferred name.
    """
    names: dict[str, str] = {}
    name_pattern = re.compile(r'_\("(.+?)"\)')

    with open(_CATALOG_DIR / "stellarium_names.dat") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            prefix = line[0:5].strip()
            obj_id = line[5:20].strip().lstrip("0") or "0"
            match = name_pattern.search(line[20:])
            if not match:
                continue

            name = match.group(1)
            key = prefix + obj_id

            # First name listed is the preferred name.
            if key not in names:
                names[key] = name

    return names


@beartype
def _read_ngc2000_catalog(names: dict[str, str]) -> list[dict[str, Any]]:
    entries = []
    with open(_CATALOG_DIR / "ngc2000.dat") as f:
        for line in f.readlines():
            entry: dict[str, Any] = {}
            entry["id"] = line[0:5].strip()
            entry["type"] = line[6:9].strip()
            entry["ra_string"] = line[10:17].strip()
            hours, minutes = entry["ra_string"].split(" ")
            entry["ra_degrees"] = Degrees(15.0 * (float(hours) + float(minutes) / 60.0))
            entry["dec_string"] = line[18:25].strip()
            whole, minutes = entry["dec_string"].split(" ")
            # The sign applies to the whole angle: "-05 23" is -(5 + 23/60).
            magnitude = abs(float(whole)) + float(minutes) / 60.0
            entry["dec_degrees"] = Degrees(
                -magnitude if whole.startswith("-") else magnitude
            )
            entry["reference"] = line[26:27].strip()
            entry["constellation"] = line[28:32].strip()

            try:
                entry["radius"] = Degrees(float(line[32:38].strip()) / 60.0 / 2.0)
            except:
                entry["radius"] = Degrees(0.0)

            try:
                entry["magnitude"] = float(line[39:44].strip())
            except:
                entry["magnitude"] = 0.0

            entry["p"] = line[44:45].strip()
            entry["comment"] = line[46:].strip()

            canonical_id = _normalize_ngc2000_id(entry["id"])
            entry["designation"] = canonical_id
            entry["named"] = canonical_id in names
            entry["name"] = names.get(canonical_id, canonical_id)

            entries.append(entry)

    return entries


@cache
def _all_entries() -> list[dict[str, Any]]:
    """All catalog entries (NGC/IC + the VizieR catalogs + NGC-less Messier
    objects), sorted by descending radius. Cached: pure function of the
    static catalog files."""
    names = _read_names_catalog()
    entries = (
        _read_ngc2000_catalog(names)
        + read_vizier_catalogs(names)
        + read_messier_entries()
    )
    return sorted(entries, key=lambda x: x["radius"], reverse=True)


@cache
def designation_pairs() -> list[tuple[str, str]]:
    """(designation, canonical name) for every designation the user might
    type — every entry's own designation plus Messier aliases. The payload
    behind /api/catalog/entries; lets search resolve "SH2 132" or "M31" to
    the name objects are actually stored under."""
    aliases = read_messier_aliases()
    pairs = []
    for entry in _all_entries():
        pairs.append((entry["designation"], entry["name"]))
        messier = aliases.get(entry["designation"])
        if messier:
            pairs.append((messier, entry["name"]))
    return pairs


DistanceAndEntry = tuple[Degrees, dict[str, Any] | None]


# TODO: This return type is terrible.
@beartype
def _find_best_object_by_size(ra: Degrees, dec: Degrees) -> DistanceAndEntry:
    entries = _all_entries()

    # A candidate must lie within 2 normalized radii of the object (distance /
    # radius) OR within an absolute half-degree of it: without knowing the
    # frame's field of view, the relative test alone strips small objects (a
    # 12-arcmin galaxy would lose its name to a 0.2 degree framing offset).
    # Objects without sizes are excluded (the by-distance fallback covers them).
    #
    # Candidates are then ranked, best first:
    #   1. curated common name beats designation-only: a frame near both a
    #      famous object and an anonymous catalog region should be called what
    #      a person would call it;
    #   2. an object CONTAINING the position beats a merely nearby one;
    #   3. among containers, the smallest (most specific) wins — otherwise
    #      giants like Barnard's Loop (r=10 deg) swallow everything inside
    #      them; among non-containers, smallest normalized radius wins.
    best_key = None
    best_distance, best_entry = Degrees(np.inf), None
    for entry in entries:
        if not entry["radius"]:
            continue
        d = distance(ra, dec, entry["ra_degrees"], entry["dec_degrees"])
        normalized_radius = d / entry["radius"]
        if normalized_radius >= 2.0 and d >= 0.5:
            continue

        inside = d < entry["radius"]
        key = (
            not entry["named"],
            not inside,
            entry["radius"] if inside else normalized_radius,
        )
        if best_key is None or key < best_key:
            best_key, best_distance, best_entry = key, d, entry

    return best_distance, best_entry


@beartype
def _find_best_object_by_distance(ra: Degrees, dec: Degrees) -> DistanceAndEntry:
    entries = _all_entries()

    best_distance = Degrees(np.inf)
    best_entry = None
    for entry in entries:
        d = distance(ra, dec, entry["ra_degrees"], entry["dec_degrees"])
        if d < best_distance:
            best_distance = d
            best_entry = entry

    return best_distance, best_entry


@beartype
def find_best_object(ra: Degrees, dec: Degrees) -> DistanceAndEntry:
    distance, entry = _find_best_object_by_size(ra, dec)
    if entry:
        return distance, entry

    return _find_best_object_by_distance(ra, dec)


if __name__ == "__main__":
    d = distance(ra_1=Degrees(0), dec_1=Degrees(0), ra_2=Degrees(360), dec_2=Degrees(0))
    print(f"distance is {d}")

    best_distance, best_entry = find_best_object(ra=Degrees(325.4), dec=Degrees(58.2))
    print("best_distance", best_distance)
    print("best_entry", best_entry)
