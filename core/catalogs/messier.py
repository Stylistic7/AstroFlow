"""Messier designations, from catalogs/messier.tsv (see its header for
provenance: OpenNGC cross-reference + two hand-added rows).

Most Messier objects are NGC/IC entries already in the catalog, so they
contribute ALIASES ("M31" resolves to NGC224's name). The handful with no
NGC/IC identity (M40, M45) become real entries of their own — this is how
the Pleiades field gets named "Pleiades" at all.
"""

import csv
from pathlib import Path
from typing import Any

from beartype import beartype

from units import Degrees

_CATALOG_DIR = Path(__file__).parent


@beartype
def _read_rows() -> list[dict[str, str]]:
    with open(_CATALOG_DIR / "messier.tsv") as f:
        return list(csv.DictReader(
            (line for line in f if not line.startswith("#")), delimiter="\t"
        ))


@beartype
def read_messier_entries() -> list[dict[str, Any]]:
    """Catalog entries for Messier objects with no NGC/IC identity."""
    return [
        {
            "id": row["M"],
            "designation": row["M"],
            "ra_degrees": Degrees(float(row["ra_degrees"])),
            "dec_degrees": Degrees(float(row["dec_degrees"])),
            "radius": Degrees(float(row["radius_degrees"])),
            "named": bool(row["name"]),
            "name": row["name"] or row["M"],
        }
        for row in _read_rows()
        if not row["id"]
    ]


@beartype
def read_messier_aliases() -> dict[str, str]:
    """NGC/IC designation -> Messier designation, e.g. "NGC224" -> "M31"."""
    return {row["id"]: row["M"] for row in _read_rows() if row["id"]}
