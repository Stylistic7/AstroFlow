"""Deep-sky catalogs fetched from VizieR (CDS, Strasbourg).

The .tsv source files in this directory were retrieved 2026-09-05 from the
VizieR ASU endpoint (https://vizier.cds.unistra.fr/viz-bin/asu-tsv) with
computed J2000 positions (-out.add=_RAJ2000,_DEJ2000); each file carries its
full query URL and bibliographic provenance in its # header.

  sharpless_vii20.tsv   VII/20    Sharpless 1959 HII regions        Diam (arcmin)
  lbn_vii9.tsv          VII/9     Lynds 1965 Bright Nebulae         Diam1/Diam2 (arcmin)
  ldn_vii7a.tsv         VII/7A    Lynds 1962 Dark Nebulae           Area (deg^2)
  barnard_vii220a.tsv   VII/220A  Barnard 1927 dark objects         Diam (arcmin)
  vdb_vii21.tsv         VII/21    van den Bergh 1966 reflection     BRadMax/RRadMax (arcmin, radii)

Entries are normalized to the shape find_best_object needs: id, ra_degrees,
dec_degrees, radius (degrees), name. Objects without a listed size get
radius 0.0 and participate only in the nearest-object fallback, matching how
ngc2000 entries without sizes behave.
"""

import math
from pathlib import Path
from typing import Any, Callable

from beartype import beartype

from units import Degrees

_CATALOG_DIR = Path(__file__).parent


def _float(cell: str) -> float:
    return float(cell) if cell.strip() else 0.0


# filename -> (row -> (display_id, stellarium name key, radius in degrees)).
# Name keys follow _read_names_catalog: 5-char prefix + zero-stripped number.
_SPECS: list[tuple[str, Callable[[dict[str, str]], tuple[str, str, float]]]] = [
    (
        "sharpless_vii20.tsv",
        lambda r: (f"Sh2-{r['Sh2']}", "SH2" + r["Sh2"], _float(r["Diam"]) / 120.0),
    ),
    (
        "lbn_vii9.tsv",
        lambda r: (
            f"LBN {r['Seq']}",
            "LBN" + r["Seq"],
            max(_float(r["Diam1"]), _float(r["Diam2"])) / 120.0,
        ),
    ),
    (
        "ldn_vii7a.tsv",
        lambda r: (
            f"LDN {r['LDN']}",
            "LDN" + r["LDN"],
            math.sqrt(_float(r["Area"]) / math.pi),
        ),
    ),
    (
        "barnard_vii220a.tsv",
        lambda r: (f"B{r['Barn']}", "B" + r["Barn"], _float(r["Diam"]) / 120.0),
    ),
    (
        "vdb_vii21.tsv",
        lambda r: (
            f"vdB {r['VdB']}",
            "VDB" + r["VdB"],
            max(_float(r["BRadMax"]), _float(r["RRadMax"])) / 60.0,
        ),
    ),
]


@beartype
def _read_tsv(filename: str) -> list[dict[str, str]]:
    """Parse a VizieR ASU tab-separated file: # comments, then a header line,
    a units line, a dashes line, and data rows."""
    with open(_CATALOG_DIR / filename) as f:
        lines = [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    header = lines[0].split("\t")
    assert lines[2].startswith("-"), f"unexpected {filename} structure: {lines[2]!r}"
    return [
        {key: cell.strip() for key, cell in zip(header, line.split("\t"))}
        for line in lines[3:]
    ]


@beartype
def read_vizier_catalogs(names: dict[str, str]) -> list[dict[str, Any]]:
    entries = []
    for filename, extract in _SPECS:
        for row in _read_tsv(filename):
            display_id, name_key, radius = extract(row)
            entries.append(
                {
                    "id": display_id,
                    "ra_degrees": Degrees(float(row["_RAJ2000"])),
                    "dec_degrees": Degrees(float(row["_DEJ2000"])),
                    "radius": Degrees(radius),
                    "named": name_key in names,
                    "name": names.get(name_key, display_id),
                }
            )
    return entries
