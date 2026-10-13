"""What an hour of sky is worth, 0..1, by filter, altitude and moon.

Numbers are Stylistic7's, measured from his own subs (2026-10-05/06): the
altitude penalty (30-40° costs ~1.3-1.4x the hours) and the moon penalty
per filter (Hα 1.5x with a bright moon far away, 1.8x at 45-70°, 3.8x
within 45°; SII 1.2-1.3x; OIII only with the moon below the horizon — his
practice; there is no measured OIII-under-moon data yet).
Floors are his rules: OIII ≥ 40°, everything else ≥ 30°.

Moonless work (`moonless`): OIII, and faint broadband — L, and R/G/B on an
LRGB project. R/G/B on a narrowband project is star colour, which the moon
costs little. The planner shoots moonless work first whenever the moon is
down (a tier above Hα/SII in both layers): those hours are the only ones
it has, while Hα and SII have every night (Stylistic7, 2026-10-12: "this is
literally the only time to shoot OIII").
"""
from __future__ import annotations

import numpy as np

NARROWBAND = {"H", "S", "O"}
FLOOR_DEG = {"O": 40.0}
DEFAULT_FLOOR_DEG = 30.0


def altitude_factor(alt: np.ndarray, filt: str) -> np.ndarray:
    floor = FLOOR_DEG.get(filt, DEFAULT_FLOOR_DEG)
    return np.select([alt < floor, alt < 40.0, alt < 50.0], [0.0, 0.72, 0.9], default=1.0)


def moonless(filt: str, kind: str) -> bool:
    """Work for moonless hours only: OIII, L, and R/G/B when the project is
    LRGB (faint signal); star colour on a narrowband project is not."""
    return filt in ("O", "L") or (filt in ("R", "G", "B") and kind == "lrgb")


def moon_rank(filt: str, kind: str) -> int:
    """0 for moonless work, 1 for the rest: the tier below the star."""
    return 0 if moonless(filt, kind) else 1


def moon_factor(filt: str, moon_alt: np.ndarray, illum: np.ndarray, sep: np.ndarray,
                kind: str = "narrowband") -> np.ndarray:
    up = moon_alt > 0
    if filt == "O":
        f = np.zeros_like(illum)               # Stylistic7's practice: OIII only with the moon down
    elif filt == "H":
        f = np.select([illum < 0.4, illum < 0.7, sep > 70, sep >= 45], [1.0, 0.9, 0.67, 0.55], default=0.26)
    elif filt == "S":
        f = np.where(illum < 0.7, 1.0, 0.8)
    elif filt in ("R", "G", "B") and not moonless(filt, kind):
        f = np.full_like(illum, 0.8)           # star colour only: the moon costs little
    else:                                      # L and R/G/B on an LRGB project: faint signal
        f = np.where(illum < 0.25, 0.5, 0.0)
    return np.where(up, f, 1.0)


def efficiency(filt: str, alt, moon_alt, illum, sep, kind: str = "narrowband") -> np.ndarray:
    return altitude_factor(np.asarray(alt), filt) * \
        moon_factor(filt, np.asarray(moon_alt), np.asarray(illum), np.asarray(sep), kind)

