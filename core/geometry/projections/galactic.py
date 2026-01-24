from typing import overload
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore

from geometry import Radians
from geometry.projection import Projection


@overload
def equatorial_to_galactic(ra: float, dec: float) -> tuple[float, float]: ...


@overload
def equatorial_to_galactic(
    ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def equatorial_to_galactic(ra: Radians, dec: Radians) -> tuple[Radians, Radians]: ...


def equatorial_to_galactic(ra: Radians, dec: Radians) -> tuple[Radians, Radians]:
    eq = SkyCoord(ra, dec, unit="rad")

    theta = np.radians((180 - eq.galactic.l.value) % 360 - 180)
    phi = np.radians(eq.galactic.b.value)

    return theta, phi


# TODO: This is akwardly-named, because it isn't a 3D to 2D projection at all.
class GalacticProjection(Projection):
    @overload
    def project(self, lon: float, lat: float) -> tuple[float, float]: ...

    @overload
    def project(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def project(self, lon: Radians, lat: Radians) -> tuple[Radians, Radians]: ...

    def project(self, lon: Radians, lat: Radians) -> tuple[Radians, Radians]:
        return equatorial_to_galactic(lon, lat)

    @overload
    def unproject(self, x: float, y: float) -> tuple[float, float]: ...

    @overload
    def unproject(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def unproject(self, x: Radians, y: Radians) -> tuple[Radians, Radians]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"GalacticProjection"
