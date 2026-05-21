"""Mosaic-level placement: project each channel master into the
mosaic's pixel-space viewbox. Single AstroFlow-native implementation
(numpy + spherical_geometry); no backend-specific variants today.

`from mosaics.placement import place_image, PlacedImage, place_tile` etc.
keeps working via the re-export from `.astroflow`.
"""
from mosaics.placement.astroflow import *  # noqa: F401, F403
