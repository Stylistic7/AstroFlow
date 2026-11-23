"""Mosaic-level merging: `oss.merge_images_oss` blends the placed panels of one
filter into a mosaic channel, `polish` finishes the blend, `level_solve` solves
the per-panel levels and `seam_metric` scores a join. `place_image` takes
PANEL_COUNT from here.
"""
from mosaics.merging.oss import PANEL_COUNT  # noqa: F401
