"""Mosaic-level merging: pairwise-combine placed images into a mosaic
channel master. PixInsight (PhotometricMosaic) default; OSS available.

`from mosaics.merging import merge_images, MergedImage` etc. keeps
working via the re-export from `.pixinsight`. Pipeline code should pick
a backend explicitly: `from mosaics.merging.pixinsight import …` or
`from mosaics.merging.oss import …`.
"""
from mosaics.merging.pixinsight import *  # noqa: F401, F403
# Underscore-prefixed re-exports needed by other modules.
from mosaics.merging.pixinsight import (  # noqa: F401
    PANEL_COUNT,
    _panel_count,
)
