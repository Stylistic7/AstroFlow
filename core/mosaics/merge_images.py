"""Compatibility shim — real code moved to `mosaics/merging/pixinsight.py`."""
from mosaics.merging.pixinsight import *  # noqa: F401, F403
# Explicit underscore-prefixed re-export: `mosaics/placement/astroflow.py`
# imports `_set_panel_count` (`import *` skips underscore names).
from mosaics.merging.pixinsight import _set_panel_count  # noqa: F401
