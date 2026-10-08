"""`analyze_master`: measurements of a finished master, for planning.

A LEAF node: it consumes a master's `ImageMetadata` (whatever stacker made
it) and nothing consumes its result, so adding it — or new analyses to it —
re-keys nothing in the graph. Results are written into the master's sidecar
as the `analysis` section and returned; readers (the night planner, the
dashboard) read the sidecar, never the graph.

    analysis = {
      "version": ANALYSIS_VERSION,
      "stack": STBLHASH of the master,          # which stack it was built from
      "filter": FILTER,
      "integration_s": master_integration_s,     # what the curve is relative to
      "visibility": {                            # metadata.visibility
        "threshold": 3.0, "scales": [1, 2, 4, 8], "bin": 2,
        "factors":  [1, 1.25, 1.5, 2, 3, 4],     # x the master's integration
        "fraction": [...],                       # field visible at each factor
      },
    }

More analyses belong here as further keys. If one becomes expensive, give
it its own leaf node so the others stay cached when it changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from decorators import node
from metadata.classes import ImageMetadata

ANALYSIS_VERSION = 1
SECTION = "analysis"
FACTORS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
BIN = 2


@node(num_cpus=1, auto_memory=True, resources={"diskcache": 0.001})
def analyze_master(master: ImageMetadata) -> dict:
    """Measure `master` and write the results into its sidecar."""
    from file_io.generic import read_image
    from file_io.sidecar import set_section
    from metadata import visibility as V
    img = V._bin(np.asarray(read_image(Path(master.filename)), np.float32).squeeze(), BIN)
    valid, maps = V.snr_maps(img)
    result = {
        "version": ANALYSIS_VERSION,
        "stack": str(master.stable_hash),
        "filter": str((master.header or {}).get("FILTER", "")),
        "integration_s": master.master_integration_s,
        "visibility": {
            "threshold": V.THRESHOLD, "scales": list(V.SCALES), "bin": BIN,
            "factors": list(FACTORS),
            "fraction": [V.visible_fraction(valid, maps, f) for f in FACTORS],
        },
    }
    set_section(Path(master.filename), SECTION, result)
    return result


def visible_gain_per_hour(analysis: dict, hours_now: float, added_h: float = 5.0) -> float | None:
    """Fraction of the field newly visible per added hour, from `hours_now`
    (the library's integration, which may exceed the master's) over the next
    `added_h` hours, read off the stored curve. None without a usable curve."""
    vis = (analysis or {}).get("visibility") or {}
    t_m = (analysis or {}).get("integration_s")
    if not vis.get("fraction") or not t_m:
        return None
    t_m_h = t_m / 3600.0
    f = np.interp([hours_now / t_m_h, (hours_now + added_h) / t_m_h], vis["factors"], vis["fraction"])
    return float(f[1] - f[0]) / added_h
