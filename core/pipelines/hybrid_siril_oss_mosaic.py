"""Hybrid pipeline: Siril stacking + OSS mosaic, no GraXpert.

An A/B fork of `hybrid_siril_graxpert_oss_mosaic.py`. **One thing differs**:
GraXpert runs nowhere — not on any master, not on the merged mosaic. The
Gaia gain, the pedestal pin, the overlap level solve, BXT and the colour
tail are the same code.

WHY (measured 2026-10-23 on the 46 MDW panels of mosaic 9310, 112
overlapping pairs): neighbours disagree by 8% of the scene's range in MDW's
own pixels, 13% after our calibration; the Gaia gains are consistent
(0.46-0.59), so the rest is per-panel GraXpert taking a different share of
Cygnus's wall-to-wall Hα for "background" in each panel. With narrowband
filters and night-sky flats there may be little gradient left for it to
find. Run both pipelines on the same mosaics and compare seams and depth.
"""
from pipelines.hybrid_graxpert_oss_mosaic import register
from pipelines.hybrid_siril_graxpert_oss_mosaic import channel, geometry, post_processed_master

PIPELINE = "Hybrid: Siril + OSS Mosaic (no GraXpert)"

for_stack, for_target, for_mosaic = register(PIPELINE, post_processed_master,
                                             channel=channel, geometry=geometry,
                                             graxpert=False)
