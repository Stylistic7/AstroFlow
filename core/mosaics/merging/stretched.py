"""Display-referred mosaic merging — equalize panels where the eye judges.

Photometry is DELIBERATELY broken here, by decision (2026-08-21): when the
panels were collected through different skies (light pollution, airglow,
moonlight, different nights), a photometrically faithful merge preserves
the variance of the OBSERVING CONDITIONS, not the flux of the sky. The
photometric record lives in the per-panel masters and the linear
pipelines; this deliverable's job is to look like the sky.

The chain, per channel: place -> AUTOSTRETCH EACH PANEL (the house STF
autostretch, parameters from that panel's own statistics — the
zeroth-order equalizer: every panel's sky lands at the same output level
and its noise at the same output contrast, absorbing offset, gain, and
regime differences without modeling them) -> the SAME fused joint solve
(`_blend_many(level_solve="overlap")`: overlap-difference constants,
one exact global solve — plane fitting retired 2026-08-31, it fit scene
as sky on nebulosity-filled fields — operating in stretched space where
the residuals it minimizes are exactly the residuals the viewer sees)
-> feather. Output is tagged
COLORSPC "display": already stretched, viewer applies taste, not a second
autostretch.

This automates the hand-workflow behind most seamless amateur mosaics:
stretch each pane individually, then blend.
"""

import numpy as np

from decorators import node
from decisions.settings import ProcessingConfig
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
import structlog

log = structlog.get_logger()


def _stretch_nonzero(arr: "np.ndarray") -> "np.ndarray":
    """Per-panel autostretch that respects the no-data convention: statistics
    and mapping come from NONZERO pixels only (zero is `place_image`'s
    no-data fill, and letting it into the median would poison the black
    point); zeros stay exactly zero."""
    from mosaics.color.palettes import autostretch
    out = np.zeros_like(arr, dtype=np.float32)
    mask = arr > 0
    if mask.any():
        out[mask] = autostretch(arr[mask])
    return out


@node(auto_memory=True, plate_solve=False)
def autostretch_panel(
    mosaic: Mosaic, target, filter: str, viewbox, config: ProcessingConfig,
    placed: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """One placed panel, autostretched on its OWN statistics (bbox-on-disk
    in, bbox-on-disk out, so the blend reads it exactly like a placement).
    Per-panel pure: no joint coupling, no solver — equalization here is a
    property of the stretch, not a fit."""
    from pathlib import Path
    from mosaics.placement.panel_io import read_placed_panel, write_placed_panel

    arr, offset_xy, canvas = read_placed_panel(placed.filename)
    stretched = _stretch_nonzero(arr)
    out = Path(autostretch_panel.filename_fn(
        mosaic, target, filter, viewbox, config, placed))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_placed_panel(out, stretched, offset_xy, canvas,
                       {"FILTER": filter, "COLORSPC": "display"})
    return WithSidecar(TiffFilePath(out))


@node(display="Mosaic", final=True, auto_memory=True, plate_solve=False)  # mosaic projection — non-square pixels
def merge_images_stretched(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The fused matched merge over AUTOSTRETCHED panels — level matching
    and feathering in display space, where the solver's objective equals
    the viewer's. Thin twin of `merge_images_matched` (that op's body is
    hash-frozen for the linear pipelines; this one owns the display tag)."""
    from pathlib import Path
    from hashing import hash_list
    from mosaics.merging.oss import (
        _blend_many, _panel_count, _set_panel_count, write_tiff_image,
    )

    assert len(inputs) >= 1
    output_filename = merge_images_stretched.filename_fn(mosaic, filter, *inputs)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    paths = [inp.filename.resolve(strict=True) for inp in inputs]
    # pedestal=None: these panels are already stretched, so there is no
    # linear black point to aim at — equalize their skies to their own
    # MEAN instead (the per-panel autostretch pins each panel's MEDIAN,
    # i.e. scene content, which measured 2026-08-22 left skies 0.030 apart
    # in display units; this is the correction for exactly that).
    result = _blend_many(paths, level_solve="overlap", pedestal=None)

    write_tiff_image(Path(output_filename), result.astype(np.float32),
                     {"FILTER": filter,
                      "STBLHASH": hash_list([inp.stable_hash for inp in inputs]),
                      "PIPELINE": "AstroFlow-OSS-Stretched",
                      "COLORSPC": "display",
                      "MATCHVER": "stretched-1",
                      "SKYVER": "overlap-2"})

    new_count = sum(_panel_count(inp) for inp in inputs)
    _set_panel_count(output_filename, new_count)

    log.info("stretched OSS merge complete", output=str(output_filename),
             panel_count=new_count, n_inputs=len(inputs))
    return WithSidecar(output_filename)


import ray


@ray.remote(num_cpus=0.1)
def merge_scope_stretched(mosaic, filter, viewbox, config, targets, *corrected,
                          dry_run=False):
    """`merge_scope` with two swaps: every placement is autostretched on its
    own statistics, and the terminal is the display-space matched merge.
    Same `delete_intermediates` semantics — placements and stretched copies
    are interiors reclaimed at scope conclusion; the merge persists."""
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.placement.astroflow import place_image

    print("### merge_scope_stretched fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in corrected)))

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, corrected)
        ]
        stretched = [
            autostretch_panel(mosaic, target, filter, viewbox, config, p)
            for target, p in zip(targets, placed)
        ]
        return merge_images_stretched(mosaic, filter, *stretched)

    return Graph(trace(_body))
