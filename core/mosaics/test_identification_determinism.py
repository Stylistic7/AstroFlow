"""Cross-process determinism of the mosaic identification hashes.

The cache keys everything on `content_hash`. If any content_hash depends on
Python's per-process `str`/`bytes` hash randomization (`PYTHONHASHSEED`) — e.g.
a `hash()` on a string, or **iterating a `set` of target-id strings** whose
order flips between processes — then a mosaic that hasn't logically changed gets
a *new* content_hash on the next app start, its whole cache subtree invalidates,
and the pipeline re-derives bit-identical outputs forever (the "runs all night,
never finishes" failure).

That bug is invisible to a single-process test: within one process
`PYTHONHASHSEED` is fixed, so every hash is stable. The only way to see it is to
recompute in **separate processes with different seeds** and compare. A handful
of seeds is enough — any str-hash/set-order dependence differs by seed 0 vs 1.
"""
import os
import subprocess
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1]

# Recompute the identification hashes from FIXED synthetic inputs and print them.
# Deliberately routes the footprint through a real `set` iteration (`for t in
# s`) — the exact `for target_id in mosaic` shape in identification.py — so a
# set-order leak into the geometry hash would surface here.
_PROBE = r"""
import numpy as np
from types import SimpleNamespace
from spherical_geometry.polygon import SingleSphericalPolygon
from hashing import hash_list
from mosaics.viewbox import compute_viewbox

target_ids = ["target_c", "target_a", "target_b"]
polys = {
    "target_a": SingleSphericalPolygon.from_radec([98, 102, 102, 98], [-1, -1, 1, 1]),
    "target_b": SingleSphericalPolygon.from_radec([101, 105, 105, 101], [-1, -1, 1, 1]),
    "target_c": SingleSphericalPolygon.from_radec([100, 104, 104, 100], [0, 0, 2, 2]),
}

# 1) the mosaic content_hash recipe (identification.py: sorted ids -> hash_list)
h_content = hash_list(sorted(target_ids))

# 2) the DERIVED viewbox, via a real set iteration into the convex-hull footprint
#    (mirrors `all_points = [... for target_id in mosaic]` with mosaic a set)
s = set(target_ids)
all_points = np.concatenate([polys[t].points[:-1] for t in s])
footprint = SingleSphericalPolygon.convex_hull(all_points)
vb = compute_viewbox("Master", SimpleNamespace(footprint=footprint, image_scale=1.5))
h_viewbox = hash_list([vb.x_min, vb.x_span, vb.y_min, vb.y_span,
                       vb.resolution, vb.num_pixels_x, vb.num_pixels_y])

print(h_content, h_viewbox)
"""


def _probe(seed: int) -> str:
    """Run the probe in a fresh process pinned to `PYTHONHASHSEED=seed`."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
        cwd=str(_CORE),
    )
    assert out.returncode == 0, f"probe (seed={seed}) failed:\n{out.stderr}"
    return out.stdout.strip()


def test_identification_hashes_are_stable_across_pythonhashseed():
    """The mosaic content_hash AND the derived viewbox hash must be identical
    across processes with different hash seeds — else caches invalidate on every
    restart even when nothing changed."""
    seeds = [0, 1, 2, 7, 12345]
    results = {seed: _probe(seed) for seed in seeds}
    distinct = set(results.values())
    assert len(distinct) == 1, (
        "content_hash is NOT stable across PYTHONHASHSEED (it will change on "
        f"the next app restart, invalidating caches):\n"
        + "\n".join(f"  seed={s}: {h}" for s, h in results.items())
    )
