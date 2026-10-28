"""Mosaics are ad hoc: any targets whose footprints overlap, on the same
train, are one mosaic. Membership is transitive — a connected component of
pairwise overlaps — with one guard: a union that would put two panel
centres more than `MAX_MOSAIC_SPAN` apart is refused, because the merge
reprojects every panel onto one tangent plane, and a 60-degree canvas is
what the Cassiopeia grid chained into Cepheus through a shared pane would
be.

Until 2026-10-07 a mosaic also had to be a regular grid: overlaps were
classified LONG_EDGE / SHORT_EDGE by stripe geometry, panes had to agree in
position angle and area, and panels were ordered into strips for a
pairwise merge. The merges have been global for a long time; the grid
rules only dropped mosaics ("Failed to order mosaic") and forbade fill-in
and hinge panes. Panel numbers come from `rise_time.panel_order`, as before.
"""
import structlog
from frozendict import frozendict
from spherical_geometry.polygon import MalformedPolygonError, SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry.vector import vector_to_lonlat
import ray
import numpy as np
import random
from typing import Any
from dataclasses import asdict
from beartype import beartype
from collections import defaultdict
from itertools import combinations

from decisions.decisions import DecisionType
from mosaics.types import Mosaic, MosaicsDict
from catalogs import position_string
from geometry import footprints
from targets.types import TargetHash, Target
from mosaics.debugging import plot_mosaic
from decorators import node
from attributes import AstrometricAttributes, ChronologyAttributes
from attributes.astrometric_attributes import healpix_cells_from_poly
from cache.context_ref import ContextRef
from units import Degrees, Radians, ArcsecondsPerPixel

logger = structlog.get_logger()
log = logger

# Largest separation between two panel centres one mosaic may hold. 75a1,
# 24 RedCat panels in a 4 x 6 grid, spans ~30 deg and the Cassiopeia grid
# ~25 deg; a chain through the Question Mark into Cepheus would pass 60.
MAX_MOSAIC_SPAN = Degrees(45.0)


@beartype
def roughly_same_image_scale(a: ArcsecondsPerPixel, b: ArcsecondsPerPixel) -> bool:
    ratio = a / b
    return bool(ratio > 0.9 and ratio < 1.1)


@node
def targets_overlap(
    a_target_ref: ContextRef[Target],
    b_target_ref: ContextRef[Target],
    min_area_frac: float = 0.03,
) -> bool:
    """Whether two targets are panels of one mosaic: the same train and
    footprints sharing at least `min_area_frac` of the smaller one — enough
    overlap to level-match; a corner graze is not a join. Position angle
    and layout are free: the merge reprojects every panel onto the
    mosaic's canvas. This node's cache key hashes its own source, so only
    a change HERE re-keys cached verdicts.

    "Same train" is the same sensor (pixel geometry, as `identify_targets`
    partitions stacks) at the same image scale (within 10%): different
    cameras never share a mosaic — Tier 2 combines across cameras, a
    mosaic does not — and the two rigs' 6200MMs are told apart by scale."""
    a_target = a_target_ref.get()
    b_target = b_target_ref.get()
    if a_target is b_target:
        return False
    if a_target.sensor != b_target.sensor:
        return False
    if not roughly_same_image_scale(a_target.image_scale, b_target.image_scale):
        return False
    a, b = a_target.footprint, b_target.footprint
    if not a.intersects_poly(b):
        return False
    try:
        inter = a.intersection(b)
    except MalformedPolygonError as e:
        # spherical_geometry gives up on degenerate pairs — coincident
        # framings, hulls sharing an edge ("remove orphan nodes", "remove
        # cut lines"; the grid rules' PA gate used to keep these from ever
        # being intersected, 2026-10-07). The question is still just "how
        # much of the smaller footprint lies inside the other", so answer
        # it by sampling.
        log.warning("targets_overlap: intersection failed, sampling instead",
                    a=a_target.stable_hash[:8], b=b_target.stable_hash[:8], error=str(e))
        return _sampled_overlap_frac(a, b) >= min_area_frac
    polys = getattr(inter, "polygons", [inter])
    if not polys:
        return False
    inter = max(polys, key=lambda p: p.area())
    return bool(inter.area() / max(min(a.area(), b.area()), 1e-30) >= min_area_frac)


def _sampled_overlap_frac(a, b, n: int = 6000) -> float:
    """Fraction of the smaller footprint's area inside the other, by Monte
    Carlo: `n` points uniform in the spherical cap around the smaller
    polygon, kept where it contains them, counted where the other does
    too. Seeded, so the verdict is a pure function of the pair. ±1% at the
    3% threshold."""
    small, other = sorted((a, b), key=lambda p: p.area())
    pts = np.asarray(small.points, float)
    c = pts.mean(0)
    c /= np.linalg.norm(c)
    cos_r = float(np.min(pts @ c))                      # cap just covering the polygon
    rng = np.random.default_rng(0)
    z = rng.uniform(cos_r, 1.0, n)                      # uniform on the sphere within the cap
    phi = rng.uniform(0.0, 2 * np.pi, n)
    u = np.cross(c, [0.0, 0.0, 1.0] if abs(c[2]) < 0.999 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(c, u)
    s = np.sqrt(1.0 - z * z)
    samples = z[:, None] * c + (s * np.cos(phi))[:, None] * u + (s * np.sin(phi))[:, None] * v
    inside_small = [p for p in samples if small.contains_point(p)]
    if not inside_small:
        return 0.0
    return sum(1 for p in inside_small if other.contains_point(p)) / len(inside_small)


def _span(targets: list[Target]) -> Degrees:
    """Largest great-circle separation between any two centres."""
    ra, dec = np.radians(np.array([t.center for t in targets], float)).T
    xyz = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
    return Degrees(float(np.degrees(np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0)).max())))


@beartype
def group_targets(
    target_refs: dict[TargetHash, ContextRef[Target]],
    target_ids: list[TargetHash],
) -> list[list[TargetHash]]:
    """Connected components of the overlap graph, singletons included.
    Pairs are tested only where they share a HEALPix cell. Edges apply in
    a fixed order, so which union the span guard refuses is deterministic."""
    assert len(target_ids) > 0
    resolved = {t: target_refs[t].get() for t in target_ids}
    target_ids_for_cell = defaultdict(list)
    for target_id in target_ids:
        assert len(resolved[target_id].healpix_cells) > 0
        for cell in resolved[target_id].healpix_cells:
            target_ids_for_cell[cell].append(target_id)

    pairs = sorted({tuple(sorted(pair)) for tids in target_ids_for_cell.values()
                    for pair in combinations(tids, 2)})
    verdicts = ray.get([targets_overlap.remote(target_refs[a], target_refs[b]) for a, b in pairs])

    component: dict[TargetHash, set[TargetHash]] = {t: {t} for t in target_ids}
    for (a, b), overlaps in zip(pairs, verdicts):
        if not overlaps or component[a] is component[b]:
            continue
        merged = component[a] | component[b]
        span = _span([resolved[t] for t in merged])
        if span > MAX_MOSAIC_SPAN:
            log.warning("Refusing to join mosaics: span too large", a=a[:8], b=b[:8],
                        span_deg=round(float(span), 1), max_deg=float(MAX_MOSAIC_SPAN))
            continue
        log.debug("Union", a=a[:8], b=b[:8])
        for t in merged:
            component[t] = merged
    groups = {id(c): sorted(c) for c in component.values()}
    result = sorted(groups.values(), key=lambda g: g[0])
    assert sum(len(r) for r in result) == len(target_ids)
    return result


@beartype
def _identify_overlapping_targets(
    target_refs: dict[TargetHash, ContextRef[Target]],
) -> list[list[TargetHash]]:
    """The groups of two or more overlapping targets, each sorted."""
    if not target_refs:
        return []
    resolved = {k: ref.get() for k, ref in target_refs.items()}
    target_ids = list(sorted(resolved.keys(), key=lambda t: resolved[t].center[0]))
    log.info("Identifying overlapping targets", num_targets=len(target_refs))
    mosaics = [m for m in group_targets(target_refs, target_ids) if len(m) > 1]
    log.info("Done grouping overlapping targets", num_mosaics=len(mosaics),
             num_grouped_targets=sum(len(m) for m in mosaics),
             num_ungrouped_targets=len(target_refs) - sum(len(m) for m in mosaics))
    for mosaic in mosaics:
        log.info("Overlapping target group", num_targets=len(mosaic))
    return mosaics


STANDALONE = "standalone"


def standalone_targets(decision_overrides) -> frozenset[str]:
    """The targets the user keeps out of every mosaic (DecisionType.MOSAIC_MEMBERSHIP)."""
    return frozenset(k.subject_id for k, v in (decision_overrides or {}).items()
                     if k.decision_type is DecisionType.MOSAIC_MEMBERSHIP and v.choice == STANDALONE)


def mosaic_from_targets(target_ids, resolved: dict[str, Target]) -> Mosaic:
    """The Mosaic of the targets `target_ids` (`resolved`: id -> Target).
    Identity is the oldest target's, content the members' contents — the
    same composition Stack and Target use — so a mosaic built from a
    SUBSET of a project's targets, or from targets holding a subset of their
    runs (`pipelines/time_lapse.py`), is keyed by exactly that membership."""
    oldest = resolved[sorted(target_ids, key=lambda t: resolved[t].oldest_date)[0]]
    stable_hash = oldest.stable_hash

    oldest_date = min(resolved[t].oldest_date for t in target_ids)
    newest_date = max(resolved[t].newest_date for t in target_ids)

    image_scale = ArcsecondsPerPixel(np.mean([resolved[t].image_scale for t in target_ids]))
    rotation_angle = Radians(np.mean([resolved[t].rotation_angle for t in target_ids]))

    all_points = [
        resolved[target_id].footprint.points[:-1] for target_id in target_ids
    ]
    all_points = np.concatenate(all_points)
    log.debug("Computing convex hull", num_targets=len(target_ids), num_points=all_points.shape[0])
    footprint = SingleSphericalPolygon.convex_hull(all_points)

    healpix_cells = healpix_cells_from_poly(footprint)

    rng = random.Random(hash(tuple(map(tuple, all_points))))
    for i in range(10):
        if SphericalPolygon.self_intersect(footprint.points):
            points = footprint.points
            rng.shuffle(points)
            footprint = SingleSphericalPolygon.convex_hull(points)
        else:
            break

    projected_footprints = frozendict(footprints.all_projections(footprint))

    center_raw = vector_to_lonlat(*np.mean(footprint.points, axis=0))
    center = (Degrees(center_raw[0]), Degrees(center_raw[1]))
    name = position_string.canonical_target_string(center)

    chronology_attributes = ChronologyAttributes(
        stable_hash, oldest_date, newest_date
    )
    astrometric_attributes = AstrometricAttributes(
        name=name,
        center=center,
        image_scale=image_scale,
        rotation_angle=rotation_angle,
        footprint=footprint,
        healpix_cells=healpix_cells,
        projected_footprints=projected_footprints,
        stable_center=oldest.stable_center,
    )

    from hashing import hash_list
    # content_hash composes the children's CONTENT hashes (mirror
    # Stack/Target), NOT their stable/identity hashes — so a Mosaic's
    # content_hash changes whenever any member target's *content* changes.
    # (target_ids stays the identity list.)
    content_hash = hash_list(sorted(resolved[t].content_hash for t in target_ids))

    # Panel order — observer-independent, sorted by RA with the
    # sequence rotated to start after the largest RA gap so the
    # 0h/24h boundary doesn't split a strip. Same numbering anywhere
    # on Earth, so the polygon labels in the viewer match the
    # "Panel N" prefix in the Telescopius / text export.
    from mosaics.rise_time import panel_order
    panel_target_ids = panel_order(
        [(t, float(resolved[t].center[0])) for t in target_ids]
    )

    attributes: dict[str, Any] = (
        asdict(chronology_attributes)
        | asdict(astrometric_attributes)
        | {
            "stable_hash": stable_hash,
            "content_hash": content_hash,
            "target_ids": tuple(sorted(target_ids)),
            "panel_target_ids": panel_target_ids,
        }
    )
    return Mosaic(**attributes)


@node
def identify_mosaics(
    standalone: frozenset[str],
    targets: dict[TargetHash, ContextRef[Target]],
    debug: bool = False,
) -> MosaicsDict:
    """Mosaics = groups of overlapping targets, less the `standalone` ones,
    which the user has said are projects of their own however they overlap."""
    targets = {k: ref for k, ref in targets.items() if k not in standalone}
    resolved: dict[str, Target] = {k: ref.get() for k, ref in targets.items()}
    log.info("Identifying mosaics", num_targets=len(resolved), num_standalone=len(standalone))

    mosaics = _identify_overlapping_targets(targets)
    if not mosaics:
        log.info("No overlapping targets found")
        return MosaicsDict({})

    log.info("Found mosaics", num_mosaics=len(mosaics))

    # TODO: Parallelize these
    output_mosaics = MosaicsDict()
    for i, mosaic in enumerate(mosaics):
        log.info("Processing mosaic", index=i, total=len(mosaics), num_targets=len(mosaic))
        output_mosaic = mosaic_from_targets(mosaic, resolved)
        stable_hash = output_mosaic.stable_hash
        log.info("Made mosaic", name=output_mosaic.name, stable_hash=stable_hash, num_targets=len(mosaic))

        if debug:
            plot_mosaic(
                resolved,
                output_mosaic,
                title=f"Mosaic {stable_hash}",
            )

        output_mosaics[stable_hash] = ContextRef.store(
            ("mosaics", stable_hash), output_mosaic
        )

    return output_mosaics
