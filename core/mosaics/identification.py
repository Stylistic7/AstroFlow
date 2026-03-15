import structlog
from frozendict import frozendict
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry.vector import vector_to_lonlat
import ray
import numpy as np
import random
from typing import Any
from dataclasses import asdict
from beartype import beartype
from collections import defaultdict
from itertools import combinations

from clustering.angles import angular_spread_mod_pi
from mosaics.types import Mosaic, MosaicHash, MosaicsDict, Strip
from catalogs import position_string
from geometry import footprints
from targets.types import TargetHash, Target, TargetsDict
from mosaics.debugging import (
    plot_targets_and_footprint,
    plot_mosaic,
    plot_targets_overlap,
)
from mosaics.disjoint_set import HashDisjointSet
from mosaics.overlap import Overlap
from decorators import cached
from attributes import AstrometricAttributes, ChronologyAttributes
from attributes.astrometric_attributes import healpix_cells_from_poly

logger = structlog.get_logger()
log = logger


@beartype
def roughly_same_area(
    a: SingleSphericalPolygon, b: SingleSphericalPolygon
) -> bool:
    ratio = a.area() / b.area()
    return bool(ratio > 0.9 and ratio < 1.1)


@beartype
def roughly_same_image_scale(a: float, b: float) -> bool:
    ratio = a / b
    return bool(ratio > 0.9 and ratio < 1.1)


@beartype
def roughly_same_position_angle(
    a: float, b: float, angular_tolerance: float
) -> bool:
    return bool(angular_spread_mod_pi([a, b]) < angular_tolerance)


@cached.remote
def targets_overlap(
    a_target: Target,
    b_target: Target,
    min_area_frac: float = 0.03,
    span_match: float = 0.75,
    thick_min: float = 0.1,
    thick_max: float = 0.3,
    angular_tolerance: float = 10.0,
    debug: bool = False,
) -> Overlap:
    if a_target is b_target:
        return Overlap.IDENTICAL

    if not roughly_same_image_scale(a_target.image_scale, b_target.image_scale):
        return Overlap.INCOMPATIBLE

    if not roughly_same_position_angle(
        a_target.rotation_angle, b_target.rotation_angle, angular_tolerance
    ):
        return Overlap.INCOMPATIBLE

    a, b = a_target.footprint, b_target.footprint

    if not roughly_same_area(a, b):
        return Overlap.INCOMPATIBLE

    if not a.intersects_poly(b):
        return Overlap.NON_OVERLAPPING

    inter = a.intersection(b)
    polys = getattr(inter, "polygons", [inter])
    if not polys:
        return Overlap.NON_OVERLAPPING
    inter = max(polys, key=lambda p: p.area())

    # reject tiny overlaps: real, but not useful for mosaics
    if inter.area() / max(min(a.area(), b.area()), 1e-30) < min_area_frac:
        return Overlap.INCOMPATIBLE

    # tangent basis at intersection centroid, in XYZ
    ptsI = np.array(list(inter.points), float)  # (N,3) unit vectors
    c = ptsI.mean(0)
    c /= np.linalg.norm(c)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(c @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(c, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    e2 /= np.linalg.norm(e2)
    E = np.vstack([e1, e2])  # 2x3; rows are basis vectors

    def spans_in_plane(poly):
        pts = np.array(list(poly.points), float)  # (N,3)
        xy = pts @ E.T  # (N,2) in common plane
        d = xy.max(0) - xy.min(0)  # (dx, dy)
        return float(abs(d[0])), float(abs(d[1]))

    ax, ay = spans_in_plane(a)
    bx, by = spans_in_plane(b)
    ix, iy = spans_in_plane(inter)

    # decide which direction is "span" vs "thickness" for the intersection stripe
    if iy >= ix:
        span, thick = iy, ix  # stripe runs along e2
        span_axis_is_y = True
    else:
        span, thick = ix, iy  # stripe runs along e1
        span_axis_is_y = False

    def scores_for_frame(wx, wy):
        # wx,wy are that frame's extents along e1,e2
        if wx == 0 or wy == 0:
            return 0.0, 0.0

        # frame intrinsic long/short sides
        if wx >= wy:
            L, S = wx, wy
            span_L, span_S = ix, iy
        else:
            L, S = wy, wx
            span_L, span_S = iy, ix

        # how much of each side is covered by the intersection bounding box
        cov_L = span_L / L  # ∈ (0,1]
        cov_S = span_S / S  # ∈ (0,1]

        # decide whether the stripe runs more along the frame's long or short side
        stripe_along_L = span_L >= span_S

        long_score = 0.0
        short_score = 0.0

        if stripe_along_L:
            # long-edge hypothesis: stripe runs along long side, thin on short side
            thin_short = cov_S  # stripe thickness vs short side
            if cov_L >= span_match and thick_min <= thin_short <= thick_max:
                long_score = cov_L
        else:
            # short-edge hypothesis: stripe runs along short side, thin on long side
            thin_long = cov_L  # stripe thickness vs long side
            if cov_S >= span_match and thick_min <= thin_long <= thick_max:
                short_score = cov_S

        return long_score, short_score

    long_a, short_a = scores_for_frame(ax, ay)
    long_b, short_b = scores_for_frame(bx, by)

    # symmetric aggregation: swapping a,b leaves these unchanged
    long_score = max(long_a, long_b)
    short_score = max(short_a, short_b)

    if long_score >= span_match and short_score < span_match:
        result = Overlap.LONG_EDGE
    elif short_score >= span_match and long_score < span_match:
        result = Overlap.SHORT_EDGE
    elif long_score >= span_match and short_score >= span_match:
        result = (
            Overlap.LONG_EDGE
            if long_score >= short_score
            else Overlap.SHORT_EDGE
        )
    else:
        result = Overlap.INCOMPATIBLE

    if debug:
        title = (
            f"targets_overlap {a_target.stable_hash} {b_target.stable_hash}\n"
            f"{ax=:.3f} {ay=:.3f} {bx=:.3f} {by=:.3f} {ix=:.3f} {iy=:.3f}\n"
            f"{span=:.3f} {thick=:.3f} {span_axis_is_y=}\n"
            f"{long_score=:.3f} {short_score=:.3f} {result=}"
        )

        plot_targets_overlap(a_target, b_target, title=title)

    return result


@beartype
def group_targets(
    targets: TargetsDict,
    target_ids: list[TargetHash],
    allowed_overlaps: list[Overlap],
    angular_tolerance: float,
) -> list[list[TargetHash]]:
    assert len(target_ids) > 0

    target_ids_for_cell = defaultdict(list)
    for target_id in target_ids:
        target = targets[target_id]
        assert len(target.healpix_cells) > 0
        for cell in target.healpix_cells:
            target_ids_for_cell[cell].append(target_id)

    assert len(target_ids_for_cell) > 0

    disjoint_set = HashDisjointSet()
    # TODO: Parallelize these.
    for cell, tids in target_ids_for_cell.items():
        if len(tids) < 2:
            continue

        for a, b in combinations(sorted(tids), 2):
            overlap = ray.get(
                targets_overlap.remote(
                    targets[a], targets[b], angular_tolerance=angular_tolerance
                )
            )
            if overlap in allowed_overlaps:
                print(f"### union {a=} {b=} with overlap {overlap}")
                disjoint_set.union(a, b)

    mosaics = disjoint_set.groups()

    singletons = [[t] for t in target_ids if not any(t in m for m in mosaics)]

    result = mosaics + singletons
    assert sum(len(r) for r in result) == len(target_ids)
    return result


# def test_commutativity():
#   targets = context.grouped_targets

#   for a in targets.keys():
#     for b in targets.keys():
#       a_to_b = ray.get(targets_overlap.remote(targets[a], targets[b]))
#       b_to_a = ray.get(targets_overlap.remote(targets[b], targets[a]))
#       assert a_to_b == b_to_a, f"### {a_to_b=} {b_to_a=}"


@beartype
def find_strips(
    targets: TargetsDict,
    target_ids: list[TargetHash],
    angular_tolerance: float,
    debug: bool = False,
) -> tuple[Strip, ...]:
    strips = group_targets(
        targets, target_ids, [Overlap.LONG_EDGE], angular_tolerance
    )
    if debug:
        for i, strip in enumerate(strips):
            print(f"### Strip {i} is {strip}")
        plot_targets_and_footprint(targets, target_ids, None, title="Mosaic")

    # TODO: Maybe use this same algorithm to order the strips themselves to make
    # sure the output strips can be merged with PhotometricMergeMosaic.
    @beartype
    def order_strip(strip: set[TargetHash]) -> Strip:
        strip_length = len(strip)
        strip = list(sorted(strip, key=lambda t: targets[t].center[0]))

        if len(strip) < 3:
            return Strip(strip)

        def overlap_count(target_id):
            return sum(
                1
                for tid in strip
                if ray.get(
                    targets_overlap.remote(
                        targets[target_id],
                        targets[tid],
                        angular_tolerance=angular_tolerance,
                    )
                )
                == Overlap.LONG_EDGE
            )

        overlap_counts = {tid: overlap_count(tid) for tid in strip}
        print(f"### overlap_counts {overlap_counts}")
        assert set(overlap_counts.values()) == {1, 2}
        ones = [k for k, v in overlap_counts.items() if v == 1]
        assert len(ones) == 2
        end_target_id = ones[0]

        output_strip = [end_target_id]
        strip.remove(end_target_id)
        while len(output_strip) < strip_length:
            print(f"### order_strip len(output_strip) {len(output_strip)}")
            starting_length = len(output_strip)
            for target_id in strip:
                overlap = ray.get(
                    targets_overlap.remote(
                        targets[output_strip[-1]], targets[target_id]
                    )
                )
                if overlap == Overlap.LONG_EDGE:
                    if debug:
                        print(f"### added {target_id} to strip")
                    output_strip.append(target_id)
                    strip.remove(target_id)
                    break
                else:
                    print(
                        f"### end of strip target {output_strip[-1]} "
                        f"has overlap {overlap} with target {target_id}"
                    )

            assert len(output_strip) != starting_length, (
                "### order_strip did not make progress"
            )

        assert len(output_strip) == strip_length
        return Strip(output_strip)

    output_strips = []
    for i, strip in enumerate(strips):
        print(f"### Ordering strip {i} / {len(strips)}")
        ordered_strip = order_strip(set(strip))
        if debug:
            print(f"### before ordering, strip is {strip}")
            print(f"### after ordering, strip is {ordered_strip}")
        output_strips.append(ordered_strip)

    # TODO: This is a bit of a hack. Consider more principled ways of sorting
    # strips.
    def min_dec(strip: list[TargetHash]):
        return min(targets[t].center[1] for t in strip)

    # Sort each strip individually by increasing RA.
    strips = [tuple(sorted(s, key=lambda t: targets[t].center[0])) for s in strips]

    # Sort the strips by increasing Dec.
    return tuple(sorted(strips, key=min_dec))


@beartype
def _identify_overlapping_targets(
    targets: TargetsDict, angular_tolerance: float, debug: bool = False
) -> list[list[TargetHash]]:
    if not targets:
        return []

    # Sort the targets by their center RA.
    target_ids = list(
        sorted(targets.keys(), key=lambda t: targets[t].center[0])
    )
    log.info("Identifying overlapping targets", num_targets=len(targets))

    mosaics = group_targets(
        targets,
        target_ids,
        [Overlap.LONG_EDGE, Overlap.SHORT_EDGE],
        angular_tolerance,
    )

    num_grouped_targets = sum(len(m) for m in mosaics)
    num_ungrouped_targets = len(targets) - num_grouped_targets
    log.info(
        "Done grouping overlapping targets",
        num_mosaics=len(mosaics),
        num_grouped_targets=num_grouped_targets,
        num_ungrouped_targets=num_ungrouped_targets,
    )

    mosaics = [sorted(m) for m in mosaics if len(m) > 1]
    for mosaic in mosaics:
        print(f"### Group of {len(mosaic)} overlapping targets:", mosaic)

    return mosaics


@cached.remote
def identify_mosaics(
    angular_tolerance: float, targets: dict, debug: bool = False
) -> dict[MosaicHash, Mosaic]:
    if not isinstance(targets, TargetsDict):
        targets = TargetsDict(targets)
    print(f"### Identifying mosaics from {len(targets)} targets")

    # test_commutativity()

    mosaics = _identify_overlapping_targets(targets, angular_tolerance, debug)
    if not mosaics:
        print("### No overlapping targets found")
        return MosaicsDict()

    print(f"Found {len(mosaics)} mosaics")

    # TODO: Parallelize these
    output_mosaics = {}
    for i, mosaic in enumerate(mosaics):
        print(
            f"### Processing mosaic {i} / {len(mosaics)} "
            f"with {len(mosaic)} targets: {mosaic}"
        )

        stable_hash = targets[
            sorted(mosaic, key=lambda t: targets[t].oldest_date)[0]
        ].stable_hash

        if debug:
            plot_targets_and_footprint(
                targets,
                mosaic,
                None,
                title=f"Mosaic {stable_hash} before ordering",
            )

        oldest_date = min(targets[t].oldest_date for t in mosaic)
        newest_date = min(targets[t].newest_date for t in mosaic)

        image_scale = np.mean([targets[t].image_scale for t in mosaic])
        rotation_angle = np.mean([targets[t].rotation_angle for t in mosaic])

        # This doesn't seem work for polygons with holes in them.
        # TODO: Consider using multi_union but falling back to convex hull when
        # it doesn't work?
        # old_footprint = next(
        #     SphericalPolygon.multi_union([
        #         SphericalPolygon(targets[target_id].footprint)
        #         for target_id in mosaic
        #     ]).points).tolist()

        try:
            strips = find_strips(targets, mosaic, angular_tolerance, debug)
            print("### strips", strips)
        except AssertionError:
            print(f"### Failed to order mosaic")
            continue

        for target_id in mosaic:
            print(
                f"### Target {target_id} "
                f"has {len(targets[target_id].footprint.points)} points"
            )

        all_points = [
            targets[target_id].footprint.points[:-1] for target_id in mosaic
        ]
        all_points = np.concatenate(all_points)
        print(
            f"### Mosaic composed of {len(mosaic)} targets "
            f"and {all_points.shape} points"
        )
        footprint = SingleSphericalPolygon.convex_hull(all_points)

        healpix_cells = healpix_cells_from_poly(footprint)

        # TODO: This is turbo weird, but appears to work. Just randomize the
        # points until the `convex_hull()` operation is successful.
        for i in range(10):
            if SphericalPolygon.self_intersect(footprint.points):
                points = footprint.points
                random.shuffle(points)
                footprint = SingleSphericalPolygon.convex_hull(points)
            else:
                print(
                    f"### convex_hull successful after {i} iterations "
                    f"with {len(footprint.points)} points"
                )
                break

        # assert not SphericalPolygon.self_intersect(footprint.points)
        projected_footprints = frozendict(footprints.all_projections(footprint))

        center = vector_to_lonlat(*np.mean(footprint.points, axis=0))
        name = position_string.canonical_target_string(center)

        chronology_attributes = ChronologyAttributes(
            stable_hash, oldest_date, newest_date
        )
        astrometric_attributes = AstrometricAttributes(
            name=name,
            center=tuple(center),
            image_scale=image_scale,
            rotation_angle=rotation_angle,
            footprint=footprint,
            healpix_cells=healpix_cells,
            projected_footprints=projected_footprints,
        )

        attributes: dict[str, Any] = (
            asdict(chronology_attributes)
            | asdict(astrometric_attributes)
            | {
                "stable_hash": stable_hash,
                "target_ids": tuple(sorted(mosaic)),
                "strips": strips,
            }
        )

        output_mosaic = Mosaic(**attributes)
        print(f"### Made output_mosaic {output_mosaic}")

        if debug:
            plot_mosaic(
                targets,
                output_mosaic,
                title=f"Mosaic {stable_hash} after ordering",
            )

        output_mosaics[stable_hash] = output_mosaic

    return output_mosaics
