"""Pipeline: Siril stacking + the site's sky-glow model + OSS mosaic.

A fork of `hybrid_siril_graxpert_oss_mosaic.py`. **One thing differs**: no
GraXpert. Each subframe has its own predicted sky glow — the site's light
pollution and, with the Moon up, scattered moonlight (`light_pollution`) —
subtracted before Siril registers and stacks it. The masters therefore come
out of the integration already gradient-corrected, and the post-stacking
calibration applies the SPFC gain and the pedestal only
(`register(..., panel_graxpert=False)`).

WHY (measured 2026-11-02 against the MDW Hα survey on the 24 75a1 panels it
covers): a data-driven background extraction on a nebula-rich master takes
the nebula for background (GraXpert left 3.2x the gradient on panel b98d);
the site model, fitted from the library's own subframes, left 0.81 of the
uncorrected gradient on the median panel (GraXpert 0.94) and 0.70 on the
moonlit-heavy ones once the Moon was modelled. Run both pipelines on the
same mosaic and compare.
"""
from dependency_graph.tracing import workflow_group
from light_pollution.model import SiteModel, SiteModels
from light_pollution.sites import site_of
from metadata.master_analysis import analyze_master
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image
from pipelines.hybrid_graxpert_oss_mosaic import register
from pipelines.hybrid_siril_graxpert_oss_mosaic import _by_exposure
from stacks.calibration_selection import calibrate
from stacks.foreign import foreign_master, foreign_post_processed_master, is_foreign
from stacks.post_processing.graxpert_stack import post_stack_calibrate
from stacks.stacking.siril import GLOW_CODE, integrate_with_siril
from targets.merging.oss import calibrate_channel, hdr_merge_masters, merge_target_channel_oss

PIPELINE = "Siril + Site Model + OSS Mosaic"


def models_for(stack, table: SiteModels | None) -> tuple[SiteModel | str, ...]:
    """The distinct models of the sites the stack's frames state, in hash
    order (argument order folds into the integration's key), then the
    correction's code hash. A frame at a site without a model, or stating
    none, is simply stacked as it is; a stack no model applies to carries
    no code hash either, so a change to the correction leaves it alone."""
    if table is None:
        return ()
    found = {m.content_hash: m for meta in stack.all_metadata().values()
             if (m := table.model_for(site_of(meta.header))) is not None}
    return (*(found[k] for k in sorted(found)), GLOW_CODE) if found else ()


def solved_master(stack, config):
    """Calibrate (Siril masters) → Siril-integrate with the sky glow taken
    out of each sub → plate-solve."""
    master = plate_solve_image(integrate_with_siril(
        stack, config, *calibrate(stack, config, integrate=integrate_with_siril),
        *models_for(stack, config.get_site_models())))
    analyze_master(master)
    return master


@workflow_group
def post_processed_master(stack, config):
    """`solved_master` → SPFC gain and pedestal (no background extraction)."""
    if is_foreign(stack):
        return foreign_post_processed_master(stack, config, post_stack_calibrate)
    master = solved_master(stack, config)
    return inherit_plate_solution(post_stack_calibrate(stack, config, master), master)


def channel(target, filter, stacks, config):
    """As the GraXpert fork's channel, on glow-free raw masters."""
    if len(stacks) == 1:
        master = post_processed_master(stacks[0], config)
        return inherit_plate_solution(
            merge_target_channel_oss(target, filter, tuple(stacks), config, master), master)
    ordered = _by_exposure(stacks)
    if any(is_foreign(s) for s in ordered):
        masters = [post_processed_master(s, config) for s in ordered]
        return inherit_plate_solution(
            hdr_merge_masters(target, filter, ordered, config, *masters), masters[0])
    raws = [solved_master(s, config) for s in ordered]
    merged = inherit_plate_solution(hdr_merge_masters(target, filter, ordered, config, *raws), raws[0])
    return inherit_plate_solution(calibrate_channel(target, filter, ordered, config, merged), raws[0])


def geometry(target, filter, stacks, config):
    first = _by_exposure(stacks)[0]
    return foreign_master(first) if is_foreign(first) else solved_master(first, config)


for_stack, for_target, for_mosaic = register(PIPELINE, post_processed_master,
                                             channel=channel, geometry=geometry, panel_graxpert=False)
