"""Hybrid pipeline: Siril stacking + GraXpert gradients + OSS mosaic.

An A/B fork of `hybrid_graxpert_oss_mosaic.py`. **One thing differs**: each
stack is integrated by Siril (`stacks.stacking.siril.integrate_with_siril`)
instead of PixInsight's WBPP (`stack_files`), and so are its bias, dark and
flat masters: nothing in this pipeline runs WBPP. The post-stack calibration
(SPFC + GraXpert), the channel combine, BXT, placement, the merge and the
colour tail are the same code — registered through that module's `register`.

WHY (measured 2026-10-08 on panel 6 OIII, 45 subs, uncalibrated): Siril
integrated in 26 s on an idle machine, where WBPP took 12.5 min with the app
running. After GraXpert, Siril's master revealed more of the field at >=3
sigma (21.0% vs 19.9%), with stars slightly larger (2.19 px vs 1.98 px FWHM).
Local normalisation made no visible difference after GraXpert. The recipe is
`stacks.stacking.siril.STACK_OPTIONS`. Run both pipelines on the same objects
and compare whole mosaics.
"""
from dependency_graph.tracing import workflow_group
from metadata.master_analysis import analyze_master
from metadata.solve_nodes import inherit_plate_solution, plate_solve_image
from pipelines.hybrid_graxpert_oss_mosaic import register
from stacks.calibration_selection import calibrate
from stacks.post_processing.graxpert_stack import post_stack_calibrate
from stacks.stacking.siril import integrate_with_siril

PIPELINE = "Hybrid: Siril + GraXpert + OSS Mosaic"


@workflow_group
def post_processed_master(stack, config):
    """Calibrate (Siril masters) → Siril-integrate → plate-solve → post-stacking calibration.
    The solve is its own light node so the integration's CPUs and memory
    are free while solve-field runs on one core."""
    cal_masters = calibrate(stack, config, integrate=integrate_with_siril)
    master = plate_solve_image(integrate_with_siril(stack, config, *cal_masters))
    analyze_master(master)   # leaf: writes the master's `analysis` sidecar section
    return inherit_plate_solution(post_stack_calibrate(stack, config, master), master)


for_stack, for_target, for_mosaic = register(PIPELINE, post_processed_master)
