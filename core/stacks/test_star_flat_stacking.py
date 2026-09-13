"""`scale_star_flat` and `stack_files` together, on real frames, through the
graph — the way the pipeline runs them.

Setup (as `test_orion_pane`): the three real Orion Pane 1 subs (ZWO
ASI6200MM, H, 300 s) copied into the test library with synthetic darks at
the subs' own pedestal, identified by the compute graph into one light
Stack and one dark calibration stack. The star flat is a synthetic 3%
pattern written as a WBPP Master Flat for the sensor (a real solve needs
25+ subs with photometry), and the subs' light above the pedestal carries
that pattern printed in at 1.3x, so the fit has a known answer on real sky.

Covered:
  * `calibrate()` on the real Stack traces build_star_flat -> scale_star_flat
    -> stack_files, with the dark master as the fit's pedestal input;
  * the node body executes as a Ray task (beartype on the real types, the
    ephemeral flag, the filename_fn) and records k ~ 1.3 in its sidecar;
  * `stack_files` (mock WBPP) consumes the scaled flat — matched by the
    group's STBLHASH that the scaled file carries — and produces the master;
  * the scaled flat, being ephemeral and interior, is reclaimed once the
    pipeline graph that consumed it concludes (as a child of the graph the
    runtime was given — `run_pipeline`'s shape), while the master stays.
"""
import datetime
from pathlib import Path

import numpy as np
import pytest

from context import context
from metadata.classes import ImageMetadata
from testing import copy_testdata_frames, make_synthetic_frames

K_TRUE = 1.3
STAR_FLAT_ID = "1234567890123456"     # an all-digit hash: the XISF reader must keep it a str


def _pedestal(path) -> int:
    """The real frames' pedestal (bias level): the 0.1th percentile of the
    first sub — these Orion subs sit ~590 ADU with only ~45 ADU of sky above
    it, so the synthetic dark must match it to a few ADU or the residual
    pedestal, not the pattern, would set the fitted gain."""
    import astropy.io.fits as pyfits
    return int(round(np.percentile(pyfits.getdata(path).astype(np.float64), 0.1)))


def _print_pattern_into(paths, pattern, k, pedestal):
    """Multiply the LIGHT above the pedestal by pattern^k in place (uint16,
    clipped) — the sensor's response acts on photons, not on the bias."""
    import astropy.io.fits as pyfits
    for p in paths:
        with pyfits.open(p, mode="update", mmap=False) as f:
            light = f[0].data.astype(np.float64) - pedestal
            f[0].data = np.clip(pedestal + light * (1.0 + k * (pattern - 1.0)), 0, 65535).astype(np.uint16)
            f.flush()


def _sensor_pattern(shape, seed=5, amp=0.03):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal(shape), 60)
    return 1.0 + amp * (f - f.mean()) / f.std()


def _identify_orion_with_darks(pattern=None):
    """Real Orion Pane 1 subs + synthetic darks in the library, identified
    by an empty-cart graph run. Returns (stack, dark_stack, sub_paths)."""
    from compute_graph import build_compute_graph
    from dependency_graph import run_graph

    input_dir = context.input_dirs[0]
    copy_testdata_frames(target_name="Orion Pane 1", dest_dir=input_dir, add_noise=False,
                         start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43))
    subs = sorted(str(p) for p in Path(input_dir).glob("Light_Orion*.fit"))
    pedestal = _pedestal(subs[0])
    if pattern is not None:
        _print_pattern_into(subs, pattern, K_TRUE, pedestal)
    # The synthetic darks sit at the real pedestal, so the mock master dark
    # (the first dark frame, +0.5 ADU of hash noise) removes it in the fit.
    make_synthetic_frames(prefix="dark", number=5, image_type="Dark", fill_value=pedestal,
                          instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
                          temperature=-15, dest_dir=input_dir)
    run_graph(build_compute_graph(), debug=True)
    assert len(context.stacks) == 1 and len(context.calibration_stacks) == 1
    stack = next(iter(context.stacks.values())).get()
    dark = next(iter(context.calibration_stacks.values()))
    assert dark.image_type == "dark"
    return stack, dark, subs


def _config():
    """The ProcessingConfig the graph builder hands the pipelines."""
    from cache.context_ref import ContextRef
    from decisions.settings import ProcessingConfig
    settings = context.settings
    return ProcessingConfig(
        calibration_stacks=ContextRef.store_large(context.calibration_stacks or {}),
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )


def _write_star_flat(pattern, dest, stack) -> ImageMetadata:
    """The group's star flat as `build_star_flat` writes it, and its
    ImageMetadata as the runtime hands it to a consumer."""
    from stacks.starflat import write_master_flat
    write_master_flat(pattern.astype(np.float32), dest, {
        "IMAGETYP": "Master Flat", "FILTER": "H", "INSTRUME": "ZWO ASI6200MM Pro",
        "FOCALLEN": 351.0, "XBINNING": 1, "YBINNING": 1, "EXPOSURE": 1.0, "EXPTIME": 1.0,
        "STBLHASH": STAR_FLAT_ID, "SFLATSUB": 0, "SFLATCEL": 256,
    })
    return ImageMetadata(content_hash="starflat-content", stable_hash=STAR_FLAT_ID, filename=Path(dest),
                         header={"IMAGETYP": "Master Flat", "FILTER": "H", "STBLHASH": STAR_FLAT_ID},
                         width_pixels=stack.width_pixels, height_pixels=stack.height_pixels,
                         channels=1, dtype=np.dtype("float32"), plate_solution=None)


def _star_flat_stack(stack, subs):
    """A minted-looking StarFlatStack for the light stack's group, so
    `resolve_calibration` picks it as the flat."""
    import dataclasses
    from frozendict import frozendict
    from hashing import hash_list
    from metadata.star_photometry import SCHEMA
    from stacks.run import RunsDict
    from stacks.star_flat import StarFlatStack
    from units import Celsius, Seconds
    return StarFlatStack(
        content_hash=hash_list(sorted(subs)), image_type="flat", camera=stack.camera, filter=stack.filter,
        mean_temperature=Celsius(-15.0), exposure=Seconds(1.0), integration=Seconds(0.0),
        width_pixels=stack.width_pixels, height_pixels=stack.height_pixels, stable_hash=STAR_FLAT_ID,
        oldest_date=stack.oldest_date, newest_date=stack.newest_date, state=frozendict(), contents=RunsDict(),
        camera_id_hash=stack.camera_id_hash, focal_length=351.0, member_files=tuple(subs),
        photometry_schema=SCHEMA)


def _ops(graph):
    from dependency_graph.trace_paths import _op_name_for_node
    return {n: _op_name_for_node(n) for n in graph}


@pytest.mark.exhaustive
def test_calibrate_traces_the_scaled_star_flat_on_a_real_stack(test_context):
    """With the group's star flat in the calibration stacks, `calibrate()` on
    the real Orion stack traces build_star_flat -> scale_star_flat ->
    stack_files, and the fit node gets the dark master."""
    from dependency_graph.tracing import trace
    from stacks.calibration_selection import calibrate
    from stacks.stacking.pixinsight import stack_files

    stack, dark, subs = _identify_orion_with_darks()
    sf = _star_flat_stack(stack, subs)
    context.calibration_stacks = {**context.calibration_stacks, sf.stable_hash: sf}
    config = _config()

    def pipeline(stack, config):
        return stack_files(stack, config, *calibrate(stack, config))

    graph = trace(pipeline, stack, config)
    ops = _ops(graph)
    by_op = {}
    for n, name in ops.items():
        by_op.setdefault(name, []).append(n)
    assert set(by_op) >= {"build_star_flat", "scale_star_flat", "stack_files"}, set(by_op)
    scale = by_op["scale_star_flat"][0]
    scale_deps = {ops[d] for d in graph[scale]}
    assert scale_deps == {"build_star_flat", "stack_files"}, scale_deps   # the flat, and the dark's master
    light = [n for n in by_op["stack_files"] if scale in graph[n]]
    assert len(light) == 1, "the light stack's stack_files takes the SCALED flat, not the group's"
    assert not any(by_op["build_star_flat"][0] in graph[n] for n in by_op["stack_files"]), (
        "no stack_files may divide by the unscaled group flat")


@pytest.mark.exhaustive
def test_scaled_star_flat_runs_into_stack_files_on_real_frames(test_context, tmp_path):
    """The node body and mock WBPP on the real subs with a known printed
    gain: k ~ 1.3 in the scaled flat's sidecar, the group's STBLHASH carried
    into the file WBPP matches on, the master produced, and the ephemeral
    scaled flat reclaimed once the graph that consumed it completes."""
    from dependency_graph import run_graph
    from dependency_graph.trace_paths import resolve_output_paths
    from dependency_graph.tracing import trace
    from file_io.generic import read_header
    from file_io.sidecar import get_section
    from stacks.calibration_selection import calibrate
    from stacks.star_flat import GAIN_SECTION, gain_args, scale_star_flat
    from stacks.stacking.pixinsight import stack_files

    pattern = _sensor_pattern((6388, 9576))
    stack, dark, subs = _identify_orion_with_darks(pattern)
    flat_meta = _write_star_flat(pattern, tmp_path / "group_flat.xisf", stack)
    config = _config()

    # Phase 1: the scale node as a graph TERMINAL, so its output survives to be read.
    def scale_only(stack, config):
        dark_master = calibrate(stack, config)[0]
        return scale_star_flat(*gain_args(stack, flat_meta, dark_master))

    g1 = trace(scale_only, stack, config)
    run_graph(g1, debug=True)
    ops1 = _ops(g1)
    scaled_path = next(p for n, p in resolve_output_paths(g1).items() if ops1[n] == "scale_star_flat")
    assert Path(scaled_path).exists()
    fit = get_section(Path(scaled_path), GAIN_SECTION)
    assert fit["n_used"] == 3 and fit["dark"] is True and "reason" not in fit, fit
    assert abs(fit["k"] - K_TRUE) < 0.12, fit
    hdr = read_header(Path(scaled_path))
    assert str(hdr["STBLHASH"]) == STAR_FLAT_ID and hdr["IMAGETYP"] == "master flat"   # normalised on read

    # Phase 2: the same node INTERIOR to stack_files, inside a graph that is a
    # CHILD of the one run_graph is given — the shape `run_pipeline` gives every
    # cart item's pipeline. The runtime reclaims a graph's ephemerals when that
    # graph concludes as a child (a top-level graph never does), sparing its
    # terminal; `@workflow_group` is UI metadata only and creates no child.
    import ray
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import record

    @ray.remote(num_cpus=0.1)
    def pipeline_graph(stack, config, flat, dry_run=False):
        del dry_run
        from dependency_graph.tracing import trace as _trace
        from stacks.calibration_selection import calibrate as _calibrate
        from stacks.star_flat import gain_args as _gain_args, scale_star_flat as _scale
        from stacks.stacking.pixinsight import stack_files as _stack_files

        def _body():
            dark_master = _calibrate(stack, config)[0]
            scaled = _scale(*_gain_args(stack, flat, dark_master))
            return _stack_files(stack, config, dark_master, scaled)
        return Graph(_trace(_body))

    def top(stack, config):
        return record(pipeline_graph, stack, config, flat_meta)

    g2 = trace(top, stack, config)
    run_graph(g2, debug=True)
    master = next(p for p in Path(context.output_dir).rglob("stack_files/*.xisf")
                  if str(read_header(p).get("STBLHASH")) == stack.stable_hash)
    assert master.exists(), "mock WBPP produced no master for the light stack"
    assert not Path(scaled_path).exists(), (
        "the ephemeral scaled flat must be reclaimed once the pipeline graph that consumed it concluded")
