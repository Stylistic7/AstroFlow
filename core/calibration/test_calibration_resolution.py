"""Stronger tests around calibration-source selection by `resolve_calibration`.

Each test exercises a real selection path:

  1. Directory affinity finds an external master in the lights' ancestor tree
     → `FilenameDependency` to the master file.
  2. No external master in the tree → fall through to chronological
     matching on the raw calibration stacks AstroFlow discovered →
     `StackIdDependency` to the calibration stack hash.
  3. User override beats both directory affinity and chronological default.
  4. Per-kind independence: bias from the directory tree, dark / flat from
     the chronological matcher (or vice versa).
  5. Stale cached defaults: if the per-object decision cache still holds a
     stack hash that's no longer in `calibration_stacks` (because it got
     filtered out at discovery time, e.g. a sub-min-frames straggler), the
     resolver doesn't blow up — it silently skips the missing calibration
     for that kind.
"""
from __future__ import annotations

from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np
import pytest
from frozendict import frozendict

from compute_graph.compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from decisions.decisions import (
    DecisionKey, DecisionPoint, DecisionType, ObjectType, _decisions_key,
)
from decisions.helpers import calibration_override
from decisions.resolve import resolve_calibration
from stacks.composite_stack import Stack
from stacks.types import FilenameDependency, StackIdDependency
from testing import make_synthetic_frames


def _make_master_fits(path: Path, image_type: str, *, exposure: float | None = None,
                     filter: str | None = None, temperature: float = -20.0,
                     camera: str = "ZWO ASI6200MM Pro") -> Path:
    """Write a tiny FITS file with `BITPIX=-32` (the master signal).

    The strengthened `is_master()` will recognize this as a master via the
    BITPIX check even though the IMAGETYP is just "Bias" / "Dark" / "Flat".
    Filename is set up to start with `Master*` per the user's convention so
    `find_image_files` skips it for raw-frame discovery.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = pyfits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    hdu.header["IMAGETYP"] = f"Master {image_type.capitalize()}"
    hdu.header["INSTRUME"] = camera
    hdu.header["CCD-TEMP"] = temperature
    if exposure is not None:
        hdu.header["EXPOSURE"] = exposure
    if filter is not None:
        hdu.header["FILTER"] = filter
    hdu.writeto(path, overwrite=True)
    return path


def _resolve_first_light(stack: Stack, overrides=None):
    """Run resolve_calibration on the given light stack and return the
    resolved Run's deps dict."""
    overrides = overrides if overrides is not None else frozendict()
    resolved = resolve_calibration(stack, context.calibration_stacks, overrides)
    run = next(iter(resolved.contents.values()))
    return dict(run.dependencies)


def _make_lights_with_raw_cals(target_dir: Path, *, n_lights: int = 3,
                                temperature: float = -20.0):
    """Drop synthetic light frames and matching raw bias / dark / flat frames
    into a target directory. Returns nothing — the caller runs
    `build_compute_graph` to ingest them.
    """
    lights_dir = target_dir / "Lights"
    cals_dir = target_dir / "RawCals"
    lights_dir.mkdir(parents=True, exist_ok=True)
    cals_dir.mkdir(parents=True, exist_ok=True)

    make_synthetic_frames(
        number=n_lights, image_type="Light",
        instrument="ZWO ASI6200MM Pro",
        temperature=temperature, exposure=120, filter="G",
        dest_dir=str(lights_dir),
    )
    make_synthetic_frames(
        number=5, image_type="Bias",
        instrument="ZWO ASI6200MM Pro",
        temperature=temperature,
        dest_dir=str(cals_dir),
    )
    make_synthetic_frames(
        number=5, image_type="Dark",
        instrument="ZWO ASI6200MM Pro",
        temperature=temperature, exposure=120,
        dest_dir=str(cals_dir),
    )
    make_synthetic_frames(
        number=5, image_type="Flat",
        instrument="ZWO ASI6200MM Pro",
        temperature=temperature, exposure=1, filter="G",
        dest_dir=str(cals_dir),
    )


def _the_one_light_stack() -> Stack:
    light_stacks = [
        ref.get() for ref in context.stacks.values()
        if isinstance(ref.get(), Stack) and ref.get().image_type == "light"
    ]
    assert len(light_stacks) == 1, f"Expected 1 light stack, got {len(light_stacks)}"
    return light_stacks[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.exhaustive
def test_directory_affinity_picks_external_master(test_context) -> None:
    """When an external master sits in the lights' ancestor tree, it wins
    over the chronologically-nearest AstroFlow-stacked calibration stack."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)

    # Drop an external bias master in a sibling Calibration Masters/ — the
    # canonical layout that directory-affinity is meant to recognize.
    masters_dir = target_dir / "Calibration Masters"
    master_path = _make_master_fits(
        masters_dir / "MasterBias_-20c.fit",
        image_type="bias", temperature=-20.0,
    )

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    assert "bias" in deps
    assert isinstance(deps["bias"], FilenameDependency), (
        f"Expected directory-affinity FilenameDependency, got {deps['bias']!r}"
    )
    assert deps["bias"].filename == str(master_path.resolve())


@pytest.mark.exhaustive
def test_falls_back_to_chronological_when_no_external_master(test_context) -> None:
    """No external master in the lights' tree → chronological match wins."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    assert "bias" in deps
    assert isinstance(deps["bias"], StackIdDependency), (
        f"Expected chronological StackIdDependency, got {deps['bias']!r}"
    )
    # The chronological choice must reference an existing calibration stack.
    assert deps["bias"].stack_id in context.calibration_stacks


@pytest.mark.exhaustive
def test_override_beats_directory_affinity(test_context) -> None:
    """User override pins the calibration even when an external master
    would otherwise win."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    _make_master_fits(
        target_dir / "Calibration Masters" / "MasterBias_-20c.fit",
        image_type="bias", temperature=-20.0,
    )

    run_graph(build_compute_graph())
    light_stack = _the_one_light_stack()

    # Pick a chronological-AstroFlow-stacked bias; pin it via override.
    raw_bias_id = next(
        sid for sid, s in context.calibration_stacks.items()
        if s.image_type == "bias"
    )
    run_hash = next(iter(light_stack.contents.keys()))
    overrides = frozendict(
        calibration_override("bias", light_stack.stable_hash, run_hash, raw_bias_id)
    )

    deps = _resolve_first_light(light_stack, overrides=overrides)
    assert deps["bias"] == StackIdDependency(stack_id=raw_bias_id)


@pytest.mark.exhaustive
def test_per_kind_independence(test_context) -> None:
    """Directory affinity provides bias only; dark and flat fall through
    to chronological matching independently."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    bias_master = _make_master_fits(
        target_dir / "Calibration Masters" / "MasterBias_-20c.fit",
        image_type="bias", temperature=-20.0,
    )
    # No master dark or master flat in the tree.

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    # Bias is from the directory tree.
    assert isinstance(deps["bias"], FilenameDependency)
    assert deps["bias"].filename == str(bias_master.resolve())
    # Dark and flat are AstroFlow-stacked, found via chronological matching.
    assert isinstance(deps["dark"], StackIdDependency)
    assert isinstance(deps["flat"], StackIdDependency)
    assert deps["dark"].stack_id in context.calibration_stacks
    assert deps["flat"].stack_id in context.calibration_stacks


@pytest.mark.exhaustive
def test_resolve_handles_stale_cached_choice(test_context) -> None:
    """If the per-object decision cache still holds a stack hash that's no
    longer in calibration_stacks (e.g. one filtered out at discovery), the
    resolver must not crash. The bias dep is silently dropped; downstream
    stack_files runs without that calibration kind rather than KeyError'ing
    inside `_setup_run`.
    """
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    run_graph(build_compute_graph())

    light_stack = _the_one_light_stack()
    run_hash = next(iter(light_stack.contents.keys()))

    # Manually populate the per-object decision cache with a bias choice
    # pointing to a hash that doesn't exist in calibration_stacks. This is
    # what would happen if a previous session resolved a now-filtered-out
    # 1-frame straggler stack.
    stale_hash = "deadbeefcafef00d"
    assert stale_hash not in context.calibration_stacks

    bias_key = DecisionKey.calibration("bias", light_stack.stable_hash, run_hash)
    cached_dp = DecisionPoint(
        key=bias_key,
        decision_type=DecisionType.CALIBRATION_BIAS,
        default_choice=stale_hash,
        title="Calibration: bias",
        description="Stale cached default for test",
        object_type=ObjectType.STACK,
        object_id=light_stack.stable_hash,
        subject_id=run_hash,
    )
    context.or_equals(
        _decisions_key(ObjectType.STACK, light_stack.stable_hash),
        {bias_key: cached_dp},
    )

    deps = _resolve_first_light(light_stack)
    # The stale bias choice was dropped — no bias dep on the run.
    assert "bias" not in deps
    # Dark and flat were resolved fresh and reference real calibration
    # stacks, demonstrating the failure is per-kind, not whole-run.
    assert isinstance(deps["dark"], StackIdDependency)
    assert deps["dark"].stack_id in context.calibration_stacks
    assert isinstance(deps["flat"], StackIdDependency)
    assert deps["flat"].stack_id in context.calibration_stacks
