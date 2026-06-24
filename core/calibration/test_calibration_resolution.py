"""Stronger tests around calibration-source selection by `resolve_calibration`.

Calibration topology is Landmann §7.1 Case 2 (matched-exposure darks): a
LIGHT stack resolves exactly `{dark, flat}` and NEVER a master bias — the
master dark already carries the bias signal, so a separate MB subtraction
is the §8.2 anti-pattern (`decisions/resolve.py:_CALIBRATIONS_FOR_IMAGE_TYPE`).
So these tests drive the selection mechanisms through the `dark` (and `flat`)
light dependencies rather than `bias`.

Each test exercises a real selection path:

  1. Directory affinity finds an external master dark in the lights' ancestor
     tree → `FilenameDependency` to the master file.
  2. No external master in the tree → fall through to chronological
     matching on the raw calibration stacks AstroFlow discovered →
     `StackIdDependency` to the calibration stack hash.
  3. User override beats both directory affinity and chronological default.
  4. Per-kind independence: dark from the directory tree, flat from the
     chronological matcher.
  5. Stale cached defaults: if the per-object decision cache still holds a
     stack hash that's no longer in `calibration_stacks` (because it got
     filtered out at discovery time, e.g. a sub-min-frames straggler), the
     resolver doesn't blow up — it resolves fresh to a real stack.
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

    # Drop an external master dark in a sibling Calibration Masters/ — the
    # canonical layout that directory-affinity is meant to recognize. Match
    # the lights' exposure (120 s) + temperature so dark affinity accepts it.
    masters_dir = target_dir / "Calibration Masters"
    master_path = _make_master_fits(
        masters_dir / "MasterDark_120s_-20c.fit",
        image_type="dark", exposure=120, temperature=-20.0,
    )

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    assert "dark" in deps
    assert isinstance(deps["dark"], FilenameDependency), (
        f"Expected directory-affinity FilenameDependency, got {deps['dark']!r}"
    )
    assert deps["dark"].filename == str(master_path.resolve())


@pytest.mark.exhaustive
def test_falls_back_to_chronological_when_no_external_master(test_context) -> None:
    """No external master in the lights' tree → chronological match wins, and
    a light resolves exactly the Case 2 set `{dark, flat}` (never bias)."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    # Case 2 topology guard: a light's calibration is dark + flat, no bias.
    assert set(deps) == {"dark", "flat"}, (
        f"Expected Case 2 light deps {{dark, flat}}, got {set(deps)}"
    )
    assert isinstance(deps["dark"], StackIdDependency), (
        f"Expected chronological StackIdDependency, got {deps['dark']!r}"
    )
    # The chronological choice must reference an existing calibration stack.
    assert deps["dark"].stack_id in context.calibration_stacks


@pytest.mark.exhaustive
def test_override_beats_directory_affinity(test_context) -> None:
    """User override pins the calibration even when an external master
    would otherwise win."""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    _make_master_fits(
        target_dir / "Calibration Masters" / "MasterDark_120s_-20c.fit",
        image_type="dark", exposure=120, temperature=-20.0,
    )

    run_graph(build_compute_graph())
    light_stack = _the_one_light_stack()

    # Pick a chronological-AstroFlow-stacked dark; pin it via override.
    raw_dark_id = next(
        sid for sid, s in context.calibration_stacks.items()
        if s.image_type == "dark"
    )
    run_hash = next(iter(light_stack.contents.keys()))
    overrides = frozendict(
        calibration_override("dark", light_stack.stable_hash, run_hash, raw_dark_id)
    )

    deps = _resolve_first_light(light_stack, overrides=overrides)
    assert deps["dark"] == StackIdDependency(stack_id=raw_dark_id)


@pytest.mark.exhaustive
def test_per_kind_independence(test_context) -> None:
    """Directory affinity provides the dark only; flat falls through to
    chronological matching independently. (Lights take no bias under Case 2.)"""
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    dark_master = _make_master_fits(
        target_dir / "Calibration Masters" / "MasterDark_120s_-20c.fit",
        image_type="dark", exposure=120, temperature=-20.0,
    )
    # No master flat in the tree.

    run_graph(build_compute_graph())

    deps = _resolve_first_light(_the_one_light_stack())
    # Case 2: a light never takes a bias dependency.
    assert "bias" not in deps
    # Dark is from the directory tree.
    assert isinstance(deps["dark"], FilenameDependency)
    assert deps["dark"].filename == str(dark_master.resolve())
    # Flat is AstroFlow-stacked, found via chronological matching.
    assert isinstance(deps["flat"], StackIdDependency)
    assert deps["flat"].stack_id in context.calibration_stacks


@pytest.mark.exhaustive
def test_resolve_handles_stale_cached_choice(test_context) -> None:
    """The per-object decision cache is never read as the source of truth
    (it exists only for the UI). So even if it holds a stack hash that's
    no longer in calibration_stacks (e.g. one filtered out at discovery),
    that stale entry cannot poison resolution at all: the dark is computed
    fresh and lands on a real calibration stack — never the stale hash,
    never KeyError'ing downstream in `_setup_run`.
    """
    target_dir = Path(context.input_dirs[0]) / "Cyg4"
    _make_lights_with_raw_cals(target_dir)
    run_graph(build_compute_graph())

    light_stack = _the_one_light_stack()
    run_hash = next(iter(light_stack.contents.keys()))

    # Manually populate the per-object decision cache with a dark choice
    # pointing to a hash that doesn't exist in calibration_stacks. This is
    # what would happen if a previous session resolved a now-filtered-out
    # 1-frame straggler stack.
    stale_hash = "deadbeefcafef00d"
    assert stale_hash not in context.calibration_stacks

    dark_key = DecisionKey.calibration("dark", light_stack.stable_hash, run_hash)
    cached_dp = DecisionPoint(
        key=dark_key,
        decision_type=DecisionType.CALIBRATION_DARK,
        default_choice=stale_hash,
        title="Calibration: dark",
        description="Stale cached default for test",
        object_type=ObjectType.STACK,
        object_id=light_stack.stable_hash,
        subject_id=run_hash,
    )
    context.or_equals(
        _decisions_key(ObjectType.STACK, light_stack.stable_hash),
        {dark_key: cached_dp},
    )

    deps = _resolve_first_light(light_stack)
    # The stale cached choice is ignored entirely (cache is never the
    # source of truth): the dark is resolved fresh to a REAL calibration
    # stack — not the stale hash, not dropped.
    assert isinstance(deps["dark"], StackIdDependency)
    assert deps["dark"].stack_id != stale_hash
    assert deps["dark"].stack_id in context.calibration_stacks
    # Flat likewise resolves fresh to a real calibration stack. (Case 2:
    # a light takes no bias dependency.)
    assert "bias" not in deps
    assert isinstance(deps["flat"], StackIdDependency)
    assert deps["flat"].stack_id in context.calibration_stacks
