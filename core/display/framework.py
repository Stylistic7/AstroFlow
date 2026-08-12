"""Display-referred finish framework — the F2 half of the scene→display seam.

A *finish* takes a scene-linear image (fetched from the pipeline by TAP NAME) and
maps it to display space. It is a composition of per-pixel OPS. Each op:
  * DECLARES its params (name, range, default, step, label) — the single source of
    truth from which the 3×3 grid's axes and the sliders are generated. No knob is
    hand-listed on the frontend; it is *extracted* from the Python definition.
  * carries a **numpy** backend (bakes the exact archival deliverable) AND a
    **WGSL** snippet (drives the real-time grid) — same function, two evaluators,
    one definition so they can't drift.

See the project_scene_display_seam memory for the whole architecture. This is the
minimal framework; ops live in display/finishes.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Param:
    """One knob, with the hints the 3×3 grid needs to advance it sensibly.

    `scale` says HOW to nudge the axis:
      * "linear" — one grid step is ± `step` (an absolute amount). Right for a
        knob whose sensible increment is constant across its range (e.g. a
        saturation in [0, 3]).
      * "log" — one grid step is × (1 ± `step`), i.e. `step` is a FRACTION of the
        current value. Right for anything whose sensible increment scales with
        its magnitude — a black point sitting at 0.002 and a stretch at 0.25 both
        move by the same %, so a fixed step can't be small for both.
    `step` is that per-axis grid step (absolute for linear, a fraction for log);
    the grid's Spread slider is a global multiplier on top. `lo`/`hi` clamp the
    value; the slider is drawn with its own fine granularity. `label` is the short
    tag on a grid cell."""
    name: str
    default: float
    lo: float
    hi: float
    step: float
    label: str
    scale: str = "linear"


@dataclass
class DisplayOp:
    """A per-pixel op. `numpy_fn(rgb, values) -> rgb` bakes; `wgsl` is a statement
    block operating on a mutable `var c: vec3<f32>` and reading each param by name
    (the generated shader binds those names as f32 uniforms). `wgsl_helpers` are
    function defs emitted once (e.g. the MTF helper)."""
    name: str
    params: list[Param]
    numpy_fn: Callable[[np.ndarray, dict], np.ndarray]
    wgsl: str
    wgsl_helpers: str = ""


@dataclass
class DisplayFinish:
    """A named composition of ops with a declared TAP-NAME input contract.

    MULTI-SOURCE finishes (len(inputs) > 1) add a COMPOSE stage: `compose` is a
    WGSL statement block that runs before the op chain with `src0..srcN`
    (vec4 samples of each input tap, in `inputs` order) in scope. Its contract:
    declare `var c: vec3<f32>` (the chain input), and optionally write `aux`
    (an f32 local, pre-declared 0.0) — the side channel a later op consumes
    (the luminance for an LRGB lab op, say). All op snippets share ONE function
    body, so `aux` is just a local — no new plumbing. `compose_params` are the
    compose stage's own knobs; `compose_np(sources, v)` is its numpy twin
    (returns the chain rgb; stashes aux as `v["__aux"]`)."""
    name: str
    inputs: list[str]            # tap names it consumes (the F1↔F2 interface)
    ops: list[DisplayOp] = field(default_factory=list)
    compose: str = ""
    compose_params: list[Param] = field(default_factory=list)
    compose_np: "Callable[[list, dict], np.ndarray] | None" = None

    def params(self) -> list[Param]:
        """Every knob across the pipeline, in order — THE grid axes / sliders.
        Extracted, never hand-maintained. Compose knobs come first (they run
        first)."""
        return list(self.compose_params) + [p for op in self.ops for p in op.params]

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params()}

    def bake(self, sources: "list[np.ndarray] | np.ndarray",
             values: dict | None = None) -> np.ndarray:
        """Exact numpy evaluation — the archival deliverable. `sources` is one
        array per input tap (a bare array means single-source); `values`
        overrides defaults per param."""
        if isinstance(sources, np.ndarray):
            sources = [sources]
        v = self.defaults()
        if values:
            v.update(values)
        if self.compose_np is not None:
            rgb = self.compose_np(sources, v)
        else:
            rgb = sources[0]
        for op in self.ops:
            rgb = op.numpy_fn(rgb, v)
        return rgb

    def to_wgsl(self) -> str:
        """Assemble the real-time shader chunk from the ops' snippets. Emits a
        self-contained trio the frontend splices verbatim into its tile shader:
          * `struct DisplayParams` — one f32 field per param, IN param order, so
            the frontend packs its uniform buffer by index with no name lookup.
          * the ops' `wgsl_helpers` (mtf, etc.).
          * `fn displayFinish(src0.., P)` — one vec4 arg per input tap; aliases
            each param to a bare `let` so the op snippets read plain names while
            the values come from the bound uniform. Same names the numpy backend
            keys on, one definition, so bake and shader can't drift. The
            frontend generates the matching fs_main from `spec.inputs`."""
        ps = self.params()
        struct = "\n".join(f"    {p.name}: f32," for p in ps)
        aliases = "\n".join(f"    let {p.name} = P.{p.name};" for p in ps)
        helpers = "\n".join(op.wgsl_helpers for op in self.ops if op.wgsl_helpers)
        args = ", ".join(f"src{i}: vec4<f32>" for i in range(len(self.inputs)))
        compose = self.compose or "var c: vec3<f32> = src0.rgb;"
        body = "\n    ".join(op.wgsl for op in self.ops)
        return (
            f"struct DisplayParams {{\n{struct}\n}}\n\n"
            f"{helpers}\n\n"
            f"fn displayFinish({args}, P: DisplayParams) -> vec3<f32> {{\n"
            f"{aliases}\n"
            f"    var aux: f32 = 0.0;\n"
            f"    {compose}\n    {body}\n    return c;\n}}"
        )

    def spec(self) -> dict:
        """JSON descriptor the frontend consumes: the input contract, the extracted
        params (→ axes + sliders), and the assembled WGSL (→ shader)."""
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "params": [asdict(p) for p in self.params()],
            "wgsl": self.to_wgsl(),
        }
