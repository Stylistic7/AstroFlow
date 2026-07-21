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
    """A named composition of ops with a declared TAP-NAME input contract."""
    name: str
    inputs: list[str]            # tap names it consumes (the F1↔F2 interface)
    ops: list[DisplayOp] = field(default_factory=list)

    def params(self) -> list[Param]:
        """Every knob across the pipeline, in order — THE grid axes / sliders.
        Extracted, never hand-maintained."""
        return [p for op in self.ops for p in op.params]

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params()}

    def bake(self, rgb: np.ndarray, values: dict | None = None) -> np.ndarray:
        """Exact numpy evaluation — the archival deliverable. `values` overrides
        defaults per param."""
        v = self.defaults()
        if values:
            v.update(values)
        for op in self.ops:
            rgb = op.numpy_fn(rgb, v)
        return rgb

    def to_wgsl(self) -> str:
        """Assemble the real-time shader chunk from the ops' snippets. Emits a
        self-contained trio the frontend splices verbatim into its tile shader:
          * `struct DisplayParams` — one f32 field per param, IN param order, so
            the frontend packs its uniform buffer by index with no name lookup.
          * the ops' `wgsl_helpers` (mtf, etc.).
          * `fn displayFinish(cIn, P)` — aliases each param to a bare `let` so the
            op snippets read plain names (nice authoring) while the values come
            from the bound uniform. Same names the numpy backend keys on, one
            definition, so bake and shader can't drift."""
        ps = self.params()
        struct = "\n".join(f"    {p.name}: f32," for p in ps)
        aliases = "\n".join(f"    let {p.name} = P.{p.name};" for p in ps)
        helpers = "\n".join(op.wgsl_helpers for op in self.ops if op.wgsl_helpers)
        body = "\n    ".join(op.wgsl for op in self.ops)
        return (
            f"struct DisplayParams {{\n{struct}\n}}\n\n"
            f"{helpers}\n\n"
            f"fn displayFinish(cIn: vec3<f32>, P: DisplayParams) -> vec3<f32> {{\n"
            f"    var c = cIn;\n{aliases}\n    {body}\n    return c;\n}}"
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
