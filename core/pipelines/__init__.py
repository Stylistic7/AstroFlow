"""User-defined pipelines: discovery, decorators, registry.

User pipelines live as `.py` files in `core/pipelines/` (built-in) and
`~/AstroFlow/pipelines/` (user). An `@pipeline("Display Name")` decorator
on the entry function registers it under the given name. The display
name is selected per-shopping-cart-entry via the 4th tuple element:
`("mosaic", mosaic_id, mode_str, pipeline_display_name)`.

Bundled implementation names (`"PixInsight"`, `"OSS"`, `"AstroFlow"`)
and registered pipeline names share the same dropdown surface — the
backend dispatcher in `compute_graph.build_main_processing_graph`
resolves each kind via an `Implementation` enum lookup with a
`get_pipeline(...)` fallback.

Pipeline selection is per-entry (not global) so the same mosaic can
appear in the cart multiple times with different pipelines for
side-by-side comparison; the chosen name never flows into any cached
@node op's arg hash.

See `~/.claude/plans/splendid-munching-rainbow.md` for the design.
"""
