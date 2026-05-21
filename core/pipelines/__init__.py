"""User-defined pipelines: discovery, decorators, registry.

User pipelines live as `.py` files in `core/pipelines/` (built-in) and
`~/AstroFlow/pipelines/` (user). `@pipeline("Display Name", processes=Cls)`
on an entry function registers it under that (name, class) pair; one
pipeline name can declare multiple entry points (one per Stack /
Target / Mosaic class it handles). The dropdown collapses them to one
user-visible name; the cart-branch dispatcher looks up the right
entry point by class.

Pipeline selection is per-cart-entry (not global), so the same mosaic
can appear in the cart multiple times with different pipelines for
side-by-side comparison; the pipeline name never flows into any
cached @node op's arg hash.

See `~/.claude/plans/splendid-munching-rainbow.md` for the design.
"""
