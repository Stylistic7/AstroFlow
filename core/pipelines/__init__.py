"""User-defined pipelines: discovery, decorators, registry.

User pipelines live as `.py` files in `core/pipelines/` (built-in) and
`~/AstroFlow/pipelines/` (user). An `@pipeline("Display Name")` decorator
on the entry function registers it under the given name. The dev setting
`DevelopmentSettings.pipeline_name` selects which one is dispatched
when a `"mosaic"` cart entry is processed.

See `~/.claude/plans/splendid-munching-rainbow.md` for the design.
"""
