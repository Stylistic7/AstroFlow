"""Re-export of the global context for use in user pipelines.

The context carries `calibration_stacks`, `stacks`, `targets`, `mosaics`,
`input_dirs`, and similar process-wide state. Read it inside pipeline
bodies via `context.calibration_stacks` etc.
"""
from context import local_context as context

__all__ = ["context"]
