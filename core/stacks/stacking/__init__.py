"""Stack-level stacking. PixInsight default; OSS available.

Existing `from stacks.stacking import stack_files` keeps working via
the re-export from `.pixinsight`. Pipeline code should import the
backend directly (`from stacks.stacking.pixinsight import stack_files`
or `from stacks.stacking.oss import stack_files`).
"""
from stacks.stacking.pixinsight import *  # noqa: F401, F403
