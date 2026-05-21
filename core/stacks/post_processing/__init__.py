"""Stack-level post-processing (plate-solve + flux-calibrate + gradient).

`from stacks.post_processing import post_process_stack` re-exports the
PixInsight variant; pipeline code should pick a backend explicitly via
`from stacks.post_processing.pixinsight import post_process_stack` or
`from stacks.post_processing.oss import post_process_stack`.
"""
from stacks.post_processing.pixinsight import *  # noqa: F401, F403
