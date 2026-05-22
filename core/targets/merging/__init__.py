"""Target-level merging: HDR-combine per-stack masters into one per-
(target, filter) channel master. PixInsight default; OSS available.

`from targets.merging import merge_target_channel` re-exports the
PixInsight variant; pipeline code should pick a backend explicitly via
`from targets.merging.pixinsight import merge_target_channel` or
`from targets.merging.oss import merge_target_channel_oss`.
"""
from targets.merging.pixinsight import *  # noqa: F401, F403
# Underscore-prefixed re-exports needed by other modules.
from targets.merging.pixinsight import _merge_target_channel_narrow  # noqa: F401
