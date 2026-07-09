"""max_calls=1 worker isolation (real Ray). A profiled `auto_memory` op must run
in a FRESH worker so the absolute RSS that `measure_performance` records reflects
only that op — never a big op that previously ran on the same reused worker. This
is the live proof behind the decorator wiring asserted in `test_memory_estimate.py`.

Heavy (spins the Ray test cluster + allocates GBs) -> marked `serial` so it runs in
the serial phase, not a parallel shard. Fresh cache per test (make_test_context
calls context.clear()), so distinct args below always execute (never cache-hit an
old pid)."""
import os

import numpy as np
import psutil
import pytest
import ray

from decorators import node

_GB = 1024 ** 3


@node(auto_memory=True)
def _bloat_worker(n_bytes: int) -> tuple:
    # Allocate + hold a large array so the worker's RSS actually grows; sample it
    # while it's still resident. np.ones touches every page.
    big = np.ones(n_bytes // 8, dtype=np.float64)
    rss = psutil.Process(os.getpid()).memory_info().rss
    return (os.getpid(), int(rss), float(big[0]))   # keep `big` alive past the sample


@node(auto_memory=True)
def _small_worker(tag: int) -> tuple:
    # Allocates ~nothing; on a fresh worker its RSS is just the import baseline.
    # `tag` only varies the call_hash so repeated calls actually re-execute.
    return (os.getpid(), int(psutil.Process(os.getpid()).memory_info().rss))


@pytest.mark.serial
def test_profiled_op_isolated_from_prior_bloat(test_context):
    # Both are profiled auto_memory ops (names not in _FIXED_MEMORY_BY_OP), so
    # each gets max_calls=1: the 3 GB bloat worker EXITS after its task, and the
    # tiny op lands on a FRESH worker that cannot carry the 3 GB. Without the fix
    # the small op could reuse the bloated worker and record ~3 GB — the exact
    # pollution this change eliminates.
    bloat_pid, bloat_rss, _ = ray.get(_bloat_worker.remote(3 * _GB))
    assert bloat_rss > int(2.5 * _GB), f"bloat not resident: {bloat_rss / _GB:.2f} GB"

    small_pid, small_rss = ray.get(_small_worker.remote(0))
    assert small_pid != bloat_pid, "small reused the bloated worker (max_calls=1 not applied)"
    assert small_rss < bloat_rss - _GB, (
        f"small inherited the bloat footprint: "
        f"small={small_rss / _GB:.2f} GB, bloat={bloat_rss / _GB:.2f} GB"
    )


@pytest.mark.serial
def test_profiled_op_gets_a_fresh_worker_each_call(test_context):
    # max_calls=1 => the worker exits after each task => a distinct pid per call.
    pids = [ray.get(_small_worker.remote(i))[0] for i in range(3)]
    assert len(set(pids)) == 3, f"expected 3 fresh workers, got pids={pids}"
