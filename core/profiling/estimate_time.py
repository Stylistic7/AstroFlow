import ray
import numpy as np

from context import context
from beartype import beartype


@beartype
def estimate_time(fn_name: str, remaining: int = 1) -> float | None:
    """Estimate seconds to complete `remaining` tasks of function `fn_name`.

    Uses historical wall_time_s from the profile data collected by
    @measure_performance, combined with per-node CPU counts to model
    parallel throughput across the Ray cluster.
    """
    if remaining <= 0:
        return 0.0

    # Collect all profile samples for this function across all nodes.
    all_profiles = context.get_matching(("profile", fn_name))
    if not all_profiles:
        return None

    # Flatten all wall_time_s samples.
    wall_times = []
    for key, samples in all_profiles.items():
        for sample in samples:
            if "wall_time_s" in sample:
                wall_times.append(sample["wall_time_s"])

    if not wall_times:
        return None

    mean_task_time = np.mean(wall_times)
    if mean_task_time <= 0:
        return None

    # Sum throughput across all alive nodes.
    # Each node can run min(remaining, num_cpus) tasks in parallel.
    total_throughput = 0.0
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        num_cpus = node.get("Resources", {}).get("CPU", 0)
        if num_cpus > 0:
            total_throughput += num_cpus / mean_task_time  # tasks per second

    if total_throughput <= 0:
        return None

    return remaining / total_throughput
