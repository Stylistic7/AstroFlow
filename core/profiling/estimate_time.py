import ray
import numpy as np

from context import context
from beartype import beartype
from hashing import hash_source_code


@beartype
def estimate_time(
    fn_name: str,
    remaining: int = 1,
    quality: str | None = None,
    fn_hash: str | None = None,
) -> float | None:
    """Estimate seconds to complete `remaining` tasks of function `fn_name`.

    Uses historical wall_time_s from the profile data collected by
    @measure_performance, combined with the Ray resources the function
    requires (captured from @node's ray_opts) to model correct parallel
    throughput across the Ray cluster.
    """
    if remaining <= 0:
        return 0.0

    if fn_hash is None:
        return None

    quality_value = str(quality) if quality is not None else None
    key = ("profile", fn_hash, fn_name, quality_value)
    samples = context.get(key, [])

    if not samples:
        return None

    wall_times: list[float] = []
    ray_opts: dict | None = None
    for sample in samples:
        if "wall_time_s" in sample:
            wall_times.append(sample["wall_time_s"])
        if ray_opts is None and "ray_opts" in sample:
            ray_opts = sample["ray_opts"]

    if not wall_times:
        return None

    mean_task_time = float(np.mean(wall_times))
    if mean_task_time <= 0:
        return None

    num_cpus_per_task = float((ray_opts or {}).get("num_cpus", 1) or 1)
    custom_resources = (ray_opts or {}).get("resources") or {}

    total_concurrent = 0.0
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        node_res = node.get("Resources", {})
        node_slots = node_res.get("CPU", 0) / num_cpus_per_task
        for resource, amount in custom_resources.items():
            if amount > 0:
                available = node_res.get(resource, 0)
                node_slots = min(node_slots, available / amount)
        total_concurrent += node_slots

    if total_concurrent <= 0:
        return None

    throughput = total_concurrent / mean_task_time
    return remaining / throughput
