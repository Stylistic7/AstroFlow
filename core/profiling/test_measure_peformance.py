import ray
from pathlib import Path
from PIL import Image
import tempfile

from context import context, get_cache
from cache.cached_decorator import make_cached_decorator, Injected
from hashing import hash_list
from decorators import measure_performance
from testing.sidecar_hasher import SidecarHasher


def _compute_content_hash(serialized_func: bytes, *args, **kwargs) -> str:
    kwargs_list = sorted(kwargs.items())
    return hash_list([serialized_func, *args, *kwargs_list])


sidecar_hasher = SidecarHasher()
task_with_context = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=sidecar_hasher,
    call_hasher=_compute_content_hash,
)


@task_with_context.remote
@measure_performance
def make_large_image_remote(
    shape: tuple[int, ...], output_dir: Injected[Path]
) -> Path:
    context.num_make_large_image_runs = context.num_make_large_image_runs + 1
    filename = output_dir.value / "foo.png"
    img = Image.new("L", shape, color=128)
    img.save(filename)
    print(f"Wrote image to filename {filename}")
    return filename


def test_decorator(test_context):
    from ray.util import get_node_ip_address

    context.num_make_large_image_runs = 0
    fn_name = "make_large_image_remote"
    node_ip = get_node_ip_address()
    perf_key = ("profile", node_ip, fn_name)

    context[perf_key] = []

    with tempfile.TemporaryDirectory() as output_dir:
        injected_context = {"output_dir": Path(output_dir)}
        ray.get(
            make_large_image_remote.remote(
                (50, 50), injected_context=injected_context
            )
        )

    history = context.get(perf_key, [])
    assert len(history) > 0
    assert isinstance(history[0], dict)
