import ray
from pathlib import Path
from PIL import Image
import tempfile

from context import context, get_cache
from cache.cached_decorator import make_cached_decorator, Injected
from hashing import hash_list
from decorators import measure_performance


def _compute_content_hash(serialized_func: bytes, *args, **kwargs) -> str:
    kwargs_list = sorted(kwargs.items())
    return hash_list([serialized_func, *args, *kwargs_list])


task_with_context = make_cached_decorator(
    cache_provider=get_cache,
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
    from hashing import hash_source_code

    context.num_make_large_image_runs = 0

    with tempfile.TemporaryDirectory() as output_dir:
        injected_context = {"output_dir": Path(output_dir)}
        ray.get(
            make_large_image_remote.remote(
                (50, 50), injected_context=injected_context
            )
        )

    # Profile keys are ("profile", fn_hash, fn_name, quality, node_ip).
    # Match by prefix since node_ip varies by test environment.
    matches = context.get_matching(("profile",))
    histories = [v for v in matches.values() if v]
    assert len(histories) > 0, f"No profile samples recorded; matches={matches}"
    assert isinstance(histories[0][0], dict)
