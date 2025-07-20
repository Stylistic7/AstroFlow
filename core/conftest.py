import ray
import pytest
import time

from testing import make_test_context


def pytest_collection_modifyitems(config, items):
  exclusive = [item for item in items if 'exclusive' in item.keywords]
  normal = [item for item in items if 'exclusive' not in item.keywords]
  items[:] = normal + exclusive  # run normal tests first


@pytest.fixture(scope="session", autouse=True)
def init_ray(worker_id):
  # Setup code
  assert not ray.is_initialized()
  from ray_infra.ray_local import enable_test_mode
  enable_test_mode()
  assert ray.is_initialized()


@pytest.fixture(autouse=True)
def test_context(init_ray):
  print(f'### running context fixture')

  # Setup code
  assert ray.is_initialized()
  test_context = make_test_context()

  yield test_context

  # Teardown code
  test_context.input_dir_obj.cleanup()
  test_context.output_dir_obj.cleanup()
