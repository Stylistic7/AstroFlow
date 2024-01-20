# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import ray
ray.init(
    address="local",
    resources={"diskcache": 1},
    ignore_reinit_error=True,
    runtime_env={"env_vars": {
        "ASTROFLOW_MOCK_CACHE": "true"
    }},
)
# yapf: enable

from make_test_frames import make_test_frames
import image_metadata
import ray_diskcache
import unittest
import tempfile

class TestRayDiskcache(unittest.TestCase):

    def setUp(self):
        ray_diskcache.clear()
        cache.clear()

    def tearDown(self):
        ray_diskcache.clear()

    def test_global_dict(self):
        image_hashes = ray_diskcache.get_global("image_hashes", {})
        self.assertEqual(len(image_hashes), 0)

        ray_diskcache.set_global('image_hashes', {'foo': 'bar'})
        image_hashes = ray_diskcache.get_global('image_hashes', {})
        self.assertEqual(len(image_hashes), 1)


if __name__ == "__main__":
    unittest.main()
