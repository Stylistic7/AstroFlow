import glob
import unittest
import tempfile
import random
import os

import image_metadata
from make_test_frames import make_test_frames


class TestImageHashes(unittest.TestCase):
  def test_existing_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(project_dir=tmp_dir)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(sorted(filenames), sorted(image_hashes.values()))

  def test_added_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=10, project_dir=tmp_dir)
      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 10)

      make_test_frames(number=10, project_dir=tmp_dir)
      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 20)

  def test_deleted_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=20, project_dir=tmp_dir)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 20)

      random.shuffle(filenames)
      for filename in filenames[:5]:
        os.remove(filename)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(
          filenames, image_hashes)
      self.assertEqual(len(image_hashes), 20)

      self.assertEqual(sum(1 for f in image_hashes.values() if f), 15)

  def test_moved_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=20, project_dir=tmp_dir)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 20)
      self.assertTrue(all(os.path.exists(f) for f in image_hashes.values()))

      new_dir = os.path.join(tmp_dir, 'new_dir')
      os.mkdir(new_dir)
      random.shuffle(filenames)
      for filename in filenames[:5]:
        os.rename(filename, os.path.join(new_dir, os.path.basename(filename)))

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 20)
      self.assertTrue(all(os.path.exists(f) for f in image_hashes.values()))

      self.assertEqual(sum(1 for f in image_hashes.values() if 'new_dir' in f),
                       5)


class TestFitsHeaders(unittest.TestCase):
  def test_existing_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(project_dir=tmp_dir, exposure=30)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(sorted(filenames), sorted(image_hashes.values()))

      fits_headers = image_metadata.read_fits_headers(image_hashes, {})
      self.assertEqual(sorted(fits_headers.keys()),
                       sorted(image_hashes.keys()))

      self.assertTrue(all(f['EXPOSURE'] == 30 for f in fits_headers.values()))

  def test_added_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=10, project_dir=tmp_dir)
      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 10)
      fits_headers = image_metadata.read_fits_headers(image_hashes, {})
      self.assertEqual(len(fits_headers), 10)

      make_test_frames(number=10, project_dir=tmp_dir)
      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(len(image_hashes), 20)
      fits_headers = image_metadata.read_fits_headers(image_hashes, {})
      self.assertEqual(len(fits_headers), 20)


class TestPlateSolving(unittest.TestCase):
  def test_existing_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(project_dir=tmp_dir, exposure=30)

      filenames = glob.glob(f'{tmp_dir}/**/*.fit', recursive=True)
      image_hashes = image_metadata.compute_image_hashes(filenames, {})
      self.assertEqual(sorted(filenames), sorted(image_hashes.values()))

      plate_solutions = image_metadata.plate_solve(image_hashes.keys(),
                                                   image_hashes, {})
      self.assertEqual(sorted(plate_solutions.keys()),
                       sorted(image_hashes.keys()))


if __name__ == '__main__':
  unittest.main()
