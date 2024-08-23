import unittest
import numpy as np

from . import ransac


class TestRansac(unittest.TestCase):

  def test_three_unit_vectors(self):
    centers = np.array([[0, 0], [1, 0], [0, 1]])
    bases = ransac._find_all_possible_bases(centers)

    expected_bases = [np.array([[0., 1.], [1., 0.]])]
    print('### expected_bases', expected_bases)

    self.assertEqual(len(bases), 1)
    self.assertTrue(
        np.array_equal(np.sort(bases[0].flat), np.sort(expected_bases[0].flat)))

  def test_three_long_vectors(self):
    centers = np.array([[0, 0], [2, 0], [0, 2]])
    bases = ransac._find_all_possible_bases(centers)

    expected_bases = [np.array([[0., 1.], [1., 0.]])]
    print('### expected_bases', expected_bases)

    self.assertEqual(len(bases), 1)
    self.assertTrue(
        np.array_equal(np.sort(bases[0].flat), np.sort(expected_bases[0].flat)))

  def test_four_long_vectors(self):
    centers = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
    bases = ransac._find_all_possible_bases(centers)

    expected_bases = [np.array([[0., 1.], [1., 0.]])]
    print('### expected_bases', expected_bases)

    self.assertEqual(len(bases), 1)
    self.assertTrue(
        np.array_equal(np.sort(bases[0].flat), np.sort(expected_bases[0].flat)))

  def test_non_orthogal(self):
    v1 = np.array([1, 0])
    v2 = np.array([1, 0.1])

    centers = np.array([[0, 0], v1, v1 + v2, 2 * v1 + v2, v1 + 2 * v2,
                        2 * v1 + 2 * v2])
    print('centers', centers)

    bases = ransac._find_all_possible_bases(centers)

    # TODO: Make this test invariant to the order of the basis vectors.
    expected_bases = [np.array([[1, 0], [0, 1]])]
    print('### expected_bases', expected_bases)

    self.assertEqual(len(bases), 1)
    self.assertTrue(
        np.array_equal(np.sort(bases[0].flat), np.sort(expected_bases[0].flat)))


if __name__ == "__main__":
  unittest.main()
