import unittest
import operator

import clustering


class TestClustering(unittest.TestCase):

  def test_cluster_continuous(self):
    items = [(0.31, 'c'), (0.32, 'd'), (0.45, 'e'), (0.35, 'f'), (0.40, 'g'),
             (0.5, 'h'), (0.1, 'a'), (0.09, 'b')]
    clusters = clustering.cluster_continuous(items,
                                             epsilon=0.2,
                                             key=operator.itemgetter(0))
    self.assertEqual(clusters, [[(0.09, 'b'), (0.1, 'a')],
                                [(0.31, 'c'), (0.32, 'd'), (0.35, 'f'),
                                 (0.4, 'g'), (0.45, 'e'), (0.5, 'h')]])

  def test_cluster_discrete(self):
    items = [('a', '2600'), ('b', '2600'), ('c', '6200'), ('d', '6200')]
    clusters = clustering.cluster_discrete(items, key=operator.itemgetter(1))
    self.assertEqual(
        clusters, {
            '2600': [('a', '2600'), ('b', '2600')],
            '6200': [('c', '6200'), ('d', '6200')]
        })

  def test_items_in_distribution(self):
    items = [(0.31, 'c'), (0.32, 'd'), (0.45, 'e'), (0.35, 'f'), (0.40, 'g')]
    items_to_test = [(0.3, 'i'), (0.28, 'j'), (0.5, 'h'), (0.1, 'a'),
                     (0.09, 'b')]
    items_in_distribution = clustering.items_in_distribution(
        items, items_to_test, max_std_devs=2.0, key=operator.itemgetter(0))
    self.assertEqual(items_in_distribution, [(0.3, 'i'), (0.28, 'j')])


if __name__ == "__main__":
  unittest.main()
