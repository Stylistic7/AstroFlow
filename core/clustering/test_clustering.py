import unittest
import operator
import datetime

import clustering


class TestClustering(unittest.TestCase):

  def test_cluster_discrete(self):
    items = [('a', '2600'), ('b', '2600'), ('c', '6200'), ('d', '6200')]
    clusters = clustering.cluster_discrete(items, key=operator.itemgetter(1))
    self.assertEqual(clusters, [[('a', '2600'),
                                 ('b', '2600')], [('c', '6200'),
                                                  ('d', '6200')]])

  def test_cluster_continuous(self):
    items = [(0.31, 'c'), (0.32, 'd'), (0.45, 'e'), (0.35, 'f'), (0.40, 'g'),
             (0.5, 'h'), (0.1, 'a'), (0.09, 'b')]
    clusters = clustering.cluster_continuous(items,
                                             epsilon=0.2,
                                             key=operator.itemgetter(0))
    self.assertEqual(clusters, [[(0.09, 'b'), (0.1, 'a')],
                                [(0.31, 'c'), (0.32, 'd'), (0.35, 'f'),
                                 (0.4, 'g'), (0.45, 'e'), (0.5, 'h')]])

  def test_cluster_continuous_dates(self):
    run_0 = [(datetime.datetime(2024, 2, 7, 0, 0, 0) +
              datetime.timedelta(seconds=300 * i), i) for i in range(5)]
    run_1 = [(datetime.datetime(2024, 2, 8, 0, 0, 0) +
              datetime.timedelta(seconds=300 * i), 100 + i) for i in range(5)]
    clusters = clustering.cluster_continuous(
        run_0 + run_1,
        epsilon=datetime.timedelta(minutes=15),
        key=operator.itemgetter(0))
    self.assertEqual(clusters, [[(datetime.datetime(2024, 2, 7, 0, 0), 0),
                                 (datetime.datetime(2024, 2, 7, 0, 5), 1),
                                 (datetime.datetime(2024, 2, 7, 0, 10), 2),
                                 (datetime.datetime(2024, 2, 7, 0, 15), 3),
                                 (datetime.datetime(2024, 2, 7, 0, 20), 4)],
                                [(datetime.datetime(2024, 2, 8, 0, 0), 100),
                                 (datetime.datetime(2024, 2, 8, 0, 5), 101),
                                 (datetime.datetime(2024, 2, 8, 0, 10), 102),
                                 (datetime.datetime(2024, 2, 8, 0, 15), 103),
                                 (datetime.datetime(2024, 2, 8, 0, 20), 104)]])

  def test_items_in_distribution(self):
    items = [(0.31, 'c'), (0.32, 'd'), (0.45, 'e'), (0.35, 'f'), (0.40, 'g')]
    items_to_test = [(0.3, 'i'), (0.28, 'j'), (0.5, 'h'), (0.1, 'a'),
                     (0.09, 'b')]
    items_in_distribution = clustering.items_in_distribution(
        items, items_to_test, max_std_devs=2.0, key=operator.itemgetter(0))
    self.assertEqual(items_in_distribution, [(0.3, 'i'), (0.28, 'j')])

  def test_cluster_dates_with_gap(self):
    dates = [(datetime.datetime(2023, 4, 14, 5, 53, 1, 550185), 1),
             (datetime.datetime(2023, 4, 14, 6, 16, 35, 498587), 2),
             (datetime.datetime(2023, 4, 14, 6, 22, 53, 380184), 3),
             (datetime.datetime(2023, 4, 14, 6, 24, 54, 625657), 4),
             (datetime.datetime(2023, 4, 14, 6, 27, 21, 381694), 5),
             (datetime.datetime(2023, 4, 14, 6, 29, 22, 655187), 6),
             (datetime.datetime(2023, 4, 14, 6, 31, 53, 451560), 7),
             (datetime.datetime(2023, 4, 14, 6, 33, 54, 638653), 8),
             (datetime.datetime(2023, 4, 14, 6, 36, 11, 369856), 9),
             (datetime.datetime(2023, 4, 14, 6, 38, 12, 652277), 10),
             (datetime.datetime(2023, 4, 14, 6, 40, 41, 366753), 11),
             (datetime.datetime(2023, 4, 14, 6, 42, 42, 613899), 12),
             (datetime.datetime(2023, 4, 14, 6, 45, 15, 362782), 13),
             (datetime.datetime(2023, 4, 14, 6, 47, 16, 562603), 14)]
    clusters = clustering.cluster_continuous(
        dates,
        epsilon=datetime.timedelta(minutes=15),
        key=operator.itemgetter(0))
    self.assertEqual(
        clusters, [[(datetime.datetime(2023, 4, 14, 5, 53, 1, 550185), 1)],
                   [(datetime.datetime(2023, 4, 14, 6, 16, 35, 498587), 2),
                    (datetime.datetime(2023, 4, 14, 6, 22, 53, 380184), 3),
                    (datetime.datetime(2023, 4, 14, 6, 24, 54, 625657), 4),
                    (datetime.datetime(2023, 4, 14, 6, 27, 21, 381694), 5),
                    (datetime.datetime(2023, 4, 14, 6, 29, 22, 655187), 6),
                    (datetime.datetime(2023, 4, 14, 6, 31, 53, 451560), 7),
                    (datetime.datetime(2023, 4, 14, 6, 33, 54, 638653), 8),
                    (datetime.datetime(2023, 4, 14, 6, 36, 11, 369856), 9),
                    (datetime.datetime(2023, 4, 14, 6, 38, 12, 652277), 10),
                    (datetime.datetime(2023, 4, 14, 6, 40, 41, 366753), 11),
                    (datetime.datetime(2023, 4, 14, 6, 42, 42, 613899), 12),
                    (datetime.datetime(2023, 4, 14, 6, 45, 15, 362782), 13),
                    (datetime.datetime(2023, 4, 14, 6, 47, 16, 562603), 14)]])


if __name__ == "__main__":
  unittest.main()
