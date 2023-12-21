import unittest

import throughput


class TestThroughput(unittest.TestCase):
  def test_one_value(self):
    count, mean = 0, 0
    count, mean = throughput._add_sample(count, mean, 10)

    self.assertEqual(count, 1)
    self.assertEqual(mean, 10)

  def test_two_values(self):
    count, mean = 0, 0
    count, mean = throughput._add_sample(count, mean, 10)
    count, mean = throughput._add_sample(count, mean, 20)

    self.assertEqual(count, 2)
    self.assertEqual(mean, 15)


if __name__ == '__main__':
  unittest.main()
