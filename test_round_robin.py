import unittest

from round_robin import round_robin_by_directory, chunked


class TestRoundRobin(unittest.TestCase):
  def test_round_robin(self):
    # These are intentionally out of order.
    filenames = [
        '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/baz.fit', '/a/b/bar.fit',
        '/a/b/quz.fit', '/a/b/c/bar.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit'
    ]

    ordered = list(round_robin_by_directory(filenames, cluster=1))
    expected = [
        '/a/b/bar.fit', '/a/b/c/bar.fit', '/a/b/baz.fit', '/a/b/c/baz.fit',
        '/a/b/foo.fit', '/a/b/c/foo.fit', '/a/b/quz.fit', '/a/b/c/quz.fit'
    ]
    self.assertEqual(ordered, expected)

    ordered = list(round_robin_by_directory(filenames, cluster=2))
    expected = [
        '/a/b/bar.fit', '/a/b/baz.fit', '/a/b/c/bar.fit', '/a/b/c/baz.fit',
        '/a/b/foo.fit', '/a/b/quz.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit'
    ]
    self.assertEqual(ordered, expected)

    ordered = list(round_robin_by_directory(filenames, cluster=3))
    expected = [
        '/a/b/bar.fit', '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/bar.fit',
        '/a/b/c/baz.fit', '/a/b/c/foo.fit', '/a/b/quz.fit', '/a/b/c/quz.fit'
    ]
    self.assertEqual(ordered, expected)

    ordered = list(round_robin_by_directory(filenames, cluster=4))
    expected = [
        '/a/b/bar.fit', '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/quz.fit',
        '/a/b/c/bar.fit', '/a/b/c/baz.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit'
    ]
    self.assertEqual(ordered, expected)

  def test_chunked(self):
    filenames = [
        '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/baz.fit', '/a/b/bar.fit',
        '/a/b/quz.fit', '/a/b/c/bar.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit'
    ]

    chunks = chunked(filenames, size=5)

    expected_chunks = [[
        '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/baz.fit', '/a/b/bar.fit',
        '/a/b/quz.fit'
    ], ['/a/b/c/bar.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit']]

    for chunk, expected_chunk in zip(chunks, expected_chunks):
      self.assertEqual(list(chunk), list(expected_chunk))

  def test_chunked_round_robin(self):
    filenames = [
        '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/baz.fit', '/a/b/bar.fit',
        '/a/b/quz.fit', '/a/b/c/bar.fit', '/a/b/c/foo.fit', '/a/b/c/quz.fit'
    ]

    chunks = chunked(round_robin_by_directory(filenames, cluster=3), size=5)

    expected_chunks = [[
        '/a/b/bar.fit', '/a/b/baz.fit', '/a/b/foo.fit', '/a/b/c/bar.fit',
        '/a/b/c/baz.fit'
    ], ['/a/b/c/foo.fit', '/a/b/quz.fit', '/a/b/c/quz.fit']]

    for chunk, expected_chunk in zip(chunks, expected_chunks):
      self.assertEqual(list(chunk), list(expected_chunk))


if __name__ == '__main__':
  unittest.main()
