import unittest

class TestShellScript(unittest.TestCase):

  def test_hang(self):
    self.assertEqual(True, False)

if __name__ == "__main__":
  unittest.main()
