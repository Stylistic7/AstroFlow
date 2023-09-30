import unittest

from stack import Stack


class TestStack(unittest.TestCase):
  def test_function_modifying_stack(self):
    stacks = []
    stacks.append(Stack(target="Foo"))
    stacks.append(Stack(target="Bar"))

    self.assertFalse(stacks[1].enable)

    def modify_stack(stack):
      if stack.target == "Bar":
        stack.enable = True
      if stack.target == "Foo":
        stack.master = "foo.xisf"

    for stack in stacks:
      modify_stack(stack)

    foo, bar = stacks
    self.assertFalse(foo.enable)
    self.assertEqual(foo.master, "foo.xisf")

    self.assertTrue(bar.enable)
    self.assertEqual(bar.master, None)


if __name__ == '__main__':
  unittest.main()
