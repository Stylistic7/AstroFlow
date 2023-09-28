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

    for stack in stacks:
      modify_stack(stack)

    self.assertFalse(stacks[0].enable)
    self.assertTrue(stacks[1].enable)


if __name__ == '__main__':
  unittest.main()
