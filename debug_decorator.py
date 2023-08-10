def debug(func):
  def wrapper(*args, **kwargs):
    flow, *foo = args

    # args = arguments passed to the step. For example: in def join(self, inputs), args would be (self, inputs)
    # flow = the `self` object passed into each step in a Flow
    # *_ is needed to allow this line to work regardless of whether we pass only 1 (e.g. `self`) or more parameters (e.g. `self, inputs`)

    def repr(x):
      if type(x) is list:
        return f"list[{len(x)}]"
      else:
        return f"'{str(x)}'"

    if len(foo) == 0:
      for index, num_tasks, input in flow.foreach_stack():
        if type(input) is str:
          input_string = f"'{input}'"
        elif type(input) is list:
          return repr(input)
        elif type(input) is tuple:
          input_string = '(' + ', '.join([repr(x) for x in input]) + ')'
        else:
          input_string = f'{str(input)} [type {type(input).__name__}]'

        print(f'Foreach frame: task {index+1} / {num_tasks}: '
              f'input {input_string}')

    func(*args, **kwargs)

  wrapper.__name__ = func.__name__  # This prevents an invalid self.next() transition error: "Step specifies a transition to an unknown step, wrapper."

  return wrapper
