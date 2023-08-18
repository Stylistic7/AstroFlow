def compact_repr(L):
  def inner(element):
    if type(element) is str:
      return f"'{element}'"
    elif type(element) is list:
      return f"list[{len(element)}]"
    elif type(element) is frozenset:
      return f"frozenset[{len(element)}]"
    elif type(element) is tuple:
      return '(' + ', '.join([inner(x) for x in element]) + ')'
    else:
      return f'{str(element)} [type {type(element).__name__}]'

  return inner(L)
