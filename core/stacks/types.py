from typing import NewType

StackHash = NewType('StackHash', str)

DependenciesDict = dict[str, StackHash]
