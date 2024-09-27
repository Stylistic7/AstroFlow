from typing import NewType

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)

DependenciesDict = dict[str, StackHash]
