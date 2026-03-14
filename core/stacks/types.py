from typing import NewType
from dataclasses import dataclass
from beartype import beartype

StackHash = NewType("StackHash", str)


@beartype
@dataclass(frozen=True)
class StackIdDependency:
    stack_id: StackHash


@beartype
@dataclass(frozen=True)
class FilenameDependency:
    filename: str


StackDependency = StackIdDependency | FilenameDependency
DependenciesDict = dict[str, StackDependency]
