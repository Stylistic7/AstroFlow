import itertools
import operator
from typing import Any, Callable, TypeVar

T = TypeVar('T')


def cluster_discrete(
    items: list[T], key: Callable[[T], Any] = operator.itemgetter(1)
) -> list[list[T]]:
  items = sorted(items, key=key)
  return [list(g) for _, g in itertools.groupby(items, key=key)]
