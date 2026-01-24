import operator
import numpy as np
from typing import Any, Callable, TypeVar, Protocol

T = TypeVar("T")

S = TypeVar("S", bound="SupportsAddAndLessEqual")


class SupportsAddAndLessEqual(Protocol):
    def __add__(self: T, other: T) -> T: ...

    def __lt__(self: T, other: T) -> bool: ...

    def __le__(self: T, other: T) -> bool: ...

    def __gt__(self: T, other: T) -> bool: ...

    def __ge__(self: T, other: T) -> bool: ...


def cluster_continuous(
    items: list[T],
    epsilon: SupportsAddAndLessEqual,
    key: Callable[[T], SupportsAddAndLessEqual] = operator.itemgetter(1),
) -> list[list[T]]:
    if len(items) == 0:
        return []

    clusters = []
    items_sorted = sorted(items, key=key)
    curr_point: T = items_sorted[0]
    curr_cluster = [curr_point]
    for point in items_sorted[1:]:
        if key(point) <= key(curr_point) + epsilon:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
        curr_point = point
    clusters.append(curr_cluster)

    return clusters


def items_in_distribution(
    items: list[T],
    items_to_test: list[T],
    max_std_devs: float = 2.0,
    slop: float = 0.0,
    key: Callable[[T], Any] = operator.itemgetter(1),
) -> list[T]:
    values = [key(i) for i in items]
    mean, std = np.mean(values), np.std(values)
    return [
        i
        for i in items_to_test
        if abs(key(i) - mean) <= slop or abs(key(i) - mean) / std <= max_std_devs
    ]
