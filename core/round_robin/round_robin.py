import os
import itertools
from collections.abc import Generator, Iterable
from typing import Any
from beartype import beartype


@beartype
def round_robin_by_directory(
    filenames: list[str], cluster: int = 3
) -> Generator[str, None, None]:
    filenames = sorted(filenames)
    dirs = sorted(set(os.path.dirname(f) for f in filenames))

    while filenames:
        for dir in dirs:
            files_in_dir = [f for f in filenames if os.path.dirname(f) == dir]
            for filename in files_in_dir[:cluster]:
                yield filename
                filenames.remove(filename)


@beartype
def chunked(
    iterable: Iterable[Any], size: int = 10
) -> Generator[itertools.chain, None, None]:
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))
