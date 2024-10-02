from typing import NewType, TypeAlias

MosaicHash = NewType('MosaicHash', str)
Template: TypeAlias = tuple[int | None, int | None]
TileIndices: TypeAlias = tuple[int, int]
Pair: TypeAlias = list[TileIndices]
