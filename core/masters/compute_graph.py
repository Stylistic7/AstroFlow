import ray

from context import context
from decorators import cached
from metadata.classes import MetadataDict
from metadata.image_types import is_master
from masters.types import Master, MastersDict


@cached.remote
def identify_masters(metadata: MetadataDict) -> MastersDict:
  return {
      h: Master.from_frame(h, metadata)
      for h, m in metadata.items()
      if is_master(m)
  }


@ray.remote
def store_masters(masters: MastersDict) -> None:
  context.grouped_masters = masters
  print(f'### store_masters: Stored '
        f'{len(masters)} masters; total is now {len(context.grouped_masters)}')
  return masters
