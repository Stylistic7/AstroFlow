import os
from dataclasses import dataclass

from context import context
from file_io.generic import read_image, read_header


@dataclass(frozen=True)
class Artifact:
  key: str = None
  filename: str = None
  content_hash: str = None


# TODO: This should maybe raise Exceptions.
def read_artifact_image(artifact: Artifact):
  print(f'### read_artifact_image {artifact}')

  if artifact.filename:
    if not os.path.exists(artifact.filename):
      print(f'### Filename {artifact.filename} does not exist')
      return

    if artifact.content_hash:
      header = read_header(artifact.filename)
      if header['CONTHASH'] == artifact.content_hash:
        print(f'### Found artifact {artifact.filename}')
        return read_image(artifact.filename)
      else:
        print(f'### File {artifact.filename} did not have expected '
              f'content hash.')
        return

  elif artifact.key:
    if context.contains(artifact.key):
      return context.get(artifact.key, keep_in_memory=False)
    else:
      print(f'### Cache does not contain artifact key {artifact.key}')
      return
