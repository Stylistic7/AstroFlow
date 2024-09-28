import os
import io
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
from functools import singledispatch
import datetime
from pathlib import Path

import numpy as np
import astropy.io.fits as pyfits  # type: ignore

from context import context, WriteMode
from file_io.generic import read_image, read_header
from file_io.generic import get_content_hash
from file_io.normalize_header import normalize_header


class ArtifactNotFoundException(Exception):
  pass


class WrongContentHashException(Exception):
  pass


Key = tuple[str, ...]
Header = dict[str, Any]


@dataclass(frozen=True)
class FileArtifact:
  filename: str
  content_hash: str


@dataclass(frozen=True)
class CacheArtifact:
  key: Key
  content_hash: str


Artifact = FileArtifact | CacheArtifact


def read_artifact(artifact: Artifact) -> tuple[np.ndarray, Header]:
  return read_artifact_image(artifact), read_artifact_header(artifact)


@singledispatch
def read_artifact_header(artifact: Artifact) -> Header:
  assert False, f'Bad artifact type {type(artifact)}'


@read_artifact_header.register
def _(artifact: FileArtifact) -> Header:
  return read_header(artifact.filename)


# TODO: Move file_obj operations into file_io module.
@read_artifact_header.register
def _(artifact: CacheArtifact) -> Header:
  if not context.contains(artifact.key):
    raise ArtifactNotFoundException(f'Key {artifact.key} not found in context')

  file_obj = io.BytesIO(context.get(artifact.key, keep_in_memory=False))

  fits = pyfits.open(file_obj, mmap=False)
  header = fits[0].header
  fits.close()

  return normalize_header(header)


@singledispatch
def read_artifact_image(artifact: Artifact) -> np.ndarray:
  assert False, f'Bad artifact type {type(artifact)}'


@read_artifact_image.register
def _(artifact: FileArtifact) -> np.ndarray:
  print(f'### read_artifact_image artifact {artifact}')
  if artifact.content_hash:
    content_hash = get_content_hash(artifact.filename)
    print(f'### read_artifact_image filename {artifact.filename} has '
          f'content_hash {content_hash}')

    if content_hash != artifact.content_hash:
      raise WrongContentHashException(
          f'Filename {artifact.filename} did not have expected content hash '
          f'{artifact.content_hash}')

  return read_image(artifact.filename)


# TODO: Move file_obj operations into file_io module.
@read_artifact_image.register
def _(artifact: CacheArtifact) -> np.ndarray:
  print(f'### read_artifact_image artifact {artifact}')
  if not context.contains(artifact.key):
    raise ArtifactNotFoundException(f'Key {artifact.key} not found in context')

  file_obj = io.BytesIO(context.get(artifact.key, keep_in_memory=False))

  fits = pyfits.open(file_obj, mmap=False)
  image = fits[0].data
  fits.close()

  if artifact.content_hash is not None:
    content_hash = fits[0].header['CONTHASH'] if 'CONTHASH' in fits[
        0].header else None
    if content_hash != artifact.content_hash:
      raise WrongContentHashException(
          f'Key {artifact.key} had incorrect content hash '
          f'{content_hash}, expected {artifact.content_hash}')

  return image


@singledispatch
def write_artifact(
    artifact: Artifact,
    image: np.ndarray,
    header: Optional[Dict[str, Any]] = None,
) -> None:
  assert False, f'Bad artifact type {type(artifact)}'


@write_artifact.register
def _(artifact: FileArtifact,
      image: np.ndarray, header: Optional[Dict[str, Any]] = None) -> None:
  hdu = pyfits.PrimaryHDU(image)

  if header is not None:
    for k, v in header.items():
      if type(v) is datetime.datetime:
        hdu.header[k] = v.isoformat()
      else:
        hdu.header[k] = v

  if artifact.content_hash is not None:
    hdu.header['CONTHASH'] = artifact.content_hash

  Path(artifact.filename).parent.mkdir(parents=True, exist_ok=True)
  hdu.writeto(artifact.filename, overwrite=True)
  print(f'### Wrote artifact {artifact}')


@write_artifact.register
def _(artifact: CacheArtifact,
      image: np.ndarray, header: Optional[Dict[str, Any]] = None) -> None:
  hdu = pyfits.PrimaryHDU(image)

  if header is not None:
    for k, v in header.items():
      if type(v) is datetime.datetime:
        hdu.header[k] = v.isoformat()
      else:
        hdu.header[k] = v

  if artifact.content_hash is not None:
    hdu.header['CONTHASH'] = artifact.content_hash

  file_obj = io.BytesIO()
  hdu.writeto(file_obj)

  context.set(artifact.key, file_obj.getvalue(), write_mode=WriteMode.DISK_ONLY)
  print(f'### Wrote artifact {artifact}')


@singledispatch
def artifact_exists(artifact: Artifact) -> bool:
  assert False, f'Bad artifact type {type(artifact)}'


@artifact_exists.register
def _(artifact: FileArtifact) -> bool:
  if not os.path.exists(artifact.filename):
    return False

  if artifact.content_hash is not None:
    content_hash = get_content_hash(artifact.filename)
    print(f'### filename content_hash {content_hash} '
          f'artifact content_hash {artifact.content_hash}')
    if content_hash != artifact.content_hash:
      return False

  return True


@artifact_exists.register
def _(artifact: CacheArtifact) -> bool:
  if not context.contains(artifact.key):
    return False

  if artifact.content_hash is not None:
    value = context.get(artifact.key, keep_in_memory=False)

    if type(value) is not bytes:
      return True

    file_obj = io.BytesIO(value)
    fits = pyfits.open(file_obj, mmap=False)
    header = normalize_header(fits[0].header)
    fits.close()

    if 'CONTHASH' not in header or header['CONTHASH'] != artifact.content_hash:
      return False

  return True
