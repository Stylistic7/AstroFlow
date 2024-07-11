import os
import io
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
from functools import singledispatch
import datetime

import numpy as np
import astropy.io.fits as pyfits  # type: ignore

from context import context
from file_io.generic import read_image, read_header
from file_io.generic import get_content_hash
from file_io.normalize_header import normalize_header


class ArtifactNotFoundException(Exception):
  pass


class WrongContentHashException(Exception):
  pass


@dataclass(frozen=True)
class FileArtifact:
  filename: str
  content_hash: Optional[str | None] = None


@dataclass(frozen=True)
class CacheArtifact:
  key: Tuple[Any]
  content_hash: Optional[str | None] = None


Artifact = FileArtifact | CacheArtifact


@singledispatch
def read_artifact_header(artifact: Artifact):
  pass


@read_artifact_header.register
def _(artifact: FileArtifact):
  return read_header(artifact.filename)


# TODO: Move file_obj operations into file_io module.
@read_artifact_header.register
def _(artifact: CacheArtifact):
  if not context.contains(artifact.key):
    raise ArtifactNotFoundException(f'Key {artifact.key} not found in context')

  file_obj = io.BytesIO(context.get(artifact.key, keep_in_memory=False))

  fits = pyfits.open(file_obj)
  header = fits[0].header
  fits.close()

  return normalize_header(header)


@singledispatch
def read_artifact_image(artifact: Artifact):
  assert False, f'Bad artifact type {type(artifact)}'


@read_artifact_image.register
def _(artifact: FileArtifact):
  if artifact.content_hash:
    content_hash = get_content_hash(artifact.filename)

    if content_hash != artifact.content_hash:
      raise WrongContentHashException(
          f'Filename {artifact.filename} did not have expected content hash '
          f'{artifact.content_hash}')

  return read_image(artifact.filename)


# TODO: Move file_obj operations into file_io module.
@read_artifact_image.register
def _(artifact: CacheArtifact):
  if not context.contains(artifact.key):
    raise ArtifactNotFoundException(f'Key {artifact.key} not found in context')

  file_obj = io.BytesIO(context.get(artifact.key, keep_in_memory=False))

  fits = pyfits.open(file_obj)
  image = fits[0].data
  fits.close()

  if artifact.content_hash is not None:
    if 'CONTHASH' not in fits[0].header or fits[0].header[
        'CONTHASH'] != artifact.content_hash:
      raise WrongContentHashException(
          f'Key {artifact.key} had incorrect content hash')

  return image


@singledispatch
def write_artifact(artifact: Artifact, header: Dict[str, Any],
                   image: np.ndarray):
  assert False, f'Bad artifact type {type(artifact)}'


@write_artifact.register
def _(artifact: FileArtifact, header: Dict[str, Any],
      image: np.ndarray):
  hdu = pyfits.PrimaryHDU(image)

  for k, v in header.items():
    if type(v) is datetime.datetime:
      hdu.header[k] = v.isoformat()
    else:
      hdu.header[k] = v

  hdu.writeto(artifact.filename, overwrite=True)


@write_artifact.register
def _(artifact: CacheArtifact, header: Dict[str, Any],
      image: np.ndarray):
  hdu = pyfits.PrimaryHDU(image)

  for k, v in header.items():
    if type(v) is datetime.datetime:
      hdu.header[k] = v.isoformat()
    else:
      hdu.header[k] = v

  file_obj = io.BytesIO()
  hdu.writeto(file_obj)

  context[artifact.key] = file_obj.getvalue()


@singledispatch
def artifact_exists(artifact: Artifact):
  assert False, f'Bad artifact type {type(artifact)}'


@artifact_exists.register
def _(artifact: FileArtifact):
  if not os.path.exists(artifact.filename):
    return False

  if artifact.content_hash is not None:
    if get_content_hash(artifact.filename) != artifact.content_hash:
      return False

  return True


@artifact_exists.register
def _(artifact: CacheArtifact):
  if not context.contains(artifact.key):
    return False

  if artifact.content_hash is not None:
    file_obj = io.BytesIO(context.get(artifact.key, keep_in_memory=False))

    fits = pyfits.open(file_obj)
    header = normalize_header(fits[0].header)
    fits.close()

    if header['CONTHASH'] != artifact.content_hash:
      return False

  return True
