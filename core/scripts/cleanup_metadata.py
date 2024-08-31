from context import get_cache

cache = get_cache()

for k in cache.iterkeys():
  if type(k) is not tuple or k[0] != 'image_metadata':
    continue

  metadata = cache[k]

  if 'IMAGETYP' not in metadata.header:
    print(f'### Found image without IMAGETYP {metadata.filename}')
    continue

  if 'placed' in metadata.header['IMAGETYP']:
    print(f'### Found placed file {metadata.filename}')
    del cache[k]
