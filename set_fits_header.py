import astropy.io.fits as pyfits
from pathlib import Path

path = "/Users/nobody/Projects/Astro/Dark Library/ZWO ASI2600MM Pro"
filenames = list(str(f) for f in Path(path).rglob("*.fit"))

for filename in filenames:
  with pyfits.open(filename, mode='update') as fits:
    fits[0].header['INSTRUME'] = 'ZWO ASI2600MM Pro'
    fits[0].header['IMAGETYP'] = 'Dark'
    if 'FILTER' in fits[0].header:
      del fits[0].header['FILTER']
