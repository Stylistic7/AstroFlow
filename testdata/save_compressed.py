import numpy as np
import astropy.io.fits as pyfits
from pathlib import Path

for pane in range(1, 5):
  path = f"/Users/nobody/Projects/AstroFlow/testdata"
  for i, filename in enumerate(Path(path).rglob(f"Light_Pane {pane}*")):
    fits = pyfits.open(filename)
    image = fits[0].data
    kwargs = {f"orion_pane_{pane}_{i}": image}
    np.savez_compressed(f'orion_pane_{pane}_{i}.npz', **kwargs)
