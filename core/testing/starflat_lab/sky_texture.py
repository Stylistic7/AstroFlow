"""Sky-texture judge (no PI): nebula-poor 256px cells, etalon band, for label->dir pairs given as argv."""
import glob, sys, numpy as np
from scipy.ndimage import gaussian_filter
from xisf import XISF
S = "/private/tmp/claude-501/-Users-nobody-Projects-AstroFlow-core/5db806d6-08c6-42ad-ad88-13dbd5e0934e/scratchpad"
H, W = 6388, 9576
def load(d):
    p = [q for q in glob.glob(f"{S}/{d}/master/masterLight*mono.xisf") if "autocrop" not in q][0]
    a = np.array(XISF(p).read_image(0), dtype=np.float32); return a[..., 0] if a.ndim == 3 else a
def cells(img, cell):
    nx, ny = W // cell, H // cell
    return np.nanmedian(img[:ny*cell, :nx*cell].reshape(ny, cell, nx, cell), axis=(1, 3))
def hp(A, s): Af = np.where(np.isfinite(A), A, np.nanmedian(A)); return Af - gaussian_filter(Af, s)
arms = dict(a.split("=") for a in sys.argv[1:])
base = load(next(iter(arms.values())))
for cell in (256, 128):
    c0 = cells(np.where(base > 0, base, np.nan), cell); poor = np.isfinite(c0) & (c0 < np.nanpercentile(c0, 35))
    print(f"SKY TEXTURE, nebula-poor cells (same mask), {cell}px cells, etalon band (>~1000px removed):")
    for label, d in arms.items():
        img = load(d); c = cells(np.where(img > 0, img, np.nan), cell); T = hp(np.log(np.where(np.isfinite(c) & (c > 0), c, np.nanmedian(c))), 1000 / cell)
        print(f"  {label:26s}: {100*np.std(T[poor]):.3f}%")
