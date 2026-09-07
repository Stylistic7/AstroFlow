"""e2e.py <out_dir> <lights_dir_with_RUN_x> <darks_dir> <template_master_flat.xisf> <full_flat.fits>
WBPP three arms (none / etalon-only / full) with darks, then judge the OUTPUT
masters: nebula-poor etalon-band sky texture + fixed-stretch visual. Run with
nothing else on the PixInsight slot."""
import glob, os, shutil, sys, numpy as np, astropy.io.fits as pf
from pathlib import Path
from scipy.ndimage import gaussian_filter
from xisf import XISF
from pix.tasks.wbpp import run_wbpp
from stacks.starflat import MASTER_FLAT_NAME, etalon_only, write_master_flat_xisf

out, lights, darks, template, full_fits = sys.argv[1:6]; H, W = 6388, 9576; CELL = 256
name = MASTER_FLAT_NAME.format(w=W, h=H, f="H")
full = np.array(pf.getdata(full_fits), dtype=np.float32); full /= np.median(full)
flats = {"noflat": None, "etalon": etalon_only(full), "full": full}
for arm, fl in flats.items():
    d = os.path.join(out, f"e2e_{arm}"); shutil.rmtree(d, ignore_errors=True); shutil.copytree(lights, d)
    os.makedirs(os.path.join(d, "DARK")); [shutil.copy(f, os.path.join(d, "DARK")) for f in glob.glob(os.path.join(darks, "*.fit"))]
    if fl is not None:
        os.makedirs(os.path.join(d, "FLAT")); write_master_flat_xisf(fl, template, os.path.join(d, "FLAT", name))
    shutil.rmtree(d + "_out", ignore_errors=True); run_wbpp(Path(d), Path(d + "_out"), slots=[1]); print(f"{arm} done", flush=True)
def loadm(arm):
    p = [q for q in glob.glob(os.path.join(out, f"e2e_{arm}_out/master/masterLight*mono.xisf")) if "autocrop" not in q][0]
    a = np.array(XISF(p).read_image(0), dtype=np.float32); return np.array(a[..., 0] if a.ndim == 3 else a)
def cells(img): return np.array([[np.median(img[j*CELL:(j+1)*CELL, i*CELL:(i+1)*CELL]) for i in range(W // CELL)] for j in range(H // CELL)])
def hp(A): return A - gaussian_filter(A, 4)
M = {a: loadm(a) for a in flats}; sky0 = cells(M["noflat"]); poor = sky0 < np.percentile(sky0, 35)
for a in flats:
    T = hp(np.log(np.maximum(cells(M[a]), 1e-9))); print(f"{a:8s}: nebula-poor etalon-band texture {100*np.std(T[poor]):.3f}%")
