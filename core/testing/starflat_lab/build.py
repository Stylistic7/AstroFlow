"""build.py <out_dir> <template_master_flat.xisf> [cell] [smooth_sigma_cells]
Joint solve on <out_dir>/stars.npy, split-half convergence, smoothed response
FITS and a WBPP Master Flat XISF."""
import os, sys, numpy as np, astropy.io.fits as pf
from stacks.starflat import MASTER_FLAT_NAME, response_image, solve_star_flat, split_half, write_master_flat_xisf

out, template = sys.argv[1], sys.argv[2]
cell = int(sys.argv[3]) if len(sys.argv) > 3 else 256; sig = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
H, W = 6388, 9576
stars = np.load(os.path.join(out, "stars.npy"))
sol = solve_star_flat(stars, (H, W), cell=cell)
r_raw, r_blob = split_half(stars, sol)
print(f"clean: {sol.n_stars} stars from {len(sol.kept_subs)} subs (dropped {len(sol.dropped_subs)} contaminated); residual {100*sol.residual_scatter:.2f}%; colour {100*sol.colour_term:+.2f}%/mag")
print(f"stars/cell median {np.nanmedian(sol.counts[sol.counts>0]):.0f}; per-cell noise {100*np.nanmedian(sol.uncertainty):.2f}%; split-half r = {r_raw:+.3f} raw / {r_blob:+.3f} etalon band")
resp = response_image(sol, smooth_sigma_cells=sig)
pf.PrimaryHDU(resp).writeto(os.path.join(out, "starflat_response.fits"), overwrite=True)
write_master_flat_xisf(resp, template, os.path.join(out, MASTER_FLAT_NAME.format(w=W, h=H, f="H")))
np.save(os.path.join(out, "cellmap.npy"), np.stack([sol.log_gain, sol.counts, sol.uncertainty]))
print(f"wrote starflat_response.fits (range {resp.min():.3f}-{resp.max():.3f}) and the Master Flat XISF")
