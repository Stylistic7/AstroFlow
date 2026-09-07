"""Build the H star flat from every harvested 6200MM @350mm H sub (sidecar
star_photometry), judge it against the 295-sub CURRENT_BEST on held-out subs,
and write response/etalon FITS + a master flat XISF + a comparison figure."""
import glob, json, os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/nobody/Projects/AstroFlow/core")
import astropy.io.fits as pf
from scipy.ndimage import gaussian_filter, map_coordinates
from stacks.starflat import gather_star_photometry, solve_star_flat, split_half, response_image, etalon_only, write_master_flat_xisf
from metadata.star_photometry import SCHEMA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = os.environ.get("STARFLAT_LAB_DIR", os.path.dirname(os.path.abspath(__file__))); H, W = 6388, 9576; t0 = time.time()   # data/products dir
subs = []
for j in glob.glob("/Volumes/AstroRaw/Raw/**/Light*6200MM*.fit.json", recursive=True):
    try:
        sc = json.load(open(j))["astroflow"]; sp = sc.get("star_photometry") or {}
        if sp.get("data") is None or sp.get("schema") != SCHEMA: continue
        h = sc["image_metadata"]["header"]
        if "6200" not in str(h.get("INSTRUME", "")) or not (344 <= float(h.get("FOCALLEN", 0)) <= 352): continue
        if str(h.get("FILTER", "")).strip().upper()[:1] != "H": continue
        subs.append(j[:-5])
    except Exception: pass
subs.sort(); print(f"{len(subs)} harvested H subs ({time.time()-t0:.0f}s)", flush=True)
stars = gather_star_photometry(subs); print(f"{len(stars)} stars", flush=True)
np.save(f"{S}/h_all_stars_xp.npy", stars); open(f"{S}/h_all_subs_xp.txt", "w").write("\n".join(subs))
def cellmap_rms(g, x, y, cell):
    nx, ny = W // cell + 1, H // cell + 1
    cl = (np.clip(y.astype(int), 0, H - 1) // cell) * nx + (np.clip(x.astype(int), 0, W - 1) // cell)
    R = np.full(nx * ny, np.nan)
    for ci in np.unique(cl):
        m = cl == ci
        if m.sum() >= 30: R[ci] = np.median(g[m])
    R = R.reshape(ny, nx); Rf = np.where(np.isfinite(R), R, np.nanmedian(R)); Rf -= np.nanmedian(Rf)
    hp = Rf - gaussian_filter(Rf, 1000 / cell)
    return 100 * np.nanstd(Rf), 100 * np.nanstd(hp)
def heldout(resp_img, test):
    """cell-structure rms (all scales, etalon band) of held-out per-star residuals after dividing by resp_img."""
    G = test[:, 4]; m = (G >= 8) & (G <= 12); t = test[m]
    g = np.log(t[:, 8]) - np.log(map_coordinates(resp_img, [np.clip(t[:, 1], 0, H - 1), np.clip(t[:, 0], 0, W - 1)], order=1))
    col = t[:, 5]; A = np.vstack([np.ones_like(g), col]).T; g = g - A @ np.linalg.lstsq(A, g, rcond=None)[0]
    for s in np.unique(t[:, 10]):
        ms = t[:, 10] == s; g[ms] -= np.median(g[ms])
    return cellmap_rms(g, t[:, 0], t[:, 1], 256)
sid = stars[:, 10]; even, odd = stars[sid % 2 == 0], stars[sid % 2 == 1]   # STAR_COLUMNS: x y ra dec G colour fwhm signal scale snr sub
# full solves at four cell sizes (+ a faint-window variant), in parallel processes
from multiprocessing import get_context   # fork: workers inherit the 15 M-star array instead of re-running this script
CASES = [(256, (8.0, 12.0)), (128, (8.0, 12.0)), (64, (8.0, 12.0)), (32, (8.0, 12.0)), (64, (8.0, 13.5))]
def _solve_case_even(args):
    cell, win = args; sol = solve_star_flat(even, (H, W), cell=cell, mag_window=win); return (cell, win), sol, (np.nan, np.nan)
def _solve_case(args):
    cell, win = args; sol = solve_star_flat(stars, (H, W), cell=cell, mag_window=win); return (cell, win), sol, split_half(stars, sol)
sols = {}
with get_context("fork").Pool(len(CASES)) as pool:
    for key, sol, (r_raw, r_blob) in pool.imap_unordered(_solve_case, CASES):
        sols[key] = sol; cell, win = key
        print(f"cell {cell:3d} window {win}: kept {len(sol.kept_subs)} dropped {len(sol.dropped_subs)} subs; {sol.n_stars} stars; stars/cell median {np.nanmedian(sol.counts):.0f}; "
              f"per-cell unc median {100*np.nanmedian(sol.uncertainty):.3f}%; colour term {sol.colour_term:+.4f}; scatter {100*sol.residual_scatter:.2f}%; split-half r_raw {r_raw:.3f} r_blob {r_blob:.3f}  ({time.time()-t0:.0f}s)", flush=True)
resp = {c: response_image(sols[(c, (8.0, 12.0))], smooth_sigma_cells=0.7) for c in (256, 128, 64, 32)}
resp["64_deep"] = response_image(sols[(64, (8.0, 13.5))], smooth_sigma_cells=0.7)
old = pf.getdata(f"{S}/starflat_response_H_v2.fits").astype(np.float32)   # the PI-derived v2 flat (128 px) is the reference now
# held-out judge: solve on even subs, test on odd subs (disjoint)
with get_context("fork").Pool(len(CASES)) as pool:
    ev = {key: response_image(sol, smooth_sigma_cells=0.7) for key, sol, _ in pool.imap_unordered(_solve_case_even, CASES)}
print("\nHELD-OUT (odd subs) cell-structure rms  [all scales / etalon band]:")
print(f"  no correction        : %.3f%% / %.3f%%" % heldout(np.ones((H, W), np.float32), odd))
print(f"  PI-derived v2 flat   : %.3f%% / %.3f%%" % heldout(old, odd))
for key in CASES: print(f"  new even-half, {key[0]:3d}px window {key[1]}: %.3f%% / %.3f%%" % heldout(ev[key], odd))
# agreement old vs new
for c in (256, 128, 64, 32):
    lo, ln = np.log(old[::8, ::8]), np.log(resp[c][::8, ::8]); print(f"  corr(log old, log new{c}) all-scales {np.corrcoef(lo.ravel(), ln.ravel())[0,1]:.3f}; etalon band {np.corrcoef((lo-gaussian_filter(lo,125)).ravel(), (ln-gaussian_filter(ln,125)).ravel())[0,1]:.3f}")
best = 128
pf.PrimaryHDU(resp[best]).writeto(f"{S}/starflat_response_H_xp.fits", overwrite=True)
pf.PrimaryHDU(etalon_only(resp[best])).writeto(f"{S}/starflat_H_xp_ETALON_ONLY.fits", overwrite=True)
for c in (256, 64, 32): pf.PrimaryHDU(resp[c]).writeto(f"{S}/starflat_response_H_xp_cell{c}.fits", overwrite=True)
tmpl = sorted(glob.glob(f"{S}/**/masterFlat*.xisf", recursive=True) + glob.glob("/Volumes/AstroProcessed/Processed/**/masterFlat*mono.xisf", recursive=True))
if tmpl:
    write_master_flat_xisf(resp[best], tmpl[0], f"{S}/masterFlat_BIN-1_{W}x{H}_FILTER-H_mono_xp.xisf"); print("master flat XISF written from template", tmpl[0][-60:])
# figure: old vs new (full + etalon-only), counts, uncertainty
fig, ax = plt.subplots(2, 3, figsize=(24, 11), facecolor="#111")
def show(a, img, title, lo, hi, cmap="gray"):
    ax[a].imshow(img, vmin=lo, vmax=hi, cmap=cmap, origin="lower"); ax[a].set_title(title, color="w", fontsize=12); ax[a].axis("off")
show((0, 0), old[::4, ::4], "PI-derived star flat v2 (2028 subs, 128 px)", 0.85, 1.05)
show((0, 1), resp[best][::4, ::4], f"XP-engine star flat ({len(subs)} subs, {best} px cells)", 0.85, 1.05)
show((0, 2), resp[64][::4, ::4], f"NEW, 64 px cells", 0.85, 1.05)
show((1, 0), etalon_only(old)[::4, ::4], "PI-derived etalon-only (±1.5%)", 0.985, 1.015)
show((1, 1), etalon_only(resp[best])[::4, ::4], "XP-engine etalon-only (±1.5%)", 0.985, 1.015)
show((1, 2), etalon_only(resp[64])[::4, ::4], "NEW 64 px etalon-only (±1.5%)", 0.985, 1.015)
plt.tight_layout(); plt.savefig(f"{S}/starflat_H_xp_vs_pi_comparison.png", dpi=80, facecolor="#111"); print(f"figure written ({time.time()-t0:.0f}s)")
