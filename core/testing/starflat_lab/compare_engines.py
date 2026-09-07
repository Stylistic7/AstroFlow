"""compare_engines.py <catalog.npz> <sub.fit> [...]: run the PixInsight-free photometry
engine on each sub and compare, star by star, with the PI SPFC section already in the
sub's sidecar (the backfill's truth). Reports what matters for the star flat: the
scatter of log(scale_ours / scale_PI), its slope, and any dependence on position,
brightness, colour or FWHM; plus 256 px cell-map agreement and timing."""
import json, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
import astropy.io.fits as pf
from astropy.wcs import WCS
from astropy.io import fits
from photometry.gaia_catalog import GaiaCatalog
from photometry.forced_photometry import measure
from file_io.sidecar import get_section

cat = GaiaCatalog(Path(sys.argv[1]))
for sub in sys.argv[2:]:
    sub = Path(sub); t0 = time.time()
    hdr = fits.getheader(sub); img = pf.getdata(sub).astype(np.float32) / 65535.0; t_read = time.time() - t0
    ps = get_section(sub, "plate_solution")["data"]["header"]
    wcs = WCS(fits.Header({k: v for k, v in ps.items() if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}))
    filt = str(hdr["FILTER"]).strip().upper()[:1]
    H, W = img.shape
    st = cat.stars_in_field(wcs, W, H, filt)
    m = measure(img, st["x"], st["y"]); t_all = time.time() - t0
    ok = m["ok"] & (st["catalog_flux"] > 0) & (m["snr"] > 40)
    ours_x, ours_y, ours_scale = m["x"][ok], m["y"][ok], m["signal"][ok] / st["catalog_flux"][ok]
    G, col, fw = st["G"][ok], st["colour"][ok], m["fwhm"][ok]
    obj = get_section(sub, "star_photometry")   # PI's v1 section is the truth, whatever the current schema
    pi = np.column_stack([np.asarray(obj["data"]["columns"][k], float) for k in ("x", "y", "mag_G", "colour", "fwhm", "signal", "scale")]) if isinstance(obj, dict) and obj.get("data") else None
    print(f"\n{sub.name}: {filt}; catalogue stars in field {len(st['x'])}, measured ok {ok.sum()}; PI section {0 if pi is None else len(pi)} stars; read {t_read:.2f}s, total {t_all:.2f}s")
    if pi is None: continue
    # match by position (< 1.5 px)
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.c_[pi[:, 0], pi[:, 1]]).query(np.c_[ours_x, ours_y], distance_upper_bound=1.5)
    mm = np.isfinite(d); j = j[mm]
    ls = np.log(ours_scale[mm]) - np.log(pi[j, 6]); ls0 = ls - np.median(ls)
    win = (G[mm] >= 8) & (G[mm] <= 12)
    s = 1.4826 * np.median(np.abs(ls0[win])); slope = np.polyfit(np.log(pi[j, 6][win]), np.log(ours_scale[mm][win]), 1)[0]
    print(f"  matched {mm.sum()} (median offset {np.median(d[mm]):.2f} px); of PI's stars {100*mm.sum()/len(pi):.0f}% recovered")
    print(f"  log(scale ours/PI), 8<=G<=12: rms {100*s:.2f}%  p99 |resid| {100*np.percentile(np.abs(ls0[win]), 99):.2f}%  slope {slope:.4f}")
    for name, v in (("x", ours_x[mm]), ("y", ours_y[mm]), ("G", G[mm]), ("BP-RP", col[mm]), ("fwhm", fw[mm]), ("log signal", np.log(m["signal"][ok][mm]))):
        print(f"    corr(resid, {name:10s}) {np.corrcoef(ls0[win], v[win])[0,1]:+.3f}")
    print(f"  signal ours/PI median {np.exp(np.median(np.log(m['signal'][ok][mm] / pi[j, 5]))):.4f}; fwhm ours/PI median {np.median(fw[mm] / pi[j, 4]):.3f}")
    for lo, hi in ((8, 9), (9, 10), (10, 11), (11, 12)):
        b = (G[mm] >= lo) & (G[mm] < hi)
        if b.sum() > 10: print(f"    G {lo}-{hi}: n={b.sum():4d} rms {100*1.4826*np.median(np.abs(ls0[b]-np.median(ls0[b]))):.2f}%")
    # 256 px cell maps of log scale (per-sub zero-point removed): agreement
    cell = 256; nx = W // cell + 1
    def cellmap(x, y, v):
        cl = (np.clip(y.astype(int), 0, H - 1) // cell) * nx + (np.clip(x.astype(int), 0, W - 1) // cell); R = np.full((H // cell + 1) * nx, np.nan)
        for ci in np.unique(cl):
            k = cl == ci
            if k.sum() >= 5: R[ci] = np.median(v[k])
        return R
    Ro = cellmap(ours_x[mm][win], ours_y[mm][win], np.log(ours_scale[mm][win])); Rp = cellmap(pi[j, 0][win], pi[j, 1][win], np.log(pi[j, 6][win]))
    k = np.isfinite(Ro) & np.isfinite(Rp); dR = (Ro - Rp)[k]; dR -= np.median(dR)
    print(f"  256px cell maps: {k.sum()} cells, rms difference {100*np.std(dR):.3f}%  (PI map rms {100*np.std(Rp[k]-np.median(Rp[k])):.3f}%)")
