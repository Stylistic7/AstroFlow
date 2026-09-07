"""compare_engines_batch.py <n_subs> [workers]: score the PixInsight-free engine against the PI
SPFC sections (schema v1) of `n_subs` randomly chosen subs, in parallel; per-sub metrics to CSV and
distribution summaries to stdout. The metric that matters for the star flat is the per-star
log(scale ours / PI) scatter in the 8-12 window (noise-limited ~1.2%) with no position trend."""
import glob, json, random, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from pathlib import Path
from scipy.spatial import cKDTree

def score(sub):
    from astropy.io import fits
    from photometry.forced_photometry import measure
    from photometry.harvest import catalog, wcs_from_sidecar
    from file_io.sidecar import get_section
    try:
        t0 = time.time(); sub = Path(sub)
        obj = get_section(sub, "star_photometry")
        if not (isinstance(obj, dict) and obj.get("schema") == "astroflow.starphotometry.v1" and obj.get("data")): return None
        c = obj["data"]["columns"]; pi = np.column_stack([np.asarray(c[k], float) for k in ("x", "y", "mag_G", "colour", "fwhm", "signal", "scale")])   # PI v1: 7 columns
        with fits.open(sub, memmap=False) as h:
            hdr = h[0].header; img = h[0].data.astype(np.float32) / 65535.0
        H, W = img.shape; filt = str(hdr["FILTER"]).strip().upper()[:1]
        st = catalog().stars_in_field(wcs_from_sidecar(sub), W, H, filt); m = measure(img, st["x"], st["y"]); dt = time.time() - t0
        ok = m["ok"] & (st["catalog_flux"] > 0) & (m["snr"] > 40)
        x, y, sc, G = m["x"][ok], m["y"][ok], m["signal"][ok] / st["catalog_flux"][ok], st["G"][ok]
        d, j = cKDTree(np.c_[pi[:, 0], pi[:, 1]]).query(np.c_[x, y], distance_upper_bound=1.5); mm = np.isfinite(d); j = j[mm]
        if mm.sum() < 50: return {"sub": sub.name, "filter": filt, "n_pi": len(pi), "matched": int(mm.sum()), "secs": dt}
        ls = np.log(sc[mm]) - np.log(pi[j, 6]); ls0 = ls - np.median(ls); win = (G[mm] >= 8) & (G[mm] <= 12); bright = (G[mm] >= 8) & (G[mm] <= 10)
        rob = lambda v: 100 * 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 5 else np.nan
        return {"sub": sub.name, "filter": filt, "n_pi": len(pi), "matched": int(mm.sum()), "secs": dt,
                "rms_8_12": rob(ls0[win]), "rms_8_10": rob(ls0[bright]),
                "corr_x": float(np.corrcoef(ls0[win], x[mm][win])[0, 1]) if win.sum() > 5 else np.nan,
                "corr_y": float(np.corrcoef(ls0[win], y[mm][win])[0, 1]) if win.sum() > 5 else np.nan,
                "corr_G": float(np.corrcoef(ls0[win], G[mm][win])[0, 1]) if win.sum() > 5 else np.nan,
                "median_ratio": float(np.exp(np.median(ls)))}
    except Exception as e:
        return {"sub": Path(sub).name, "error": str(e)[:120]}

if __name__ == "__main__":
    n = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    subs = sorted(glob.glob("/Volumes/AstroRaw/Raw/**/Light*.fit.json", recursive=True))
    with_pi = []
    for p in subs:
        try:
            s = json.load(open(p)).get("astroflow", {}).get("star_photometry") or {}
            if s.get("schema") == "astroflow.starphotometry.v1" and s.get("data"): with_pi.append(p[:-5])
        except Exception: pass
    random.seed(7); pick = random.sample(with_pi, min(n, len(with_pi)))
    print(f"{len(with_pi)} subs have a PI (v1) section; scoring {len(pick)} at random with {workers} workers", flush=True)
    t0 = time.time(); rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(score, pick, chunksize=2), 1):
            if r: rows.append(r)
            if i % 200 == 0: print(f"  {i}/{len(pick)} ({(time.time()-t0):.0f}s)", flush=True)
    import csv
    keys = sorted({k for r in rows for k in r}); out = Path(sys.argv[0]).with_name("compare_engines_batch.csv")
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    good = [r for r in rows if "rms_8_12" in r and np.isfinite(r["rms_8_12"])]; err = [r for r in rows if "error" in r]; few = [r for r in rows if "rms_8_12" not in r and "error" not in r]
    print(f"\nscored {len(good)} subs; {len(few)} with <50 matched stars; {len(err)} errors ({(time.time()-t0)/60:.1f} min, {np.median([r['secs'] for r in rows if 'secs' in r]):.2f} s/sub median engine time)")
    if err: print("  example errors:", [e["error"] for e in err[:3]])
    for k, lab in (("rms_8_12", "per-star rms 8<=G<=12 [%]"), ("rms_8_10", "per-star rms 8<=G<=10 [%]"), ("corr_x", "corr(resid, x)"), ("corr_y", "corr(resid, y)"), ("corr_G", "corr(resid, G)"), ("matched", "stars matched")):
        v = np.array([r[k] for r in good if np.isfinite(r.get(k, np.nan))])
        if not len(v): continue
        print(f"  {lab:28s}: median {np.median(v):8.3f}   p10 {np.percentile(v,10):8.3f}   p90 {np.percentile(v,90):8.3f}   worst {v.max():8.3f}")
    by_f = {}
    for r in good: by_f.setdefault(r["filter"], []).append(r["rms_8_12"])
    print("  per filter, median rms 8-12 [%]:", {f: round(float(np.median(v)), 2) for f, v in sorted(by_f.items())}, {f: len(v) for f, v in sorted(by_f.items())})
    if err: print("  example errors:", [e["error"] for e in err[:3]])
