"""harvest.py <sub_list.txt> <out_dir> [tolerance_px]
SPFC per-star gains for every plate-solved raw sub in the list; writes
<out_dir>/stars.npy (STAR_COLUMNS with sub id = line index) and subs.txt.
Resume-safe: subs whose CSV exists are skipped. Do NOT run concurrently with
WBPP on the same PixInsight slot."""
import os, sys, time, numpy as np
from stacks.starflat import harvest_sub

lst, out = sys.argv[1], sys.argv[2]; tol = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
subs = [l.strip() for l in open(lst) if l.strip()]; os.makedirs(out, exist_ok=True)
rows, done, t0 = [], [], time.time()
for i, p in enumerate(subs):
    d = os.path.join(out, f"{i:05d}")
    try:
        if os.path.exists(os.path.join(d, "SPFC-data-0.csv")):
            a = np.genfromtxt(os.path.join(d, "SPFC-data-0.csv"), delimiter=",", names=True).reshape(-1)
            tab = np.column_stack([a["x"], a["y"], a["alpha"], a["delta"], a["mag_G"], a["mag_BP"], a["mag_RP"], (a["img_fwhmx"] + a["img_fwhmy"]) / 2, a["img_signal"], a["scale"]])
        else:
            tab, _ = harvest_sub(p, d, search_tolerance_px=tol)
        rows.append(np.column_stack([tab, np.full(len(tab), i)])); done.append((i, p, len(tab)))
    except Exception as e:
        done.append((i, p, 0)); print(f"  {i} FAILED {str(e)[:70]}", flush=True)
    if i % 25 == 24 or i == len(subs) - 1:
        np.save(os.path.join(out, "stars.npy"), np.vstack(rows)); open(os.path.join(out, "subs.txt"), "w").write("\n".join(f"{a}\t{b}\t{c}" for a, b, c in done))
        print(f"  ... {i+1}/{len(subs)} subs, {sum(len(r) for r in rows)} stars, {(time.time()-t0)/60:.1f} min", flush=True)
print("HARVEST DONE")
