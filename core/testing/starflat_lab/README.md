# Star-flat lab (2026-09-06 checkpoint)

Method and findings for measuring the 6200MM Hα **etalon** (and vignetting)
from Gaia stars instead of a flat panel. Library code: `stacks/starflat.py`.
Drivers here reproduce the experiment on any sub list. Full narrative and
numbers: memory `superflat-etalon-derived`.

## Why
* The Hα etalon (silicon interference, sensor-fixed, ~0.6% rms, 64–400 px
  blobs + ~0.1% fine rings) is invisible to a continuum panel flat (0.16%,
  uncorrelated) and untouched by GraXpert/MGC. A sky superflat captures it
  only with heavy contamination (centred targets → a 2.6% central dome).
* SPFC already measures every star against its Gaia-XP-synthetic Hα flux;
  `generateTextFiles=True` exports the per-star gains (`scale`).

## Workflow
1. `harvest.py <sub_list.txt> <out_dir>` — SPFC per-star gains for each
   plate-solved raw sub (tolerance 12 px). Checkpointed; resume-safe.
   **Never run alongside WBPP on the same PixInsight slot** (300 s lock
   timeout → subs fail). Only subs with an accurate WCS: the ±90° handoff
   mirrors the WCS (production bug, `write_astrometric_solution.py`), so
   those subs mis-match and get dropped by the contamination filter.
2. `build.py <out_dir>` — joint solve, split-half convergence, smoothed
   response FITS and Master Flat XISF (template = any WBPP master flat).
3. `e2e.py <out_dir> <flat.fits>` — WBPP with darks, three arms
   (none / etalon-only / full), judged by nebula-poor sky texture and by
   SPFC photometric flatness of the OUTPUT masters, plus a fixed-stretch
   visual. This is the only trustworthy judge: dividing a finished,
   registered, un-dark-subtracted master afterwards reads the opposite.

## Results (295 clean subs, 377,839 stars, 256 px cells)
* split-half r = 0.965 raw / 0.924 etalon band (17 subs 0.38 → 69 0.81 →
  114 0.86 → 295 0.92); 397 stars/cell, 0.15%/cell.
* End-to-end (10 Cyg 1 subs + 20 darks): nebula-poor etalon-band texture
  no-flat 0.388% → etalon-only 0.335% → **full 0.317%**; output-master
  star flatness 1.42% → 1.35% → **1.20%**; after production GraXpert the
  gain persists (0.39% → 0.32%) — GraXpert leaves the etalon untouched.
* Held-out photometry on disjoint subs: star flat removes a 0.58%-rms
  component (the etalon amplitude); the panel flat removes nothing.
* Density: 1 star / 318 px², corners ~93% of centre; 128 px cells are
  S/N ~3.5 at the full library, 64 px needs the ±90° fix + übercal.
