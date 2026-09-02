"""Headless Gaia DR3/SP (XPSD) cone search returning stars WITH XP spectra.

The catalog framework used by scripts (AstronomicalCatalogs.jsh) reads only
scalar magnitude columns, so CatalogStarGenerator cannot render narrowband
fluxes. The Gaia *process* can: a DR3/SP search returns, per source, the
sampled BP/RP spectrum as the 10th row element (verified 2026-09-01 —
row = ra, dec, parallax, pmra, pmdec, magG, magBP, magRP, flags, spectrum[]),
with the wavelength grid in databaseSpectrumStart/Step (nm). This module
runs that search through the recipe machinery and hands back plain Python.

Flux units are the database's spectral flux densities (W·m⁻²·nm⁻¹ scale;
~1e-14 for a mag-8 star). Consumers that only need ratios — the SPFC
round-trip lab — never convert them.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


_SEARCH_JS = """
let g = new Gaia;
g.command = "search";
// The recipe engine does not populate process enum constants
// (Gaia.prototype.DataRelease_3_SP is undefined here); 3 is DR3/SP,
// verified 2026-09-01 by spectra coming back in the source rows.
let dr = Gaia.prototype.DataRelease_3_SP;
g.dataRelease = (typeof dr == "number") ? dr : 3;
g.centerRA = {center_ra};
g.centerDec = {center_dec};
g.radius = {radius};
g.magnitudeLow = {magnitude_low};
g.magnitudeHigh = {magnitude_high};
g.sourceLimit = {source_limit};
g.exclusionFlags = 0;
g.inclusionFlags = 0;
g.verbosity = 1;
g.generateTextOutput = false;
if (!g.executeGlobal())
    throw new Error("Gaia DR3/SP search failed");

let f = new File;
f.createForWriting("{output_path}");
f.outTextLn("{{");
f.outTextLn('"spectrum_start_nm": ' + g.databaseSpectrumStart + ",");
f.outTextLn('"spectrum_step_nm": ' + g.databaseSpectrumStep + ",");
f.outTextLn('"stars": [');
let first = true;
for (let i = 0; i < g.sources.length; ++i) {{
    let s = g.sources[i];
    if (!isFinite(s[0]) || !isFinite(s[1]) || !isFinite(s[5]))
        continue;
    let spec = s[9];
    let bad = false;
    for (let j = 0; j < spec.length; ++j)
        if (!isFinite(spec[j])) {{ bad = true; break; }}
    if (bad)
        continue;
    if (!first)
        f.outTextLn(",");
    first = false;
    f.outText('[' + s[0] + ',' + s[1] + ',' + s[5] + ',[' +
              spec.join(',') + ']]');
}}
f.outTextLn("");
f.outTextLn("]}}");
f.close();
"""


def query_xpsd_stars(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    magnitude_high: float,
    magnitude_low: float = -1.5,
    source_limit: int = 20_000,
) -> dict:
    """Cone-search the local Gaia DR3/SP database, spectra included.

    Returns ``{"wavelength_nm": np.ndarray, "stars": [(ra_deg, dec_deg,
    g_mag, spectrum: np.ndarray), ...]}`` with every spectrum sampled on
    the shared ``wavelength_nm`` grid.
    """
    from astropy.io import fits
    from pix.image import Image
    from pix.recipe import Step

    # Cleaned in the finally below; a query killed mid-write strands the
    # dir (155 MB of stars.json), which the janitor's gaia_xpsd spec reaps.
    tmp = Path(tempfile.mkdtemp(prefix="gaia_xpsd_"))
    anchor = tmp / "anchor.fits"
    fits.PrimaryHDU(data=np.zeros((16, 16), np.float32)).writeto(anchor)
    out_json = tmp / "stars.json"

    try:
        img = Image.open(anchor)
        img.recipe.add_step(Step(
            name="gaia xpsd search",
            javascript=_SEARCH_JS.format(
                center_ra=float(center_ra_deg),
                center_dec=float(center_dec_deg),
                radius=float(radius_deg),
                magnitude_low=float(magnitude_low),
                magnitude_high=float(magnitude_high),
                source_limit=int(source_limit),
                output_path=out_json,
            ),
        ))
        img.save_to(tmp / "anchor_out.xisf")
        raw = json.loads(out_json.read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not raw["stars"]:
        raise RuntimeError(
            f"Gaia DR3/SP search returned no stars at "
            f"({center_ra_deg}, {center_dec_deg}) r={radius_deg}"
        )
    n = len(raw["stars"][0][3])
    wavelength_nm = (raw["spectrum_start_nm"]
                     + raw["spectrum_step_nm"] * np.arange(n))
    return {
        "wavelength_nm": wavelength_nm,
        "stars": [
            (ra, dec, g_mag, np.asarray(spec, dtype=np.float64))
            for ra, dec, g_mag, spec in raw["stars"]
        ],
    }
