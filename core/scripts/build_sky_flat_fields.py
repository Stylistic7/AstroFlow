"""Build the catalogue of blank-sky fields for night-sky flats.

Run once (needs the Gaia XP catalogue and, for the brightest stars Gaia
lacks, network access to the CDS name resolver):

    python -m scripts.build_sky_flat_fields

Writes `stacks/sky_flat_fields.json`, which ships with the code: field choice
is a pure function of that file, the rig and the night.

A night-sky flat wants the emptiest sky there is: far from the Milky Way and
from the ecliptic (no planet or asteroid wandering through the panes),
with no bright star whose halo or reflections would survive the median,
and no nebula, dark cloud or large galaxy in the frame. Candidates are
spread evenly over the whole sky (both hemispheres, so any site has some),
those that fail a hard rule are dropped, the rest are scored by how many
stars they hold, and the best one in every cell of 1 h of RA x 30 deg of Dec
is kept, so on any night some field is well placed from anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parents[1] / "stacks" / "sky_flat_fields.json"
N_CANDIDATES = 8000
# Two tiers, each judged at its own clearance radius: WIDE for short focal
# lengths (the 6200MM at 350 mm has a 3.5 deg half-diagonal) and NARROW for
# long ones (1.0 deg at 1230 mm). One radius for both starved the narrow rigs
# of fields wherever a bright star or a galaxy group sits a few degrees off.
TIERS = (("wide", 4.0), ("narrow", 1.5))
MIN_GAL_LAT = 25.0          # |b|: the Milky Way's stars and dust stay out
MIN_ECL_LAT = 20.0          # |beta|: planets keep within ~7 deg of the ecliptic, most asteroids within ~20
BRIGHT_G = 5.5              # no Gaia star brighter than this inside the radius
NAKED_EYE_MARGIN = 1.5      # extra clearance (deg) around the brightest stars Gaia lacks
OBJECT_MIN_RADIUS = 0.1     # deg: nebulae, dark clouds and galaxies larger than this are avoided
COUNT_G = 11.0              # the score: stars brighter than this inside the radius
DEC_BANDS = (-90, -60, -30, 0, 30, 60, 90)

# Brighter than about G 2.7, where the Gaia XP catalogue starts. Positions come
# from the CDS resolver when this script runs; the list only names them.
BRIGHTEST = """Sirius Canopus Arcturus Vega Capella Rigel Procyon Achernar Betelgeuse
Hadar Altair Acrux Aldebaran Antares Spica Pollux Fomalhaut Deneb Mimosa Regulus Adhara
Castor Gacrux Shaula Bellatrix Elnath Miaplacidus Alnilam Alnair Alnitak Alioth Dubhe
Mirfak Wezen Sargas Avior Alkaid Menkalinan Atria Alhena Peacock Mirzam Alphard Polaris
Hamal Algieba Diphda Nunki Menkent Mirach Alpheratz Rasalhague Kochab Saiph Denebola
Algol Suhail Alphecca Mizar Sadr Schedar Eltanin Mintaka Caph Dschubba Merak Ankaa Enif
Scheat Sabik Phecda Aludra Markeb Markab Zosma Arneb Alderamin""".split()
BRIGHTEST += ["alf Cen", "Kaus Australis", "del Vel", "bet Gru", "gam Cen", "iot Car",
              "gam Cas", "eps Cyg", "eps Sco", "kap Sco", "bet Sco"]


def unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def chord(deg):
    return 2 * np.sin(np.radians(deg) / 2)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    dec = np.degrees(np.arcsin(1 - 2 * i / n))
    ra = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
    return ra, dec


def brightest_stars():
    from astropy.coordinates import SkyCoord
    out = []
    for name in BRIGHTEST:
        try:
            c = SkyCoord.from_name(name)
            out.append((name, float(c.ra.deg), float(c.dec.deg)))
        except Exception as e:                        # a name the resolver does not know
            print(f"  could not resolve {name}: {e}", file=sys.stderr)
    return out


def extended_objects():
    from catalogs import messier, ngc2000_catalog as ngc, vizier_catalogs as viz
    names = ngc._read_names_catalog()
    objs = [(o["ra_degrees"], o["dec_degrees"], o.get("radius") or 0.0, o.get("designation", o.get("id")))
            for o in ngc._read_ngc2000_catalog(names) + viz.read_vizier_catalogs(names)]
    big = [o for o in objs if o[2] >= OBJECT_MIN_RADIUS]
    for m in messier.read_messier_entries():          # every Messier object, whatever its size
        try:
            big.append((float(m["ra_degrees"]), float(m["dec_degrees"]), max(float(m.get("radius") or 0.2), 0.2), m.get("designation", "M")))
        except (KeyError, TypeError, ValueError):
            pass
    return big


def main() -> int:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from photometry.harvest import catalog

    print("loading Gaia ...", flush=True)
    cat = catalog()
    G = np.asarray(cat.G)
    counted = G < COUNT_G
    bright = G < BRIGHT_G
    star_tree = cKDTree(unit(np.asarray(cat.ra)[counted], np.asarray(cat.dec)[counted]))
    bright_tree = cKDTree(unit(np.asarray(cat.ra)[bright], np.asarray(cat.dec)[bright]))
    print("resolving the brightest stars ...", flush=True)
    naked = brightest_stars()
    naked_tree = cKDTree(unit([s[1] for s in naked], [s[2] for s in naked]))
    objs = extended_objects()
    obj_xyz = unit([o[0] for o in objs], [o[1] for o in objs])
    obj_rad = np.array([o[2] for o in objs])

    ra, dec = fibonacci_sphere(N_CANDIDATES)
    xyz = unit(ra, dec)
    sky = SkyCoord(ra * u.deg, dec * u.deg)
    b = sky.galactic.b.deg
    beta = sky.transform_to("barycentricmeanecliptic").lat.deg
    keep = (np.abs(b) >= MIN_GAL_LAT) & (np.abs(beta) >= MIN_ECL_LAT)
    out = []
    for tier, radius in TIERS:
        r = chord(radius)
        fields = []
        for i in np.flatnonzero(keep):
            if bright_tree.query_ball_point(xyz[i], r):
                continue
            if naked_tree.query_ball_point(xyz[i], chord(radius + NAKED_EYE_MARGIN)):
                continue
            sep = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(obj_xyz - xyz[i], axis=1) / 2, 0, 1)))
            if np.any(sep < radius + obj_rad):
                continue
            n = len(star_tree.query_ball_point(xyz[i], r))
            fields.append((float(ra[i]), float(dec[i]), float(b[i]), float(beta[i]), n))
        print(f"{tier}: {len(fields)} of {int(keep.sum())} candidates pass the hard rules", flush=True)
        best = {}
        for f in fields:
            cell = (int(f[0] // 15), int(np.digitize(f[1], DEC_BANDS)))
            if cell not in best or f[4] < best[cell][4]:
                best[cell] = f
        for (h, band), (fra, fdec, fb, fbeta, n) in sorted(best.items()):
            sign = "+" if fdec >= 0 else "-"
            out.append({
                "id": f"{tier[0].upper()}{int(round(fra / 15 * 10)):03d}{sign}{int(round(abs(fdec))):02d}",
                "tier": tier, "clear_radius_deg": radius,
                "ra_deg": round(fra, 4), "dec_deg": round(fdec, 4),
                "gal_lat_deg": round(fb, 1), "ecl_lat_deg": round(fbeta, 1),
                "stars_brighter_than_g11": n,
            })
    OUT.write_text(json.dumps({
        "about": "Blank-sky fields for night-sky flats; written by scripts/build_sky_flat_fields.py",
        "rules": {"tiers": dict(TIERS), "min_gal_lat_deg": MIN_GAL_LAT, "min_ecl_lat_deg": MIN_ECL_LAT, "no_star_brighter_than_g": BRIGHT_G,
                  "naked_eye_margin_deg": NAKED_EYE_MARGIN, "object_min_radius_deg": OBJECT_MIN_RADIUS,
                  "score": f"Gaia stars brighter than G {COUNT_G} within the tier's radius"},
        "fields": out,
    }, indent=1) + "\n")
    print(f"wrote {len(out)} fields to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
