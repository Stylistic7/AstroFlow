import csv
import re

import spatial_association

_ra_regex = re.compile("(\d+)hr (\d+)' (\d+)\"")
_dec_regex = re.compile("(\d+)º (\d+)' (\d+)\"")


def _parse_ra(ra_string):
    h, m, s = _ra_regex.match(ra_string).groups()
    return 15.0 * (float(h) + float(m) / 60.0 + float(s) / 3600.0)


def _parse_dec(dec_string):
    d, m, s = _dec_regex.match(dec_string).groups()
    return float(d) + float(m) / 60.0 + float(s) / 3600.0


filename = "/Users/nobody/Projects/Astro/sadr_pano.csv"

positions = []
ids = []

with open(filename, newline="\n") as csvfile:
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    pane_index = 0
    for row in reader:
        ra, dec = _parse_ra(row["RA"]), _parse_dec(row["DEC"])
        print(f"RA {ra} DEC {dec}")
        positions.append((ra, dec))
        ids.append(f"pane_index_{pane_index}")

spatial_groups = spatial_association.find_components(
    ids, positions, min_component_members=1
)
for position_string, g in spatial_groups:
    print(f"Spatial group '{position_string}' has {len(g)} light frames")
