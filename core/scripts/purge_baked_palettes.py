"""One-shot cleanup after the baked-palette removal (2026-08-12): delete
every file and version-log chain for the displays the colour tail no longer
produces. Version-log entries PIN files against the orphan sweep by design,
so a deliberate delete must clear the chains too (the sweep then handles any
stragglers on later passes).

Ray-free: reads/writes the durable cache directly, like the sweep CLI.

    PYTHONPATH=. python -m scripts.purge_baked_palettes [--delete]
"""
from __future__ import annotations

import os
import sys

DEAD_DISPLAYS = {
    "Hubble", "Foraxx", "Hubble Starless", "Foraxx Starless", "Stars",
    "RGB", "LRGB", "HaRGB", "HaLRGB",
}


def main() -> None:
    from context.cache import get_durable_cache
    from products import version_log

    delete = "--delete" in sys.argv
    dc = get_durable_cache()
    total_bytes = 0
    total_files = 0
    chains = 0
    for k in list(dc):
        if not (isinstance(k, tuple) and k and k[0] == "product_versions"):
            continue
        # key = ("product_versions", object_type, object_id, pipeline,
        #        display, filter)
        if len(k) < 5 or str(k[4]) not in DEAD_DISPLAYS:
            continue
        chains += 1
        for e in dc.get(k) or []:
            for f in version_log._files_of(e.get("path", "")):
                try:
                    sz = os.path.getsize(f)
                except OSError:
                    continue
                total_files += 1
                total_bytes += sz
                if delete:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
        if delete:
            del dc[k]
    action = "deleted" if delete else "would delete"
    print(f"{chains} dead-display version chains; {action} {total_files} "
          f"files, {total_bytes / 1e9:.1f} GB")
    if not delete:
        print("Dry run. Re-run with --delete.")


if __name__ == "__main__":
    main()
