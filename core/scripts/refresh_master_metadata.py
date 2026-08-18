"""Refresh master sidecars with the fields compute_image_metadata now emits.

Existing stacked masters never re-run their metadata computation: it executes
inside `stack_files`, whose cache-hit returns the old pickled result — so
sidecars written before `noise_estimate` / the numberOfImages-based
`master_integration_s` existed stay stale forever (only a physical re-stack
refreshes them). This script performs the same canonical reads the op does —
header via `read_xisf_header`, integration via
`read_master_integration_seconds`, MRS noise via the inline `NOISE00`
keyword — and rewrites ONLY those fields in each sidecar's image_metadata
section. Pixel statistics are untouched (same pixels, still valid).

Safe to run against a live app: pure file reads + sidecar section writes, no
Ray, no context import. Idempotent; a future re-stack overwrites with
identical values.

Usage:  python scripts/refresh_master_metadata.py [--dry-run] [DIR]
        (default DIR: /Volumes/AstroProcessed/Processed/stack_files)
"""
from __future__ import annotations

import sys
from pathlib import Path

from file_io.master_integration import read_master_integration_seconds
from file_io.xisf_file_io import read_xisf_header
from metadata.sidecar_io import get_section, set_section


def refresh(master: Path, dry_run: bool = False) -> str:
    obj = get_section(master, "image_metadata")
    if not isinstance(obj, dict) or not obj:
        return "no-sidecar-section"  # never stacked/analyzed -> op will do it
    header = read_xisf_header(master)
    integration = read_master_integration_seconds(master, header)
    noise = header.get("NOISE00")
    noise = float(noise) if isinstance(noise, (int, float, str)) and str(noise) else None
    changed = []
    if integration is not None and obj.get("master_integration_s") != integration:
        obj["master_integration_s"] = integration
        changed.append(f"integration={integration:.0f}s")
    if noise is not None and obj.get("noise_estimate") != noise:
        obj["noise_estimate"] = noise
        changed.append(f"noise={noise:.3e}")
    if not changed:
        return "up-to-date"
    if not dry_run:
        set_section(master, "image_metadata", obj)
    return ", ".join(changed)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    root = Path(args[0]) if args else Path("/Volumes/AstroProcessed/Processed/stack_files")
    masters = sorted(root.glob("*.xisf"))
    counts: dict[str, int] = {}
    for m in masters:
        try:
            result = refresh(m, dry_run=dry)
        except Exception as e:  # one bad file must not stop the pass
            result = f"ERROR: {e}"
            print(f"  {m.name}: {result}")
        key = result.split(":")[0].split(",")[0].split("=")[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"{'DRY RUN — ' if dry else ''}{len(masters)} masters: {counts}")


if __name__ == "__main__":
    main()
