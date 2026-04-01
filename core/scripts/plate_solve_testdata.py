"""Re-generate sidecars for all testdata FITS files.

Run this whenever the sidecar schema changes or testdata is added:
    python scripts/plate_solve_testdata.py

Each light frame gets plate-solved via solve-field, then full
ImageMetadata (including plate solution) is written to the JSON sidecar.
This ensures that compute_image_metadata can return instantly from the
sidecar cache during tests, with no FITS I/O or plate solving.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plate_solving
from file_io.sidecar import load_sidecar, save_sidecar, sidecar_path_for
from file_io.generic import read_header, get_content_hash, read_image_geometry
from metadata.sidecar_io import write_image_metadata
from metadata.classes import ImageMetadata
import metadata.image_types as image_types

testdata_dir = Path(__file__).resolve().parent.parent / "testdata" / "input"
fits_files = sorted(testdata_dir.glob("*.fit"))
print(f"Found {len(fits_files)} FITS files in {testdata_dir}")

failed = []
for filename in fits_files:
    # Clear stale sections so we always write fresh.
    sc_path = sidecar_path_for(filename)
    root = load_sidecar(sc_path)
    root["astroflow"].pop("plate_solution", None)
    root["astroflow"].pop("image_metadata", None)
    save_sidecar(sc_path, root)

    # Plate-solve light frames.
    header = read_header(filename)
    plate_solution = None
    if image_types.is_light_frame(header):
        plate_solution = plate_solving.solve_field(filename, debug=True)
        if plate_solution is None:
            print(f"FAILED: {filename.name}")
            failed.append(filename)
            continue

    # Write full ImageMetadata to sidecar.
    content_hash = get_content_hash(filename)
    geometry = read_image_geometry(filename)
    meta = ImageMetadata(
        content_hash=content_hash,
        stable_hash=header.get("STBLHASH", content_hash),
        filename=filename,
        header=header,
        width_pixels=geometry.width_pixels,
        height_pixels=geometry.height_pixels,
        channels=geometry.channels,
        dtype=geometry.dtype,
        plate_solution=plate_solution,
    )
    write_image_metadata(meta)

    stars = f", {plate_solution.num_stars} stars" if plate_solution else ""
    print(f"OK: {filename.name}{stars}")

if failed:
    print(f"\n{len(failed)} files failed to solve:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"\nAll {len(fits_files)} files ready.")
