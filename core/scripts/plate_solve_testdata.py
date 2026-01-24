import pickle
from pathlib import Path

import plate_solving

for filename in Path("testdata/input").rglob("*.fit"):
    filename = filename.resolve()
    plate_solution = plate_solving.solve_field(
        filename, allow_sidecars=False, debug=True
    )
    assert plate_solution is not None
    sidecar = filename.with_suffix(".pkl")
    with open(sidecar, "wb") as f:
        print(f"### Wrote plate-solving sidecar {sidecar}")
        pickle.dump(plate_solution, f)
