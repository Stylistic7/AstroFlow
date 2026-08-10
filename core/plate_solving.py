from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path
import warnings
import astropy.wcs  # type: ignore
from astropy.wcs import FITSFixedWarning  # type: ignore
import astropy.io.fits as pyfits  # type: ignore

warnings.filterwarnings("ignore", category=FITSFixedWarning)
from dataclasses import dataclass, is_dataclass, asdict
import numpy as np
import traceback
from typing import Any
from collections.abc import Mapping

from beartype import beartype

from file_io.conversion import convert_xisf_to_fits
from file_io.sidecar import get_section, set_section, sidecar_path_for
from units import Degrees, Radians, ArcsecondsPerPixel
_PLATESOLUTION_SCHEMA = "astroflow.platesolution.v2"


class PlateSolveFailedException(Exception):
    """Raised when solve-field fails on a light frame that was expected
    to solve. Frames carrying the DONTSOLV header (synthetic noise used
    by tests) do not raise — they simply return None from solve_field."""


@dataclass(frozen=True)
class PlateSolution:
    image_filename: str
    header: dict
    num_stars: int
    center: tuple[Degrees, Degrees]
    image_scale: ArcsecondsPerPixel
    rotation_angle: Radians

    def __setstate__(self, state):
        # Drop fields removed in v2 (e.g. footprint) so old pickled
        # PlateSolution objects can still be deserialized.
        valid_fields = {f.name for f in self.__dataclass_fields__.values()}
        for key in list(state):
            if key not in valid_fields:
                del state[key]
        for key, value in state.items():
            object.__setattr__(self, key, value)


@beartype
def _ensure_naxis(header: dict[str, Any]) -> None:
    """Ensure NAXIS, NAXIS1, NAXIS2 are present in the WCS header.

    solve-field outputs IMAGEW/IMAGEH but not always NAXIS/NAXIS1/NAXIS2.
    Astropy WCS needs these for footprint_contains() and to avoid warnings.
    """
    if "NAXIS1" not in header and "IMAGEW" in header:
        header["NAXIS1"] = header["IMAGEW"]
    if "NAXIS2" not in header and "IMAGEH" in header:
        header["NAXIS2"] = header["IMAGEH"]
    if "NAXIS" not in header and "NAXIS1" in header:
        header["NAXIS"] = 2


@beartype
def get_center(header: dict[str, Any]) -> tuple[Degrees, Degrees]:
    return Degrees(header["CRVAL1"]), Degrees(header["CRVAL2"])


@beartype
def get_rotation_angle(header: dict[str, Any]) -> Radians:
    return Radians(np.arctan2(header["CD1_2"], header["CD1_1"]))


@beartype
def get_image_scale(header: dict[str, Any]) -> ArcsecondsPerPixel:
    scale_x = np.linalg.norm(np.array([header["CD1_1"], header["CD2_1"]]))
    scale_y = np.linalg.norm(np.array([header["CD1_2"], header["CD2_2"]]))
    assert np.isclose(scale_x, scale_y, atol=0.05 / 3600, rtol=0), (
        f"scale_x {scale_x} is not close to scale_y {scale_y}"
    )

    return ArcsecondsPerPixel(scale_x * 3600.0)


@beartype
def get_num_stars(axy_file: Path | str) -> int:
    fits = pyfits.open(axy_file, mmap=False)
    table = fits[1].data
    fits.close()
    return table.shape[0]



@beartype
def simplify_header(header: Mapping[str, Any]) -> dict[str, Any]:
    output_header = {}
    for k in header.keys():
        if "COMMENT" in k or "HISTORY" in k:
            continue
        output_header[k] = header[k]

    return output_header


@beartype
def _solve_field(
    filename: Path | str, arguments: list[str], debug: bool = False
) -> PlateSolution | None:
    # TODO: Only copy the file to a temp dir once.
    with tempfile.TemporaryDirectory() as temp_dir:
        if os.path.splitext(filename)[-1] in [".xisf"]:
            dest_filename = Path(
                os.path.join(temp_dir, os.path.basename(filename))
            ).with_suffix(".fits")
            convert_xisf_to_fits(filename, dest_filename)
        else:
            dest_filename = Path(
                os.path.join(temp_dir, os.path.basename(filename))
            )
            os.symlink(Path(filename).resolve(), dest_filename)

        try:
            command = arguments + [dest_filename]
            if debug:
                print("### command " + " ".join([str(i) for i in command]))

            ret = subprocess.run(command, capture_output=True)

            if debug:
                print(ret.stdout.decode("utf-8"))
        except Exception:
            print(
                f"Could not plate solve file {dest_filename} due to exception"
            )
            traceback.print_exc()
            raise

        wcs_file = Path(dest_filename).with_suffix(".wcs")
        axy_file = Path(dest_filename).with_suffix(".axy")
        if os.path.exists(wcs_file) and os.path.exists(axy_file):
            fits = pyfits.open(wcs_file, mmap=False)
            header = simplify_header(fits[0].header)
            _ensure_naxis(header)
            try:
                num_stars = get_num_stars(axy_file)
                soln = PlateSolution(
                    image_filename=str(filename),
                    header=header,
                    center=get_center(header),
                    image_scale=get_image_scale(header),
                    rotation_angle=get_rotation_angle(header),
                    num_stars=num_stars,
                )
                print(
                    f"### Sucessfully plate-solved file {filename}, "
                    f"has {num_stars} stars"
                )
                return soln
            except Exception:
                print(f"Could not plate solve file {filename} due to exception")
                traceback.print_exc()
                raise
            finally:
                fits.close()
        else:
            # TODO: Emit warnings through Metaflow's reporting system.
            # print(f'Could not plate-solve file {filename}')
            return None


# TODO: Make a "junk drawer" video that shows everything that can't be plate-
# solved. The user can look at it and decide whether to improve the solver's
# settings, or to throw the images away.
@beartype
def _plate_solution_to_obj(ps: PlateSolution) -> dict[str, Any]:
    d = asdict(ps) if is_dataclass(ps) else dict(ps)  # type: ignore[arg-type]
    c = d.get("center")
    if isinstance(c, tuple):
        d["center"] = list(c)
    return {"schema": _PLATESOLUTION_SCHEMA, "data": d}


@beartype
def _plate_solution_from_obj(obj: dict[str, Any]) -> PlateSolution:
    if obj.get("schema") != _PLATESOLUTION_SCHEMA:
        raise ValueError(
            f"Unsupported plate_solution schema: {obj.get('schema')!r}"
        )
    d = obj["data"]
    if "header" in d and isinstance(d["header"], dict):
        _ensure_naxis(d["header"])
    if "center" in d:
        c = d["center"]
        if isinstance(c, list):
            c = tuple(c)
        d["center"] = (Degrees(c[0]), Degrees(c[1]))
    if "image_scale" in d and d["image_scale"] is not None:
        d["image_scale"] = ArcsecondsPerPixel(d["image_scale"])
    if "rotation_angle" in d and d["rotation_angle"] is not None:
        d["rotation_angle"] = Radians(d["rotation_angle"])
    d.pop("footprint", None)
    return PlateSolution(**d)


@beartype
def solve_field(
    filename: Path | str,
    debug: bool = False,
) -> PlateSolution | None:
    filename = Path(filename)
    assert str(filename)

    # Always consult sidecar first.
    ps_obj = get_section(filename, "plate_solution")
    if isinstance(ps_obj, dict):
        try:
            if ps_obj.get("data") is None:
                # Previously failed to solve — don't retry.
                return None
            ps = _plate_solution_from_obj(ps_obj)
            print(
                f"### Returning plate-solution from sidecar "
                f"{sidecar_path_for(filename)}"
            )
            return ps
        except Exception as e:
            # Corrupt / incompatible section → recompute.
            print(f"### plate_solving: ignoring cached solution for {filename}: {e}")
            pass

    # Synthetic noise frames opt out of solving via the DONTSOLV header.
    from file_io.generic import read_header
    header = read_header(filename)
    if header.get("DONTSOLV"):
        print(f"### Skipping solve for {filename} (DONTSOLV set)")
        return None

    # EVERY solve caps the source list and gets a generous CPU budget.
    # This used to be master-only (gated on is_master header divination),
    # leaving every framework-PRODUCED deep image — BXT'd channel masters,
    # blends — at solve-field's uncapped defaults. On a dense field that is
    # the blind quad-match blow-up: the stuck Cygnus SII panel extracted
    # 38,004 sources and hit "CPU time limit reached!" deterministically in
    # TWO universes (it was the real cause of the pre-wipe "S never builds"
    # mystery), while --objs 1000 solved the same image in 177 s at
    # log-odds 342. The brightest ~1000 sources are ample for ANY solve
    # (it needs dozens) and make subframes faster too — so one arg set for
    # everything, no image-class guessing.
    base = [
        "solve-field",
        "--overwrite",
        "--no-plots",
        "--new-fits",
        "none",
        "--scale-low",
        "0.1",
        "--scale-high",
        "180",
        "--scale-units",
        "degwidth",
        "--tweak-order",
        "2",
        "--crpix-center",
        "--cpulimit",
        "600",
        "--objs",
        "1000",
    ]
    # RETRY LADDER, universal (no image-class guessing). Rung 1 is the
    # standard extraction. Rung 2 exists for nebula-dominated narrowband
    # fields (WR134/Veil OIII + SII, 2026-08-09): at --downsample 2 the
    # extractor fills its 1000-source budget with FILAMENT KNOTS, the quads
    # match nothing, and a perfectly solvable frame burns the full CPU
    # budget — 26 of 27 "unsolvable" archive subs were this. --downsample 4
    # smooths the small-scale nebulosity away; real stars survive and the
    # same frame solves blind. Tombstone only after BOTH rungs fail.
    result = None
    for downsample in ("2", "4"):
        result = _solve_field(
            filename, base + ["--downsample", downsample], debug,
        )
        if result is not None:
            if downsample != "2":
                print(f"### plate-solve rescued by --downsample {downsample}: "
                      f"{filename}")
            break

    if result is None:
        print(f"Could not plate-solve file {filename} (all retry rungs)")
        set_section(
            filename, "plate_solution", {"schema": _PLATESOLUTION_SCHEMA, "data": None}
        )
        return None

    set_section(
        filename, "plate_solution", _plate_solution_to_obj(result)
    )
    print(f"### Updated AstroFlow sidecar for {filename}")

    return result


if __name__ == "__main__":
    solve_field(Path(sys.argv[1]), debug=True)
