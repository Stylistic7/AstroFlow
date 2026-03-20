from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path
import astropy.wcs  # type: ignore
import astropy.io.fits as pyfits  # type: ignore
import warnings
from dataclasses import dataclass, is_dataclass, asdict
import numpy as np
import traceback
from typing import Any

from file_io.conversion import convert_xisf_to_fits
from file_io.sidecar import get_section, set_section, sidecar_path_for
from units import Degrees, Radians, ArcsecondsPerPixel

_PLATESOLUTION_SCHEMA = "astroflow.platesolution.v2"


@dataclass
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
        self.__dict__.update(state)


def _ensure_naxis(header: dict[str, Any]) -> None:
    """Copy IMAGEW/IMAGEH to NAXIS1/NAXIS2 if missing.

    solve-field outputs IMAGEW/IMAGEH but not always NAXIS1/NAXIS2.
    Astropy WCS needs NAXIS1/NAXIS2 for footprint_contains() and similar.
    """
    if "NAXIS1" not in header and "IMAGEW" in header:
        header["NAXIS1"] = header["IMAGEW"]
    if "NAXIS2" not in header and "IMAGEH" in header:
        header["NAXIS2"] = header["IMAGEH"]


def get_center(header: dict[str, Any]) -> tuple[Degrees, Degrees]:
    return Degrees(header["CRVAL1"]), Degrees(header["CRVAL2"])


def get_rotation_angle(header: dict[str, Any]) -> Radians:
    return Radians(np.arctan2(header["CD1_2"], header["CD1_1"]))


def get_image_scale(header: dict[str, Any]) -> ArcsecondsPerPixel:
    scale_x = np.linalg.norm(np.array([header["CD1_1"], header["CD2_1"]]))
    scale_y = np.linalg.norm(np.array([header["CD1_2"], header["CD2_2"]]))
    assert np.isclose(scale_x, scale_y, atol=0.05 / 3600, rtol=0), (
        f"scale_x {scale_x} is not close to scale_y {scale_y}"
    )

    return ArcsecondsPerPixel(scale_x * 3600.0)


def get_num_stars(axy_file: Path | str) -> int:
    fits = pyfits.open(axy_file, mmap=False)
    table = fits[1].data
    fits.close()
    return table.shape[0]


def simplify_header(header: dict[str, Any]) -> dict[str, Any]:
    output_header = {}
    for k in header.keys():
        if "COMMENT" in k or "HISTORY" in k:
            continue
        output_header[k] = header[k]

    return output_header


def _solve_field(
    filename: Path | str, arguments: list[str], debug: bool = False
) -> PlateSolution | Exception | None:
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
        except Exception as e:
            print(
                f"Could not plate solve file {dest_filename} due to exception"
            )
            traceback.print_exc()
            return e

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
            except Exception as e:
                print(f"Could not plate solve file {filename} due to exception")
                traceback.print_exc()
                return e
            finally:
                fits.close()
        else:
            # TODO: Emit warnings through Metaflow's reporting system.
            # print(f'Could not plate-solve file {filename}')
            return None


# TODO: Make a "junk drawer" video that shows everything that can't be plate-
# solved. The user can look at it and decide whether to improve the solver's
# settings, or to throw the images away.
def _plate_solution_to_obj(ps: PlateSolution) -> dict[str, Any]:
    d = asdict(ps) if is_dataclass(ps) else dict(ps)  # type: ignore[arg-type]
    c = d.get("center")
    if isinstance(c, tuple):
        d["center"] = list(c)
    return {"schema": _PLATESOLUTION_SCHEMA, "data": d}


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


def solve_field(
    filename: Path | str, debug: bool = False
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
        except Exception:
            # Corrupt / incompatible section → recompute.
            pass

    trial_arguments = [
        [
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
            "--downsample",
            "2",
            "--objs",
            "1000",
            "--tweak-order",
            "2",
            "--crpix-center",
        ]
    ]

    final_result: PlateSolution | None = None
    for arguments in trial_arguments:
        result = _solve_field(filename, arguments, debug)
        if type(result) is PlateSolution:
            final_result = result
            break

    if final_result is None:
        print(f"Could not plate-solve file {filename}")
        set_section(
            filename, "plate_solution", {"schema": _PLATESOLUTION_SCHEMA, "data": None}
        )
        return None

    # Persist result unconditionally.
    set_section(
        filename, "plate_solution", _plate_solution_to_obj(final_result)
    )
    print(f"### Updated AstroFlow sidecar for {filename}")

    return final_result


if __name__ == "__main__":
    solve_field(Path(sys.argv[1]), debug=True)
