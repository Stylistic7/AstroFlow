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
from dataclasses import dataclass, is_dataclass, asdict, replace
import numpy as np
import traceback
from typing import Any
from collections.abc import Mapping

from beartype import beartype

import node_settings
from third_party_tools import tool_path
from file_io.conversion import convert_xisf_to_fits
from file_io.links import link_or_copy
from file_io.sidecar import get_section, set_section, sidecar_path_for
from units import Degrees, Radians, ArcsecondsPerPixel
_PLATESOLUTION_SCHEMA = "astroflow.platesolution.v2"


class PlateSolveFailedException(Exception):
    """Raised when solve-field fails on a light frame that was expected
    to solve. Frames carrying the DONTSOLV header (synthetic noise used
    by tests) do not raise — they simply return None from solve_field."""


@dataclass(frozen=True)
class PlateSolution:
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
def _solver_arg(arg) -> str:
    """An argument as solve-field reads it. On Windows it is a Cygwin program
    (app/windows/build_solve_field.sh): it takes `C:/...` paths, but splits
    names only on '/'."""
    return str(arg).replace("\\", "/") if sys.platform == "win32" else str(arg)


def _solver_env() -> dict[str, str]:
    """solve-field's Python steps (removelines, uniformize) run on this
    interpreter: the Windows bundle's wrappers read ASTROFLOW_PYTHON."""
    return {**os.environ, "ASTROFLOW_PYTHON": sys.executable, "CYGWIN": "nodosfilewarning"}


def _solve_field(
    filename: Path | str, arguments: list[str], debug: bool = False,
    verify: PlateSolution | None = None,
) -> PlateSolution | None:
    """One `solve-field` run. `verify` = an existing solution to REFINE:
    the quad search is skipped (`--verify`), so a large source budget and a
    high SIP order cost no search time and cannot blow up."""
    # TODO: Only copy the file to a temp dir once.
    with tempfile.TemporaryDirectory() as temp_dir:
        # solve-field always gets FITS, and is told so (`--fits-image`):
        # any other format it converts through Python and Netpbm helpers
        # that AstroFlow.app doesn't ship (app/build_app.sh).
        suffix = Path(filename).suffix.lower()
        dest_filename = Path(temp_dir) / (Path(filename).stem + ".fits")
        if suffix == ".xisf":
            convert_xisf_to_fits(filename, dest_filename)
        elif suffix in (".fits", ".fit", ".fts"):
            link_or_copy(filename, dest_filename)
        else:
            from file_io.generic import read_image
            pixels = np.asarray(read_image(Path(filename)), np.float32)
            pyfits.PrimaryHDU(pixels.mean(axis=2) if pixels.ndim == 3 else pixels).writeto(dest_filename)  # colour: luminance

        if verify is not None:
            # a header-only FITS carrying the WCS keywords (NAXISn are what
            # `_ensure_naxis` added for astropy; a data-less HDU must not
            # carry them)
            seed = Path(temp_dir) / "seed.wcs"
            hdr = pyfits.Header({k: v for k, v in verify.header.items()
                                 if isinstance(v, (int, float, str))
                                 and k not in ("COMMENT", "HISTORY", "NAXIS", "NAXIS1", "NAXIS2")})
            hdr["IMAGEW"], hdr["IMAGEH"] = verify.header.get("NAXIS1", 0), verify.header.get("NAXIS2", 0)
            pyfits.PrimaryHDU(header=hdr).writeto(seed)
            arguments = arguments + ["--verify", str(seed)]

        try:
            command = [_solver_arg(a) for a in [*arguments, "--temp-dir", temp_dir, dest_filename]]
            if debug:
                print("### command " + " ".join(command))

            ret = subprocess.run(command, capture_output=True, env=_solver_env())

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
            # Loud: what solve-field said, since a failed solve otherwise
            # leaves no trace of WHY (a missing index folder, a broken step).
            output = (ret.stdout + ret.stderr).decode("utf-8", "replace").strip().splitlines()
            print(f"### solve-field did not solve {filename} (exit {ret.returncode}); its last lines:\n"
                  + "\n".join(output[-12:]))
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
    # Sidecars are write-once and never rewritten, so every one already on
    # disk still carries `image_filename` — an absolute path that nothing
    # ever read. Dropping it here is the same move `footprint` needed when
    # it was removed; without it every historical sidecar raises TypeError.
    d.pop("footprint", None)
    d.pop("image_filename", None)
    return PlateSolution(**d)


@beartype
def _header_hints(header: Mapping[str, Any]) -> list[str]:
    """solve-field search PRIORS from the capture header, or [] to stay
    fully blind. Numeric RA/DEC (degrees) bound the sky search to a 5-deg
    cone; FOCALLEN+XPIXSZ give a pixel-scale window (+/-15%). Conservative
    on parse: any missing or non-numeric card (e.g. sexagesimal OBJCTRA
    spellings) simply yields no hints — never a guess."""
    try:
        ra = float(header.get("RA"))
        dec = float(header.get("DEC"))
    except (TypeError, ValueError):
        return []
    if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
        return []
    hints = ["--ra", f"{ra:.5f}", "--dec", f"{dec:.5f}", "--radius", "5"]
    try:
        focal = float(header.get("FOCALLEN"))
        pixsz = float(header.get("XPIXSZ"))
    except (TypeError, ValueError):
        return hints
    if focal > 0 and pixsz > 0:
        scale = 206.265 * pixsz / focal
        hints += ["--scale-units", "arcsecperpix",
                  "--scale-low", f"{scale * 0.85:.3f}",
                  "--scale-high", f"{scale * 1.15:.3f}"]
    return hints


@beartype
def _solve_flattened(
    filename: Path, args: list[str], debug: bool
) -> PlateSolution | None:
    """Solve a background-subtracted twin of `filename` (see rung 3 note
    at the call site). Background = block-downsampled median (x4, size 9
    ~ a 36 px kernel at full scale), bilinearly upsampled — O(n) even on
    gigapixel blends, star-resistant where a plain Gaussian is not. The
    twin is a temp FITS; ONLY the solve runs against it, so no sidecar or
    cache state ever attaches to the temp path."""
    import tempfile

    import numpy as np
    from scipy.ndimage import median_filter, zoom

    try:
        import astropy.io.fits as fits

        from file_io.generic import read_image

        img = np.asarray(read_image(filename)).astype(np.float32)
        if img.ndim == 3:  # solve on luminance for color inputs
            img = img.mean(axis=2)
        bg_small = median_filter(img[::4, ::4], size=9)
        bg = zoom(bg_small, 4, order=1)[: img.shape[0], : img.shape[1]]
        if bg.shape != img.shape:  # dims not divisible by 4 — pad by edge
            pad = np.pad(bg, ((0, img.shape[0] - bg.shape[0]),
                              (0, img.shape[1] - bg.shape[1])), mode="edge")
            bg = pad
        flat = (img - bg + 0.01).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="solve_flat_") as td:
            twin = Path(td) / "flattened.fits"
            fits.PrimaryHDU(flat).writeto(twin)
            for downsample in ("2", "4"):
                result = _solve_field(
                    twin, args + ["--downsample", downsample], debug
                )
                if result is not None:
                    return result
    except Exception as e:
        # A broken flatten must never mask the ladder's real verdict.
        print(f"### flatten rung failed for {filename}: {e}")
    return None


# A TAN-SIP's own keywords: replaced wholesale when the Gaia refit wins.
_WCS_KEY_PREFIXES = ("CTYPE", "CUNIT", "CRVAL", "CRPIX", "CD1_", "CD2_", "PC1_", "PC2_",
                     "CDELT", "A_", "B_", "AP_", "BP_", "LONPOLE", "LATPOLE",
                     "WCSAXES", "RADESYS", "EQUINOX", "MJDREF")


def _gaia_checked(filename: Path,
                  result: PlateSolution) -> PlateSolution:
    """`result`, or a Gaia refit of it when `result` fails the solution check.

    solve-field fits its distortion only to the reference stars of the one
    index tile that matched, so a field crossing a tile edge gets an
    unconstrained corner: LBN 522 and LBN 521 OIII (2026-10-14) sat 3.5 and
    1.6 px off there, and post_stack_calibrate refused both, blocking their
    whole mosaic channel. The check is Gaia, against the pixels, over nine
    cells. The refit replaces the solution ONLY when it then passes; a
    solution that already passes is never touched."""
    from astropy.wcs import WCS
    from file_io.generic import read_image
    from photometry.harvest import catalog
    from photometry.solution_check import check_solution, refit

    # Geometry only: the check and the refit read catalogue POSITIONS and the
    # G-magnitude cut, never a passband's flux column, so any passband key
    # will do. (Requiring the image's FILTER broke on BXT'd channel TIFFs,
    # which carry FILTER='none' — 2026-10-14.)
    key = catalog().filters[0]
    image = np.asarray(read_image(filename), np.float32)
    if image.ndim == 3:
        image = image[:, :, 0]
    wcs = WCS(pyfits.Header({k: v for k, v in result.header.items()
                             if isinstance(v, (int, float, str))}))
    before = check_solution(image, wcs, key)
    if before.ok:
        return result
    fitted = refit(image, wcs, key)
    after = check_solution(image, fitted, key) if fitted is not None else None
    if after is None or not after.ok:
        print(f"### plate-solve: Gaia refit did not rescue {filename}: "
              f"solver {before}; refit {after}")
        return result
    print(f"### plate-solve: Gaia refit replaced the solver's distortion for "
          f"{filename}: {before} -> {after}")
    fit = fitted.to_header(relax=True)
    cd = fitted.wcs.cd if fitted.wcs.has_cd() else fitted.wcs.get_pc() * fitted.wcs.cdelt[:, None]
    new = {k: v for k, v in result.header.items() if not k.startswith(_WCS_KEY_PREFIXES)}
    new.update({k: fit[k] for k in fit
                if not k.startswith(("PC1_", "PC2_", "CDELT", "DATE", "MJD")) and k not in ("COMMENT", "HISTORY")})
    new.update({"CD1_1": float(cd[0, 0]), "CD1_2": float(cd[0, 1]),
                "CD2_1": float(cd[1, 0]), "CD2_2": float(cd[1, 1])})
    _ensure_naxis(new)
    # The fit chooses its own reference point (pinning it to the image centre
    # made the SIP solve diverge on LBN 521), so CRVAL is no longer the image
    # centre; `center` stays what it means — the sky at the middle pixel.
    mid = fitted.pixel_to_world((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    return replace(
        result, header=new, center=(Degrees(float(mid.ra.deg)), Degrees(float(mid.dec.deg))),
        image_scale=get_image_scale(new), rotation_angle=get_rotation_angle(new))


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
    # TWO STAGES. FIND: the quad search, capped at the brightest 1000 sources
    # and a quadratic SIP — bounded, robust, the ladder below. REFINE: the
    # found solution is handed back to solve-field with `--verify`, which
    # skips the search and re-fits the distortion from EVERY star it can
    # match under a 5000-source budget at SIP order 3. A reducer's
    # distortion is cubic in radius and a quadratic diverges at the corners
    # (2026-09-20, the 0.63"/px 6200MM train: 1 px inside r=700, 21 px at
    # r=5500 — forced photometry measured sky at the edge and
    # post_stack_calibrate refused four masters every pass). Fitting the
    # cubic in the FIND stage was worse (23 matches at --downsample 4:
    # noise everywhere); the refine matched 3-8x more stars — the capped
    # brightest-1000 of a deep master are saturated cores and knots. Order
    # 3, not 4 (2026-09-21, 74-image stratified sample, 15 optical trains,
    # judged by `photometry.solution_check`): the two agree to 0.1 px
    # wherever both fit, order 3 is 6 s faster, and a quartic OVERFITS a
    # sparse corner — the 540 mm 2600MM M42 frame came out 7 px off in one
    # corner at order 4 and 0.9 px at order 3. Systematic displacement
    # after the refine: 0.1-0.5 px on every train but the 2000 mm 6200MM,
    # where 2 of 5 subs sit at 1.3-1.7 px in one corner at either order.
    # The app's own index files plus any extra folder (node_settings); with
    # neither (a checkout never set up through the app), solve-field's own
    # astrometry.cfg.
    config = node_settings.ASTROMETRY_CONFIG
    base = [
        tool_path("solve-field"),
        *(["--config", str(config)] if config.exists() else []),
        "--fits-image",
        "--overwrite",
        "--no-plots",
        "--new-fits",
        "none",
        "--tweak-order",
        "2",
        "--crpix-center",
        "--cpulimit",
        "600",
        "--objs",
        "1000",
    ]
    blind_scale = [
        "--scale-low", "0.1", "--scale-high", "180",
        "--scale-units", "degwidth",
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
    # Rung 0 — header-HINTED search, when the capture header carries usable
    # priors. Hints narrow WHERE the solver looks; star quads still decide
    # the solution, so a wrong header costs one fast failed rung, never a
    # wrong answer ("trust the pixels" intact — the header only
    # accelerates). Measured (vdB15 Lum, 2026-08-21): blind burned the full
    # 600 s CPU budget per rung and never converged; the same master with
    # header hints solved in <30 s at log-odds 540.
    hints = _header_hints(header)
    if hints:
        scale_hinted = any(a == "--scale-units" for a in hints)
        rung0 = base + hints + ([] if scale_hinted else blind_scale)
        result = _solve_field(filename, rung0 + ["--downsample", "4"], debug)
        if result is not None:
            print(f"### plate-solve via header hints: {filename}")
    if result is None:
        for downsample in ("2", "4"):
            result = _solve_field(
                filename,
                base + blind_scale + ["--downsample", downsample], debug,
            )
            if result is not None:
                if downsample != "2":
                    print(f"### plate-solve rescued by --downsample "
                          f"{downsample}: {filename}")
                break

    if result is None:
        # Rung 3 — background-FLATTENED twin, for fields whose dynamic
        # range is dominated by smooth bright nebulosity (lab panel 15 Hα,
        # 2026-09-01: unsolvable through every rung above, yet the same
        # pixels minus a 33 px median background solved instantly with 267
        # stars). Downsampling (rung 2) removes small knots but cannot
        # remove LARGE-scale structure, which skews the extractor's
        # thresholding until no coherent star set survives. The twin
        # shares the original's pixel grid, so its WCS transfers verbatim.
        result = _solve_flattened(filename, base + blind_scale, debug)
        if result is not None:
            print(f"### plate-solve rescued by background flatten: "
                  f"{filename}")

    if result is None:
        print(f"Could not plate-solve file {filename} (all retry rungs)")
        set_section(
            filename, "plate_solution", {"schema": _PLATESOLUTION_SCHEMA, "data": None}
        )
        return None

    # REFINE (see `base`): the distortion fit, seeded by the find. A refine
    # that fails leaves the find in place — it is a better solution than
    # none, and `check_solution` below is what says whether it is good enough.
    refine_args = [a for a in base if a not in ("--tweak-order", "2", "--objs", "1000")] + [
        "--tweak-order", "3", "--objs", "5000", "--downsample", "2",
    ] + (hints if hints else []) + ([] if (hints and any(a == "--scale-units" for a in hints)) else blind_scale)
    refined = _solve_field(filename, refine_args, debug, verify=result)
    if refined is not None:
        result = refined
    else:
        print(f"### plate-solve: refine failed, keeping the find: {filename}")
    result = _gaia_checked(filename, result)

    set_section(
        filename, "plate_solution", _plate_solution_to_obj(result)
    )
    print(f"### Updated AstroFlow sidecar for {filename}")

    return result


if __name__ == "__main__":
    solve_field(Path(sys.argv[1]), debug=True)
