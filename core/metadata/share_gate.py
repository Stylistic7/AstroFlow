"""Share readiness: whether a stack's master belongs in a shared master
folder, and so in the group's all-sky coadd, with every reason spelled out
up front.

Proper coaddition weights a master by its own noise and PSF, so DEPTH is
never a reason to refuse one: a shallow or blurry master simply contributes
less, and rejecting it only loses sensitivity (Zackay & Ofek 2017). What
makes the map WORSE is a master whose errors are not the noise its weight
assumes -- a systematic the coadd cannot know about. Those are what this
gate refuses, the same things the survey pipelines guard against before an
exposure enters a coadd: an astrometric solution that holds (every
pipeline), enough stars for a photometric calibration and a photometric
night (SDSS Stripe 82, DES), a PSF model that holds (Rubin's makeWarp
guards), and outlier rejection in the stack. Everything the weights handle
it lets through with a warning the user can read.

BLOCKING -- the master is not shared, and the user is told why:
  solve    no plate solution: the coadd cannot place it
  flux     no Gaia flux calibration, or its scatter above MAX_SCATTER
           (the pipeline's own refusal): the flux scale is unknown
  psf      the PSF could not be measured (too few stars): the weight model
           is unknown
  frames   fewer than MIN_FRAMES subframes: no outlier rejection, so trails
           and cosmic rays become permanent
  flat     stacked without a flat: vignetting and dust shadows no level
           solve can remove
  clipped  more than MAX_CLIPPED of the pixels at full scale: saturated or
           stretched, not linear
WARNINGS -- shared, with the caveat shown:
  blurry   PSF wider than BLURRY x the train's typical: it contributes only
           the coarser detail
  sharp    PSF narrower than SHARP x the train's typical: check the measure
  flux     scatter above WARN_SCATTER: transparency varied within the stack
  body     calibrated with another camera body's master
  frames   fewer than FEW_FRAMES subframes: weak outlier rejection

The inputs are what travels with a master (its sidecar: `analysis`,
`spfc_calibration`, `plate_solution`, `image_metadata.statistics`), the
stack's frame count and its concerns, and the train's typical PSF -- so the
same verdict is reached by the publisher, the API and a consumer of the
folder. `readiness` is pure; `master_inputs` reads a sidecar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_SCATTER = 0.25          # photometry.harvest._SPFC_MAX_SCATTER: the calibration's own refusal
WARN_SCATTER = 0.10
MIN_FRAMES = 5
FEW_FRAMES = 10
MAX_CLIPPED = 0.01
BLURRY = 1.5
SHARP = 0.6
PSF_MIN_STARS = 10          # metadata.master_analysis.PSF_MIN_STARS


def _reason(code: str, title: str, detail: str, **fields) -> dict:
    return {"code": code, "title": title, "detail": detail, **fields}


def readiness(*, analysis: dict | None, cert: dict | None, solved: bool, n_frames: int | None,
              concerns: list[dict], statistics: dict | None, train_fwhm: float | None) -> dict:
    """`{"ready", "blocking": [...], "warnings": [...]}` for one master."""
    blocking: list[dict] = []
    warnings: list[dict] = []
    if not solved:
        blocking.append(_reason("solve", "Not plate-solved",
                                "The coadd places a master by its astrometric solution; without one it cannot be used."))
    if not cert:
        blocking.append(_reason("flux", "No flux calibration",
                                "No Gaia flux calibration was possible (too few matchable stars: clouds, starvation or bad "
                                "pointing), so the master's flux scale is unknown and it cannot be weighted against the others."))
    else:
        scatter = float(cert.get("sigma") or 0.0)
        if scatter > MAX_SCATTER:
            blocking.append(_reason("flux", "Flux calibration scatter too high",
                                    f"The Gaia calibration's star-to-star scatter is {scatter:.2f}, above {MAX_SCATTER:.2f}: "
                                    "transparency varied so much within the stack that its flux scale is not trustworthy.",
                                    value=scatter, limit=MAX_SCATTER))
        elif scatter > WARN_SCATTER:
            warnings.append(_reason("flux", "Flux calibration scatter",
                                    f"The Gaia calibration's star-to-star scatter is {scatter:.2f}, above {WARN_SCATTER:.2f}: "
                                    "transparency varied within the stack, so its flux scale is less certain than usual.",
                                    value=scatter, limit=WARN_SCATTER))
    psf = (analysis or {}).get("psf") or {}
    fwhm = psf.get("fwhm_arcsec")
    if not fwhm:
        n = int(psf.get("n_stars") or 0)
        blocking.append(_reason("psf", "PSF not measurable",
                                f"Only {n} usable stars were found (at least {PSF_MIN_STARS} are needed) so the PSF width is "
                                "unknown; the coadd weights every master by its PSF and cannot admit one it cannot model.",
                                value=n, limit=PSF_MIN_STARS))
    elif train_fwhm:
        if fwhm > BLURRY * train_fwhm:
            warnings.append(_reason("blurry", "Blurrier than this train's usual",
                                    f"PSF {fwhm:.1f}″ against the train's typical {train_fwhm:.1f}″ (seeing or focus). It is "
                                    "still worth sharing: the coadd takes only the coarser detail it has.",
                                    value=fwhm, limit=BLURRY * train_fwhm))
        elif fwhm < SHARP * train_fwhm:
            warnings.append(_reason("sharp", "Sharper than this train's usual",
                                    f"PSF {fwhm:.1f}″ against the train's typical {train_fwhm:.1f}″: unusually sharp, "
                                    "so the measurement is worth a look.",
                                    value=fwhm, limit=SHARP * train_fwhm))
    if n_frames is not None:
        if n_frames < MIN_FRAMES:
            blocking.append(_reason("frames", "Too few subframes",
                                    f"{n_frames} subframes: outlier rejection needs at least {MIN_FRAMES}, so satellite trails "
                                    "and cosmic rays would become permanent in the map.",
                                    value=n_frames, limit=MIN_FRAMES))
        elif n_frames < FEW_FRAMES:
            warnings.append(_reason("frames", "Few subframes",
                                    f"{n_frames} subframes: outlier rejection is weak below {FEW_FRAMES}.",
                                    value=n_frames, limit=FEW_FRAMES))
    for c in concerns or []:
        title = str(c.get("event") or c.get("title") or "")
        if title.startswith("Stacking with no") and "flat" in (c.get("missing") or {}):
            blocking.append(_reason("flat", "Stacked without a flat",
                                    "No flat master was applied, so vignetting and dust shadows are still in the master; "
                                    "the map's level solve cannot remove a bowl."))
        elif title.startswith("Calibrated with another camera body"):
            warnings.append(_reason("body", "Calibrated with another body's master",
                                    "A bias, dark or flat from another camera body calibrated this stack: hot pixels, the "
                                    "etalon and window dust are the other body's."))
    clipped = float((statistics or {}).get("clipped_high_fraction") or 0.0)
    if clipped > MAX_CLIPPED:
        blocking.append(_reason("clipped", "Not linear",
                                f"{100 * clipped:.1f}% of the pixels sit at full scale: the master is saturated or has "
                                "been stretched, and the coadd needs linear data.",
                                value=clipped, limit=MAX_CLIPPED))
    return {"ready": not blocking, "blocking": blocking, "warnings": warnings}


def master_inputs(path: Path) -> dict:
    """What a master's sidecar says of it, for `readiness`."""
    from file_io.sidecar import get_section
    try:
        analysis = get_section(path, "analysis")
        cert = get_section(path, "spfc_calibration")
        solution = get_section(path, "plate_solution")
        meta = get_section(path, "image_metadata") or {}
    except Exception:          # noqa: BLE001 — no sidecar, or unreadable: nothing is known
        analysis = cert = solution = None
        meta = {}
    return {"analysis": analysis if isinstance(analysis, dict) else None,
            "cert": cert if isinstance(cert, dict) else None,
            "solved": isinstance(solution, dict) and solution.get("data") is not None,
            "statistics": meta.get("statistics") if isinstance(meta, dict) else None}


def concerns_from_cache(cache, stack_id: str) -> list[dict]:
    """A stack's concerns as `@capture_logs` stored them, at
    `("concerns", "Stack", stable_hash)` → `{op: [concern, ...]}`."""
    entry = cache.get(("concerns", "Stack", stack_id), {})
    if not isinstance(entry, dict):
        return []
    return [c for op_concerns in entry.values() for c in op_concerns]


def train_of(ui_stack: dict) -> tuple:
    """The optical train a typical PSF is taken over: focal length binned at
    the matcher's 2% (`same_optical_train`), and the filter."""
    focal = ui_stack.get("focal_length")
    return (int(round(np.log(float(focal)) / np.log(1.02))) if focal else None, ui_stack.get("filter"))


def stacks_readiness(ui_stacks: dict, masters: list[tuple[str, str, Path]], concerns_of) -> dict[str, dict]:
    """Readiness of every stack in `masters` (`(stack_id, pipeline, path)`),
    the train's typical PSF being the median over the same masters."""
    inputs = {sid: master_inputs(path) for sid, _, path in masters}
    by_train: dict[tuple, list[float]] = {}
    for sid, inp in inputs.items():
        fwhm = ((inp["analysis"] or {}).get("psf") or {}).get("fwhm_arcsec")
        if fwhm and sid in ui_stacks:
            by_train.setdefault(train_of(ui_stacks[sid]), []).append(float(fwhm))
    out = {}
    for sid, inp in inputs.items():
        ui = ui_stacks.get(sid, {})
        peers = by_train.get(train_of(ui), [])
        out[sid] = readiness(
            analysis=inp["analysis"], cert=inp["cert"], solved=inp["solved"],
            n_frames=len(ui["filenames"]) if ui.get("filenames") is not None else None,
            concerns=concerns_of(sid), statistics=inp["statistics"],
            train_fwhm=float(np.median(peers)) if len(peers) >= 3 else None,
        )
    return out
