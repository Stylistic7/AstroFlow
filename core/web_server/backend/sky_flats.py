"""Night-sky flats panel: what each rig has, where to shoot the next one, and
a button that writes the plan into the rig's ASIAir.

  * `GET  /api/sky_flats` — per rig and filter: the night-sky flats built so
    far (with the quality the build recorded) and the exposure the rig
    usually uses in that filter.
  * `GET  /api/sky_flats/fields` — the blank fields best placed for a rig on
    a night (`stacks/sky_flat_plans.rank_fields`).
  * `POST /api/sky_flats/plan` — five SkyFlat panes around a field for one
    filter and exposure, previewed with `dry_run`, otherwise ADDED to the
    rig's ASIAir as a new, disabled plan beside whatever is armed (enabling it
    is a tap on the tablet). Frames shot under those pane names group into one
    Stack that is a night-sky flat source without anyone promoting it.

Everything is read from the UI summaries and the device list — no stack is
loaded on a request.
"""
from __future__ import annotations

import datetime as _dt
import statistics

from flask import Blueprint, abort, jsonify, request

from context import local_context as context

sky_flats_bp = Blueprint("sky_flats", __name__)

NARROWBAND = ("H", "S", "O")
DEFAULT_EXPOSURE_S = 300.0
DEFAULT_FRAMES_PER_PANE = 4
FOCAL_TOLERANCE = 0.03


def _rigs() -> list[dict]:
    from planner.inventory import PIXEL_UM, SENSOR
    out = []
    for r in context.asiair_devices or []:
        focal = r.get("focal_len_mm")
        if not focal:
            continue
        out.append({"key": r["key"], "name": r.get("name") or r["key"], "ip": r.get("ip"),
                    "focal_mm": float(focal), "pixel_um": PIXEL_UM, "sensor": list(SENSOR)})
    return out


def _rig(key: str) -> dict:
    for r in _rigs():
        if r["key"] == key:
            return r
    abort(404, description=f"no rig {key!r}")


def _same_train(focal, rig_focal: float) -> bool:
    try:
        return abs(float(focal) - rig_focal) <= FOCAL_TOLERANCE * rig_focal
    except (TypeError, ValueError):
        return False


def _role(filt) -> str:
    from pix.tasks.filters import normalize_filter
    return normalize_filter(str(filt)) or str(filt)


def _sky_flat_record(sf: dict) -> dict:
    from file_io.sidecar import get_section
    from stacks.sky_flat import sky_flat_master_path
    from stacks.sky_flat_ramp import ramp_of
    path = sky_flat_master_path(sf["stable_hash"], context.output_dir)
    q = (get_section(path, "sky_flat") or {}) if path else {}
    try:
        ramp = ramp_of(path) if path else None
    except (OSError, ValueError):
        ramp = None
    return {
        "ramp_pct": ramp and ramp["ramp_pct"], "ramp_angle_deg": ramp and ramp["angle_deg"],
        "id": sf["stable_hash"], "source_stack_id": sf.get("source_stack_id"),
        "frames": sf.get("n_frames"), "newest_date": sf.get("newest_date"),
        "built": path is not None, "good": q.get("good"), "rms": q.get("rms"),
        "motes_cleaned": len(q.get("motes") or []),
    }


@sky_flats_bp.route("/api/sky_flats")
def list_sky_flats():
    lights = (context.ui_stacks or {}).values()
    cals = (context.ui_calibration_stacks or {}).values()
    rigs = []
    for rig in _rigs():
        exposures: dict[str, list[float]] = {}
        for s in lights:
            if _same_train(s.get("focal_length"), rig["focal_mm"]) and s.get("exposure"):
                exposures.setdefault(_role(s.get("filter")), []).append(float(s["exposure"]))
        flats: dict[str, list[dict]] = {}
        for c in cals:
            if c.get("kind") == "sky" and _same_train(c.get("focal_length"), rig["focal_mm"]):
                flats.setdefault(_role(c.get("filter")), []).append(_sky_flat_record(c))
        filters = []
        for f in sorted(set(exposures) | set(flats), key=lambda f: ("HSOLRGB".find(f) % 99, f)):
            exp = statistics.median(exposures[f]) if f in exposures else DEFAULT_EXPOSURE_S
            filters.append({
                "filter": f,
                "exposure_s": exp,
                "frames_per_pane": DEFAULT_FRAMES_PER_PANE,
                "sky_flats": sorted(flats.get(f, []), key=lambda x: str(x.get("newest_date")), reverse=True),
            })
        from stacks.sky_flat_plans import field_of_view_deg
        fov = field_of_view_deg(rig["focal_mm"], rig["pixel_um"], tuple(rig["sensor"]))
        rigs.append({**rig, "fov_deg": [round(fov[0], 2), round(fov[1], 2)], "filters": filters})
    return jsonify({"rigs": rigs})


def _night(arg: str | None) -> _dt.date:
    if arg:
        return _dt.date.fromisoformat(arg)
    from planner.inventory import site
    lon, _ = site(context)
    local = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=lon / 15.0)
    return local.date()


@sky_flats_bp.route("/api/sky_flats/fields")
def ranked_fields():
    from planner.inventory import site
    from stacks.sky_flat_plans import plan_minutes, rank_fields
    rig = _rig(request.args.get("rig", ""))
    exposure = float(request.args.get("exposure_s", DEFAULT_EXPOSURE_S))
    frames = int(request.args.get("frames_per_pane", DEFAULT_FRAMES_PER_PANE))
    night = _night(request.args.get("date"))
    lon, lat = site(context)
    minutes = plan_minutes(exposure, frames)
    return jsonify({
        "rig": rig["key"], "date": night.isoformat(), "plan_minutes": round(minutes),
        "fields": rank_fields(rig["focal_mm"], rig["pixel_um"], tuple(rig["sensor"]),
                              night, lon, lat, minutes),
    })


@sky_flats_bp.route("/api/sky_flats/plan", methods=["POST"])
def push_sky_flat_plan():
    from asiair.plan_push import ExposureRow, PlanTarget, build_plan, next_plan_id, push_plan, wheel_slots
    from asiair.rpc_session import AsiairSession
    from asiair.rsa_sign import default_key_path
    from stacks.sky_flat_plans import (
        PANES, field_by_id, field_of_view_deg, pane_centres, pane_names, plan_minutes,
    )

    body = request.get_json(force=True)
    rig = _rig(body.get("rig", ""))
    filt = str(body.get("filter") or "").strip()
    if not filt:
        abort(400, description="missing filter")
    exposure = float(body.get("exposure_s") or DEFAULT_EXPOSURE_S)
    frames = int(body.get("frames_per_pane") or DEFAULT_FRAMES_PER_PANE)
    if exposure <= 0 or frames <= 0:
        abort(400, description="exposure and frames per pane must be positive")
    try:
        field = field_by_id(str(body.get("field_id")))
    except KeyError as e:
        abort(400, description=str(e))
    night = _night(body.get("date"))
    _, short = field_of_view_deg(rig["focal_mm"], rig["pixel_um"], tuple(rig["sensor"]))
    targets = [PlanTarget(name, ra, dec, (ExposureRow(filt, exposure, frames),))
               for name, (ra, dec) in zip(pane_names(filt, field["id"]), pane_centres(field, short))]
    name = f"AstroFlow sky flat {filt} {field['id']} {night.isoformat()}"
    summary = {"rig": rig["key"], "filter": filt, "field": field, "panes": PANES,
               "frames": PANES * frames, "exposure_s": exposure,
               "plan_minutes": round(plan_minutes(exposure, frames)),
               "targets": [{"name": t.name, "ra_deg": round(t.ra_deg, 5), "dec_deg": round(t.dec_deg, 5)}
                           for t in targets]}
    if not rig.get("ip"):
        abort(409, description=f"rig {rig['key']} has no known address")
    key = default_key_path()
    if not key.exists():
        abort(409, description=f"no ASIAir signing key at {key}")
    with AsiairSession(rig["ip"]) as session:
        if not session.verify(key):
            abort(502, description=f"ASIAir {rig['ip']} refused the handshake")
        slots = wheel_slots(session)
        if filt not in slots:
            abort(400, description=f"filter {filt!r} is not in the wheel {sorted(slots)}")
        existing = session.call("get_plan") or []
        like = next((p for p in existing if p.get("enable")), existing[0] if existing else None)
        plan = build_plan(next_plan_id(session), name, targets, slots, night, like=like)
        if body.get("dry_run", True):
            return jsonify({"dry_run": True, "plan_name": name, **summary})
        try:
            result = push_plan(session, plan)
        except RuntimeError as e:
            abort(502, description=str(e))
    return jsonify({"dry_run": False, "plan_name": name, **summary, **result})


@sky_flats_bp.route("/api/sky_flats/<sky_flat_id>/thumbnail.png")
def sky_flat_thumbnail(sky_flat_id):
    """A small, stretched picture of the BUILT night-sky flat, to eyeball it:
    vignetting, dust shadows, a ramp. The flat is near 1.0 everywhere, so it
    is shown with a percentile stretch (0.5-99.5% of its own values), not the
    sky autostretch, which assumes a dark background."""
    import io

    import numpy as np
    from flask import send_file
    from PIL import Image
    from file_io.generic import read_image
    from stacks.sky_flat import sky_flat_master_path

    path = sky_flat_master_path(sky_flat_id, context.output_dir)
    if path is None:
        abort(404, description=f"night-sky flat {sky_flat_id} is not built")
    width = max(64, min(int(request.args.get("w", 360)), 1200))
    img = np.asarray(read_image(path))
    if img.ndim == 3:
        img = img[..., 0]
    img = np.asarray(img[::8, ::8], np.float32)
    ok = np.isfinite(img) & (img > 0)
    lo, hi = np.percentile(img[ok], [0.5, 99.5]) if ok.any() else (0.0, 1.0)
    v = np.clip((np.nan_to_num(img) - lo) / max(hi - lo, 1e-9), 0, 1)
    pic = Image.fromarray((v * 255).astype(np.uint8))
    pic.thumbnail((width, width))
    buf = io.BytesIO()
    pic.save(buf, format="PNG")
    buf.seek(0)
    resp = send_file(buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "max-age=3600"
    return resp
