"""Publishing masters into the shared master folders.

At the end of every full pass (`state.actor`), each shared master folder
(`context.shared_dirs`) is brought to hold exactly the post-stack-calibrated
masters this instance currently has for the targets it shares into that
folder: a master missing there is copied in with its sidecar (payload last,
atomically — `file_io.publish`), and a master this instance published before
but no longer has (re-stacked under a new call hash, or dropped from the
cart) is deleted, so the other members' sync tools carry the deletion and
the group's mosaic re-forms without it. The folder is a mirror of a slice of
this instance's products, recomputed from them every pass; the only memory
kept is the durable set of call hashes published per folder, which is what
makes a file in a shared folder MINE to delete — never a name, never a
stable hash (a remote-observatory dataset two members both own gives two
masters one stable hash, and the other member's is not mine to touch).

WHICH MASTERS. Opt-in only: the user marks a stack, a target or a mosaic
for a folder (`DecisionKey.sharing`), and a target or mosaic choice sends
every component stack, the way quarantine does. Nothing is published
unmarked, whatever it overlaps. Only local stacks publish (a foreign stack's
master is someone else's), and only masters that carry a pipeline stamp and
a plate solution (what a consumer needs to admit them).

WHAT IS WRITTEN. `<call hash>.<ext>` beside `<call hash>.<ext>.json`: the
producer's own sidecar (metadata, plate solution, calibration certificate),
plus a `shared` section with what the master itself cannot say — the stack's
first and last night, its effective integration, the subframe count, the
pipeline's name, and OBSERVER for credit. Nothing else in the app ever
writes into a shared folder.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import structlog

from stacks.foreign import SHARED_SECTION

log = structlog.get_logger()

SHARED = "shared"
_PUBLISHED_NS = "shared_published"   # durable: ("shared_published", <folder>) → {call_hash: file name}


def _published_key(folder: Path) -> tuple:
    return (_PUBLISHED_NS, str(folder))


def sharing_choices(decision_overrides) -> dict[tuple[str, str], set[str]]:
    """`{(object_type, object_id): {folder, ...}}` from the user's decisions."""
    from decisions.decisions import DecisionType
    out: dict[tuple[str, str], set[str]] = {}
    for key, override in (decision_overrides or {}).items():
        if key.decision_type == DecisionType.SHARING and str(override.choice) == SHARED:
            out.setdefault((str(key.object_type.value), str(key.parent_stack_id)), set()).add(str(key.subject_id))
    return out


def stacks_shared_into(folder: str, ui_mosaics: dict, ui_targets: dict, ui_stacks: dict, choices: dict) -> set[str]:
    """The ids of the LOCAL stacks published into `folder`: marked
    themselves, or inside a marked target, or inside a target of a marked
    mosaic."""
    target_ids = {oid for (kind, oid), folders in choices.items()
                  if kind == "target" and folder in folders and oid in ui_targets}
    for (kind, oid), folders in choices.items():
        if kind == "mosaic" and folder in folders and oid in ui_mosaics:
            target_ids.update(t for t in ui_mosaics[oid].get("target_ids", ()) if t in ui_targets)
    stack_ids = {sid for (kind, oid), folders in choices.items()
                 if kind == "stack" and folder in folders for sid in (oid,)}
    for tid in target_ids:
        stack_ids.update(ui_targets[tid].get("stack_ids", ()))
    return {sid for sid in stack_ids if sid in ui_stacks and not ui_stacks[sid].get("foreign")}


def current_masters(stack_ids: set[str]) -> list[tuple[str, str, Path]]:
    """`(stack_id, pipeline, path)` of every current post-processed master
    of the given local stacks, from the product version log (head-only)."""
    from context.cache import get_durable_cache
    dc = get_durable_cache()
    out = []
    for k in list(dc):
        if not (isinstance(k, tuple) and len(k) == 6 and k[0] == "product_versions"
                and k[1] == "Stack" and k[2] in stack_ids and k[4] == "Post-Processed"):
            continue
        entries = dc.get(k) or []
        if not entries:
            continue
        path = Path(str(entries[-1].get("path", "")))
        if path.is_file():
            out.append((str(k[2]), str(k[3]), path))
    return sorted(out, key=lambda e: (e[0], e[1], str(e[2])))


def _publishable(path: Path) -> bool:
    """A master a consumer can admit: its SIDECAR (what travels with the
    copy) carries the stamp, both hashes and a plate solution."""
    from file_io.sidecar import get_section
    from hashing.pipeline_stamp import stamp_of
    from metadata.sidecar_io import read_image_metadata
    try:
        meta = read_image_metadata(path)
    except Exception:
        return False
    if meta is None:
        return False
    header = meta.header
    solution = get_section(path, "plate_solution")
    return bool(stamp_of(header) and header.get("CALLHASH") and header.get("STBLHASH")
                and isinstance(solution, dict) and solution.get("data") is not None)


def _shared_section(stack_id: str, pipeline: str, ui_stack: dict, header: dict) -> dict:
    return {
        "observer": str(header.get("OBSERVER") or "").strip() or None,
        "oldest_date": ui_stack.get("oldest_date"),
        "newest_date": ui_stack.get("newest_date"),
        "integration_s": ui_stack.get("integration"),
        "n_frames": len(ui_stack.get("filenames") or ()),
        "pipeline": pipeline,
        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def reconcile_folder(folder: Path, masters: list, published: dict, ui_stacks: dict) -> dict:
    """Make `folder` hold exactly `masters` (`(stack_id, pipeline, path)`) of
    ours: copy in what is missing, retire what we published before and no
    longer have. Returns the new published record `{call_hash: file name}`."""
    from file_io.publish import publish_pair_atomically
    from file_io.sidecar import set_section, sidecar_path_for
    from metadata.sidecar_io import read_image_metadata

    published = dict(published)
    wanted: dict[str, Path] = {}
    for stack_id, pipeline, path in masters:
        if not _publishable(path):
            continue
        wanted[path.stem] = path
        dest = folder / path.name
        try:
            if dest.exists() and dest.stat().st_size == path.stat().st_size and path.stem in published:
                continue
            publish_pair_atomically(path, dest, sidecar_path_for(path), sidecar_path_for(dest))
            section = _shared_section(stack_id, pipeline, ui_stacks.get(stack_id, {}),
                                      read_image_metadata(path).header)
            set_section(dest, SHARED_SECTION, section)   # after the payload: the sidecar's size stamp is the real one
            published[path.stem] = path.name
            log.info("published shared master", folder=str(folder), file=path.name,
                     stack=stack_id[:8], pipeline=pipeline)
        except OSError as e:
            log.warning("publishing a shared master failed", file=str(path), dest=str(dest), error=str(e))
    for call_hash, name in list(published.items()):
        if call_hash in wanted:
            continue
        stale = folder / name
        try:
            stale.unlink(missing_ok=True)
            sidecar_path_for(stale).unlink(missing_ok=True)
            log.info("retired shared master", folder=str(folder), file=name)
            del published[call_hash]
        except OSError as e:
            log.warning("retiring a shared master failed", file=str(stale), error=str(e))
    return published


def publish_shared_masters(ctx) -> None:
    """Bring every shared master folder up to date with this instance's
    masters (see the module docstring). Never raises: a folder that cannot
    be reached is left as it is, and so is its published set."""
    from metadata.library_scan import dir_available, store_record
    from metadata.shared_masters import scan_shared_dir

    folders = [Path(d) for d in (ctx.shared_dirs or [])]
    if not folders:
        return
    ui_mosaics = ctx.ui_mosaics or {}
    ui_targets = ctx.ui_targets or {}
    ui_stacks = ctx.ui_stacks or {}
    choices = sharing_choices(ctx.settings.decision_overrides)

    for folder in folders:
        if not dir_available(folder) or not folder.is_dir():
            log.warning("shared master folder unavailable — not publishing", folder=str(folder))
            continue
        key = _published_key(folder)
        stack_ids = stacks_shared_into(str(folder), ui_mosaics, ui_targets, ui_stacks, choices)
        published = reconcile_folder(folder, current_masters(stack_ids), ctx.get(key) or {}, ui_stacks)
        ctx[key] = published
        # The folder now differs from the last scan by exactly our own files,
        # which the next pass dedupes against our stacks: record it so the
        # hourly check does not drain a pass over our own publish.
        store_record(scan_shared_dir(folder))
        log.info("shared master folder reconciled", folder=str(folder),
                 stacks=len(stack_ids), published=len(published))


def shared_dirs_changed(ctx) -> bool:
    """Whether any reachable shared master folder differs from its last
    scan — the hourly poll's reason to drain and re-identify."""
    from metadata.library_scan import dir_available, load_record
    from metadata.shared_masters import scan_shared_dir
    for folder in (ctx.shared_dirs or []):
        if not dir_available(folder):
            continue
        known = load_record(folder)
        if known is None or known.fingerprint != scan_shared_dir(folder).fingerprint:
            return True
    return False
