"""Rank every mosaic and standalone target by integration depth vs. cart.

The processing-triage question: "what's worth carting next?" A mosaic is
ranked as ONE object (its panels' hours summed) because carting the mosaic
is what produces the panel-blended deliverable — carting its panels
separately would spend the same stacking and throw away the blend. Targets
that belong to a mosaic are therefore reported under it, never twice.

    PYTHONPATH=. python -m scripts.survey_library [--min-hours 20]

Ray-free apart from the context read; safe to run while a pass is going,
though the numbers reflect whatever inventory the pass has rebuilt so far.
"""
from __future__ import annotations

import argparse


def _hours(stacks, ids) -> tuple:
    """(hours, n_stacks, filters, first_date, last_date) for a stack-id set."""
    hrs, n, filters, dates = 0.0, 0, set(), []
    for sid in ids:
        s = stacks.get(sid)
        if s is None:
            continue
        hrs += (getattr(s, "integration", 0) or 0) / 3600.0
        n += 1
        filters.add(str(s.filter))
        dates.append(s.oldest_date)
    return (hrs, n, sorted(filters),
            min(dates).date() if dates else None,
            max(dates).date() if dates else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-hours", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    from context import context
    stacks = {sid: r.get() for sid, r in (context.stacks or {}).items()}
    targets = {tid: r.get() for tid, r in (context.targets or {}).items()}
    mosaics = {mid: r.get() for mid, r in (context.mosaics or {}).items()}
    cart = list(context.shopping_cart or [])
    carted = {(t, i) for t, i, _ in cart}

    print(f"inventory: {len(targets)} targets, {len(mosaics)} mosaics, "
          f"{len(stacks)} stacks, {len(cart)} cart items")

    in_mosaic = {}
    rows = []
    for mid, m in mosaics.items():
        ids = []
        for tid in m.target_ids:
            in_mosaic[tid] = mid
            t = targets.get(tid)
            if t:
                ids += list(t.stack_ids)
        hrs, n, fl, d0, d1 = _hours(stacks, ids)
        rows.append((hrs, "mosaic", m.name, mid, len(m.target_ids), n, fl,
                     d0, d1, ("mosaic", mid) in carted))

    for tid, t in targets.items():
        if tid in in_mosaic:
            continue                      # counted under its mosaic
        hrs, n, fl, d0, d1 = _hours(stacks, t.stack_ids)
        rows.append((hrs, "target", t.name, tid, 1, n, fl, d0, d1,
                     ("target", tid) in carted))

    rows.sort(reverse=True)
    nb_set = {"H", "O", "S"}
    print(f"\n{'hours':>6}  {'kind':6} {'name':24} {'id':16} {'pan':>3} "
          f"{'stk':>3}  {'cart':5} {'palette':10} filters")
    shown = 0
    for hrs, kind, name, oid, panels, n, fl, d0, d1, is_carted in rows:
        if hrs < args.min_hours or shown >= args.limit:
            continue
        shown += 1
        classes = {f[:1].upper() for f in fl}
        nb = len(classes & nb_set)
        bb = len(classes & {"R", "G", "B", "L"})
        palette = ("SHO" if nb == 3 else f"NB×{nb}" if nb else "—")
        if bb >= 3:
            palette += "+RGB"
        print(f"{hrs:6.1f}  {kind:6} {name[:24]:24} {oid:16} {panels:3d} "
              f"{n:3d}  {'YES' if is_carted else '—':5} {palette:10} "
              f"{','.join(fl)[:28]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
