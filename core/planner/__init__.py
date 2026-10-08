"""The night planner: what each rig should shoot tonight, from the cart alone.

Five layers, each readable on its own:

    inventory   what could be shot: every (panel, filter) of the cart's objects,
                with the hours it still needs (impure: reads context + sidecars)
    sky         tonight's time slots and where each target is in them
    efficiency  what an hour is worth given filter, altitude and moon
    schedule    the per-rig plan: blocks chosen greedily, one meridian flip
                at most, with costs for touching stacks and slewing
    report      the plan as text

Only `inventory` touches AstroFlow state; the rest are pure functions of
their arguments, so the planner can be replayed against any past night.
"""
