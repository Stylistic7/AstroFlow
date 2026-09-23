"""Single source of truth for the calibration dependency topology.

Landmann §7.1 Case 2 (matched-exposure darks). BOTH calibration-association
paths read this one map, so they can never drift:

  * light → cal:  `decisions/resolve.py:resolve_calibration`
  * cal  → cal:  `calibration/calibration_frames.py:associate_calibration_stacks`

(They used to carry separate copies; the copies diverged and left master
darks bias-pre-calibrated against the §6.2 rule — exactly the bug this
unification removes.)

  * `bias: ()`            — a master bias depends on nothing.
  * `dark: ()`            — integrate raw darks WITHOUT pre-calibrating them.
                            Landmann §6.2: "NEVER pre-calibrate the MasterDark"
                            — a pre-calibrated MD trips WBPP's §8.2
                            undercorrection when paired with master-bias-for-
                            lights.
  * `flat: ("bias",)`     — bias-calibrate CMOS short flats (Landmann §6.3.1).
  * `light: ("dark", "flat", "fringe")` — NO master bias for lights. Case 2's
                            closed form is `(L − MD) · s0 / MF` with no MB
                            term; staging an MB into the lights' WBPP working
                            dir would double-subtract bias against a master
                            dark that already contains it (§8.2). With this
                            topology WBPP's auto-detect lands on the documented
                            Case 2 settings.
  * `fringe: ("dark", "flat")` — OPTIONAL, and unlike the others it is
                            nobody's default: a light resolves one only when
                            the user has built one for its optical train. A
                            fringe frame is SUBTRACTIVE like a dark but scaled
                            to each frame's SKY, because the interference it
                            corrects modulates the monochromatic sky and not
                            the total flux (see `stacks/fringe.py`). Kept a
                            distinct calibration type all the way through the
                            codebase: how a stacker applies it is the
                            stacker's business — the PixInsight backend folds
                            it into the dark it hands WBPP because WBPP has no
                            other subtractive slot, and that quirk must not
                            leak out of `stacks/stacking/pixinsight.py`.
                            It takes a dark and a flat of its own because its
                            members ARE lights, calibrated exactly as the
                            science frames are so that what survives is the
                            residual the flat could not remove. NOT a fringe:
                            that would be a cycle, and there is nothing to
                            subtract from a frame that defines the pattern.
"""

CALIBRATION_TOPOLOGY: dict[str, tuple[str, ...]] = {
    "bias": (),
    "dark": (),
    "flat": ("bias",),
    "fringe": ("dark", "flat"),
    "light": ("dark", "flat", "fringe"),
}

# Calibration types a light may resolve to NOTHING without that being a fault.
# Every other dependency missing is worth a concern; a fringe frame exists only
# where the user deliberately built one.
OPTIONAL_CALIBRATION: frozenset[str] = frozenset({"fringe"})
