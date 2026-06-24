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
  * `light: ("dark", "flat")` — NO master bias for lights. Case 2's closed
                            form is `(L − MD) · s0 / MF` with no MB term;
                            staging an MB into the lights' WBPP working dir
                            would double-subtract bias against a master dark
                            that already contains it (§8.2). With this topology
                            WBPP's auto-detect lands on the documented Case 2
                            settings.
"""

CALIBRATION_TOPOLOGY: dict[str, tuple[str, ...]] = {
    "bias": (),
    "dark": (),
    "flat": ("bias",),
    "light": ("dark", "flat"),
}
