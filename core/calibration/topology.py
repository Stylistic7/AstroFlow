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
  * `light: ("dark", "flat")` — NO master bias for lights. Case 2's
                            closed form is `(L − MD) · s0 / MF` with no MB
                            term; staging an MB into the lights' WBPP working
                            dir would double-subtract bias against a master
                            dark that already contains it (§8.2). With this
                            topology WBPP's auto-detect lands on the documented
                            Case 2 settings.
  * `sky: ("dark",)`      — a NIGHT-SKY FLAT: dithered blank-sky lights,
                            stars median-rejected, averaged and normalised
                            (`stacks/sky_flat.py`). Its members ARE lights, so
                            it takes a dark — and nothing else. It is a FLAT,
                            not a fourth calibration type: what makes it
                            special is its ILLUMINANT, not its arithmetic.
                            Because it is lit by the night sky's emission
                            lines, it captures the sensor's etalon at the same
                            spectral weighting the science frames see it, which
                            a panel or twilight flat (continuum, suppressed by
                            sinc(W/dlambda)) cannot. It therefore corrects
                            vignetting, dust AND the etalon in one division,
                            for the background and the object alike. Measured
                            2026-09-25 on two panels: dividing beats
                            subtracting a scaled map, which only ever flattens
                            the background and leaves the object modulated.
"""

CALIBRATION_TOPOLOGY: dict[str, tuple[str, ...]] = {
    "bias": (),
    "dark": (),
    "flat": ("bias",),
    "sky": ("dark",),
    "light": ("dark", "flat"),
}
