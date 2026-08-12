"""The stock finishes — the five canonical looks, each a short per-pixel
function over the linear scene formats. These are the reference for
user-authored finishes (`~/AstroFlow/finishes/`): copy one, rename it,
change the math.

Conventions that make the knobs pleasant:
  * stretch params named `blackpoint/whitepoint/midtones` (linked) or
    `blackS/whiteS/midS` … (per-channel) get image-adaptive starting values
    from the viewer's STF seeding — free, by name.
  * `star_s` rides the WITH-STARS combine: `combine + (star_s − 1)·stars`
    is the plain photo at 1, starless at 0, star reduction between.
  * raw Hα is band `.g` of the zero-filled S,H,O narrowband layout.
"""
from __future__ import annotations

from finishes.framework import Param, finish
from finishes.stdlib import (
    maximum, minimum, mtf, replace_lightness, saturate, screen, vec3,
)


def _stretch(x, bp, wp, mid):
    """Black/white-point + midtones — the per-channel building block."""
    return mtf(saturate((x - bp) / maximum(wp - bp, 1e-6)), mid)


def _stars_layer(stars, star_s, star_mid):
    """Stretched, intensity-scaled RGB stars ready to screen over a palette."""
    return saturate(star_s * mtf(saturate(stars), star_mid))


@finish(name="Hubble", inputs=("Linear SHO", "Linear RGB Stars"),
        alt_primaries=("Linear Narrowband Starless",))
def hubble(nb, stars,
           blackS=Param(0.004, 0.0, 0.05, step=0.03, label="black S", scale="log"),
           whiteS=Param(0.03, 0.005, 0.3, step=0.03, label="white S", scale="log"),
           midS=Param(0.25, 0.05, 0.95, step=0.015, label="mid S", scale="log"),
           blackH=Param(0.004, 0.0, 0.05, step=0.03, label="black H", scale="log"),
           whiteH=Param(0.03, 0.005, 0.3, step=0.03, label="white H", scale="log"),
           midH=Param(0.25, 0.05, 0.95, step=0.015, label="mid H", scale="log"),
           blackO=Param(0.004, 0.0, 0.05, step=0.03, label="black O", scale="log"),
           whiteO=Param(0.03, 0.005, 0.3, step=0.03, label="white O", scale="log"),
           midO=Param(0.25, 0.05, 0.95, step=0.015, label="mid O", scale="log"),
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           star_mid=Param(0.25, 0.05, 0.95, step=0.03, label="star mid", scale="log"),
           scnr=Param(0.9, 0.0, 1.0, step=0.1, label="green→gold")):
    """Classic Hubble palette: R=SII, G=Hα, B=OIII, per-channel stretch,
    RGB stars screened back in."""
    s = _stretch(nb.r, blackS, whiteS, midS)
    h = _stretch(nb.g, blackH, whiteH, midH)
    o = _stretch(nb.b, blackO, whiteO, midO)
    # SCNR (average-neutral): green → gold, the classic SHO move —
    # without it Hα's dominance renders the whole palette green.
    h = minimum(h, (1.0 - scnr) * h + scnr * 0.5 * (s + o))
    return screen(vec3(s, h, o), _stars_layer(stars, star_s, star_mid))


@finish(name="Foraxx", inputs=("Linear SHO", "Linear RGB Stars"),
        alt_primaries=("Linear Narrowband Starless",))
def foraxx(nb, stars,
           blackS=Param(0.004, 0.0, 0.05, step=0.03, label="black S", scale="log"),
           whiteS=Param(0.03, 0.005, 0.3, step=0.03, label="white S", scale="log"),
           midS=Param(0.25, 0.05, 0.95, step=0.015, label="mid S", scale="log"),
           blackH=Param(0.004, 0.0, 0.05, step=0.03, label="black H", scale="log"),
           whiteH=Param(0.03, 0.005, 0.3, step=0.03, label="white H", scale="log"),
           midH=Param(0.25, 0.05, 0.95, step=0.015, label="mid H", scale="log"),
           blackO=Param(0.004, 0.0, 0.05, step=0.03, label="black O", scale="log"),
           whiteO=Param(0.03, 0.005, 0.3, step=0.03, label="white O", scale="log"),
           midO=Param(0.25, 0.05, 0.95, step=0.015, label="mid O", scale="log"),
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           star_mid=Param(0.25, 0.05, 0.95, step=0.03, label="star mid", scale="log"),
           scnr=Param(0.9, 0.0, 1.0, step=0.1, label="green→gold")):
    """Foraxx dynamic palette: per-pixel data-dependent narrowband weights
    (`~x = 1−x`): R = (O^~O)·S + ~(O^~O)·H; G blends H/O by (O·H)^~(O·H);
    B = O. Inputs saturated first so pow never sees a negative base."""
    s = saturate(_stretch(nb.r, blackS, whiteS, midS))
    h = saturate(_stretch(nb.g, blackH, whiteH, midH))
    o = saturate(_stretch(nb.b, blackO, whiteO, midO))
    wr = o ** (1.0 - o)
    r = wr * s + (1.0 - wr) * h
    m = o * h
    wg = m ** (1.0 - m)
    g = wg * h + (1.0 - wg) * o
    g = minimum(g, (1.0 - scnr) * g + scnr * 0.5 * (r + o))
    return screen(vec3(r, g, o), _stars_layer(stars, star_s, star_mid))


@finish(name="RGB", inputs=("Linear RGB", "Linear RGB Stars"))
def rgb(combine, stars,
        star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
        blackpoint=Param(0.002, 0.0, 0.1, step=0.03, label="black", scale="log"),
        whitepoint=Param(0.5, 0.02, 1.0, step=0.03, label="white", scale="log"),
        midtones=Param(0.15, 0.01, 0.95, step=0.015, label="mid", scale="log")):
    """The broadband colour photo: star intensity on the with-stars combine
    (1 = plain, 0 = starless), then a LINKED stretch — one curve for all
    three channels, so the stretch never shifts hue."""
    c = combine + (star_s - 1.0) * stars
    return _stretch(c, blackpoint, whitepoint, midtones)


@finish(name="LRGB", inputs=("Linear RGB", "Linear RGB Stars", "Linear L"))
def lrgb(combine, stars, lum,
         star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
         blackpoint=Param(0.002, 0.0, 0.1, step=0.03, label="black", scale="log"),
         whitepoint=Param(0.5, 0.02, 1.0, step=0.03, label="white", scale="log"),
         midtones=Param(0.15, 0.01, 0.95, step=0.015, label="mid", scale="log"),
         lum_mid=Param(0.15, 0.01, 0.95, step=0.015, label="L mid", scale="log"),
         chroma=Param(1.2, 0.0, 3.0, step=0.1, label="chroma")):
    """LRGB combination: colour from the stretched RGB, lightness replaced
    by the (separately stretched) luminance master in CIE Lab."""
    c = _stretch(combine + (star_s - 1.0) * stars,
                 blackpoint, whitepoint, midtones)
    la = _stretch(lum.r, blackpoint, whitepoint, lum_mid)
    return replace_lightness(c, la, chroma)


@finish(name="HaLRGB",
        inputs=("Linear RGB", "Linear RGB Stars", "Linear Narrowband",
                "Linear L"))
def halrgb(combine, stars, nb, lum,
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           ha_k=Param(0.4, 0.0, 2.0, step=0.1, label="Hα→R"),
           ha_q=Param(0.0, 0.0, 1.0, step=0.05, label="continuum"),
           blackpoint=Param(0.002, 0.0, 0.1, step=0.03, label="black", scale="log"),
           whitepoint=Param(0.5, 0.02, 1.0, step=0.03, label="white", scale="log"),
           midtones=Param(0.15, 0.01, 0.95, step=0.015, label="mid", scale="log"),
           lum_mid=Param(0.15, 0.01, 0.95, step=0.015, label="L mid", scale="log"),
           chroma=Param(1.2, 0.0, 3.0, step=0.1, label="chroma")):
    """The galaxy special: Hα line flux into R with live continuum
    subtraction (`ha_q`), sampling RAW narrowband — star removal never
    touches the HII knots — then the LRGB combination."""
    c = combine + (star_s - 1.0) * stars
    r2 = c.r + ha_k * maximum(nb.g - ha_q * c.r, 0.0)
    c = _stretch(vec3(r2, c.g, c.b), blackpoint, whitepoint, midtones)
    la = _stretch(lum.r, blackpoint, whitepoint, lum_mid)
    return replace_lightness(c, la, chroma)
