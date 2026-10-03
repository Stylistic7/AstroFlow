"""The stock finishes — the canonical looks, each a short per-pixel
function over the linear scene formats, plus the CHANNEL finishes (H, S, O,
R, G, B, L): one band of a tap, autostretched, RGB stars screened back in.
These are the reference for user-authored finishes
(`~/AstroFlow/finishes/`): copy one, rename it, change the math.

Registration order is the default-view preference: the first finish that
binds for an object is its landing view.

Conventions that make the knobs pleasant:
  * stretch params named `blackpoint/whitepoint/midtones` (linked) or
    `blackS/whiteS/midS` … (per-channel) get image-adaptive starting values
    from the viewer's STF seeding — free, by name.
  * scene formats are STARLESS + STARS layer pairs (no with-stars format is
    baked): `star_s` scales the stars layer added back onto the starless —
    `starless + star_s·stars` is the plain photo at 1 (placement and merging
    are linear, so the layer sum reproduces the composite exactly), starless
    at 0, star reduction between. A missing stars format binds black, which
    degrades gracefully to the starless view.
  * Hα is band `.g` of the zero-filled S,H,O narrowband layout.
"""
from __future__ import annotations

import inspect

from finishes.framework import Param, finish
from finishes.stdlib import (
    maximum, minimum, mtf, replace_lightness, saturate, screen, vec3, where,
)


def _stretch(x, bp, wp, mid):
    """Black/white-point + midtones — the per-channel building block."""
    return mtf(saturate((x - bp) / maximum(wp - bp, 1e-6)), mid)


def _stars_layer(stars, star_s, star_mid, star_white):
    """Stretched, intensity-scaled RGB stars ready to screen over a palette.

    `star_white` NORMALIZES before the midtones curve, and is the whole
    reason stars are visible: the layer is LINEAR data on the pinned
    photometric gauge (sky 0.005, star pixels ~0.005-0.5), not a 0-1
    display image. Curving raw values crushed them — measured
    2026-09-03: a p99.9 star at 0.005 came out of `mtf(0.25, ·)` at
    0.0149, so even star_s=2 screened a 3% brightening over the palette
    and the slider looked dead. Dividing by the bright anchor (`high`
    from channelStats, ~0.0089) maps a bright star to 1.0 first, so the
    curve then does what it was written to do."""
    return saturate(star_s * mtf(saturate(stars / maximum(star_white, 1e-6)),
                                 star_mid))


@finish(name="Hubble", inputs=("Linear SHO", "Linear RGB Stars"),
        alt_primaries=("Linear Narrowband Starless",),
        optional=("Linear RGB Stars",))
def hubble(nb, stars,
           blackS=Param(0.00496, 0.0, 0.05, step=0.03, label="black S", scale="log"),
           whiteS=Param(0.0089, 0.002, 0.3, step=0.03, label="white S", scale="log"),
           midS=Param(0.088, 0.05, 0.95, step=0.015, label="mid S", scale="log"),
           blackH=Param(0.00496, 0.0, 0.05, step=0.03, label="black H", scale="log"),
           whiteH=Param(0.0089, 0.002, 0.3, step=0.03, label="white H", scale="log"),
           midH=Param(0.088, 0.05, 0.95, step=0.015, label="mid H", scale="log"),
           blackO=Param(0.00496, 0.0, 0.05, step=0.03, label="black O", scale="log"),
           whiteO=Param(0.0089, 0.002, 0.3, step=0.03, label="white O", scale="log"),
           midO=Param(0.088, 0.05, 0.95, step=0.015, label="mid O", scale="log"),
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           star_mid=Param(0.25, 0.05, 0.95, step=0.03, label="star mid", scale="log"),
           star_white=Param(0.0089, 0.001, 0.5, step=0.03, label="star white", scale="log"),
           scnr=Param(0.9, 0.0, 1.0, step=0.1, label="green→gold")):
    """Classic Hubble palette: R=SII, G=Hα, B=OIII, per-channel stretch,
    RGB stars screened back in."""
    s = _stretch(nb.r, blackS, whiteS, midS)
    h = _stretch(nb.g, blackH, whiteH, midH)
    o = _stretch(nb.b, blackO, whiteO, midO)
    # SCNR (average-neutral): green → gold, the classic SHO move —
    # without it Hα's dominance renders the whole palette green.
    h = minimum(h, (1.0 - scnr) * h + scnr * 0.5 * (s + o))
    return screen(vec3(s, h, o), _stars_layer(stars, star_s, star_mid, star_white))


_E = 2.718281828459045
_GHS_B = 15.0


def _ghs_pinned(x, sky, sig, top, level, neb, strength):
    """One channel of the pinned GHS stretch (designed with Stylistic7 on the
    75a1 Hα mosaic, 2026-10-02): a FIXED GHS shape -- b = 15, SP just left
    of the sky peak (sky − 1.5σ) as in Siril's GHS tutorial, D from
    `strength` = ln(D + 1) -- followed by a linear black/white map SOLVED so
    the sky median lands exactly on `level` and the bright anchor `top`
    exactly on `neb`. Sky and nebula then move independently: each slider
    holds its own point fixed whatever the others do. `sky`, `sig` and `top`
    are data anchors seeded from the channel's stats; `neb` > 1 clips the
    brightest nebula to white on purpose.

    Only the GHS's own shape matters here: its normalisation constants
    cancel in the two-point solve, so f is used un-normalised."""
    d = maximum(_E ** strength - 1.0, 0.05)
    sp = sky - 1.5 * sig

    def t(u):   # the b > 0 GHS integral, u >= 0
        return 1.0 - (1.0 + _GHS_B * d * maximum(u, 0.0)) ** (-1.0 / _GHS_B)

    def f(v):   # point-symmetric about SP
        return where(v < sp, -t(sp - v), t(v - sp))

    f_sky = f(sky)
    scale = (neb - level) / maximum(f(top) - f_sky, 1e-12)
    return saturate((f(x) - f_sky) * scale + level)


@finish(name="Hubble Pinned GHS", inputs=("Linear SHO", "Linear RGB Stars"),
        alt_primaries=("Linear Narrowband Starless",),
        optional=("Linear RGB Stars",))
def hubble_ghs(nb, stars,
               levelS=Param(0.04, -0.1, 0.2, step=0.005, label="sky level", group="SII (red)"),
               nebS=Param(1.0, 0.5, 1.5, step=0.01, label="nebula level", group="SII (red)"),
               strS=Param(4.0, 0.0, 10.0, step=0.05, label="stretch", group="SII (red)"),
               skyS=Param(0.005, 0.0, 0.05, step=0.03, label="sky", scale="log", group="SII (red)", kind="data"),
               sigS=Param(3e-5, 1e-7, 1e-2, step=0.03, label="noise σ", scale="log", group="SII (red)", kind="data"),
               topS=Param(0.0089, 0.001, 0.5, step=0.03, label="bright anchor", scale="log", group="SII (red)", kind="data"),
               levelH=Param(0.04, -0.1, 0.2, step=0.005, label="sky level", group="Hα (green)"),
               nebH=Param(1.0, 0.5, 1.5, step=0.01, label="nebula level", group="Hα (green)"),
               strH=Param(4.0, 0.0, 10.0, step=0.05, label="stretch", group="Hα (green)"),
               skyH=Param(0.005, 0.0, 0.05, step=0.03, label="sky", scale="log", group="Hα (green)", kind="data"),
               sigH=Param(3e-5, 1e-7, 1e-2, step=0.03, label="noise σ", scale="log", group="Hα (green)", kind="data"),
               topH=Param(0.0089, 0.001, 0.5, step=0.03, label="bright anchor", scale="log", group="Hα (green)", kind="data"),
               levelO=Param(0.04, -0.1, 0.2, step=0.005, label="sky level", group="OIII (blue)"),
               nebO=Param(1.0, 0.5, 1.5, step=0.01, label="nebula level", group="OIII (blue)"),
               strO=Param(4.0, 0.0, 10.0, step=0.05, label="stretch", group="OIII (blue)"),
               skyO=Param(0.005, 0.0, 0.05, step=0.03, label="sky", scale="log", group="OIII (blue)", kind="data"),
               sigO=Param(3e-5, 1e-7, 1e-2, step=0.03, label="noise σ", scale="log", group="OIII (blue)", kind="data"),
               topO=Param(0.0089, 0.001, 0.5, step=0.03, label="bright anchor", scale="log", group="OIII (blue)", kind="data"),
               star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars", group="Stars"),
               star_mid=Param(0.25, 0.05, 0.95, step=0.03, label="star mid", scale="log", group="Stars"),
               star_white=Param(0.0089, 0.001, 0.5, step=0.03, label="star white", scale="log", group="Stars"),
               scnr=Param(0.9, 0.0, 1.0, step=0.1, label="green→gold", group="Colour")):
    """Hubble palette (R=SII, G=Hα, B=OIII) with the pinned GHS stretch per
    channel: sky level, nebula level and stretch are independent knobs.
    The anchors (sky<R>, sig<R>, top<R>) are kind="data": measured from each
    starless channel's sky median, sky noise and bright level, never shown."""
    s = _ghs_pinned(nb.r, skyS, sigS, topS, levelS, nebS, strS)
    h = _ghs_pinned(nb.g, skyH, sigH, topH, levelH, nebH, strH)
    o = _ghs_pinned(nb.b, skyO, sigO, topO, levelO, nebO, strO)
    h = minimum(h, (1.0 - scnr) * h + scnr * 0.5 * (s + o))
    return screen(vec3(s, h, o), _stars_layer(stars, star_s, star_mid, star_white))


@finish(name="Foraxx", inputs=("Linear SHO", "Linear RGB Stars"),
        alt_primaries=("Linear Narrowband Starless",),
        optional=("Linear RGB Stars",))
def foraxx(nb, stars,
           blackS=Param(0.00496, 0.0, 0.05, step=0.03, label="black S", scale="log"),
           whiteS=Param(0.0089, 0.002, 0.3, step=0.03, label="white S", scale="log"),
           midS=Param(0.088, 0.05, 0.95, step=0.015, label="mid S", scale="log"),
           blackH=Param(0.00496, 0.0, 0.05, step=0.03, label="black H", scale="log"),
           whiteH=Param(0.0089, 0.002, 0.3, step=0.03, label="white H", scale="log"),
           midH=Param(0.088, 0.05, 0.95, step=0.015, label="mid H", scale="log"),
           blackO=Param(0.00496, 0.0, 0.05, step=0.03, label="black O", scale="log"),
           whiteO=Param(0.0089, 0.002, 0.3, step=0.03, label="white O", scale="log"),
           midO=Param(0.088, 0.05, 0.95, step=0.015, label="mid O", scale="log"),
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           star_mid=Param(0.25, 0.05, 0.95, step=0.03, label="star mid", scale="log"),
           star_white=Param(0.0089, 0.001, 0.5, step=0.03, label="star white", scale="log"),
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
    return screen(vec3(r, g, o), _stars_layer(stars, star_s, star_mid, star_white))


@finish(name="RGB", inputs=("Linear RGB Starless", "Linear RGB Stars"),
        optional=("Linear RGB Stars",))
def rgb(starless, stars,
        star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
        blackpoint=Param(0.00496, 0.0, 0.1, step=0.03, label="black", scale="log"),
        whitepoint=Param(0.0089, 0.002, 1.0, step=0.03, label="white", scale="log"),
        midtones=Param(0.088, 0.01, 0.95, step=0.015, label="mid", scale="log")):
    """The broadband colour photo: stars added onto the starless layer
    (1 = plain photo, 0 = starless), then a LINKED stretch — one curve
    for all three channels, so the stretch never shifts hue."""
    c = starless + star_s * stars
    return _stretch(c, blackpoint, whitepoint, midtones)


@finish(name="LRGB",
        inputs=("Linear RGB Starless", "Linear RGB Stars",
                "Linear L Starless", "Linear L Stars"),
        optional=("Linear RGB Stars", "Linear L Stars"))
def lrgb(starless, stars, lum, lum_stars,
         star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
         blackpoint=Param(0.00496, 0.0, 0.1, step=0.03, label="black", scale="log"),
         whitepoint=Param(0.0089, 0.002, 1.0, step=0.03, label="white", scale="log"),
         midtones=Param(0.088, 0.01, 0.95, step=0.015, label="mid", scale="log"),
         lum_mid=Param(0.088, 0.01, 0.95, step=0.015, label="L mid", scale="log"),
         chroma=Param(1.2, 0.0, 3.0, step=0.1, label="chroma")):
    """LRGB combination: colour from the stretched RGB layers, lightness
    replaced by the (separately stretched) luminance in CIE Lab. `star_s`
    rides BOTH layer pairs, so stars keep their lightness in the L plane
    exactly as they keep their chroma in RGB."""
    c = _stretch(starless + star_s * stars,
                 blackpoint, whitepoint, midtones)
    la = _stretch(lum.r + star_s * lum_stars.r,
                  blackpoint, whitepoint, lum_mid)
    return replace_lightness(c, la, chroma)


@finish(name="HaLRGB",
        inputs=("Linear RGB Starless", "Linear RGB Stars",
                "Linear Narrowband Starless", "Linear L Starless"),
        optional=("Linear RGB Stars",))
def halrgb(starless, stars, nb, lum,
           star_s=Param(1.0, 0.0, 2.0, step=0.1, label="stars"),
           ha_k=Param(0.4, 0.0, 2.0, step=0.1, label="Hα→R"),
           ha_q=Param(0.0, 0.0, 1.0, step=0.05, label="continuum"),
           blackpoint=Param(0.00496, 0.0, 0.1, step=0.03, label="black", scale="log"),
           whitepoint=Param(0.0089, 0.002, 1.0, step=0.03, label="white", scale="log"),
           midtones=Param(0.088, 0.01, 0.95, step=0.015, label="mid", scale="log"),
           lum_mid=Param(0.088, 0.01, 0.95, step=0.015, label="L mid", scale="log"),
           chroma=Param(1.2, 0.0, 3.0, step=0.1, label="chroma")):
    """The galaxy special: Hα line flux into R with live continuum
    subtraction (`ha_q`) from the starless narrowband, then the LRGB
    combination. All four input slots are spent, so the L stars ride in
    as a PROXY — the RGB stars' mean is added into the linear lightness
    (L star flux ≈ broadband star flux; exact would need a fifth slot).
    Compact HII knots that native SXT classifies as stars land in the
    narrowband STARS layer and are excluded from the Hα injection — the
    lab's confetti gate watches that boundary."""
    c = starless + star_s * stars
    r2 = c.r + ha_k * maximum(nb.g - ha_q * c.r, 0.0)
    c = _stretch(vec3(r2, c.g, c.b), blackpoint, whitepoint, midtones)
    star_l = star_s * (stars.r + stars.g + stars.b) / 3.0
    la = _stretch(lum.r + star_l, blackpoint, whitepoint, lum_mid)
    return replace_lightness(c, la, chroma)


# ── channel finishes ──────────────────────────────────────────────────────
# One body serves every level: a mono tap (a stack's Post-Processed master,
# a target's Channel) is broadcast to (r, r, r) by the viewer and the bake,
# so `band` reads the same value whichever tap binds. Primaries are in
# preference order; the catalog binds a per-filter tap to the entry whose
# filter normalises to `role`. "Mosaic" is a pipeline's finished per-filter
# mosaic (the PI post-processed channel, or the Subframe preview's merge);
# "Master" is the fallback while a stack's Post-Processed is still building.
_NARROWBAND = ("Linear SHO", "Linear Narrowband Starless", "Mosaic Starless",
               "Post-Processed", "Channel", "Mosaic", "Master")
_BROADBAND = ("Linear RGB Starless", "Mosaic Starless",
              "Post-Processed", "Channel", "Mosaic", "Master")
_LUM = ("Linear L Starless", "Mosaic Starless",
        "Post-Processed", "Channel", "Mosaic", "Master")


def _channel(role: str, band: str, primaries: tuple):
    """A single filter's channel, autostretched, with NO stars layer: a
    channel view shows that filter's (starless) data and nothing else
    (Stylistic7, 2026-10-01). Where the primary is a with-stars master
    (Post-Processed / Master), its own stars are simply part of the data."""
    def body(c, **k):
        x = _stretch(getattr(c, band), k["black" + role], k["white" + role],
                     k["mid" + role])
        return vec3(x, x, x)

    kw = inspect.Parameter.KEYWORD_ONLY
    body.__name__ = role
    body.__signature__ = inspect.Signature([
        inspect.Parameter("c", inspect.Parameter.POSITIONAL_ONLY),
        inspect.Parameter("black" + role, kw, default=Param(0.00496, 0.0, 0.05, step=0.03, label="black", scale="log")),
        inspect.Parameter("white" + role, kw, default=Param(0.0089, 0.002, 1.0, step=0.03, label="white", scale="log")),
        inspect.Parameter("mid" + role, kw, default=Param(0.088, 0.05, 0.95, step=0.015, label="mid", scale="log")),
    ])
    return finish(name=role, inputs=(primaries[0],), alt_primaries=primaries[1:],
                  role=role)(body)


for _role, _band in (("H", "g"), ("S", "r"), ("O", "b")):
    _channel(_role, _band, _NARROWBAND)
for _role, _band in (("R", "r"), ("G", "g"), ("B", "b")):
    _channel(_role, _band, _BROADBAND)
_channel("L", "r", _LUM)


@finish(name="Master", inputs=("Master",), role="none")
def master(c,
           blackpoint=Param(0.00496, 0.0, 0.1, step=0.03, label="black", scale="log"),
           whitepoint=Param(0.0089, 0.002, 1.0, step=0.03, label="white", scale="log"),
           midtones=Param(0.088, 0.01, 0.95, step=0.015, label="mid", scale="log")):
    """A filterless master (dark, bias): the autostretched integration."""
    return _stretch(c, blackpoint, whitepoint, midtones)
