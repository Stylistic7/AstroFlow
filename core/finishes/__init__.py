"""User-authored display finishes — Python traced to WGSL.

A *finish* is a short per-pixel Python function that turns the linear scene
formats (Linear RGB, Linear RGB Stars, Linear Narrowband, Linear L, Linear
SHO) into a display-ready colour image. ONE definition drives everything:

  * traced once at import with symbolic proxies → the WGSL the viewer runs
    live, with every `Param` a slider/uniform;
  * evaluated directly on numpy arrays → the server-side baker (dashboard
    previews today, full-res exports later).

Drop a file in `~/AstroFlow/finishes/` to add or REPLACE a look (same
name + primary input overrides a stock finish). You are running that
Python — read what you install. Author surface:

    from finishes import finish, Param, vec3, where, mtf, screen, ...

    @finish(name="My HOO", inputs=("Linear Narrowband Starless",))
    def my_hoo(nb, mid=Param(0.15, 0.01, 0.95, scale="log")):
        h, o = nb.g, nb.b
        return vec3(mtf(h, mid), mtf(o, mid), mtf(o, mid))

Params are SYMBOLIC at trace time (they become GPU uniforms — the slider
moves them without re-tracing), so `if some_param:` raises; per-pixel
branching is `where(cond, a, b)`.
"""
from finishes.framework import Color, Param, finish
from finishes.stdlib import (
    absolute, clamp, dot, exp, log, maximum, minimum, mix, mtf, saturate,
    screen, sqrt, replace_lightness, vec3, where, white,
)

__all__ = [
    "Color", "Param", "finish",
    "absolute", "clamp", "dot", "exp", "log", "maximum", "minimum", "mix",
    "mtf", "saturate", "screen", "sqrt", "replace_lightness", "vec3",
    "where", "white",
]
