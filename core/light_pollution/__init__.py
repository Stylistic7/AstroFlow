"""The sky glow a site puts into every frame — light pollution and moonlight —
as a small model per observing site, fitted from the library's own subframes
and subtracted from each subframe before it is stacked.

`sites`    where a frame was shot (the header's SITELAT/SITELONG) and the sites of a library
`measure`  one run's subframes -> their tilts, and the glow shapes' tilts at each frame's moment
`model`    the site model: city dome shape + moonlight, fit, hysteresis, a frame's surface
`graph`    the library-wide fold, scheduled beside the dust epochs
"""
