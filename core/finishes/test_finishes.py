"""The finishes package: tracer parity, WGSL structure, validation errors,
math pins, and registry discovery. The load-bearing idea: `bake()` runs the
author's ORIGINAL Python on numpy, `interpret()` evaluates the TRACED GRAPH
under WGSL semantics — when they agree, the emitted shader computes what the
author wrote (select-arg order, splats, promotion, CSE all pinned without a
GPU)."""
import numpy as np
import pytest

from finishes.framework import FinishDefinitionError, Param, finish
from finishes.registry import finish_specs, get_finish, refresh_finishes
from finishes.stdlib import NpVec, _mtf_np, _replace_lightness_np
from finishes.tracer import FinishTraceError


def _srcs(n, seed=7, scale=0.05, size=16):
    rng = np.random.default_rng(seed)
    return [(rng.random((size, size, 3)) * scale).astype(np.float32)
            for _ in range(n)]


STOCK = [("RGB", 2), ("LRGB", 4), ("HaLRGB", 4), ("Hubble", 2), ("Hubble Pinned GHS", 2), ("Emission Hues", 2), ("Foraxx", 2),
         ("H", 1), ("S", 1), ("O", 1), ("R", 1), ("G", 1), ("B", 1), ("L", 1),
         ("Master", 1)]


class TestParity:
    """bake (direct numpy) ≡ interpret (traced graph, WGSL semantics)."""

    @pytest.mark.parametrize("name,nsrc", STOCK)
    def test_stock_finishes_at_defaults_and_jittered(self, name, nsrc):
        f = get_finish(name)
        rng = np.random.default_rng(3)
        for seed in (1, 2):
            sources = _srcs(nsrc, seed=seed)
            values = {p.name: float(np.clip(
                p.default * (0.7 + 0.6 * rng.random()), p.lo, p.hi))
                for p in f.params}
            a = f.bake(sources, values)
            b = f.interpret(sources, values)
            assert np.abs(a - b).max() < 1e-5

    def test_where_select_order_and_swizzles(self):
        """A micro-finish exercising the riskiest emissions: comparison →
        select (REVERSED arg order in WGSL), swizzle reads, vec3 assembly."""
        from finishes.stdlib import vec3, where

        @finish(name="_t_where", inputs=("Linear RGB",))
        def t(c, k=Param(0.5, 0.0, 1.0)):
            hot = where(c.r > k, c.r, c.b)
            return vec3(hot, c.g, k - c.b)

        (src,) = _srcs(1, scale=1.0)
        for k in (0.1, 0.9):
            assert np.abs(t.bake([src], {"k": k})
                          - t.interpret([src], {"k": k})).max() < 1e-6
        assert "select(" in t.wgsl


class TestWgslStructure:
    def test_signature_struct_order_and_helpers(self):
        f = get_finish("HaLRGB")
        w = f.wgsl
        assert ("fn displayFinish(src0: vec4<f32>, src1: vec4<f32>, "
                "src2: vec4<f32>, src3: vec4<f32>, P: DisplayParams) "
                "-> vec3<f32>") in w
        # struct fields in signature order = uniform packing order
        names = [p.name for p in f.params]
        idx = [w.index(f"    {n}: f32,") for n in names]
        assert idx == sorted(idx)
        # helper dedup: two mtf3 consumers, ONE fn mtf / fn mtf3
        assert w.count("fn mtf(") == 1
        assert w.count("fn mtf3(") == 1
        assert w.count("fn replaceLightness(") == 1
        assert w.count("{") == w.count("}")

    def test_alt_primary_copies_share_wgsl(self):
        a = get_finish("Hubble", primary="Linear SHO")
        b = get_finish("Hubble", primary="Linear Narrowband Starless")
        assert a is not None and b is not None
        assert a.wgsl == b.wgsl and a.finish_hash == b.finish_hash
        assert b.inputs[0] == "Linear Narrowband Starless"
        assert a.inputs[1:] == b.inputs[1:]

    def test_specs_serve_the_contract(self):
        for s in finish_specs():
            assert set(s) == {"name", "inputs", "optional", "params", "wgsl",
                              "role", "finish_hash"}
            assert set(s["optional"]) <= set(s["inputs"])
            for p in s["params"]:
                assert {"name", "default", "lo", "hi", "step", "label",
                        "scale"} <= set(p)


class TestValidation:
    def test_if_on_param_raises_friendly(self):
        with pytest.raises(FinishTraceError, match="where"):
            @finish(name="_t_if", inputs=("Linear RGB",))
            def t(c, k=Param(1.0, 0, 2)):
                if k > 0.5:
                    return c
                return c

    def test_too_many_inputs(self):
        with pytest.raises(FinishDefinitionError, match="at most"):
            @finish(name="_t_many", inputs=("A", "B", "C", "D", "E"))
            def t(a, b, c, d, e):
                return a

    def test_reserved_param_name(self):
        with pytest.raises(FinishDefinitionError, match="reserved"):
            @finish(name="_t_res", inputs=("Linear RGB",))
            def t(c, mtf=Param(1.0, 0, 2)):
                return c

    def test_non_vec3_return(self):
        with pytest.raises(FinishDefinitionError, match="vec3"):
            @finish(name="_t_scalar", inputs=("Linear RGB",))
            def t(c):
                return c.r

    def test_arg_count_mismatch(self):
        with pytest.raises(FinishDefinitionError, match="positional"):
            @finish(name="_t_argc", inputs=("Linear RGB", "Linear L"))
            def t(c):
                return c


class TestMathPins:
    """Ported invariants from the deleted display/test_ops.py."""

    def test_star_intensity_algebra(self):
        """Layer semantics (2026-09-02): inputs are STARLESS + STARS;
        star_s=1 reproduces the composite (starless + stars), star_s=0
        the starless layer alone."""
        f = get_finish("RGB")
        starless_src, stars = _srcs(2)
        v = f.defaults()
        plain = f.bake([starless_src, stars], {**v, "star_s": 1.0})
        starless = f.bake([starless_src, stars], {**v, "star_s": 0.0})
        ref = _mtf_np(np.clip(((starless_src + stars) - v["blackpoint"])
                              / (v["whitepoint"] - v["blackpoint"]), 0, 1),
                      v["midtones"])
        assert np.abs(plain - ref).max() < 1e-6
        ref0 = _mtf_np(np.clip((starless_src - v["blackpoint"])
                               / (v["whitepoint"] - v["blackpoint"]), 0, 1),
                       v["midtones"])
        assert np.abs(starless - ref0).max() < 1e-6

    def test_ha_continuum_clamp(self):
        f = get_finish("HaLRGB")
        combine, stars, nb, lum = _srcs(4)
        nb = nb.copy()
        nb[..., 1] = combine[..., 0] * 0.5      # Hα ≤ continuum everywhere
        v = f.defaults()
        boosted = f.bake([combine, stars, nb, lum],
                         {**v, "ha_k": 1.0, "ha_q": 2.0})
        base = f.bake([combine, stars, nb, lum], {**v, "ha_k": 0.0})
        assert np.abs(boosted - base).max() < 1e-6

    def test_lrgb_own_lightness_identity(self):
        """replace_lightness with the colour's own luminance Y and chroma=1
        reproduces the colour (grey (y,y,y) has exactly L*(Y=y))."""
        rng = np.random.default_rng(5)
        c = (rng.random((8, 8, 3)) * 0.8 + 0.1).astype(np.float64)
        y = c @ np.array([0.2126729, 0.7151522, 0.0721750])
        out = _replace_lightness_np(c, y, 1.0)
        assert np.abs(out - c).max() < 2e-3

    def test_mtf_matches_reference(self):
        # reference casts to float32; the stdlib computes float64 → the
        # comparison is bounded by float32 epsilon, not exactness.
        from mosaics.color.palettes import _mtf as reference
        x = np.linspace(0, 1, 257)
        for m in (0.1, 0.25, 0.5, 0.9):
            assert np.abs(_mtf_np(x, m) - reference(x, m)).max() < 1e-6


class TestNpVec:
    def test_swizzles_and_field_broadcast(self):
        a = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
        v = NpVec(a)
        assert np.array_equal(v.r, a[..., 0])
        out = v + v.r          # field broadcasts along channels
        assert np.array_equal(out.a, a + a[..., 0][..., None])
        out2 = 1.0 - v          # scalar rsub
        assert np.array_equal(out2.a, 1.0 - a)


class TestRegistry:
    def test_user_dir_loading_and_override(self, tmp_path, monkeypatch):
        import finishes.registry as reg
        (tmp_path / "AstroFlow" / "finishes").mkdir(parents=True)
        (tmp_path / "AstroFlow" / "finishes" / "mine.py").write_text(
            "from finishes import finish, Param, mtf, saturate\n"
            "@finish(name='RGB', inputs=('Linear RGB Starless',"
            " 'Linear RGB Stars'))\n"
            "def rgb(c, stars, mid=Param(0.3, 0.01, 0.95)):\n"
            "    return mtf(saturate(c), mid)\n"
        )
        (tmp_path / "AstroFlow" / "finishes" / "broken.py").write_text(
            "this is not python\n")
        monkeypatch.setattr(reg.Path, "home", lambda: tmp_path)
        try:
            refresh_finishes()
            f = get_finish("RGB", primary="Linear RGB Starless")
            authored = [p.name for p in f.params if not p.builtin]
            assert authored == ["mid"], \
                "user file must OVERRIDE the stock RGB"
            # A user-authored finish goes through the SAME composition as a
            # stock one — the framework's common first pass is injected here
            # too, so both run the identical traced path.
            assert [p.name for p in f.params if p.builtin] == [
                "trim_ragged", "trim_floor", "show_clipping"]
            assert get_finish("Hubble") is not None, \
                "a broken sibling file must not abort discovery"
        finally:
            monkeypatch.undo()
            refresh_finishes()


class TestCommonFirstPass:
    """The framework's common first pass, composed in front of every finish."""

    def _hubble(self):
        from finishes.registry import discover_finishes, get_finish
        discover_finishes()
        return get_finish("Hubble")

    def test_injected_into_every_finish_as_a_builtin_toggle(self):
        from finishes.registry import discover_finishes, finish_specs
        discover_finishes()
        for s in finish_specs():
            builtin = [p for p in s["params"] if p.get("builtin")]
            assert [p["name"] for p in builtin] == [
                "trim_ragged", "trim_floor", "show_clipping"], s["name"]
            assert builtin[0]["kind"] == "toggle", s["name"]
            # LAST, so author param order — and therefore the uniform packing
            # index every author already depends on — is untouched.
            assert s["params"][-1]["name"] == "show_clipping", s["name"]
            assert s["params"][-3]["name"] == "trim_ragged", s["name"]

    def test_zeroes_pixels_missing_a_required_band(self):
        import numpy as np
        f = self._hubble()
        # covered, missing H, missing O, missing S
        nb = np.array([[[0.02, 0.03, 0.02], [0.02, 0.0, 0.02],
                        [0.02, 0.03, 0.0], [0.0, 0.03, 0.02]]], dtype=np.float32)
        out = f.bake([nb, np.zeros_like(nb)], {"trim_ragged": 1.0})[0]
        assert out[0].max() > 0, "a fully covered pixel must survive"
        assert np.all(out[1:] == 0), "ragged pixels must go black"

    def test_toggle_off_restores_the_ragged_pixels(self):
        import numpy as np
        f = self._hubble()
        nb = np.array([[[0.02, 0.03, 0.02], [0.02, 0.0, 0.02]]], dtype=np.float32)
        stars = np.zeros_like(nb)
        on = f.bake([nb, stars], {"trim_ragged": 1.0})[0]
        off = f.bake([nb, stars], {"trim_ragged": 0.0})[0]
        assert np.all(on[1] == 0)
        assert off[1].max() > 0, "off must render the partial pixel's colour"
        assert np.allclose(on[0], off[0]), "covered pixels are untouched either way"

    def test_optional_inputs_are_not_tested(self):
        """A missing OPTIONAL input binds black; testing it would blank the
        whole image instead of its rim."""
        import numpy as np
        f = self._hubble()
        assert "Linear RGB Stars" in f.optional
        nb = np.full((1, 2, 3), 0.02, dtype=np.float32)
        out = f.bake([nb, np.zeros_like(nb)], {"trim_ragged": 1.0})[0]
        assert out.max() > 0, "an absent star layer must not black out the image"

    def test_traced_into_the_wgsl(self):
        """It goes through the tracer like any author op — not spliced into
        the shader template — so WGSL and numpy come from one source."""
        f = self._hubble()
        assert "trim_ragged: f32" in f.wgsl
        assert "trim_floor: f32" in f.wgsl
        assert "select(" in f.wgsl

    def test_trim_floor_zeroes_the_coverage_ramp(self):
        """`trim_floor` extends the exact-zero coverage test to a level:
        partial-coverage rolloff pixels (between no-data zero and the sky
        pedestal — pyramid downsampling manufactures them at the mosaic
        boundary) render as no-data once the floor is seeded, so a black
        point below the pedestal never exposes a bright colored rim
        (2026-09-03). Default 0.0 keeps the historical behavior."""
        import numpy as np
        f = self._hubble()
        # NOTE width 4, not 3: a width-3 image's channel slices have
        # last-dim 3 and _rewrap would treat them as colour vectors.
        # Values sit ABOVE the finish's default black points so the
        # stretch itself never blanks them — only the floor may.
        nb = np.array([[[0.02] * 3, [0.006] * 3,
                        [0.0] * 3, [0.02] * 3]], dtype=np.float32)
        stars = np.zeros_like(nb)
        seeded = f.bake([nb, stars], {"trim_ragged": 1.0, "trim_floor": 0.01})[0]
        assert seeded[0].max() > 0, "sky must survive the floor"
        assert np.all(seeded[1] == 0), "ramp pixels must render as no-data"
        assert np.all(seeded[2] == 0)
        assert seeded[3].max() > 0
        legacy = f.bake([nb, stars], {"trim_ragged": 1.0})[0]
        assert legacy[1].max() > 0, "default floor 0.0 = historical behavior"


class TestChannelFinishes:
    """The degenerate finishes — one band of a tap, autostretched, stars
    screened back in — named by filter ROLE, one body per role serving a
    3-band join and a mono tap alike."""

    def test_every_role_is_registered_for_its_family(self):
        from finishes.registry import all_finishes
        prim = {}
        for f in all_finishes():
            if f.role is not None:
                prim.setdefault(f.name, []).append(f.inputs[0])
        for role in "HSO":
            assert prim[role][:2] == ["Linear SHO", "Linear Narrowband Starless"]
            assert "Post-Processed" in prim[role] and "Master" in prim[role]
        for role in "RGB":
            assert prim[role][0] == "Linear RGB Starless"
        assert prim["L"][0] == "Linear L Starless"
        assert prim["Master"] == ["Master"]
        for f in all_finishes():
            assert (f.role is None) == (f.name in {"Hubble", "Hubble Pinned GHS", "Emission Hues", "Foraxx", "RGB", "LRGB", "HaLRGB"})

    def test_h_reads_band_g_of_the_join_and_the_whole_mono_tap(self):
        """`nb.g` is Hα in the S,H,O layout; a 1-band source is broadcast so
        the same body shows a stack's Post-Processed master."""
        f = get_finish("H", primary="Linear SHO")
        rng = np.random.default_rng(5)
        sho = (rng.random((8, 8, 3)) * 0.05).astype(np.float32)
        mono = sho[..., 1:2].copy()
        a = f.bake([sho]); b = f.bake([mono])
        assert np.allclose(a, b) and np.allclose(a[..., 0], a[..., 1])
        assert np.abs(f.bake([sho]) - f.interpret([sho])).max() < 1e-5
        # a different band changes nothing for H …
        sho2 = sho.copy(); sho2[..., 0] *= 3.0
        assert np.allclose(f.bake([sho2]), a)
        # … and everything for S
        assert not np.allclose(get_finish("S").bake([sho2]), get_finish("S").bake([sho]))

    def test_trim_tests_only_the_bands_the_finish_reads(self):
        """A zero-filled S band (a partial narrowband set) must not blank the
        H view; Hubble, which reads all three, still trims it."""
        h, hub = get_finish("H", primary="Linear SHO"), get_finish("Hubble", primary="Linear SHO")
        nb = np.array([[[0.0, 0.03, 0.02], [0.02, 0.03, 0.02], [0.02, 0.0, 0.02], [0.02, 0.03, 0.02]]], np.float32)
        stars = np.zeros_like(nb)
        out_h = h.bake([nb], {"trim_ragged": 1.0})[0]
        assert out_h[0].max() > 0 and np.all(out_h[2] == 0)
        out_hub = hub.bake([nb, stars], {"trim_ragged": 1.0})[0]
        assert np.all(out_hub[0] == 0) and np.all(out_hub[2] == 0) and out_hub[1].max() > 0

    def test_channel_finishes_mix_in_no_stars(self):
        """A single-channel view is that filter's data alone: one input, no
        star-layer parameters (2026-10-01)."""
        for role in ("H", "S", "O", "R", "G", "B", "L"):
            f = get_finish(role)
            assert len(f.inputs) == 1 and not f.optional, role
            assert not any(k.startswith("star") for k in f.defaults()), role
        rgb = np.full((4, 4, 3), 0.006, np.float32)   # between the default black and white
        out = get_finish("R").bake([rgb])
        assert out.shape == (4, 4, 3) and np.allclose(out[..., 0], out[..., 1])

    def test_master_finish_is_role_none(self):
        f = get_finish("Master")
        assert f.role == "none" and f.inputs == ("Master",)
        mono = np.full((4, 4, 1), 0.01, np.float32)
        out = f.bake([mono])
        assert out.shape == (4, 4, 3) and np.allclose(out[..., 0], out[..., 2])


def test_hubble_ghs_pins_sky_and_nebula_independently():
    """Each channel's sky median lands exactly on its sky level and its bright
    anchor exactly on its nebula level, whatever the stretch (2026-10-02)."""
    f = get_finish("Hubble Pinned GHS")
    rng = np.random.default_rng(1)
    sky, sig, top = 0.005, 3e-5, 0.0089
    nb = (sky + rng.normal(0, sig, (16, 16, 3))).astype(np.float32)
    nb[0, 0], nb[0, 1] = top, sky
    v = dict(f.defaults(), scnr=0.0, star_s=0.0)
    for r in "SHO":
        v.update({f"sky{r}": sky, f"sig{r}": sig, f"top{r}": top})
    for level, neb, strength in ((0.02, 1.0, 4.0), (0.12, 1.0, 1.0), (0.04, 0.7, 6.0)):
        for r in "SHO":
            v.update({f"level{r}": level, f"neb{r}": neb, f"str{r}": strength})
        out = f.bake([nb, np.zeros_like(nb)], v)
        assert np.allclose(out[0, 1], level, atol=1e-5)
        assert np.allclose(out[0, 0], neb, atol=1e-5)


def test_pinned_ghs_shows_three_controls_per_channel():
    """The anchors are measured from the image (kind="data"), so each channel
    offers only sky level, nebula level and stretch (Stylistic7, 2026-10-02)."""
    f = get_finish("Hubble Pinned GHS")
    for r in "SHO":
        shown = [p.name for p in f.params if p.kind != "data" and p.name[-1] == r and not p.builtin]
        assert sorted(shown) == sorted([f"level{r}", f"neb{r}", f"str{r}"]), shown
        assert all(p.kind == "data" for p in f.params if p.name in (f"sky{r}", f"sig{r}", f"top{r}"))


def test_emission_hues_adds_each_line_in_its_own_colour():
    """A pixel lit by one line shows that line's hue; lines add; an
    overflowing sum is scaled down by its own max, keeping its hue
    (Stylistic7, 2026-10-03)."""
    f = get_finish("Emission Hues")
    sky, sig, top = 0.005, 3e-5, 0.0089
    nb = np.full((4, 4, 3), sky, np.float32)
    nb[0, 0] = (top, sky, sky)    # SII only
    nb[0, 1] = (sky, top, sky)    # Hα only
    nb[0, 2] = (sky, sky, top)    # OIII only
    nb[0, 3] = (top, top, top)    # all three
    v = dict(f.defaults(), star_s=0.0)
    for r in "SHO":
        v.update({f"sky{r}": sky, f"sig{r}": sig, f"top{r}": top, f"level{r}": 0.0, f"neb{r}": 1.0})
    v.update(hueS=0.0, hueH=60.0, hueO=240.0, satS=1.0, satH=1.0, satO=0.5)
    out = f.bake([nb, np.zeros_like(nb)], v)
    assert np.allclose(out[0, 0], (1, 0, 0), atol=1e-5)
    assert np.allclose(out[0, 1], (1, 1, 0), atol=1e-5)
    assert np.allclose(out[0, 2], (0.5, 0.5, 1), atol=1e-5)
    assert np.allclose(out[0, 3], np.array((2.5, 1.5, 1)) / 2.5, atol=1e-5)
    assert np.allclose(out[1, 1], 0.0, atol=1e-5)


def test_emission_hues_shows_five_controls_per_line():
    f = get_finish("Emission Hues")
    for r in "SHO":
        shown = [p.name for p in f.params if p.kind != "data" and p.name[-1] == r and not p.builtin]
        assert sorted(shown) == sorted([f"level{r}", f"neb{r}", f"str{r}", f"hue{r}", f"sat{r}"]), shown


def test_show_clipping_paints_blown_and_crushed_pixels_only_where_there_is_data():
    """Built-in last pass (2026-10-03): magenta where any component renders as
    255, blue where all render as 0, untouched otherwise and on no-data pixels; a
    pure hue with zero components is not clipping."""
    f = get_finish("Emission Hues")
    sky, sig, top = 0.005, 3e-5, 0.0089
    nb = np.full((4, 4, 3), sky, np.float32)
    nb[0, 0] = (top, sky, sky)          # SII only, full: (1, 0, 0) → blown
    nb[0, 1] = (0.0065, sky, sky)       # SII only, partial: pure red, G=B=0
    nb[0, 2] = (0.0, 0.0, 0.0)          # no data
    v = dict(f.defaults(), star_s=0.0)
    for r in "SHO":
        v.update({f"sky{r}": sky, f"sig{r}": sig, f"top{r}": top, f"level{r}": 0.0, f"neb{r}": 1.0})
    v.update(levelH=-0.1)               # crush the Hα-only sky
    plain = f.bake([nb, np.zeros_like(nb)], v)
    shown = f.bake([nb, np.zeros_like(nb)], dict(v, show_clipping=1.0))
    assert np.allclose(shown[0, 0], (1, 0, 1))
    assert np.allclose(shown[0, 1], plain[0, 1]) and plain[0, 1][1] < 1e-6
    assert np.allclose(shown[0, 2], 0.0)
    assert np.allclose(shown[1, 1], (0, 0, 1))
    assert np.allclose(f.bake([nb, np.zeros_like(nb)], dict(v, show_clipping=0.0)), plain)
