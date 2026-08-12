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


STOCK = [("RGB", 2), ("LRGB", 3), ("HaLRGB", 4), ("Hubble", 2), ("Foraxx", 2)]


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
            assert set(s) == {"name", "inputs", "params", "wgsl"}
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
        f = get_finish("RGB")
        combine, stars = _srcs(2)
        v = f.defaults()
        plain = f.bake([combine, stars], {**v, "star_s": 1.0})
        starless = f.bake([combine, stars], {**v, "star_s": 0.0})
        ref = _mtf_np(np.clip((combine - v["blackpoint"])
                              / (v["whitepoint"] - v["blackpoint"]), 0, 1),
                      v["midtones"])
        assert np.abs(plain - ref).max() < 1e-6
        ref0 = _mtf_np(np.clip(((combine - stars) - v["blackpoint"])
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
            "@finish(name='RGB', inputs=('Linear RGB', 'Linear RGB Stars'))\n"
            "def rgb(c, stars, mid=Param(0.3, 0.01, 0.95)):\n"
            "    return mtf(saturate(c), mid)\n"
        )
        (tmp_path / "AstroFlow" / "finishes" / "broken.py").write_text(
            "this is not python\n")
        monkeypatch.setattr(reg.Path, "home", lambda: tmp_path)
        try:
            refresh_finishes()
            f = get_finish("RGB", primary="Linear RGB")
            assert [p.name for p in f.params] == ["mid"], \
                "user file must OVERRIDE the stock RGB"
            assert get_finish("Hubble") is not None, \
                "a broken sibling file must not abort discovery"
        finally:
            monkeypatch.undo()
            refresh_finishes()
