"""Equivalence contract for the structural source hash.

Two function sources must hash EQUAL iff they differ only in what the
normalization deliberately ignores (comments, whitespace, docstrings, quote
style, constant formatting, decorator kwargs) and DIFFERENT for any semantic
body change. Each pair below is compiled from source text via exec so the
tests exercise exactly what `ast.parse` sees.
"""
import textwrap

import pytest

from hashing.structural_hash import structural_source_hash


def _fn_from(src: str, name: str = "f"):
    ns: dict = {}
    src = textwrap.dedent(src)
    exec(compile(src, f"<structural-test-{name}>", "exec"), ns)
    fn = ns[name]
    # inspect.getsource can't read exec'd functions; feed the source through
    # a stand-in with __wrapped__-style access by attaching it.
    fn.__test_source__ = src
    return fn


def _hash_src(src: str, name: str = "f") -> str:
    """Hash source text directly through the same normalize+serialize path
    (bypasses inspect.getsource, which can't see exec'd code)."""
    import ast
    import xxhash
    from hashing import structural_hash as sh
    tree = ast.parse(textwrap.dedent(src))
    tokens: list = []
    sh._serialize(tree, tokens)
    return xxhash.xxh64("".join(tokens)).hexdigest()


BASE = """
def f(x, y=3):
    total = x + y
    return total * 2
"""


@pytest.mark.parametrize("variant", [
    # comments
    """
    def f(x, y=3):
        # a helpful comment
        total = x + y  # inline too
        return total * 2
    """,
    # blank lines + trailing whitespace
    """
    def f(x, y=3):

        total = x + y

        return total * 2
    """,
    # docstring added
    '''
    def f(x, y=3):
        """Docs are free."""
        total = x + y
        return total * 2
    ''',
    # parenthesization that the compiler discards
    """
    def f(x, y=3):
        total = (x + y)
        return (total * 2)
    """,
])
def test_cosmetic_variants_hash_equal(variant):
    assert _hash_src(variant) == _hash_src(BASE)


def test_quote_style_and_constant_format_collapse():
    a = """
    def f():
        return {'k': "v", 'n': 1e3}
    """
    b = """
    def f():
        return {"k": 'v', "n": 1000.0}
    """
    assert _hash_src(a) == _hash_src(b)


def test_decorator_kwargs_are_hash_free():
    a = """
    @node(display="Mosaic", num_cpus=2)
    def f(x):
        return x
    """
    b = """
    @node(display="Mosaic", num_cpus=4, display_name="Friendly Name")
    def f(x):
        return x
    """
    assert _hash_src(a) == _hash_src(b)


@pytest.mark.parametrize("changed", [
    # constant value
    """
    def f(x, y=3):
        total = x + y
        return total * 3
    """,
    # int -> float (semantic: division/serialization differ)
    """
    def f(x, y=3.0):
        total = x + y
        return total * 2
    """,
    # operator
    """
    def f(x, y=3):
        total = x - y
        return total * 2
    """,
    # different callee name
    """
    def f(x, y=3):
        total = g(x) + y
        return total * 2
    """,
    # local rename (deliberately NOT normalized — fail-safe)
    """
    def f(x, y=3):
        result = x + y
        return result * 2
    """,
])
def test_semantic_changes_hash_different(changed):
    assert _hash_src(changed) != _hash_src(BASE)


def test_docstring_only_body_stays_valid():
    a = '''
    def f():
        """only docs"""
    '''
    b = """
    def f():
        pass
    """
    assert _hash_src(a) == _hash_src(b)


def test_deterministic_across_calls():
    assert _hash_src(BASE) == _hash_src(BASE)


def test_real_function_via_public_api():
    """The public entry point works on a genuine module-level function and
    is stable call-to-call (id-keyed memo aside)."""
    from stacks.calibration_selection import calibrate
    h1 = structural_source_hash(calibrate)
    h2 = structural_source_hash(calibrate)
    assert h1 == h2 and len(h1) == 16


def test_fallback_on_unparseable_source(capsys):
    """Builtins have no retrievable source: raw fallback, loudly."""
    h = structural_source_hash(len)
    assert len(h) == 16
    assert "FALLBACK" in capsys.readouterr().out


def test_fallback_input_is_address_free():
    """The source-less fallback must never hash repr(func) — default reprs
    embed ' at 0x…' addresses, which flap per process. Caught live by the
    display_name smoke test: one function, two hashes, one interpreter."""
    from hashing.structural_hash import _raw_source_or_repr

    class NoSource:
        pass

    for probe in (len, NoSource()):
        s = _raw_source_or_repr(probe)
        assert " at 0x" not in s, s
