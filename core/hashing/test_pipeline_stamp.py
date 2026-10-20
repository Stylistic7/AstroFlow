"""The pipeline stamp (`hashing.pipeline_stamp`): what "the same pipeline
made this" means across users. Pinned: the science code hash ignores what
the compiler ignores and moves on a semantic edit; non-science files and
tests never move it; the stamp folds the op and its primary input's stamp."""
from pathlib import Path

from hashing.pipeline_stamp import pipeline_stamp, science_code_hash_of, stamp_of
from hashing.structural_hash import structural_text_hash


def _tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


BASE = {
    "stacks/stacking/siril.py": "def integrate(x):\n    return x * 2\n",
    "pipelines/one.py": "from stacks.stacking.siril import integrate\n",
    "pix/script.js": "var a = 1;\n",
    "plate_solving.py": "def solve(f):\n    return None\n",
    "decisions/settings.py": "PEDESTAL = 0.005\n",
}


def test_comments_docstrings_and_formatting_do_not_move_the_science_hash(tmp_path):
    a = science_code_hash_of(_tree(tmp_path / "a", BASE))
    b = science_code_hash_of(_tree(tmp_path / "b", {
        **BASE,
        "stacks/stacking/siril.py": '"""Docs."""\n\n# a comment\ndef integrate(x):\n\n    return (x * 2)   # same\n',
    }))
    assert a == b


def test_a_semantic_edit_to_a_helper_moves_the_science_hash(tmp_path):
    a = science_code_hash_of(_tree(tmp_path / "a", BASE))
    b = science_code_hash_of(_tree(tmp_path / "b", {
        **BASE, "stacks/stacking/siril.py": "def integrate(x):\n    return x * 3\n",
    }))
    assert a != b


def test_a_science_default_moves_it_and_non_science_code_does_not(tmp_path):
    a = science_code_hash_of(_tree(tmp_path / "a", BASE))
    pedestal = science_code_hash_of(_tree(tmp_path / "b", {**BASE, "decisions/settings.py": "PEDESTAL = 0.01\n"}))
    assert pedestal != a
    ui = science_code_hash_of(_tree(tmp_path / "c", {
        **BASE,
        "web_server/frontend/src/App.jsx": "export default 1;\n",
        "finishes/stdlib.py": "def white():\n    return 1.0\n",
        "mosaics/merging/oss.py": "def merge():\n    pass\n",
        "stacks/test_stacking.py": "def test_x():\n    assert 1\n",
    }))
    assert ui == a


def test_the_stamp_folds_the_op_and_the_primary_input_stamp():
    raw = pipeline_stamp("integrate_with_siril", None, code="abc")
    wbpp = pipeline_stamp("stack_files", None, code="abc")
    assert raw != wbpp
    assert pipeline_stamp("post_stack_calibrate", raw, code="abc") != pipeline_stamp("post_stack_calibrate", wbpp, code="abc")
    assert pipeline_stamp("post_stack_calibrate", raw, code="abc") == pipeline_stamp("post_stack_calibrate", raw, code="abc")
    assert pipeline_stamp("post_stack_calibrate", raw, code="abc") != pipeline_stamp("post_stack_calibrate", raw, code="abd")


def test_stamp_of_reads_the_header_keyword_as_text():
    assert stamp_of({"PIPEHASH": 1234}) == "1234"
    assert stamp_of({"PIPEHASH": ""}) is None
    assert stamp_of({}) is None
    assert stamp_of(None) is None


def test_structural_text_hash_matches_a_live_function_source():
    from hashing.structural_hash import structural_source_hash

    def f(x):
        return x + 1

    assert structural_source_hash(f) == structural_text_hash("def f(x):\n    return x + 1\n")
