"""The GraXpert pipeline's post-processing must not hold a PixInsight slot.

PixInsight is the cluster's scarce resource — 2 slots against THREE PI
operations per panel (`stack_files` + `post_process_stack` +
`post_stack_calibrate`). Measured on the 24-panel Cygnus run (2026-09-10):
`post_process_stack` sat at 30 ready / 0 running for hours while
`gpu_slots` stayed 0/5, because nothing downstream could become ready.

`post_stack_calibrate` held a slot only for PI's
SpectrophotometricFluxCalibration (and `WriteAstrometricSolution`, which
existed solely to hand PI a WCS it would accept). The in-house Gaia XP
engine reproduces SPFC to 0.3%/1.2% with no PI, so the slot is now freed
and this whole tail can run in parallel with WBPP.
"""
import inspect

from stacks.post_processing.graxpert_stack import calibrate_master, post_stack_calibrate
from stacks.post_processing.pixinsight import post_process_stack
from targets.merging.oss import calibrate_channel


def _declared_resources(op):
    from dependency_graph.dependency_graph import _declared_custom_resources
    return _declared_custom_resources(op)


def test_post_stack_calibrate_declares_no_pixinsight():
    assert "PixInsight" not in _declared_resources(post_stack_calibrate), (
        "post_stack_calibrate is holding a PixInsight slot again; the whole "
        "point of the in-house SPFC is that it does not need one"
    )


def _body_without_docstring(fn) -> str:
    """Source of a function's (or op's) body with the docstring removed — the
    docstring NAMES the PI calls it no longer makes, so a naive substring
    check would match its own explanation."""
    import ast, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(fn, "_function", fn))))
    fn = tree.body[0]
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           and isinstance(fn.body[0].value.value, str)) else fn.body
    return "\n".join(ast.unparse(n) for n in body)


def test_post_stack_calibrate_body_calls_no_pixinsight():
    """The resource declaration and the body must agree — declaring no slot
    while still launching PI would oversubscribe the real PI instances. The
    work lives in `calibrate_master`, which the target-channel op
    `calibrate_channel` shares."""
    for op in (post_stack_calibrate, calibrate_channel):
        assert "PixInsight" not in _declared_resources(op)
        assert "calibrate_master" in _body_without_docstring(op)
    src = _body_without_docstring(calibrate_master)
    for forbidden in ("from pix.", "pix.image", "SpectrophotometricFluxCalibration",
                      "WriteAstrometricSolution", "PixInsightError"):
        assert forbidden not in src, f"calibrate_master still reaches PixInsight: {forbidden}"
    assert "spfc_scale" in src, "it should compute the SPFC scale in-house"


def test_the_pi_pipeline_still_declares_its_slot():
    """The control: `post_process_stack` genuinely needs PI for
    MultiscaleGradientCorrection (MARS reference data), so it must KEEP its
    slot. A blanket removal would oversubscribe PixInsight."""
    assert _declared_resources(post_process_stack).get("PixInsight") == 1
