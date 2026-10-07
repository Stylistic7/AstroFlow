"""The sidecar never traces and never joins Ray on its own: no module in
`mcp_server` imports a pipeline builder, the tracer, or `ray` directly.
(Confirmed writes reach Ray through `state.actor.state`, lazily.)"""
import ast
from pathlib import Path

FORBIDDEN = ("compute_graph", "pipelines", "dependency_graph.trace_paths", "ray")


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_no_module_imports_a_tracer_or_ray():
    here = Path(__file__).parent
    offenders = {
        f"{p.name}: {name}"
        for p in here.glob("*.py") if not p.name.startswith("test_")
        for name in _imports(p)
        if name == "ray" or name.startswith("ray.")
        or any(name == f or name.startswith(f + ".") or name.endswith("." + f)
               for f in FORBIDDEN if f != "ray")
    }
    assert not offenders, sorted(offenders)
