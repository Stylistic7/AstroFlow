"""JavaScript code generation helpers for PixInsight processes."""
from typing import Any


class Enum(str):
    """A JavaScript enum value — rendered without quotes."""
    pass


def format_value(value: Any) -> str:
    """Format a Python value as a JavaScript literal."""
    if value is True:
        return "true"
    elif value is False:
        return "false"
    elif isinstance(value, Enum):
        return str(value)
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    return str(value)


def execute_on_views(
    process_name: str, parameters: dict[str, Any], view_ids: list[str]
) -> str:
    """Generate JS to create a process and execute it on given views."""
    lines = [f"var P = new {process_name};"]
    for k, v in parameters.items():
        lines.append(f"P.{k} = {format_value(v)};")
    lines.append("P.launch();")
    for view_id in view_ids:
        lines.append(f'P.executeOn(View.viewById("{view_id}"));')
    return "\n".join(lines)


def execute_global(process_name: str, parameters: dict[str, Any]) -> str:
    """Generate JS to create a process and execute it globally."""
    lines = [f"var P = new {process_name};"]
    for k, v in parameters.items():
        lines.append(f"P.{k} = {format_value(v)};")
    lines.append("P.executeGlobal();")
    return "\n".join(lines)
