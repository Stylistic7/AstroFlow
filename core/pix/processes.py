from typing import Any

from .context import get_next_process_number


class enum(str):
    pass


def _format(value: Any) -> str:
    if value is True:
        return "true"
    elif value is False:
        return "false"
    elif type(value) is enum:
        return str(value)
    elif type(value) is str:
        return f'"{value}"'
    elif type(value) is list:
        return "[" + ", ".join([_format(v) for v in value]) + "]"

    return str(value)


def execute_on_view_ids(
    process_name: str, parameters: dict[str, Any], view_ids: list[str]
) -> str:
    process_number = get_next_process_number()
    code = [f"var P{process_number} = new {process_name};"]
    for k, v in parameters.items():
        code.append(f"P{process_number}.{k} = {_format(v)};")

    code.append(f"P{process_number}.launch();")

    for view_id in view_ids:
        code.append(f'P{process_number}.executeOn(View.viewById("{view_id}"));')

    return "\n".join(code)


def execute_global(process_name: str, parameters: dict[str, Any]) -> str:
    process_number = get_next_process_number()
    code = [f"var P{process_number} = new {process_name};"]
    for k, v in parameters.items():
        code.append(f"P{process_number}.{k} = {_format(v)};")

    code.append(f"P{process_number}.executeGlobal();")

    return "\n".join(code)
