from flask import Blueprint, jsonify

from context import context

performance_bp = Blueprint("performance", __name__)


@performance_bp.route("/api/performance")
def performance_summary():
    """Return all profile data, grouped by function name.

    Response shape:
    {
        "functions": {
            "stack_files": {
                "samples": [
                    {
                        "wall_time_s": 12.3,
                        "cpu_time_s": 11.1,
                        "peak_rss_delta_b": 1048576,
                        "image_totals": {"count": 5, "pixels": 48000000, "bytes": 96000000},
                        "node_ip": "10.0.0.1"
                    },
                    ...
                ]
            },
            ...
        }
    }
    """
    all_profiles = context.get_matching(("profile",))

    functions: dict[str, list] = {}
    for key, samples in all_profiles.items():
        _, fn_name, node_ip = key
        if fn_name not in functions:
            functions[fn_name] = []
        for sample in samples:
            functions[fn_name].append({**sample, "node_ip": node_ip})

    return jsonify({"functions": functions})
