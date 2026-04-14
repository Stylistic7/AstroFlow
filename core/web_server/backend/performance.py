from flask import Blueprint, jsonify

from context import local_context as context

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
        # key shape: ("profile", fn_hash, fn_name, quality_or_None, node_ip)
        _, fn_hash, fn_name, quality, node_ip = key
        label = f"{fn_name} ({quality})" if quality else fn_name
        if label not in functions:
            functions[label] = []
        for sample in samples:
            functions[label].append({
                **sample, "fn_hash": fn_hash, "node_ip": node_ip, "quality": quality,
            })

    return jsonify({"functions": functions})
