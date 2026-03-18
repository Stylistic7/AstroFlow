import json
from typing import Optional

from pix.classes import ShellScriptTask, Image
from pix.context import append_task

_javascript = """\
function copyFITSKeywords(source_view_id, dest_view_id, keys) {{
    var source_window = View.viewById(source_view_id).window;
    var dest_window = View.viewById(dest_view_id).window;

    var to_copy = (keys === null)
        ? source_window.keywords
        : source_window.keywords.filter(function(k) {{ return keys.indexOf(k.name) !== -1; }});

    var dest_keywords = dest_window.keywords;
    for (var i = 0; i < to_copy.length; i++) {{
        var kw = to_copy[i];
        dest_keywords = dest_keywords.filter(function(k) {{ return k.name !== kw.name; }});
        dest_keywords.push(new FITSKeyword(kw.name, kw.value, kw.comment));
    }}
    dest_window.keywords = dest_keywords;
    console.writeln("Copied FITS keywords from " + source_view_id + " to " + dest_view_id);
}}

copyFITSKeywords("{source_view_id}", "{dest_view_id}", {keys});
"""


def CopyFITSKeywords(
    source: Image, dest: Image, keys: Optional[list[str]] = None
) -> Image:
    """Copy FITS keywords from source view to dest view.

    If keys is None, all keywords are copied. Otherwise only the named keys.
    """
    append_task(
        ShellScriptTask(
            javascript=_javascript.format(
                source_view_id=source.view_id,
                dest_view_id=dest.view_id,
                keys=json.dumps(keys),
            )
        )
    )
    return dest
