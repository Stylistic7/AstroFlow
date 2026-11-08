"""Smoke-checks an installed AstroFlow over its API: the release it runs, the
bundled tools, and a pass over its own library and cart.

    smoke_check.py BASE_URL SHA

Standard library only (the Mac's /usr/bin/python3 runs it). Exits non-zero
naming the first thing that is wrong. Used by app/release.sh.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

BUNDLED = ("solve-field", "vips", "graxpert")
DONE = {"SUCCEEDED", "SKIPPED", "RECLAIMED"}


def get(url, method="GET", timeout=60):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait(what, probe, seconds, every=10):
    end = time.time() + seconds
    while time.time() < end:
        try:
            got = probe()
            if got:
                return got
        except (OSError, ValueError):
            pass
        time.sleep(every)
    sys.exit(f"FAIL: {what} within {seconds // 60} min")


def main(base, sha):
    wait(f"the app answering as {sha[:8]}",
         lambda: get(f"{base}/api/cluster-version")["cluster_version"] == sha, 900)
    print(f"ok  running {sha[:8]}")

    tools = wait("the tools check", lambda: get(f"{base}/api/tools/check", "POST", 600).get("nodes"), 900)
    for node in tools:
        for name in BUNDLED:
            t = node["tools"][name]
            if not t["ok"]:
                sys.exit(f"FAIL: {name} on {node['host']}: {t['detail']}")
            print(f"ok  {name}: {t['detail']}")

    def idle():   # idle, and still idle a minute later: a pass can start late
        if get(f"{base}/api/compute_graph/status")["_totals"]["remaining_number"]:
            return False
        time.sleep(60)
        return get(f"{base}/api/compute_graph/status")["_totals"]["remaining_number"] == 0
    wait("the graph going idle", idle, 3600, every=20)
    cart = get(f"{base}/api/shopping_cart")
    if not cart:
        print("ok  graph idle; the cart is empty, so no pipeline ran")
    for kind, ident, pipeline in cart:
        states = get(f"{base}/api/workflow_graph/{kind}/{ident}/{urllib.parse.quote(pipeline)}",
                     timeout=300).get("state") or {}
        counts = {}
        for v in states.values():
            s = v.get("state") if isinstance(v, dict) else v
            counts[s] = counts.get(s, 0) + 1
        if not states or set(counts) - DONE:
            sys.exit(f"FAIL: {kind} {ident} {pipeline}: {counts}")
        print(f"ok  {kind} {ident[:8]} {pipeline}: {counts}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
