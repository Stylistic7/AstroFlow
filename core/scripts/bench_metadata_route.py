"""Time /api/metadata/... requests against a live web server.

Usage (with the live web server running on localhost:3000 or wherever):
    PYTHONPATH=. python -m scripts.bench_metadata_route \\
        http://localhost:3000 \\
        /api/metadata/mosaic/75a1da6463e118e6/PixInsight%20Downscaled/Merged%2013/H

Repeats the request N times, reports each. Useful for confirming the
trace @node cache is actually hitting on warm calls — first call should
match the cold trace cost (~17 s for the live mosaic), all subsequent
should be sub-second.
"""
import sys
import time
import urllib.request


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    base = sys.argv[1].rstrip("/")
    path = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    url = base + path
    print(f"GET {url}\n")

    for i in range(n):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                body = resp.read()
                status = resp.status
        except Exception as e:
            status = f"ERROR: {e}"
            body = b""
        dt = time.perf_counter() - t0
        size_kb = len(body) / 1024.0
        print(f"  request {i+1}/{n}: {dt*1000:9.1f} ms  status={status}  body={size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
