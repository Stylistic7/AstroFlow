"""Worker-side pull-deploy: mount the head's shared disks over 10 GbE and
follow the head's committed code.

Everything here is worker-initiated. The head only *advertises* (via zeroconf)
its 10 GbE IP, its share names, and its `cluster_version`; nothing in this
module runs on the head, and the head never reaches into a worker.

Mounts go to `/Volumes/<share>` — the SAME absolute path the head uses — so the
content-addressed cache and on-disk file paths agree cluster-wide. `/Volumes`
is root-owned, so we can't `mkdir` + `mount_smbfs` there as a normal user;
instead we use macOS's `mount volume` (the automounter, the same path Finder
takes), which creates the mountpoint and mounts as the user. The share is
addressed by EXPLICIT IP so the directly-connected 10 GbE route is used rather
than WiFi.
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from base_path import checkout_dir
from state.code_version import (
    is_deployable_version,
    read_synced_sha,
    verify_and_extract,
)

HTTP_PORT = 8000
_MOUNT_CONFIRM_S = 8


def _mountpoint(share: str) -> Path:
    return Path("/Volumes") / share


def _is_mounted(mp: Path) -> bool:
    return os.path.ismount(mp)


def mount_head_shares(head_ip: str, shares: list[str]) -> None:
    """Mount each `//head_ip/<share>` at `/Volumes/<share>` with NO explicit
    user in the URL — macOS then authenticates the current login session (same
    Apple ID across the LAN → passwordless). An explicit `user@` (or `guest@`)
    instead forces a per-user credential lookup that prompts / silently fails
    (the mount never appears). Verified: `mount volume "smb://<ip>/Astro"`
    mounts, `smb://nobody@<ip>/Astro` does not. Idempotent; raises if a mount
    can't be established."""
    for share in shares:
        mp = _mountpoint(share)
        if _is_mounted(mp):
            continue
        url = f"smb://{head_ip}/{share}"
        result = subprocess.run(
            ["osascript", "-e", f'mount volume "{url}"'],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mount volume {url} failed: {result.stderr.strip()}"
            )
        # `mount volume` blocks until mounted, but confirm defensively (SMB can
        # report success before the mount is visible).
        for _ in range(_MOUNT_CONFIRM_S):
            if _is_mounted(mp):
                break
            time.sleep(1)
        if not _is_mounted(mp):
            raise RuntimeError(f"{url} did not appear at {mp}")


def unmount_head_shares(shares: list[str]) -> None:
    """Best-effort force-unmount — used on quit and before remounting at a new
    head IP (a stale SMB mount otherwise blocks os.listdir for ~60 s)."""
    for share in shares:
        mp = _mountpoint(share)
        if _is_mounted(mp):
            subprocess.run(
                ["diskutil", "unmount", "force", str(mp)],
                capture_output=True,
            )


def current_synced_sha() -> str | None:
    """The SHA the worker's checkout currently holds, or None."""
    return read_synced_sha(checkout_dir())


def head_current_version(head_ip: str) -> str:
    """The SHA the head is running RIGHT NOW, read from its AUTHORITATIVE HTTP
    endpoint. Zeroconf only tells us the head's IP — a killed prior head can leave
    a GHOST zeroconf record advertising a stale SHA, so we never follow that TXT;
    we ask the live head directly. Empty/'dev'/'wt:...' when the head isn't running
    committed code."""
    url = f"http://{head_ip}:{HTTP_PORT}/api/cluster-version"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read()).get("cluster_version", "")


def pull_head_snapshot(head_ip: str) -> str:
    """Fetch the head's CURRENT frozen snapshot — WITHOUT naming a SHA — and
    extract it. The worker doesn't need to know the SHA in advance: the head
    stamps the SHA it served in the `X-AstroFlow-SHA` header, and we verify the tar
    against THAT and record it. So the (code, SHA) pair is always self-consistent,
    even if the head redeploys mid-fetch. Returns the SHA now on disk."""
    url = f"http://{head_ip}:{HTTP_PORT}/api/code-snapshot"
    with urllib.request.urlopen(url, timeout=300) as r:
        tar = r.read()
        sha = r.headers.get("X-AstroFlow-SHA", "")
    verify_and_extract(tar, sha, checkout_dir())
    return sha


def sync_to_head(head_ip: str) -> str | None:
    """Ensure the checkout holds the head's CURRENT committed code. Reads the
    version from the head's HTTP endpoint (not the ghost-prone zeroconf TXT) and
    pulls the current snapshot if we're behind. Returns the SHA now on disk, or
    None when the head's version is non-deployable (`dev`/`dirty`/`""` — nothing to
    follow)."""
    version = head_current_version(head_ip)
    if not is_deployable_version(version):
        return None
    if current_synced_sha() == version:
        return version
    return pull_head_snapshot(head_ip)
