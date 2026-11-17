"""The programs AstroFlow runs, and a trial run of each.

Every caller finds its program through `tool_path`: the location the user set
on the Third-Party Tools page (`node_settings`), else where the program
normally installs. `check_cluster` runs every trial on every node in the
cluster, so the page shows at a glance which tool on which machine doesn't
work yet, and why: the trial's own error message.

GraXpert, solve-field and libvips ship inside AstroFlow.app
(`app/build_app.sh`), which puts its own tools first on PATH; they are still
checked, because they can still fail (2026-10-16: a worker's onnxruntime built
for the wrong Python broke every background extraction it ran). PixInsight,
Siril and RC Astro are the user's to install.
"""
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import node_settings

_TIMEOUT_S = 120
# The oldest rc-astro the pipelines accept: 2.6.10 changed how it reads TIFF
# and FITS, and up to 2.6.10 it kept a copy of each compiled model per call
# (120 GB a day under a re-key, 2026-11-13); 2.6.11 fixed both.
RC_ASTRO_MIN = (2, 6, 11)


class ToolNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Tool:
    key: str
    name: str
    used_for: str
    install: str
    # Where it normally is: a fixed path, or a command found on PATH.
    default: Callable[[], str | None]
    # Runs it; returns what it reported (a version) or raises with why not.
    trial: Callable[[str], str]
    # A library, not a program: nothing to locate.
    locatable: bool = True


def _on_path(command: str) -> Callable[[], str | None]:
    return lambda: shutil.which(command)


def _graxpert() -> str | None:
    """On PATH (the app puts its own there), else where `make install-graxpert`
    puts it: the uv tool bin folder, which a Ray worker started outside a login
    shell doesn't have on its PATH."""
    # `which` with the folder: on Windows the tool is graxpert.exe.
    return shutil.which("graxpert") or shutil.which("graxpert", path=str(Path.home() / ".local" / "bin"))


def _at(path: str) -> Callable[[], str | None]:
    return lambda: path if Path(path).exists() else None


def _installed(mac: str, windows: str) -> Callable[[], str | None]:
    """Where the tool's installer puts it on this platform."""
    return _at(windows if sys.platform == "win32" else mac)


def _first(*finders: Callable[[], str | None]) -> Callable[[], str | None]:
    return lambda: next((p for p in (f() for f in finders) if p), None)


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT_S, **kwargs)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"exited {result.returncode}:\n{tail}")
    return result


def _trial_solve_field(path: str) -> str:
    version = _run([path, "--version"]).stdout.strip()
    counts = {f: sum(1 for _ in Path(f).glob("index-*.fits")) if Path(f).is_dir() else 0
              for f in node_settings.index_folders(node_settings.read())}
    if not counts:
        raise RuntimeError(f"{version}, but no index folder on this node: set one under solve-field")
    empty = [f for f, n in counts.items() if not n]
    if empty:
        raise RuntimeError(f"{version}, but no index files in {', '.join(empty)} on this node")
    return f"{version}; " + ", ".join(f"{n} index files in {f}" for f, n in counts.items())


def _trial_vips(_path: str) -> str:
    import pyvips
    total = pyvips.Image.black(64, 64).add(1).avg()
    if total != 1:
        raise RuntimeError(f"a test image averaged {total}, not 1")
    return f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}"


def _trial_graxpert(path: str) -> str:
    """A background extraction on a small synthetic frame, as the pipeline
    runs it: the AI model loads, and an output appears."""
    import numpy as np
    from astropy.io import fits

    from stacks.post_processing.graxpert_stack import GRAXPERT_AI_VERSION
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "trial.fits"
        rng = np.random.default_rng(0)
        fits.writeto(frame, (0.1 + 0.01 * rng.random((512, 512))).astype(np.float32))
        result = _run([path, str(frame), "-cli", "-cmd", "background-extraction",
                       "-ai_version", GRAXPERT_AI_VERSION, "-output", str(Path(tmp) / "out")], cwd=tmp)
        log = result.stdout + result.stderr
        if "Critical error" in log or not (Path(tmp) / "out.fits").exists():
            raise RuntimeError("\n".join(log.strip().splitlines()[-15:]))
    version = re.search(r"version: (\S+)", log)
    return f"GraXpert {version.group(1) if version else '(unknown version)'}, AI model {GRAXPERT_AI_VERSION}"


def _trial_siril(path: str) -> str:
    return _run([path, "--version"]).stdout.strip().splitlines()[-1]


def _trial_rc_astro(path: str) -> str:
    """Its help page carries the version and the licence of each product:
    the version must be `RC_ASTRO_MIN` or newer, and every product the
    pipelines use must be licensed."""
    text = _run([path, "--no-banner", "--help"]).stdout
    found = re.search(r"Version (\d+)\.(\d+)\.(\d+)", text)
    if not found:
        raise RuntimeError("rc-astro reports no version; AstroFlow needs " + ".".join(map(str, RC_ASTRO_MIN)))
    version = ".".join(found.groups())
    if tuple(map(int, found.groups())) < RC_ASTRO_MIN:
        raise RuntimeError(f"rc-astro {version} is too old: AstroFlow needs {'.'.join(map(str, RC_ASTRO_MIN))} "
                           "or newer (older versions read images differently and fill the disk with a copy of "
                           "each compiled model per call); update it from https://www.rc-astro.com")
    status = {m.group(1): m.group(2).strip() for m in re.finditer(r"\|\s*(bxt|sxt|nxt)\s*\|[^|]*\|([^|]*)\|", text)}
    unlicensed = [f"{p}: {status.get(p, 'not listed')}" for p in ("bxt", "sxt", "nxt")
                  if not re.search(r"(?<!un)licensed", status.get(p, "").lower())
                  or re.search(r"not licensed|expired", status.get(p, "").lower())]
    if unlicensed:
        raise RuntimeError(f"rc-astro {version}, but not licensed on this node:\n" + "\n".join(unlicensed))
    return f"rc-astro {version}; BXT, SXT and NXT licensed"


def _trial_pixinsight(path: str) -> str:
    """Located, not launched: starting PixInsight on every check would open
    its window on every node."""
    app = Path(path) / "PixInsight.app"
    if not (app / "Contents" / "MacOS" / "PixInsight").exists():
        raise RuntimeError(f"no PixInsight.app in {path}")
    with open(app / "Contents" / "Info.plist", "rb") as f:
        version = f"PixInsight {plistlib.load(f).get('CFBundleShortVersionString', '(unknown version)')}"
    from pix.tasks.multiscale_gradient_correction import registered_mars_databases
    databases = registered_mars_databases()
    if not databases:
        raise RuntimeError(f"{version}, but no MARS database registered in it, which its gradient correction "
                           "needs: open MultiscaleGradientCorrection in PixInsight and add one under MARS databases")
    return f"{version}; MARS: {', '.join(databases)}"


TOOLS = [
    Tool("solve-field", "astrometry.net", "plate solving: placing every frame on the sky",
         "included in AstroFlow", _on_path("solve-field"), _trial_solve_field),
    Tool("vips", "libvips", "reading and writing large images",
         "included in AstroFlow", lambda: None, _trial_vips, locatable=False),
    Tool("graxpert", "GraXpert", "background extraction (ships with AstroFlow)",
         "included in AstroFlow", _graxpert, _trial_graxpert),
    Tool("siril", "Siril", "stacking",
         "the app from https://siril.org",
         _installed("/Applications/Siril.app/Contents/MacOS/siril-cli",
                    r"C:\Program Files\Siril\bin\siril-cli.exe"), _trial_siril),
    Tool("rc-astro", "RC Astro", "BlurXTerminator, StarXTerminator, NoiseXTerminator",
         "the rc-astro installer from https://www.rc-astro.com",
         # The app's PATH is its own tools and the system's, not /usr/local/bin
         # where the installer links it: its install location too.
         _first(_on_path("rc-astro"), _at("/Applications/RC-Astro/CLI/rc-astro")), _trial_rc_astro),
    Tool("pixinsight", "PixInsight", "the PixInsight pipelines (a folder, not a program)",
         "https://pixinsight.com",
         _installed("/Applications/PixInsight", r"C:\Program Files\PixInsight"), _trial_pixinsight),
]
_BY_KEY = {t.key: t for t in TOOLS}


def tool_path(key: str) -> str:
    """Where `key` is on this node: the location set on the Third-Party Tools
    page, else its usual place. Raises when neither has it."""
    tool = _BY_KEY[key]
    path = node_settings.read().get("tool_paths", {}).get(key) or tool.default()
    if not path:
        raise ToolNotFoundError(f"{tool.name} not found on this node. Install it ({tool.install}), "
                                "or set where it is on the Third-Party Tools page.")
    return path


def pixinsight(*parts: str) -> str:
    """A path inside this node's PixInsight installation."""
    return str(Path(tool_path("pixinsight"), *parts))


def check(key: str) -> dict:
    """{"path", "ok", "detail"}: this node's trial of one tool."""
    tool = _BY_KEY[key]
    path = None
    try:
        path = tool_path(key) if tool.locatable else None
        return {"path": path, "ok": True, "detail": tool.trial(path)}
    except Exception as e:
        return {"path": path, "ok": False, "detail": f"{type(e).__name__}: {e}"}


def check_this_mac() -> dict:
    import socket
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(TOOLS)) as pool:
        results = dict(zip(_BY_KEY, pool.map(check, _BY_KEY)))
    return {"host": socket.gethostname().removesuffix(".local"), "tools": results}


def check_cluster() -> list[dict]:
    """Every tool's trial on every node in the cluster, the head first."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from ray_infra.ray_init import connect_to_ray_cluster
    connect_to_ray_cluster()  # a web worker connects on first use
    remote = ray.remote(num_cpus=0)(check_this_mac)
    head = ray.get_runtime_context().get_node_id()
    nodes = sorted((n for n in ray.nodes() if n.get("Alive")), key=lambda n: n["NodeID"] != head)
    refs = [remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote()
            for n in nodes]
    out = []
    for n, ref in zip(nodes, refs):
        try:
            out.append({"ip": n["NodeManagerAddress"], **ray.get(ref, timeout=_TIMEOUT_S + 30)})
        except Exception as e:
            out.append({"ip": n["NodeManagerAddress"], "host": None, "error": f"{type(e).__name__}: {e}"})
    return out
