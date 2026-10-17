"""Makes Homebrew-built programs and libraries inside AstroFlow.app work on a
Mac without Homebrew, and records what was shipped.

    bundle_homebrew.py LIB_DIR LICENSES_DIR SOURCES_DIR FILE[=ORIGIN]...

Each FILE is a Mach-O already in the bundle; ORIGIN is the Homebrew file it
was copied from (its formula is then recorded, and its @loader_path loads
resolve against it). Every Homebrew library
it loads — and every library those load — is copied into LIB_DIR, and each
load is pointed at that copy with an @loader_path-relative path, so the bundle
can live anywhere. Every file touched is re-signed ad hoc (arm64 refuses to
load unsigned code); the build signs again with a Developer ID when it has one.

The licence files of each Homebrew formula shipped go to
LICENSES_DIR/<formula>-<version>/, with THIRD-PARTY.md listing every formula,
its licence and its source. Each formula's source archive goes to
SOURCES_DIR, verified by checksum and listed in LICENSES_DIR/SOURCES.txt,
for publishing beside the release: the
(L)GPL ones require it, and shipping them all is simpler than sorting them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

PREFIX = Path(subprocess.run(["brew", "--prefix"], capture_output=True, text=True, check=True).stdout.strip())
CELLAR = PREFIX / "Cellar"
LICENSE_NAMES = re.compile(r"^(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE)", re.I)
# Archives upstream replaced after Homebrew built from them, so the original
# can no longer be fetched, even by Homebrew: the current archive of the same
# release, pinned by its own checksum and noted in THIRD-PARTY.md.
REROLLED = {
    # ATNF re-rolled 8.10 (2026-10-17: `brew fetch -s wcslib` fails the same way).
    "wcslib-8.10.tar.bz2": "b394d3306da9a52410f056136911496e599b5c48f1cff2f221563697077bbd8d",
}


def loads(path: Path) -> list[str]:
    out = subprocess.run(["otool", "-L", str(path)], capture_output=True, text=True, check=True).stdout
    return [line.split()[0] for line in out.splitlines()[1:] if line.strip()]


def rpaths(path: Path) -> list[str]:
    out = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True, check=True).stdout
    return re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", out)


def resolve(load: str, loader: Path) -> Path | None:
    """The real file a load names, if it is Homebrew's; None for the system's."""
    here = Path(os.path.realpath(loader)).parent
    if load.startswith("@loader_path/"):
        p = here / load[len("@loader_path/"):]
    elif load.startswith("@rpath/"):
        candidates = [Path(r.replace("@loader_path", str(here))) / load[len("@rpath/"):] for r in rpaths(loader)]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            raise SystemExit(f"{loader}: {load} is in none of its rpaths")
    elif load.startswith("@"):
        raise SystemExit(f"{loader}: cannot resolve {load}")
    else:
        p = Path(load)
    real = Path(os.path.realpath(p))
    return real if str(real).startswith(str(PREFIX)) else None


def formula_of(path: Path) -> tuple[str, str]:
    rel = Path(os.path.realpath(path)).relative_to(CELLAR)
    return rel.parts[0], rel.parts[1]


def relocate(lib_dir: Path, files: dict[Path, Path]) -> set[tuple[str, str]]:
    lib_dir.mkdir(parents=True, exist_ok=True)
    formulas, todo, done = set(), list(files), set()
    origin = dict(files)      # bundle file -> the Homebrew file it was copied from
    while todo:
        f = todo.pop()
        if f in done:
            continue
        done.add(f)
        src = origin[f]
        if str(Path(os.path.realpath(src))).startswith(str(CELLAR)):
            formulas.add(formula_of(src))
        changes = []
        for load in loads(f)[1 if f.suffix == ".dylib" else 0:]:   # a dylib's first line is its own id
            real = resolve(load, src)
            if real is None:
                continue
            dest = lib_dir / real.name
            if not dest.exists():
                shutil.copy2(real, dest)
                dest.chmod(0o755)
                origin[dest] = real
                todo.append(dest)
            rel = os.path.relpath(lib_dir, f.parent)
            changes += ["-change", load, f"@loader_path/{rel}/{real.name}"]
        args = changes + (["-id", f"@loader_path/{f.name}"] if f.parent == lib_dir else [])
        if args:
            subprocess.run(["install_name_tool", *args, str(f)], check=True, capture_output=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(f)], check=True, capture_output=True)
    return formulas


def record(formulas: set[tuple[str, str]], licenses_dir: Path, sources_dir: Path) -> None:
    licenses_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    info = {f["name"]: f for f in json.loads(subprocess.run(
        ["brew", "info", "--json=v2", *sorted(n for n, _ in formulas)],
        capture_output=True, text=True, check=True).stdout)["formulae"]}
    rows, archives = [], []
    for name, version in sorted(formulas):
        # The source must be the one the shipped binary was built from, and
        # Homebrew only knows its formula's current source.
        current = info[name]["versions"]["stable"]
        if re.sub(r"_\d+$", "", version) != current:
            raise SystemExit(f"{name} is installed at {version} but Homebrew's formula is at {current}, "
                             f"so its source archive is not the shipped one: `brew upgrade {name}` and build again.")
        stable = info[name]["urls"]["stable"]
        url, sha = stable["url"], stable.get("checksum")
        archive = sources_dir / url.rsplit("/", 1)[-1].split("?")[0]
        if not archive.exists():
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(archive)
        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        note = ""
        if sha and got != sha:
            if REROLLED.get(archive.name) != got:
                raise SystemExit(f"{archive.name}: checksum {got} differs from Homebrew's {sha}")
            note = f" (upstream re-released this archive after Homebrew built from it: sha256 {got}, not {sha})"
        # The licence files at the top of the source tree, or in its top-level
        # licenses/ folder: Homebrew does not always install them (cfitsio's
        # keg has none).
        out = licenses_dir / f"{name}-{version}"
        out.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            found = [m for m in tar.getmembers() if m.isfile() and (
                (len(Path(m.name).parts) == 2 and LICENSE_NAMES.match(Path(m.name).name))
                or (len(Path(m.name).parts) == 3 and re.match(r"^licen[cs]es$", Path(m.name).parts[1], re.I)))]
            for m in found:
                (out / Path(m.name).name).write_bytes(tar.extractfile(m).read())
        if not found:
            raise SystemExit(f"{name} {version}: no licence file at the top of {archive.name}")
        rows.append(f"| {name} | {version} | {info[name].get('license') or '?'} | {url}{note} |")
        archives.append(archive.name)
    (licenses_dir / "THIRD-PARTY.md").write_text(
        "# Programs and libraries AstroFlow.app carries\n\n"
        "Built by Homebrew. Each one's licence files are in the folder named after it; "
        "the source archives are published beside each AstroFlow release.\n\n"
        "| Formula | Version | Licence | Source |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (licenses_dir / "SOURCES.txt").write_text("\n".join(archives) + "\n")


if __name__ == "__main__":
    lib, licenses, sources, *files = sys.argv[1:]
    pairs = {Path(f.split("=", 1)[0]): Path(f.split("=", 1)[-1]) for f in files}
    record(relocate(Path(lib), pairs), Path(licenses), Path(sources))
