#!/bin/bash
# AstroFlow's Windows installer, built in a local Windows VM from this tree's
# HEAD — what app/build_app.sh is for the Mac app. Nothing comes from GitHub:
# the inputs folder holds every file the build takes (checked here against
# inputs.sha256 before any is sent); only the lock file's packages come from PyPI.
# Writes installers/AstroFlow-Setup-<version>-windows/ (setup.exe and its .bin).
#
#   app/windows/build_in_vm.sh [user@vm]        (default 192.168.64.6)
#   INPUTS=<folder>                             (default ~/Projects/AstroFlow-build-inputs)
#
# The VM (UTM "Windows Build": Windows 11 on ARM, x64 programs emulated), once:
# OpenSSH Server with ~/.ssh/astroflow_vm.pub in administrators_authorized_keys;
# uv; Cygwin x86_64 in C:\cygwin with gcc-core, make, pkg-config, zlib-devel,
# libbz2-devel, dash, zip and unzip; Inno Setup 6 (the inputs' innosetup-*.exe
# /VERYSILENT /ALLUSERS). It keeps C:\build\inputs and the unpacked index files
# in C:\build\downloads between builds.
set -euo pipefail
host=${1:-192.168.64.6}
inputs=${INPUTS:-$HOME/Projects/AstroFlow-build-inputs}
repo=$(git rev-parse --show-toplevel)
cd "$repo"
[ -z "$(git status --porcelain)" ] || { echo "commit first: the installer is built from HEAD"; exit 1; }
sha=$(git rev-parse HEAD)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
key=(-i "$HOME/.ssh/astroflow_vm" -o BatchMode=yes -o ServerAliveInterval=30 -o LogLevel=ERROR)
vm() { ssh "${key[@]}" "$host" "\$ErrorActionPreference='Stop'; \$ProgressPreference='SilentlyContinue'; $1"; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "==> AstroFlow $version (${sha:0:8}): inputs"
(cd "$inputs" && shasum -a 256 -c --quiet "$repo/app/windows/inputs.sha256")
# Send what the VM lacks or holds at another size.
have=$(vm 'New-Item -ItemType Directory -Force C:\build\inputs\data | Out-Null; Get-ChildItem -Recurse -File C:\build\inputs | % { "$($_.Length) $($_.FullName.Substring(16).Replace([char]92, [char]47))" }' | tr -d '\r')
while read -r _ name; do
    size=$(stat -f %z "$inputs/$name")
    grep -qxF "$size $name" <<<"$have" || { echo "    sending $name"; scp -q "${key[@]}" "$inputs/$name" "$host:C:/build/inputs/$name"; }
done < app/windows/inputs.sha256

echo "==> the tree and the frontend"
git archive --format=zip -o "$tmp/tree.zip" HEAD
(cd core/web_server/frontend && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent)
(cd core/web_server/frontend/build && zip -qr "$tmp/frontend.zip" .)
vm 'foreach ($d in "C:\build\src", "C:\build\dist") { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }'
scp -q "${key[@]}" "$tmp/tree.zip" "$tmp/frontend.zip" "$host:C:/build/"
vm 'Expand-Archive C:\build\tree.zip C:\build\src; Expand-Archive C:\build\frontend.zip C:\build\src\core\web_server\frontend\build'

echo "==> solve-field (Cygwin)"
vm 'C:\cygwin\bin\bash.exe -lc "cd /cygdrive/c/build/src && bash -o igncr app/windows/build_solve_field.sh C:/build/inputs"; if ($LASTEXITCODE) { throw "solve-field build failed" }'

echo "==> the install layout and setup.exe"
vm "cd C:\\build\\src; powershell -NoProfile -ExecutionPolicy Bypass -File app\\windows\\build_installer.ps1 -Tree C:\\build\\tree.zip -Sha $sha -Inputs C:\\build\\inputs -Stage C:\\build\\stage -Downloads C:\\build\\downloads -Out C:\\build\\dist; if (\$LASTEXITCODE) { throw 'installer build failed' }"

echo "==> the proof: the install layout solves a real frame"
scp -q "${key[@]}" "$(git rev-parse --show-toplevel)/app/windows/test/frame_bin4.fit" "$host:C:/build/frame.fits"
vm '$t = "C:\build\solve-test"; $st = "C:\build\stage"
if (Test-Path $t) { Remove-Item -Recurse -Force $t }; New-Item -ItemType Directory -Force "$t\tmp" | Out-Null
Move-Item C:\build\frame.fits "$t\frame.fits"
"inparallel`ncpulimit 300`nadd_path C:/build/stage/tools/astrometry-index`nautoindex`n" | Set-Content "$t\an.cfg" -NoNewline
# The app'"'"'s PATH shape: the bundle'"'"'s bin first, then its Python, then the system; no other Cygwin.
$env:Path = "$st\tools\solve-field\bin;$st\python;C:\Windows\System32;C:\Windows"
$env:ASTROFLOW_PYTHON = "$st\python\python.exe"; $env:CYGWIN = "nodosfilewarning"
solve-field.exe --config C:/build/solve-test/an.cfg --fits-image --overwrite --no-plots --new-fits none --tweak-order 2 --crpix-center --cpulimit 300 --objs 1000 --scale-units degwidth --scale-low 4 --scale-high 8 --temp-dir C:/build/solve-test/tmp C:/build/solve-test/frame.fits | Select-String "solved with|Field center: \(RA,Dec\)"
if (-not (Test-Path "$t\frame.solved")) { throw "the frame did not solve" }'

out=installers/AstroFlow-Setup-$version-windows
rm -rf "$out" && mkdir -p "$out"
scp -q "${key[@]}" "$host:C:/build/dist/*" "$out/"
echo "$sha" > "$out/RELEASE"
ls -la "$out"
