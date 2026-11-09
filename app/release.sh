#!/bin/bash
# An AstroFlow release: both installers built from HEAD, each smoke-tested in a
# clean VM, then published into installers/ (its README.md is written by hand).
# Builds in a throwaway worktree, never the live checkout (both builds rebuild
# the frontend in place).
#
#   app/release.sh
#
# Needs: the UTM VMs "Windows Build" (builds, see app/windows/build_in_vm.sh)
# and "Windows" (tests, with a user logged in on its desktop: the tray app
# needs one, so keep it suspended between releases rather than shut down); the
# tart VM astroflow-clean; ~/.ssh/astroflow_vm in all three. Both test VMs keep
# a small library and a cart, so each smoke check runs a real pass: six RC71
# e308 Pane 17 H frames in ~/Pictures/AstroFlow/Raw, and target 4c593d16 on
# "OSS Stack + OSS Mosaic" in the cart (no Siril, which users install). Also the Windows
# inputs in ~/Projects/AstroFlow-build-inputs; Homebrew's astrometry-net and vips at
# app/homebrew.lock's versions. About 2 h, 40 GB of disk and much of the RAM:
# run it while the graph is quiet.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
[ -z "$(git status --porcelain)" ] || { echo "commit first: a release is HEAD"; exit 1; }
sha=$(git rev-parse HEAD)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
free=$(df -k "$HOME" | tail -1 | awk '{print int($4 / 1048576)}')
[ "$free" -ge 40 ] || { echo "only $free GB free; a release needs about 40"; exit 1; }
utm=/Applications/UTM.app/Contents/MacOS/utmctl
tart=~/Applications/tart.app/Contents/MacOS/tart
key=(-i "$HOME/.ssh/astroflow_vm" -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
tmp=$(cd "$(mktemp -d)" && pwd -P)   # resolved: build_app.sh compares real paths
wt=$tmp/tree
cleanup() {
    "$tart" stop astroflow-clean >/dev/null 2>&1 || true
    git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true
    rm -rf "$tmp"
}
trap cleanup EXIT
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach -q "$wt" HEAD
# The test frame from this clone's LFS store, never fetched.
oid=$(git show HEAD:app/windows/test/frame_bin4.fit | sed -n 's/^oid sha256://p')
frame=$(git rev-parse --path-format=absolute --git-common-dir)/lfs/objects/${oid:0:2}/${oid:2:2}/$oid
check=$wt/app/smoke_check.py

vm_up() {   # a UTM VM's address, once it answers ssh
    "$utm" start "$1" >/dev/null 2>&1 || true
    for _ in $(seq 1 90); do
        ip=$("$utm" ip-address "$1" 2>/dev/null | grep -E '^192\.168\.64\.' | head -1 || true)
        if [ -n "$ip" ] && ssh "${key[@]}" -o ConnectTimeout=5 "$ip" hostname >/dev/null 2>&1; then
            echo "$ip"; return
        fi
        sleep 5
    done
    echo "$1 never answered ssh" >&2; return 1
}

desktop() { ssh "${key[@]}" "$1" '(Get-Process explorer -ErrorAction SilentlyContinue | Select-Object -First 1).SessionId' | grep -q .; }
echo "==> AstroFlow $version (${sha:0:8}): preflight"
tip=$(vm_up Windows)
if ! desktop "$tip"; then
    echo "    waiting for someone to log in on the \"Windows\" VM's desktop in UTM (the tray app needs one)"
    until desktop "$tip"; do sleep 60; done
fi
"$utm" suspend Windows

echo "==> Windows build"
bip=$(vm_up "Windows Build")
(cd "$wt" && app/windows/build_in_vm.sh "$bip")
ssh "${key[@]}" "$bip" 'shutdown /s /t 0' || true
win=$wt/installers/AstroFlow-Setup-$version-windows

echo "==> macOS build"
(cd "$wt" && nice -n 15 bash app/build_app.sh)
dmg=$wt/app/dist/AstroFlow-$version.dmg
if ! cmp -s "$wt/app/homebrew.lock" "$repo/app/homebrew.lock"; then
    cp "$wt/app/homebrew.lock" "$repo/app/homebrew.lock"
    echo "app/homebrew.lock changed (HOMEBREW_LOCK_UPDATE=1): commit it"
fi

echo "==> Windows test (the \"Windows\" VM)"
tip=$(vm_up Windows)
th=$tip
desktop "$tip" || { echo "FAIL: the Windows VM's desktop login is gone: log in in UTM and run again"; exit 1; }
cat > "$tmp/install.ps1" <<'EOF'
Get-Process pythonw, python, raylet, gcs_server -ErrorAction SilentlyContinue | Stop-Process -Force
$exe = Get-ChildItem C:\AstroFlow-release\AstroFlow-Setup-*.exe | Select-Object -First 1
$p = Start-Process -Wait -PassThru $exe.FullName -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=C:\AstroFlow-release\install.log"
Set-Content C:\AstroFlow-release\install.exit $p.ExitCode
EOF
cat > "$tmp/launch.ps1" <<'EOF'
# As the Start menu shortcut does: the tray launcher on the bundled Python.
Start-Process -FilePath "C:\Program Files\AstroFlow\python\pythonw.exe" `
    -ArgumentList "`"C:\Program Files\AstroFlow\AstroFlow\app\windows\AstroFlow.pyw`"" `
    -WorkingDirectory "C:\Program Files\AstroFlow\AstroFlow\core"
EOF
ssh "${key[@]}" "$th" 'if (Test-Path C:\AstroFlow-release) { Remove-Item -Recurse -Force C:\AstroFlow-release }; New-Item -ItemType Directory C:\AstroFlow-release | Out-Null'
scp -q "${key[@]}" "$win"/AstroFlow-Setup-* "$tmp/install.ps1" "$tmp/launch.ps1" "$th:C:/AstroFlow-release/"
scp -q "${key[@]}" "$frame" "$th:C:/AstroFlow-release/frame.fits"
# Detached and elevated (an ssh session's children die with it); the launch in
# the logged-in desktop session (/it), where the tray icon lives.
ssh "${key[@]}" "$th" 'schtasks /create /tn AstroFlowInstall /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\AstroFlow-release\install.ps1" /sc once /st 00:00 /rl highest /ru SYSTEM /f | Out-Null; schtasks /run /tn AstroFlowInstall | Out-Null'
until ssh "${key[@]}" "$th" 'Test-Path C:\AstroFlow-release\install.exit' | grep -q True; do sleep 20; done
got=$(ssh "${key[@]}" "$th" '(Get-Content C:\AstroFlow-release\install.exit) + " " + (Get-Content "C:\Program Files\AstroFlow\AstroFlow\RELEASE")' | tr -d '\r')
[ "$got" = "0 $sha" ] || { echo "FAIL: Windows install: exit and RELEASE are '$got'"; exit 1; }
echo "ok  installed $sha"
ssh "${key[@]}" "$th" 'schtasks /create /tn AstroFlowLaunch /tr "powershell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File C:\AstroFlow-release\launch.ps1" /sc once /st 00:00 /it /f | Out-Null; schtasks /run /tn AstroFlowLaunch | Out-Null'
/usr/bin/python3 "$check" "http://$tip:8000" "$sha"
ssh "${key[@]}" "$th" '$app = "C:\Program Files\AstroFlow"; $t = "C:\AstroFlow-release"
New-Item -ItemType Directory -Force "$t\tmp" | Out-Null
"inparallel`ncpulimit 300`nadd_path C:/Program Files/AstroFlow/tools/astrometry-index`nautoindex`n" | Set-Content "$t\an.cfg" -NoNewline
$env:Path = "$app\tools\solve-field\bin;$app\python;C:\Windows\System32;C:\Windows"
$env:ASTROFLOW_PYTHON = "$app\python\python.exe"; $env:CYGWIN = "nodosfilewarning"
solve-field.exe --config C:/AstroFlow-release/an.cfg --fits-image --overwrite --no-plots --new-fits none --tweak-order 2 --crpix-center --cpulimit 300 --objs 1000 --scale-units degwidth --scale-low 4 --scale-high 8 --temp-dir C:/AstroFlow-release/tmp C:/AstroFlow-release/frame.fits | Select-String "solved with"
if (-not (Test-Path "$t\frame.solved")) { throw "the installed solver did not solve the test frame" }'
echo "ok  the installed solver solves the test frame"
"$utm" suspend Windows    # keeps the desktop login for the next release

echo "==> macOS test (tart astroflow-clean)"
share=$tmp/share
mkdir "$share"
cp -c "$dmg" "$frame" "$share/"
"$tart" run --no-graphics --dir="share:$share" astroflow-clean >/dev/null 2>&1 &
mip=$("$tart" ip astroflow-clean --wait 300)
mh=admin@$mip
until ssh "${key[@]}" -o ConnectTimeout=5 "$mh" true 2>/dev/null; do sleep 5; done
# An upgrade in place, as a user's would be.
ssh "${key[@]}" "$mh" "set -e; pkill -f '[A]stroFlow.app' || true; sleep 3; rm -rf /Applications/AstroFlow.app
hdiutil attach -nobrowse -quiet '/Volumes/My Shared Files/share/AstroFlow-$version.dmg'
ditto /Volumes/AstroFlow/AstroFlow.app /Applications/AstroFlow.app; hdiutil detach -quiet /Volumes/AstroFlow
open -a /Applications/AstroFlow.app"
/usr/bin/python3 "$check" "http://$mip:8000" "$sha"
ssh "${key[@]}" "$mh" "set -e; R=/Applications/AstroFlow.app/Contents/Resources; t=\$(mktemp -d); mkdir \$t/tmp
cp '/Volumes/My Shared Files/share/$oid' \$t/frame.fits
printf 'inparallel\ncpulimit 300\nadd_path %s\nautoindex\n' \$R/astrometry-index > \$t/an.cfg
PATH=\$R/tools/bin:\$R/python/bin:/usr/bin:/bin solve-field --config \$t/an.cfg --fits-image --overwrite --no-plots --new-fits none --tweak-order 2 --crpix-center --cpulimit 300 --objs 1000 --scale-units degwidth --scale-low 4 --scale-high 8 --temp-dir \$t/tmp \$t/frame.fits | grep 'solved with'
test -f \$t/frame.solved"
echo "ok  the bundled solver solves the test frame"

echo "==> publish"
out=$repo/installers
mkdir -p "$out"
(cd "$win" && zip -q -0 "$out/.windows.zip.part" AstroFlow-Setup-*)
mv -f "$out/.windows.zip.part" "$out/AstroFlow-Setup-$version-windows.zip"
mv -f "$dmg" "$out/AstroFlow-$version.dmg"
rsync -a --delete "$wt/app/dist/sources/" "$out/AstroFlow-$version-sources/"
ls -la "$out"
echo "Released $sha: both installers built and smoke-tested. Update installers/README.md."
