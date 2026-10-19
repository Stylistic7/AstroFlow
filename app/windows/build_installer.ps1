# AstroFlow's Windows installer: the install layout, then setup.exe through
# Inno Setup — what app/build_app.sh does for the Mac app.
#
#   <stage>\AstroFlow\  the committed tree, stamped RELEASE, with the frontend built
#   <stage>\python\     a standalone x64 CPython 3.12 holding every package of uv.lock
#   <stage>\tools\      solve-field (the Cygwin build), libvips, GraXpert with its
#                       model, the astrometry.net index files, the Gaia XP catalogue,
#                       and bin\ with the wrappers the app finds on PATH
#
# Runs on a Windows machine with git, uv, node (the frontend is built first,
# see .github/workflows/windows-installer.yml), curl, gh (for the data release)
# and Inno Setup 6. Expects solve-field-windows-x64.zip in the repo root.
param(
    [string]$Stage = "$PWD\_stage",
    [string]$Downloads = "$PWD\_downloads",
    [string]$Out = "$PWD\_dist"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repo = $PWD.Path
$sha = (git rev-parse HEAD).Trim()
$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
$VipsVersion = "8.18.7"
$VipsSha256 = "" # filled by the first run's output below when unknown; verified when set
$PythonKey = "cpython-3.12-windows-x86_64-none"

function Fetch($url, $file) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
        & curl.exe -sSL --retry 3 --retry-delay 5 -o "$file.part" $url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
        Move-Item "$file.part" $file
    }
}

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force $Stage, "$Stage\tools\bin", $Out, $Downloads | Out-Null

Write-Host "==> AstroFlow $version ($($sha.Substring(0,8))): the tree"
git archive --format=zip -o "$Downloads\tree.zip" HEAD
Expand-Archive "$Downloads\tree.zip" -DestinationPath "$Stage\AstroFlow" -Force
$tree = "$Stage\AstroFlow"
foreach ($d in "core\testing", "core\testdata", "core\pix\testdata", "app\windows\test", ".github") {
    if (Test-Path "$tree\$d") { Remove-Item -Recurse -Force "$tree\$d" }
}
Get-ChildItem $tree -Recurse -Include "test_*.py", "conftest.py" | Remove-Item -Force
Remove-Item -Force "$tree\core\web_server\backend\static\images\GAIA_EDR3.png", "$tree\core\web_server\backend\static\images\GAIA_EDR3_darkened.jpg" -ErrorAction SilentlyContinue
Set-Content "$tree\RELEASE" $sha -NoNewline
Copy-Item -Recurse "$repo\core\web_server\frontend\build" "$tree\core\web_server\frontend\build"
Get-ChildItem "$tree\core\web_server\frontend\build" -Recurse -Filter "*.map" | Remove-Item -Force

Write-Host "==> Python: $PythonKey and the lock file's packages"
$env:UV_PYTHON_INSTALL_DIR = "$Downloads\pyroot"
uv python install $PythonKey
if ($LASTEXITCODE -ne 0) { throw "uv python install failed" }
$installed = Get-ChildItem "$Downloads\pyroot" -Directory | Where-Object Name -like "cpython-3.12*" | Select-Object -First 1
Copy-Item -Recurse $installed.FullName "$Stage\python"
$py = "$Stage\python\python.exe"
# Ours to fill: uv marks its interpreters externally managed, which uv pip install honours.
Remove-Item "$Stage\python\Lib\EXTERNALLY-MANAGED" -ErrorAction SilentlyContinue
uv export --frozen --no-dev --no-emit-project --no-hashes -o "$Downloads\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "uv export failed" }
uv pip install --python $py -r "$Downloads\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }
& $py -c "import ray, numpy, pyvips, astropy, waitress, pystray; print('python', ray.__version__, numpy.__version__)"

Write-Host "==> solve-field (astrometry.net under Cygwin)"
Expand-Archive "$repo\solve-field-windows-x64.zip" -DestinationPath "$Stage\tools" -Force

Write-Host "==> libvips $VipsVersion"
$vipsZip = "$Downloads\vips-dev-x64-all-$VipsVersion.zip"
Fetch "https://github.com/libvips/build-win64-mxe/releases/download/v$VipsVersion/vips-dev-x64-all-$VipsVersion.zip" $vipsZip
$hash = (Get-FileHash $vipsZip -Algorithm SHA256).Hash.ToLower()
Write-Host "    sha256 $hash"
if ($VipsSha256 -and $hash -ne $VipsSha256) { throw "libvips zip hash mismatch" }
Expand-Archive $vipsZip -DestinationPath "$Downloads\vips" -Force
Copy-Item -Recurse (Get-ChildItem "$Downloads\vips" -Directory | Select-Object -First 1).FullName "$Stage\tools\vips"

Write-Host "==> GraXpert: its own packages, its model, and the wrapper"
uv pip install --python $py --target "$Stage\tools\graxpert" -r "$repo\app\graxpert-requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "GraXpert install failed" }
gh release download data-v1 -p graxpert-model-1.0.1.zip -D $Downloads --skip-existing
Expand-Archive "$Downloads\graxpert-model-1.0.1.zip" -DestinationPath "$Stage\tools\graxpert-models" -Force
Set-Content "$Stage\tools\bin\graxpert.cmd" @'
@echo off
rem GraXpert on AstroFlow's Python, with only its own packages beside the stdlib.
set "here=%~dp0"
set "PYTHONPATH=%here%..\graxpert"
"%here%..\..\python\python.exe" -s -c "import sys; from graxpert.main import main; sys.argv[0] = 'graxpert'; sys.exit(main())" %*
'@ -NoNewline

Write-Host "==> astrometry.net index files"
$indexDir = "$Downloads\astrometry-index"
foreach ($name in Get-Content "$repo\app\windows\index-files.txt") {
    $series = if ($name -like "index-41*") { "4100" } else { "5000" }
    Fetch "https://data.astrometry.net/$series/$name" "$indexDir\$name"
}
New-Item -ItemType Directory -Force "$Stage\tools\astrometry-index" | Out-Null
Copy-Item "$indexDir\*.fits" "$Stage\tools\astrometry-index\"
Copy-Item "$repo\app\LICENSE-astrometry-index.txt" "$Stage\tools\astrometry-index\"

Write-Host "==> Gaia XP catalogue"
gh release download data-v1 -p gaia_xp.zip -D $Downloads --skip-existing
Expand-Archive "$Downloads\gaia_xp.zip" -DestinationPath "$Stage\tools\gaia_xp" -Force
Copy-Item "$repo\app\LICENSE-gaia.txt" "$Stage\tools\gaia_xp\"

Write-Host "==> bytecode, then setup.exe"
& $py -m compileall -q -j 0 "$Stage\python\Lib" "$Stage\tools\graxpert" "$tree\core" | Out-Null
Copy-Item "$repo\app\windows\icon.ico" "$Stage\icon.ico"
$size = [math]::Round(((Get-ChildItem $Stage -Recurse | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host "    install layout: $size GB"
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /Q "/DVersion=$version" "/DStage=$Stage" "/DOut=$Out" "/DSrc=$repo" "$repo\app\windows\AstroFlow.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-ChildItem $Out
