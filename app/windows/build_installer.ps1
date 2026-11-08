# AstroFlow's Windows installer: the install layout, then setup.exe and its
# .bin slices through Inno Setup — what app/build_app.sh does for the Mac app.
#
#   <stage>\AstroFlow\  the committed tree, stamped RELEASE, with the frontend built
#   <stage>\python\     a standalone x64 CPython 3.12 holding every package of uv.lock
#   <stage>\tools\      solve-field (the Cygwin build), libvips, GraXpert with its
#                       model, the astrometry.net index files, the Gaia XP catalogue,
#                       and bin\ with the wrappers the app finds on PATH
#
# Runs in the build VM (app/windows/build_in_vm.sh, which says what it holds),
# from the unpacked tree: the frontend already built, solve-field-windows-x64.zip
# in the root. Downloads nothing but the lock file's packages from PyPI; every
# other input comes from the inputs folder (app/windows/inputs.sha256).
param(
    [Parameter(Mandatory)][string]$Tree,    # git archive of the commit
    [Parameter(Mandatory)][string]$Sha,
    [Parameter(Mandatory)][string]$Inputs,
    [string]$Stage = "$PWD\_stage",
    [string]$Downloads = "$PWD\_downloads",
    [string]$Out = "$PWD\_dist"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repo = $PWD.Path
$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
$VipsVersion = "8.18.7"
$PythonBuild = "cpython-3.12.13+20260408-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force $Stage, "$Stage\tools\bin", $Out, $Downloads | Out-Null

Write-Host "==> AstroFlow $version ($($Sha.Substring(0,8))): the tree"
Expand-Archive $Tree -DestinationPath "$Stage\AstroFlow" -Force
$tree = "$Stage\AstroFlow"
foreach ($d in "core\testing", "core\testdata", "core\pix\testdata", "app\windows\test") {
    if (Test-Path "$tree\$d") { Remove-Item -Recurse -Force "$tree\$d" }
}
Get-ChildItem $tree -Recurse -Include "test_*.py", "conftest.py" | Remove-Item -Force
Remove-Item -Force "$tree\core\web_server\backend\static\images\GAIA_EDR3.png", "$tree\core\web_server\backend\static\images\GAIA_EDR3_darkened.jpg" -ErrorAction SilentlyContinue
Set-Content "$tree\RELEASE" $Sha -NoNewline
Copy-Item -Recurse "$repo\core\web_server\frontend\build" "$tree\core\web_server\frontend\build"
Get-ChildItem "$tree\core\web_server\frontend\build" -Recurse -Filter "*.map" | Remove-Item -Force

Write-Host "==> Python: $PythonBuild and the lock file's packages"
tar -xzf "$Inputs\$PythonBuild" -C $Stage   # unpacks as python\
if ($LASTEXITCODE -ne 0) { throw "unpacking Python failed" }
$py = "$Stage\python\python.exe"
# Ours to fill: an EXTERNALLY-MANAGED marker would stop uv pip install.
Remove-Item "$Stage\python\Lib\EXTERNALLY-MANAGED" -ErrorAction SilentlyContinue
uv export --frozen --no-dev --no-emit-project --no-hashes -o "$Downloads\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "uv export failed" }
uv pip install --python $py -r "$Downloads\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

Write-Host "==> solve-field (astrometry.net under Cygwin)"
Expand-Archive "$repo\solve-field-windows-x64.zip" -DestinationPath "$Stage\tools" -Force

Write-Host "==> libvips $VipsVersion"
Expand-Archive "$Inputs\vips-dev-x64-all-$VipsVersion.zip" -DestinationPath "$Downloads\vips" -Force
Copy-Item -Recurse (Get-ChildItem "$Downloads\vips" -Directory | Select-Object -First 1).FullName "$Stage\tools\vips"
# The packages, pyvips with the libvips just staged among them.
$env:Path = "$Stage\tools\vips\bin;$env:Path"
& $py -c "import ray, numpy, pyvips, astropy, waitress, pystray; print('python', ray.__version__, numpy.__version__, 'libvips', pyvips.API_mode, pyvips.version(0), pyvips.version(1))"
if ($LASTEXITCODE -ne 0) { throw "the bundled Python cannot import its packages" }

Write-Host "==> GraXpert: its own packages, its model, and the wrapper"
uv pip install --python $py --target "$Stage\tools\graxpert" -r "$repo\app\graxpert-requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "GraXpert install failed" }
Expand-Archive "$Inputs\data\graxpert-model-1.0.1.zip" -DestinationPath "$Stage\tools\graxpert-models" -Force
Set-Content "$Stage\tools\bin\graxpert.cmd" @'
@echo off
rem GraXpert on AstroFlow's Python, with only its own packages beside the stdlib.
set "here=%~dp0"
set "PYTHONPATH=%here%..\graxpert"
"%here%..\..\python\python.exe" -s -c "import sys; from graxpert.main import main; sys.argv[0] = 'graxpert'; sys.exit(main())" %*
'@ -NoNewline

Write-Host "==> astrometry.net index files (our zips: data.astrometry.net has no 5200 series)"
$indexDir = "$Downloads\astrometry-index"
if (-not (Test-Path "$indexDir\index-5206-47.fits")) {
    New-Item -ItemType Directory -Force $indexDir | Out-Null
    Get-ChildItem "$Inputs\data\index-*.zip" | ForEach-Object { Expand-Archive $_.FullName -DestinationPath $indexDir -Force }
}
$expected = Get-Content "$repo\app\windows\index-files.txt"
foreach ($name in $expected) {
    $bytes = [IO.File]::ReadAllBytes("$indexDir\$name")[0..5]
    if ([Text.Encoding]::ASCII.GetString($bytes) -ne "SIMPLE") { throw "$name is not a FITS file" }
}
Write-Host "    $($expected.Count) index files, all FITS"
New-Item -ItemType Directory -Force "$Stage\tools\astrometry-index" | Out-Null
Copy-Item "$indexDir\*.fits" "$Stage\tools\astrometry-index\"
Copy-Item "$repo\app\LICENSE-astrometry-index.txt" "$Stage\tools\astrometry-index\"

Write-Host "==> Gaia XP catalogue"
Expand-Archive "$Inputs\data\gaia_xp.zip" -DestinationPath "$Stage\tools\gaia_xp" -Force
Copy-Item "$repo\app\LICENSE-gaia.txt" "$Stage\tools\gaia_xp\"

Write-Host "==> bytecode, then setup.exe"
& $py -m compileall -q -j 0 "$Stage\python\Lib" "$Stage\tools\graxpert" "$tree\core" | Out-Null
Copy-Item "$repo\app\windows\icon.ico" "$Stage\icon.ico"
$size = [math]::Round(((Get-ChildItem $Stage -Recurse | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host "    install layout: $size GB"
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /Q "/DVersion=$version" "/DStage=$Stage" "/DOut=$Out" "/DSrc=$repo" "$repo\app\windows\AstroFlow.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-ChildItem $Out
