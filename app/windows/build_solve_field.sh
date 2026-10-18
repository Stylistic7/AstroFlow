#!/bin/bash
# astrometry.net's solve-field for Windows, built under Cygwin (it has no native
# Windows build; All Sky Plate Solver ships it the same way). The result is
# what app/build_app.sh takes from Homebrew for the Mac app: solve-field and
# astrometry-engine, the two Python steps solve-field runs (removelines and
# uniformize, run on AstroFlow's own Python), and the licences — plus the
# Cygwin runtime those programs need: its DLL, a shell (solve-field runs its
# helpers through /bin/sh) and cygpath for the wrappers.
#
# Runs on a Windows machine with Cygwin's gcc-core, make, pkg-config,
# zlib-devel, dash, zip and curl (.github/workflows/solve-field-windows.yml);
# writes solve-field-windows-x64.zip into the current directory.
set -euo pipefail
# Cygwin's toolchain first: a GitHub runner also has MinGW's gcc, ld and
# gfortran on PATH, and configure mixed them in (Cygwin objects, MinGW ld:
# every Cygwin runtime symbol undefined).
export PATH=/usr/bin:/bin:$PATH

AN_VERSION=0.97
AN_SHA256=e4eef1b658ba5ad462282b661c0ca3a5c538ba1716e853f7970b7b9fa4a33459
CFITSIO_VERSION=4.7.0
CFITSIO_SHA256=ce573bbea8e75b429f8c3d3e86498741ba3dc9628a1530d2f65268397ad059e8

dest=$PWD
# Under the workspace, so a failed run can upload the logs (the workflow does).
work=$dest/_build
rm -rf "$work"
mkdir -p "$work"
out=$work/solve-field
mkdir -p "$out/bin" "$out/python/astrometry/util" "$out/licenses" "$out/tmp"

fetch() {  # url file sha256
    curl -sSL -o "$2" "$1"
    echo "$3  $2" | sha256sum -c - >/dev/null
}
cd "$work"
fetch "https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-$CFITSIO_VERSION.tar.gz" cfitsio.tar.gz "$CFITSIO_SHA256"
fetch "https://github.com/dstndstn/astrometry.net/releases/download/$AN_VERSION/astrometry.net-$AN_VERSION.tar.gz" an.tar.gz "$AN_SHA256"
tar -xzf cfitsio.tar.gz
tar -xzf an.tar.gz

echo "→ cfitsio $CFITSIO_VERSION (without curl and its chain of DLLs)"
(cd "cfitsio-$CFITSIO_VERSION" \
    && ./configure --prefix="$work/cfitsio" --disable-curl --disable-shared --enable-reentrant >configure.log 2>&1 \
    && make -j"$(nproc)" >make.log 2>&1 && make install >>make.log 2>&1) \
    || { for log in "cfitsio-$CFITSIO_VERSION"/*.log; do echo "== $log"; tail -n 40 "$log"; done; exit 1; }

echo "→ astrometry.net $AN_VERSION: the solver (plot/ wants cairo, sdss/ Python; gsl is in-tree)"
cd "astrometry.net-$AN_VERSION"
export PKG_CONFIG_PATH=$work/cfitsio/lib/pkgconfig
make -j"$(nproc)" solver >make.log 2>&1 || { tail -n 60 make.log; exit 1; }
cp solver/solve-field.exe solver/astrometry-engine.exe "$out/bin/"
cp __init__.py "$out/python/astrometry/"
cp util/*.py "$out/python/astrometry/util/"
cp LICENSE "$out/licenses/astrometry.net-LICENSE"
cp "../cfitsio-$CFITSIO_VERSION/License.txt" "$out/licenses/cfitsio-License.txt"

echo "→ the Cygwin runtime"
cp /bin/dash.exe "$out/bin/sh.exe"
cp /bin/cygpath.exe "$out/bin/"
for exe in "$out"/bin/*.exe; do
    ldd "$exe" | awk '$3 ~ /^\/usr\/bin\/cyg.*\.dll$/ {print $3}'
done | sort -u | while read -r dll; do cp "$dll" "$out/bin/"; done
licence=$(find /usr/share/doc -iname 'CYGWIN_LICENSE*' | head -1)
cp "$licence" "$out/licenses/CYGWIN_LICENSE"
uname -r > "$out/licenses/cygwin-version.txt"

# removelines and uniformize, which solve-field runs by name from its own
# folder through /bin/sh: astrometry.net's Python steps on AstroFlow's Python
# (ASTROFLOW_PYTHON, set by plate_solving.py), given Windows paths.
for step in removelines uniformize; do
    cat > "$out/bin/$step" <<EOF
#!/bin/sh
# astrometry.net's $step step, which solve-field runs, on AstroFlow's Python.
here=\$(dirname "\$0")
n=\$#
i=0
while [ \$i -lt \$n ]; do
    a=\$1
    shift
    case \$a in /*|[A-Za-z]:*) a=\$(cygpath -w "\$a");; esac
    set -- "\$@" "\$a"
    i=\$((i + 1))
done
PYTHONPATH=\$(cygpath -w "\$here/../python") exec "\$ASTROFLOW_PYTHON" -s -m astrometry.util.$step "\$@"
EOF
    chmod +x "$out/bin/$step"
done
echo "$AN_VERSION" > "$out/VERSION"

(cd "$work" && zip -qr "$dest/solve-field-windows-x64.zip" solve-field)
ls -la "$dest/solve-field-windows-x64.zip"
unzip -l "$dest/solve-field-windows-x64.zip"
