#!/bin/bash
# astrometry.net's solve-field for Windows, built under Cygwin (it has no native
# Windows build; All Sky Plate Solver ships it the same way). The result is
# what app/build_app.sh takes from Homebrew for the Mac app: solve-field and
# astrometry-engine, the two Python steps solve-field runs (removelines and
# uniformize, run on AstroFlow's own Python), and the licences — plus the
# Cygwin runtime those programs need: its DLL, a shell (solve-field runs its
# helpers through /bin/sh) and cygpath for the wrappers.
#
# Runs in the build VM (app/windows/build_in_vm.sh) under Cygwin with gcc-core,
# make, pkg-config, libbz2-devel, zlib-devel, dash and zip; takes the source
# tarballs from the inputs folder (checked against app/windows/inputs.sha256 on
# the Mac) and writes solve-field-windows-x64.zip into the current directory.
set -euo pipefail
inputs=$(cygpath -u "${1:?usage: build_solve_field.sh <inputs folder>}")
# Cygwin's tools only: a machine with MinGW's gcc, ld and gfortran on PATH
# had configure mix them in (Cygwin objects linked by MinGW's ld, through
# the gfortran Cygwin lacks: every Cygwin runtime symbol undefined).
export PATH=/usr/bin:/bin

AN_VERSION=0.97
CFITSIO_VERSION=4.7.0

dest=$PWD
work=$dest/_build
rm -rf "$work"
mkdir -p "$work"
out=$work/solve-field
mkdir -p "$out/bin" "$out/python/astrometry/util" "$out/licenses" "$out/tmp"

cd "$work"
tar -xzf "$inputs/cfitsio-$CFITSIO_VERSION.tar.gz"
tar -xzf "$inputs/astrometry.net-$AN_VERSION.tar.gz"

echo "→ cfitsio $CFITSIO_VERSION (without curl and its chain of DLLs)"
(cd "cfitsio-$CFITSIO_VERSION" \
    && ./configure --prefix="$work/cfitsio" --disable-curl --disable-shared --enable-reentrant >configure.log 2>&1 \
    && make -j"$(nproc)" >make.log 2>&1 && make install >>make.log 2>&1) \
    || { for log in "cfitsio-$CFITSIO_VERSION"/*.log; do echo "== $log"; tail -n 40 "$log"; done; exit 1; }

echo "→ astrometry.net $AN_VERSION: the solver (plot/ wants cairo, sdss/ Python; gsl is in-tree)"
cd "astrometry.net-$AN_VERSION"
# Named outright, with zlib after cfitsio: a static libcfitsio needs it, and
# pkg-config lists it only under Libs.private.
export CFITS_INC="-I$work/cfitsio/include" CFITS_LIB="-L$work/cfitsio/lib -lcfitsio -lz -lm"
# For any x86-64 PC, not the build machine's CPU (the default is -march=native).
# gsl-an first: under -j the top-level Makefile races util/ against it.
export ARCH_FLAGS="-march=x86-64 -mtune=generic"
make -j"$(nproc)" gsl-an >make.log 2>&1 && make -j"$(nproc)" solver >>make.log 2>&1 \
    || { grep -nE "error:|\*\*\*" make.log | head -20; tail -n 40 make.log; exit 1; }
cp solver/solve-field.exe solver/astrometry-engine.exe "$out/bin/"
cp __init__.py "$out/python/astrometry/"
cp util/*.py "$out/python/astrometry/util/"
cp LICENSE "$out/licenses/astrometry.net-LICENSE"
cp -r "../cfitsio-$CFITSIO_VERSION/licenses" "$out/licenses/cfitsio"

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
# Pure shell: the bundle has no coreutils. The shell execs POSIX paths only.
here=\${0%/*}
python=\$(cygpath -u "\$ASTROFLOW_PYTHON")
n=\$#
i=0
while [ \$i -lt \$n ]; do
    a=\$1
    shift
    case \$a in /*|[A-Za-z]:*) a=\$(cygpath -w "\$a");; esac
    set -- "\$@" "\$a"
    i=\$((i + 1))
done
# bin's Windows path, then up: in Cygwin bin is /usr/bin, whose .. is /usr.
PYTHONPATH=\$(cygpath -w "\$here")/../python exec "\$python" -s -m astrometry.util.$step "\$@"
EOF
    chmod +x "$out/bin/$step"
done
echo "$AN_VERSION" > "$out/VERSION"

(cd "$work" && zip -qr "$dest/solve-field-windows-x64.zip" solve-field)
ls -la "$dest/solve-field-windows-x64.zip"
unzip -l "$dest/solve-field-windows-x64.zip"
