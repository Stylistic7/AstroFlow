#!/bin/bash
# Builds AstroFlow.app and a disk image to hand to other people:
#
#   app/build_app.sh            → app/dist/AstroFlow.app, app/dist/AstroFlow-<version>.dmg
#
# The app carries its own Python (uv's relocatable CPython) with every package
# from uv.lock installed inside it, so it never touches the user's Python,
# Homebrew or PATH. Not PyInstaller: AstroFlow is a cluster of Python
# processes (Ray workers, gunicorn, the MCP sidecar) that start as
# `sys.executable -m …`, which needs a real interpreter, not a frozen binary.
#
# It also carries every program and catalogue AstroFlow needs that isn't
# someone else's licensed product, so a Mac needs nothing else installed:
#
#   * GraXpert: its own packages in Resources/tools, pinned by
#     graxpert-requirements.txt.
#   * solve-field and libvips (under pyvips), copied from this Mac's Homebrew
#     with every library they load, into Resources/tools/bin and
#     Resources/lib (bundle_homebrew.py). Their licences go in
#     Resources/licenses; their source archives beside the disk image, for
#     publishing with the release.
#   * the astrometry.net index files for fields from about 20' to the whole
#     sky (Resources/astrometry-index, from ASTROMETRY_INDEX): 4107-4119 and
#     5203-5206, 4.6 GB. An ASI585MM at 2000 mm, a 19' x 11' field, solves on
#     these alone, blind, in sparse fields (tested on the M101 and M51 subs,
#     2026-10-17). Smaller fields add their own index files on the
#     Third-Party Tools page.
#   * the Gaia XP catalogue AstroFlow calibrates with (Resources/gaia_xp, from
#     GAIA_CATALOG, built by core/photometry/gaia_catalog.py).
#
# What the user installs, if they want them: PixInsight, Siril and RC Astro
# (the dashboard's Third-Party Tools page tries each tool on every Mac).
#
# Building needs Homebrew's astrometry-net and vips installed, at their
# formulas' current versions (so the published sources are the shipped ones).
# Upgrading what the app carries is Homebrew's job: `brew upgrade`, then build
# again; the build names any formula still behind. app/homebrew.lock records
# the formulas and versions the last release carried: a build that would carry
# others stops and names them; HOMEBREW_LOCK_UPDATE=1 accepts them (commit the lock).
#
# The build is the committed tree (it is stamped with the commit, RELEASE, as
# its cluster version); ALLOW_DIRTY=1 builds the working tree for a test,
# stamped `<sha>-dirty`, which no worker will join.
#
# Signing: ad-hoc by default (first launch: System Settings → Privacy &
# Security → Open Anyway). With SIGN_IDENTITY="Developer ID Application: …"
# the app is signed with the hardened runtime, and with NOTARY_PROFILE (a
# `xcrun notarytool store-credentials` profile) the disk image is notarized.
set -euo pipefail

APP_SRC="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$APP_SRC")"
DIST="$APP_SRC/dist"
WORK="$APP_SRC/build"
PYTHON_VERSION=3.12
PYOBJC="pyobjc-framework-Cocoa==11.1"
GAIA_CATALOG="${GAIA_CATALOG:-/Volumes/AstroProcessed/Catalogs/GaiaXP/gaia_xp}"
ASTROMETRY_INDEX="${ASTROMETRY_INDEX:-/Volumes/AstroProcessed/astrometry-net}"
SOURCES="$APP_SRC/.sources"   # source archives, kept between builds

for column in source_id ra dec G colour flux; do
	if [ ! -f "$GAIA_CATALOG/$column.npy" ]; then
		echo "No Gaia XP catalogue at $GAIA_CATALOG ($column.npy is missing); set GAIA_CATALOG." >&2
		exit 1
	fi
done
index_files=()
for i in 4107 4108 4109 4110 4111 4112 4113 4114 4115 4116 4117 4118 4119; do
	index_files+=("index-$i.fits")
done
for i in 5203 5204 5205 5206; do
	for hp in $(seq -w 0 47); do
		index_files+=("index-$i-$hp.fits")
	done
done
for f in "${index_files[@]}"; do
	if [ ! -f "$ASTROMETRY_INDEX/$f" ]; then
		echo "No $f in $ASTROMETRY_INDEX; set ASTROMETRY_INDEX." >&2
		exit 1
	fi
done
for formula in astrometry-net vips; do
	if ! brew list --versions "$formula" >/dev/null 2>&1; then
		echo "The build copies Homebrew's $formula into the app: brew install $formula" >&2
		exit 1
	fi
done

sha="$(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
	if [ "${ALLOW_DIRTY:-}" != 1 ]; then
		echo "The tree has uncommitted changes. Commit them, or ALLOW_DIRTY=1 for a test build." >&2
		exit 1
	fi
	sha="$sha-dirty"
fi
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
echo "→ AstroFlow $version (${sha:0:12})"

APP="$WORK/AstroFlow.app"
RES="$APP/Contents/Resources"
rm -rf "$WORK" && mkdir -p "$APP/Contents/MacOS" "$RES/AstroFlow"

echo "→ frontend"
(cd "$REPO/core/web_server/frontend" && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent)

echo "→ source"
# Tracked and new files, as `git status` sees them; no tests or their data
# and labs (core/testing), and of the frontend only its build.
(cd "$REPO" && git ls-files -co --exclude-standard -- core app/AstroFlow.py pyproject.toml uv.lock) \
	| grep -vE '^core/(testdata|pix/testdata|testing|web_server/frontend)/' \
	| grep -vE '(^|/)(test_[^/]*|conftest)\.py$' > "$WORK/files.txt"
rsync -a --files-from="$WORK/files.txt" "$REPO/" "$RES/AstroFlow/"
rsync -a "$REPO/core/web_server/frontend/build/" "$RES/AstroFlow/core/web_server/frontend/build/"
echo "$sha" > "$RES/AstroFlow/RELEASE"
cp "$APP_SRC/icon.icns" "$RES/icon.icns"

echo "→ Python $PYTHON_VERSION"
# uv's standalone CPython, which is relocatable. Never `uv python find`: it
# returns the project's .venv, built on Homebrew's Python, whose copy still
# points at Homebrew — and the packages below then land in Homebrew's
# site-packages (2026-10-16).
export UV_PYTHON_INSTALL_DIR="$APP_SRC/.python"
uv python install --quiet --managed-python "$PYTHON_VERSION"
standalone="$(ls -d "$UV_PYTHON_INSTALL_DIR"/cpython-$PYTHON_VERSION.*-macos-aarch64-none | sort -V | tail -1)"
rsync -a "$standalone/" "$RES/python/"
py="$RES/python/bin/python$PYTHON_VERSION"
prefix="$("$py" -c 'import sys; print(sys.prefix)')"
if [ "$prefix" != "$RES/python" ]; then
	echo "The bundled Python's prefix is $prefix, not the bundle; refusing to install into it." >&2
	exit 1
fi

echo "→ packages"
(cd "$REPO" && uv export --frozen --no-dev --no-hashes --no-emit-project --quiet) > "$WORK/requirements.txt"
uv pip install --quiet --python "$py" --break-system-packages -r "$WORK/requirements.txt" "$PYOBJC"

echo "→ GraXpert"
# Its own folder, not the app's site-packages: GraXpert pins its own numpy.
uv pip install --quiet --python "$py" --target "$RES/tools/graxpert" -r "$APP_SRC/graxpert-requirements.txt"
mkdir -p "$RES/tools/bin"
cat > "$RES/tools/bin/graxpert" <<'EOF'
#!/bin/sh
# GraXpert on the app's Python, with only its own packages beside the stdlib.
res="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHONPATH="$res/tools/graxpert" exec "$res/python/bin/python3" -s \
	-c 'import sys; from graxpert.main import main; sys.argv[0] = "graxpert"; sys.exit(main())' "$@"
EOF
chmod +x "$RES/tools/bin/graxpert"

echo "→ solve-field and libvips"
# solve-field and the engine it runs. Given a FITS file (`--fits-image`,
# plate_solving.py) it runs two Python steps besides, removelines and
# uniformize, which run here on the app's Python from astrometry.net's own
# package (astropy stands in for its fitsio).
an="$(brew --prefix astrometry-net)"
relocate=()
for f in solve-field astrometry-engine; do
	cp "$an/bin/$f" "$RES/tools/bin/"
	relocate+=("$RES/tools/bin/$f=$(readlink -f "$an/bin/$f")")
done
mkdir -p "$RES/tools/astrometry"
rsync -a --exclude '*.so' --exclude __pycache__ "$an"/libexec/lib/python3.*/site-packages/astrometry "$RES/tools/astrometry/"
for step in removelines uniformize; do
	cat > "$RES/tools/bin/$step" <<EOF
#!/bin/sh
# astrometry.net's $step step, which solve-field runs, on the app's Python.
res="\$(cd "\$(dirname "\$0")/../.." && pwd)"
PYTHONPATH="\$res/tools/astrometry" exec "\$res/python/bin/python3" -s -m astrometry.util.$step "\$@"
EOF
	chmod +x "$RES/tools/bin/$step"
done
# pyvips compiled its module against Homebrew's libvips, by absolute path.
vips_module="$(ls "$RES/python/lib/python$PYTHON_VERSION/site-packages/"_libvips*.so 2>/dev/null || true)"
if [ -z "$vips_module" ]; then
	echo "pyvips installed without its compiled module, so it would look for libvips at run time; brew install vips and build again." >&2
	exit 1
fi
relocate+=("$vips_module")
"$py" "$APP_SRC/bundle_homebrew.py" "$RES/lib" "$RES/licenses" "$SOURCES" "${relocate[@]}"
# libvips loads its format plugins from its build prefix; point it at an
# empty one so it never loads Homebrew's (which link Homebrew's libvips).
mkdir -p "$RES/tools/vips"

echo "→ astrometry.net index files"
mkdir -p "$RES/astrometry-index"
(cd "$ASTROMETRY_INDEX" && rsync -a "${index_files[@]}" "$RES/astrometry-index/")
cp "$APP_SRC/LICENSE-astrometry-index.txt" "$RES/astrometry-index/"

echo "→ Gaia XP catalogue"
rsync -a --exclude '.*' "$GAIA_CATALOG/" "$RES/gaia_xp/"
cp "$APP_SRC/LICENSE-gaia.txt" "$RES/gaia_xp/"

# Bytecode now, so nothing writes into the signed bundle at run time.
"$py" -m compileall -q -j 0 "$RES/python/lib" "$RES/tools/graxpert" "$RES/tools/astrometry" "$RES/AstroFlow" >/dev/null || true

echo "→ bundle"
cat > "$APP/Contents/MacOS/AstroFlow" <<'EOF'
#!/bin/sh
res="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec "$res/python/bin/python3" "$res/AstroFlow/app/AstroFlow.py" "$@"
EOF
chmod +x "$APP/Contents/MacOS/AstroFlow"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key><string>AstroFlow</string>
	<key>CFBundleDisplayName</key><string>AstroFlow</string>
	<key>CFBundleIdentifier</key><string>org.astroflow.AstroFlow</string>
	<key>CFBundleExecutable</key><string>AstroFlow</string>
	<key>CFBundleIconFile</key><string>icon.icns</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleShortVersionString</key><string>$version</string>
	<key>CFBundleVersion</key><string>${sha:0:12}</string>
	<key>LSMinimumSystemVersion</key><string>13.0</string>
	<key>LSArchitecturePriority</key><array><string>arm64</string></array>
	<key>NSHighResolutionCapable</key><true/>
	<key>NSHumanReadableCopyright</key><string>AstroFlow ${sha:0:12}</string>
	<key>NSNetworkVolumesUsageDescription</key><string>AstroFlow reads your raw frames from, and writes its results to, the folders you choose.</string>
	<key>NSRemovableVolumesUsageDescription</key><string>AstroFlow reads your raw frames from, and writes its results to, the folders you choose.</string>
	<key>NSDocumentsFolderUsageDescription</key><string>AstroFlow reads your raw frames from, and writes its results to, the folders you choose.</string>
	<key>NSDesktopFolderUsageDescription</key><string>AstroFlow reads your raw frames from, and writes its results to, the folders you choose.</string>
	<key>NSDownloadsFolderUsageDescription</key><string>AstroFlow reads your raw frames from, and writes its results to, the folders you choose.</string>
	<key>NSLocalNetworkUsageDescription</key><string>AstroFlow finds the other Macs in your AstroFlow cluster and your ASIAir on the local network.</string>
	<key>NSBonjourServices</key><array><string>_astroflow._tcp</string></array>
</dict>
</plist>
EOF

echo "→ signing"
if [ -n "${SIGN_IDENTITY:-}" ]; then
	# Every Mach-O inside first (Python, extension modules, Ray's binaries),
	# then the app around them.
	find "$RES" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0 \
		| xargs -0 file | grep 'Mach-O' | cut -d: -f1 | tr '\n' '\0' \
		| xargs -0 codesign --force --timestamp --options runtime --sign "$SIGN_IDENTITY" \
			--entitlements "$APP_SRC/entitlements.plist"
	codesign --force --timestamp --options runtime --sign "$SIGN_IDENTITY" \
		--entitlements "$APP_SRC/entitlements.plist" "$APP"
else
	codesign --force --sign - "$APP"
fi

echo "→ disk image"
rm -rf "$DIST" && mkdir -p "$DIST/dmg"
mv "$APP" "$DIST/"
cp -R "$DIST/AstroFlow.app" "$DIST/dmg/" && ln -s /Applications "$DIST/dmg/Applications"
dmg="$DIST/AstroFlow-$version.dmg"
hdiutil create -quiet -volname AstroFlow -srcfolder "$DIST/dmg" -fs APFS -format ULMO "$dmg"
rm -rf "$DIST/dmg"
if [ -n "${SIGN_IDENTITY:-}" ]; then
	codesign --force --timestamp --sign "$SIGN_IDENTITY" "$dmg"
	if [ -n "${NOTARY_PROFILE:-}" ]; then
		xcrun notarytool submit "$dmg" --keychain-profile "$NOTARY_PROFILE" --wait
		xcrun stapler staple "$dmg"
	fi
fi
# The sources of what the app carries, to publish beside the disk image.
mkdir -p "$DIST/sources"
while read -r archive; do
	cp "$SOURCES/$archive" "$DIST/sources/"
done < "$DIST/AstroFlow.app/Contents/Resources/licenses/SOURCES.txt"
du -sh "$DIST/AstroFlow.app" "$dmg" "$DIST/sources"
