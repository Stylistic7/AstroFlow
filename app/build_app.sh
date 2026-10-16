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
# It also carries GraXpert (its own packages in Resources/tools, pinned by
# graxpert-requirements.txt) and the Gaia XP catalogue AstroFlow calibrates
# with (Resources/gaia_xp, from GAIA_CATALOG, built by
# core/photometry/gaia_catalog.py).
#
# What the user still installs: Homebrew and `brew install astrometry-net
# vips` (the app checks for them on launch), and Siril and RC Astro (the
# dashboard's Third-Party Tools page tries each tool on every Mac). The
# astrometry.net index files are chosen and downloaded on the Catalogs page.
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
PYTHON_VERSION=3.13
PYOBJC="pyobjc-framework-Cocoa==11.1"
GAIA_CATALOG="${GAIA_CATALOG:-/Volumes/AstroProcessed/Catalogs/GaiaXP/gaia_xp}"

for column in source_id ra dec G colour flux; do
	if [ ! -f "$GAIA_CATALOG/$column.npy" ]; then
		echo "No Gaia XP catalogue at $GAIA_CATALOG ($column.npy is missing); set GAIA_CATALOG." >&2
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
# Tracked and new files, as `git status` sees them; no test data, and of the
# frontend only its build.
(cd "$REPO" && git ls-files -co --exclude-standard -- core app/AstroFlow.py pyproject.toml uv.lock) \
	| grep -vE '^core/(testdata|pix/testdata|web_server/frontend)/' > "$WORK/files.txt"
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

echo "→ Gaia XP catalogue"
rsync -a --exclude '.*' "$GAIA_CATALOG/" "$RES/gaia_xp/"
cp "$APP_SRC/LICENSE-gaia.txt" "$RES/gaia_xp/"

# Bytecode now, so nothing writes into the signed bundle at run time.
"$py" -m compileall -q -j 0 "$RES/python/lib" "$RES/tools/graxpert" "$RES/AstroFlow" >/dev/null || true

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
du -sh "$DIST/AstroFlow.app" "$dmg"
