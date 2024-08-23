# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['AstroFlow.py'],
    pathex=[],
    binaries=[],
    datas=[("../core/web_server/frontend/build",
            "core/web_server/backend/templates"),
           ("../core/web_server/frontend/build/static",
            "core/web_server/backend/static")],
    hiddenimports=["yaml"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Something like this which can exclude node_modules
a.datas += Tree('../core', prefix='core', excludes=['node_modules', 'testdata'])

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AstroFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AstroFlow',
)
app = BUNDLE(
    coll,
    name='AstroFlow.app',
    icon='icon.icns',
    bundle_identifier=None,
)
