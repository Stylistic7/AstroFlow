# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[("environment.yml", "."), ("expected_env.yml", ".")],
    hiddenimports=[
        'tkinter',
        '_tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)

exe = BUNDLE(
    exe,
    name='AstroFlow.app',
    bundle_identifier='com.example.AstroFlow',
    info_plist={
        'CFBundleDisplayName':
            'AstroFlow',
        'CFBundleName':
            'AstroFlow',
        'CFBundleVersion':
            '0.1.0',
        'CFBundleShortVersionString':
            '0.1.0',
        'NSPrincipalClass':
            'NSApplication',
        'CFBundleIconFile':
            'icon.icns',
        'CFBundleURLTypes': [{
            'CFBundleURLName': 'myapp',
            'CFBundleURLSchemes': ['myapp'],
        }],
    },
    icon='icon.icns',
    sources=['main.py'],
)
