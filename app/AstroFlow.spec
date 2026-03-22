# -*- mode: python ; coding: utf-8 -*-
#
# Run with: pyinstaller -y AstroFlow.spec

import os
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden_imports = collect_submodules("ray")
hidden_imports += collect_submodules("spherical_geometry")
hidden_imports += collect_submodules("xisf")
hidden_imports += collect_submodules("objc")
hidden_imports += ["AppKit", "Foundation", "PyObjCTools", "PyObjCTools.AppHelper",
                   "diskcache", "flask", "graphlib", "objsize", "plotly",
                   "shapely", "skimage", "skimage.transform",
                   "spherical_geometry", "structlog",
                   "xisf", "xxhash", "yaml"]
print(f'### hidden_imports {hidden_imports}')

datas = collect_data_files("ray") + collect_data_files("xisf")

a = Analysis(
    ['AstroFlow.py'],
    pathex=[os.path.abspath('../core')],
    binaries=[],
    datas=datas + [("../core/web_server/frontend/build",
            "core/web_server/templates"),
           ("../core/web_server/frontend/build/static",
            "core/web_server/static"),
           ("../core/web_server/backend/static/",
            "core/web_server/static"),
           ("/Users/nobody/AstroFlow_venv/bin/ray", "bin")],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Something like this which can exclude node_modules
a.datas += Tree(os.path.abspath('../core'), prefix='core', excludes=['node_modules', 'testdata'])

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
