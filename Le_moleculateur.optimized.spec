# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Le_moleculateur.py'],
    pathex=[],
    binaries=[
        (r'.conda\\Library\\bin\\libcrypto-3-x64.dll', '.'),
        (r'.conda\\Library\\bin\\libssl-3-x64.dll', '.'),
        (r'.conda\\Library\\bin\\liblzma.dll', '.'),
        (r'.conda\\Library\\bin\\libbz2.dll', '.'),
        (r'.conda\\Library\\bin\\libexpat.dll', '.'),
        (r'.conda\\Library\\bin\\ffi.dll', '.'),
        (r'.conda\\Library\\bin\\sqlite3.dll', '.'),
    ],
    datas=[('Atomic weights.xlsx', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'tkinter', 'pandas'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Le_moleculateur_optimized',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Le_moleculateur_optimized',
)
