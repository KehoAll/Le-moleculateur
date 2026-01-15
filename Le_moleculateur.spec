# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Le_moleculateur.py'],
    pathex=[],
    binaries=[('.conda\\Library\\bin\\libcrypto-3-x64.dll', '.'), ('.conda\\Library\\bin\\libssl-3-x64.dll', '.'), ('.conda\\Library\\bin\\liblzma.dll', '.'), ('.conda\\Library\\bin\\libbz2.dll', '.'), ('.conda\\Library\\bin\\libexpat.dll', '.'), ('.conda\\Library\\bin\\ffi.dll', '.'), ('.conda\\Library\\bin\\sqlite3.dll', '.')],
    datas=[],
    hiddenimports=[],
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
    a.binaries,
    a.datas,
    [],
    name='Le_moleculateur',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
