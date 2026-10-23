# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
datas += collect_data_files('rapidocr_onnxruntime')
datas += [('subtitle_eraser/assets/fonts/NotoSansSC.ttf', 'subtitle_eraser/assets/fonts'),
          ('subtitle_eraser/assets/fonts/OFL.txt', 'subtitle_eraser/assets/fonts')]
binaries += collect_dynamic_libs('onnxruntime')
tmp_ret = collect_all('windows_capture')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['tools/launch.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore'],
    noarchive=False,
    optimize=0,
)
# Qt uses the unversioned ICU API provided by Windows. Do not let a foreign
# Poppler/Conda DLL found through the build machine's PATH shadow that API.
from pathlib import Path
windows_icu = {'icuuc.dll', 'icuin.dll', 'icudt.dll', 'icu.dll'}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in windows_icu]
# The app renders NumPy images in QWidget. It does not use PDF, SVG, QML,
# Qt Quick or a virtual keyboard. Do not redistribute those optional modules
# merely because Qt's plugin discovery found them on the build machine.
def needed_qt_binary(entry):
    name = entry[0].replace('\\', '/').lower()
    leaf = name.rsplit('/', 1)[-1]
    if leaf.startswith(('qt6qml', 'qt6quick', 'qt6virtualkeyboard', 'qt6pdf', 'qt6svg')):
        return False
    return not any(part in name for part in ('pyside6/plugins/imageformats/',
                    'pyside6/plugins/iconengines/', 'pyside6/plugins/platforminputcontexts/'))
a.binaries = [entry for entry in a.binaries if needed_qt_binary(entry)]
a.datas = [entry for entry in a.datas if 'pyside6/translations/' not in entry[0].replace('\\', '/').lower()]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SubtitleEraser',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SubtitleEraser',
)
