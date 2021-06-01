from PyInstaller.compat import is_linux

hiddenimports = [
    "PySide2.QtXml"
]

if is_linux:
    datas = [
        ("/lib/qt/plugins/platforms", "platforms")
    ]
