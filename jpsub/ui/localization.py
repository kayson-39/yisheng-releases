# -*- coding: utf-8 -*-
"""统一 Qt 内置编辑菜单的中文翻译。"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def install_chinese_translations(app: QtWidgets.QApplication) -> None:
    """使用随附的 Qt 翻译，覆盖输入框和表格临时创建的编辑器。"""
    if getattr(app, "_chinese_translator", None) is not None:
        return
    translator = QtCore.QTranslator(app)
    translations = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
    if not translator.load("qtbase_zh_CN", translations):
        raise RuntimeError("缺少 Qt 中文翻译文件 qtbase_zh_CN.qm，请使用完整的软件包。")
    app.installTranslator(translator)
    # QApplication 不接管 installTranslator 的所有权，必须保留 Python 引用。
    app._chinese_translator = translator
