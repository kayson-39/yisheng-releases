# -*- coding: utf-8 -*-
"""首页文字翻译：独立输入，复用现有翻译调度与模型。"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

import httpx
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.config import Config, TRANSLATE_PRESETS
from ..core.translate import Translator, TranslationCancelled, TranslationStream
from .localization import install_chinese_translations


DIRECTIONS = {
    "ja-zh": ("ja", "zh", "日文", "中文"),
    "zh-ja": ("zh", "ja", "中文", "日文"),
}


def translation_error(exc: Exception) -> str:
    """只展示固定诊断，不把服务响应、地址或密钥放进界面和日志。"""
    detail = str(exc).lower()
    if any(marker in detail for marker in (
            "exceeds the available context size", "context_length_exceeded",
            "maximum context length", "prompt is too long", "input is too long",
            "too many tokens")):
        return "文本太长，超过当前模型的处理上限。请缩短内容或分段翻译。"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "超时" in detail or "超过等待上限" in detail:
        return "翻译超时，请检查网络或本地模型运行状态后重试。"
    if isinstance(exc, (httpx.InvalidURL, httpx.UnsupportedProtocol)) or "base url" in detail or "接口地址" in detail:
        return "翻译服务地址无效，请检查设置中的 Base URL。"
    if "api key" in detail or "apikey" in detail or "401" in detail:
        return "翻译服务缺少 API Key 或密钥无效，请检查服务设置。"
    if "gguf" in detail or "模型未下载" in detail or "尚未下载" in detail or "没有 gguf" in detail:
        return "本地翻译模型或引擎未就绪，请在设置中检查模型文件与引擎。"
    if isinstance(exc, httpx.ConnectError) or "无法连接" in detail or "未启动" in detail or "未运行" in detail:
        return "无法连接翻译服务，请检查网络或启动所选本地服务。"
    if "模型名" in detail or "未指定本地模型" in detail:
        return "翻译模型配置无效，请检查设置中的模型名称。"
    return "翻译失败，请检查模型或服务设置后重试。"


class TranslationConfig:
    """冻结一次请求的配置，设置变化只影响下一次提交。"""

    def __init__(self, cfg: Config, *, follow_audio: bool = False, direction: str = "ja-zh"):
        self.translate = copy.deepcopy(cfg.get("translate", {}) or {})
        selected = str(cfg.get("text_translate.provider", "same") or "same")
        self.follows_audio = follow_audio or selected == "same"
        if not self.follows_audio:
            self.translate["provider"] = selected
            self.translate["services"] = copy.deepcopy(cfg.get("text_translate.services", {}) or {})
        self.direction = "ja-zh" if follow_audio else direction
        if self.direction not in DIRECTIONS:
            raise ValueError("不支持的文字翻译方向")
        source, target, _source_name, _target_name = DIRECTIONS[self.direction]
        self.translate["source_lang"] = source
        self.translate["target_lang"] = target

    def direction_label(self) -> str:
        _source, _target, source_name, target_name = DIRECTIONS[self.direction]
        return f"{source_name} → {target_name}"

    def get(self, key: str, default=None):
        if not key.startswith("translate."):
            return default
        node = self.translate
        for part in key.split(".")[1:]:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def service(self, provider: str) -> dict:
        preset = TRANSLATE_PRESETS.get(provider, TRANSLATE_PRESETS["custom"])
        user = self.get(f"translate.services.{provider}", {}) or {}
        return {
            "provider": provider, "kind": preset["kind"], "label": preset["label"],
            "base_url": user.get("base_url") or preset.get("base_url", ""),
            "model": user.get("model") or preset.get("model", ""),
            "api_key": user.get("api_key", ""),
        }

    def label(self, provider: str | None = None) -> str:
        service = self.service(provider or self.get("translate.provider", "ms-free"))
        model = str(service.get("model") or "")
        if service["kind"] == "llamacpp" and model:
            model = Path(model).name
        return service["label"] + (f" · {model}" if model else "")

    def local_identity(self) -> tuple[str, str] | None:
        provider = self.get("translate.provider", "ms-free")
        service = self.service(provider)
        if service["kind"] != "llamacpp":
            return None
        from ..core.llama_engine import MODELS_LLM, preferred_variant, resolve_model, valid_model
        from ..paths import ROOT

        chosen = str(service.get("model") or "")
        # 独立模型没有选择时不能悄悄使用语音模型或目录里的默认模型。
        if not self.follows_audio and not chosen:
            raise RuntimeError("请在设置 → 文字翻译中选择 GGUF 模型。")
        if not self.follows_audio:
            selected = Path(chosen)
            if not selected.is_absolute():
                selected = ROOT / selected
            moved = MODELS_LLM / Path(chosen).name
            model = next((path for path in (selected, moved)
                          if path.suffix.lower() == ".gguf" and valid_model(path)), None)
            if model is None:
                raise RuntimeError("选定的 GGUF 模型不可用，请在设置 → 文字翻译中重新选择。")
            self.translate.setdefault("services", {}).setdefault(provider, {})["model"] = str(model)
        else:
            model = resolve_model(chosen)
        return (str(model.resolve()) if model else chosen, preferred_variant(self))


class TextTranslationJob(QtCore.QThread):
    completed = QtCore.Signal(str, str, str)

    def __init__(self, text: str, config: TranslationConfig, release_engine: bool, parent=None):
        super().__init__(parent)
        self.text = text
        self.config = config
        self.release_engine = release_engine
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()
        self.requestInterruption()

    def run(self) -> None:
        output, actual, error = "", "", ""
        try:
            if self.cancelled.is_set():
                return
            if self.release_engine:
                from ..core.llama_engine import stop_translation
                stop_translation()
            control = TranslationStream(lambda _piece: None, self.cancelled.is_set,
                                        seconds=150.0, read_timeout=5.0)
            control.checkpoint()
            translator = Translator(self.config)
            output = translator.translate(self.text, [], on_delta=control)
            control.checkpoint()
            actual = self.config.direction_label() + " · " + self.config.label(translator.last_provider or None)
        except TranslationCancelled:
            error = "已取消翻译"
        except Exception as exc:
            # 不记录输入、密钥、URL 或原始服务响应。
            logging.warning("Text translation failed (%s)", type(exc).__name__)
            error = translation_error(exc)
        finally:
            if self.release_engine:
                try:
                    from ..core.llama_engine import stop_translation
                    stop_translation()
                except Exception as exc:
                    logging.error("Text translation engine cleanup failed (%s)", type(exc).__name__)
                    error = "本地翻译引擎未能释放，请退出软件后重试。"
        if not self.cancelled.is_set():
            self.completed.emit(output, actual, error)


class TextTranslationPanel(QtWidgets.QFrame):
    collapse_requested = QtCore.Signal()
    settings_requested = QtCore.Signal()
    copied = QtCore.Signal(str)
    busy_changed = QtCore.Signal(bool)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._job: TextTranslationJob | None = None
        self._audio_active = False
        self._shutting_down = False
        self._direction = "ja-zh"
        install_chinese_translations(QtWidgets.QApplication.instance())
        self.setObjectName("card")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        layout = QtWidgets.QVBoxLayout(self)
        # 原生字体和主题会提高控件最小高度，禁止父布局把面板压到内部控件重叠。
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(5)
        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("文字翻译")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        self.cmb_direction = QtWidgets.QComboBox()
        self.cmb_direction.setAccessibleName("文字翻译方向")
        for key, (_source, _target, source_name, target_name) in DIRECTIONS.items():
            self.cmb_direction.addItem(f"{source_name} → {target_name}", key)
        heading.addWidget(self.cmb_direction)
        self.btn_swap = QtWidgets.QPushButton("⇄")
        self.btn_swap.setObjectName("ghost")
        self.btn_swap.setAccessibleName("交换翻译方向和内容")
        self.btn_swap.setToolTip("交换翻译方向，并交换原文和译文")
        self.btn_swap.clicked.connect(self.swap_direction)
        heading.addWidget(self.btn_swap)
        heading.addStretch(1)
        self.btn_model_settings = QtWidgets.QPushButton("模型设置")
        self.btn_model_settings.setObjectName("ghost")
        self.btn_model_settings.clicked.connect(self.settings_requested.emit)
        heading.addWidget(self.btn_model_settings)
        self.btn_collapse = QtWidgets.QPushButton("收起")
        self.btn_collapse.setObjectName("ghost")
        self.btn_collapse.clicked.connect(self.collapse_requested.emit)
        heading.addWidget(self.btn_collapse)
        layout.addLayout(heading)
        self.lbl_model = QtWidgets.QLabel()
        self.lbl_model.setObjectName("muted")
        self.lbl_model.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lbl_model.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                    QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.lbl_model)
        fields = QtWidgets.QHBoxLayout()
        fields.setSpacing(10)
        self.source = QtWidgets.QPlainTextEdit()
        self.output = QtWidgets.QPlainTextEdit()
        self.lbl_source = QtWidgets.QLabel()
        self.lbl_output = QtWidgets.QLabel()
        for caption, editor in ((self.lbl_source, self.source), (self.lbl_output, self.output)):
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(3)
            caption.setObjectName("muted")
            caption.setBuddy(editor)
            column.addWidget(caption)
            editor.setFixedHeight(64)
            editor.verticalScrollBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            editor.horizontalScrollBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
            column.addWidget(editor)
            fields.addLayout(column, 1)
        layout.addLayout(fields)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(7)
        self.btn_paste = QtWidgets.QPushButton("粘贴")
        self.btn_paste.setObjectName("ghost")
        self.btn_paste.clicked.connect(self.source.paste)
        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.clicked.connect(self.clear)
        actions.addWidget(self.btn_paste)
        actions.addWidget(self.btn_clear)
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setObjectName("muted")
        self.lbl_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lbl_status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                     QtWidgets.QSizePolicy.Policy.Preferred)
        actions.addWidget(self.lbl_status, 1)
        self.btn_copy = QtWidgets.QPushButton("复制译文")
        self.btn_copy.setObjectName("ghost")
        self.btn_copy.clicked.connect(self.copy_output)
        actions.addWidget(self.btn_copy)
        self.btn_translate = QtWidgets.QPushButton("翻译")
        self.btn_translate.setObjectName("accent")
        self.btn_translate.setToolTip("翻译（Ctrl+Enter）")
        self.btn_translate.clicked.connect(self.translate)
        actions.addWidget(self.btn_translate)
        layout.addLayout(actions)
        self.source.textChanged.connect(self._update_buttons)
        self.output.textChanged.connect(self._update_buttons)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.translate)
        self.cmb_direction.currentIndexChanged.connect(self._on_direction_changed)
        self._update_direction_labels()
        self.refresh_model_label()
        self._update_buttons()
        self.hide()

    def set_audio_active(self, active: bool) -> None:
        self._audio_active = bool(active)

    def refresh_model_label(self) -> None:
        config = TranslationConfig(self.cfg, direction=self._direction)
        prefix = "跟随语音：" if config.follows_audio else "独立模型："
        text = prefix + config.label()
        self.lbl_model.setText(text)
        self.lbl_model.setToolTip(text)

    def _status(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setToolTip(text)

    def _update_buttons(self) -> None:
        busy = self.is_busy() or self._shutting_down
        self.btn_translate.setEnabled(bool(self.source.toPlainText().strip()) and not busy)
        self.btn_paste.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        self.cmb_direction.setEnabled(not busy)
        self.btn_swap.setEnabled(not busy)
        self.source.setReadOnly(busy)
        self.output.setReadOnly(busy)
        self.btn_copy.setEnabled(bool(self.output.toPlainText().strip()))

    def _update_direction_labels(self) -> None:
        _source, _target, source_name, target_name = DIRECTIONS[self._direction]
        self.lbl_source.setText(f"原文（{source_name}）")
        self.lbl_output.setText(f"译文（{target_name}）")
        self.source.setPlaceholderText(f"输入或粘贴需要翻译的{source_name}")
        self.source.setAccessibleName(source_name + "原文")
        self.output.setPlaceholderText(f"{target_name}译文将显示在这里")
        self.output.setAccessibleName(target_name + "译文")

    def _on_direction_changed(self, _index: int) -> None:
        if self.is_busy() or self._shutting_down:
            with QtCore.QSignalBlocker(self.cmb_direction):
                self.cmb_direction.setCurrentIndex(self.cmb_direction.findData(self._direction))
            return
        self._direction = str(self.cmb_direction.currentData())
        self._update_direction_labels()
        self._status("已切换为：" + TranslationConfig(self.cfg, direction=self._direction).direction_label())

    def swap_direction(self) -> None:
        if self.is_busy() or self._shutting_down:
            return
        source, output = self.source.toPlainText(), self.output.toPlainText()
        direction = "zh-ja" if self._direction == "ja-zh" else "ja-zh"
        self.cmb_direction.setCurrentIndex(self.cmb_direction.findData(direction))
        self.source.setPlainText(output)
        self.output.setPlainText(source)

    def clear(self) -> None:
        if self.is_busy():
            return
        self.source.clear()
        self.output.clear()
        self._status("")

    def copy_output(self) -> None:
        text = self.output.toPlainText()
        if text.strip():
            QtWidgets.QApplication.clipboard().setText(text)
            self.copied.emit("已复制译文")

    def translate(self) -> None:
        text = self.source.toPlainText().strip()
        if self.is_busy() or self._shutting_down or not text:
            return
        config = TranslationConfig(self.cfg, direction=self._direction)
        audio = TranslationConfig(self.cfg, follow_audio=True)
        try:
            local = config.local_identity()
            different_local = local is not None and local != audio.local_identity()
            if self._audio_active and different_local:
                self._status("请先停止聆听，再使用不同的内置模型。")
                return
        except RuntimeError as exc:
            self._status(str(exc))
            return
        self.refresh_model_label()
        self.output.clear()
        self._status("正在翻译（" + config.direction_label() + "）：" + config.label())
        self._job = TextTranslationJob(text, config, different_local, self)
        self._job.completed.connect(self._on_completed)
        self._job.finished.connect(self._on_finished)
        self._update_buttons()
        self.busy_changed.emit(True)
        self._job.start()

    @QtCore.Slot(str, str, str)
    def _on_completed(self, output: str, actual: str, error: str) -> None:
        if self._shutting_down:
            return
        if error:
            self._status(error)
        else:
            self.output.setPlainText(output)
            self._status("已完成：" + actual)

    @QtCore.Slot()
    def _on_finished(self) -> None:
        job = self._job
        self._job = None
        if job is not None:
            job.deleteLater()
        self._update_buttons()
        self.busy_changed.emit(False)

    def is_busy(self) -> bool:
        return self._job is not None

    def shutdown(self) -> None:
        self._shutting_down = True
        if self._job is not None:
            self._job.cancel()
        self._update_buttons()

    def resume_after_cancelled_shutdown(self) -> None:
        self._shutting_down = False
        self._update_buttons()
