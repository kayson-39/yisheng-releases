# -*- coding: utf-8 -*-
"""主窗口：标题栏 + 工具栏 + 字幕区 + 设置抽屉 + 悬浮字幕 + 托盘。"""
from __future__ import annotations

import time
import logging
import sqlite3
from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from ..core.segment import Segment
from ..paths import ASSETS, HISTORY
from .background import BackgroundLayer
from .overlay import OverlayWindow
from .transcript import TranscriptView
from .text_translate import TextTranslationPanel
from ..core.history import HistoryStore
from ..core.export import snapshot, write_export, to_srt


class ExportJob(QtCore.QThread):
    result = QtCore.Signal(str,str)
    def __init__(self,path,segments,parent):
        super().__init__(parent);self.path=path;self.segments=segments
    def run(self):
        try:write_export(self.path,self.segments);self.result.emit(self.path,'')
        except Exception as exc:self.result.emit(self.path,str(exc))


class LevelBar(QtWidgets.QWidget):
    """音量电平条。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setFixedSize(90, 8)

    def set_level(self, v: float) -> None:
        self._level = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, e) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, 38))
        p.drawRoundedRect(rect, 4, 4)
        w = int(rect.width() * self._level)
        if w > 2:
            grad = QtGui.QLinearGradient(0, 0, rect.width(), 0)
            grad.setColorAt(0.0, QtGui.QColor("#4fd08a"))
            grad.setColorAt(0.7, QtGui.QColor("#ffd166"))
            grad.setColorAt(1.0, QtGui.QColor("#ff6b6b"))
            p.setBrush(QtGui.QBrush(grad))
            p.drawRoundedRect(QtCore.QRect(0, 0, w, rect.height()), 4, 4)


class SessionHeader(QtWidgets.QWidget):
    """复制按钮属于整场会话；窄窗口时换行，避免挤掉回到最新。"""

    def __init__(self, title, actions, latest, count, parent=None):
        super().__init__(parent)
        self._title, self._actions = title, actions
        self._latest, self._count = latest, count
        self._compact = False
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(6)
        self._grid.addWidget(title, 0, 0)
        self._grid.addWidget(actions, 0, 1)
        self._grid.addWidget(latest, 0, 2)
        self._grid.addWidget(count, 0, 3)
        self._grid.setColumnStretch(1, 1)
        actions.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                              QtWidgets.QSizePolicy.Policy.Preferred)
        for widget in (title, actions, latest, count):
            widget.installEventFilter(self)

    def _arrange(self):
        latest_width = 0 if self._latest.isHidden() else self._latest.sizeHint().width() + 8
        needed = (self._title.sizeHint().width() + self._actions.sizeHint().width()
                  + self._count.sizeHint().width() + latest_width + 24)
        compact = self.width() < needed
        if compact == self._compact:
            return
        self._compact = compact
        self._grid.removeWidget(self._actions)
        if compact:
            self._grid.addWidget(self._actions, 1, 0, 1, 4,
                                 QtCore.Qt.AlignmentFlag.AlignRight)
        else:
            self._grid.addWidget(self._actions, 0, 1)

    def resizeEvent(self, event):
        self._arrange()
        super().resizeEvent(event)

    def event(self, event):
        if hasattr(self, "_grid") and event.type() in (
                QtCore.QEvent.Type.LayoutRequest, QtCore.QEvent.Type.FontChange,
                QtCore.QEvent.Type.StyleChange):
            QtCore.QTimer.singleShot(0, self, self._arrange)
        return super().event(event)

    def eventFilter(self, obj, event):
        if event.type() in (QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide,
                             QtCore.QEvent.Type.LayoutRequest):
            QtCore.QTimer.singleShot(0, self, self._arrange)
        return super().eventFilter(obj, event)


class _SessionTimeline:
    """Map per-capture samples to one session; never use wall-clock dates for elapsed time."""
    def __init__(self):
        self.origin = None
        self.offset = None
        self.last_end = 0.0

    def bind(self, pipeline):
        if self.offset is None and pipeline is not None:
            stamp, _ = pipeline.subtitle_audio_clock
            if stamp is not None:
                if self.origin is None:
                    self.origin = stamp
                # Compute once per epoch, not once per sentence. Input buffers
                # can arrive in bursts, so never rewind behind a prior epoch.
                self.offset = max(self.last_end, stamp - self.origin)
        return self.offset or 0.0

    def position(self, pipeline):
        if pipeline is None:
            return self.last_end
        return max(self.last_end, self.bind(pipeline) + pipeline.subtitle_audio_clock[1])

    def clear_at(self, pipeline):
        if pipeline is not None:
            stamp, elapsed = pipeline.subtitle_audio_clock
            if stamp is not None:
                self.origin = stamp + elapsed
                self.offset = -elapsed


class MainWindow(QtWidgets.QWidget):
    def __init__(self, cfg):
        super().__init__(None)
        self.cfg = cfg
        self.pipeline: Pipeline | None = None
        self._retiring_pipelines = []
        self._restart_requested = False
        self._pending_translation_engine_reset = False
        self.overlay: OverlayWindow | None = None
        self._settings = None
        self._boot_done = False
        self._ui_ready = False  # 初始化期间不弹提示气泡
        self._closing = False
        self._live_jp = ""
        self._live_zh = ""
        self._drag_pos: QtCore.QPoint | None = None
        self._resize_edges = ""
        self._resize_start: tuple | None = None
        self._session_timeline = _SessionTimeline()
        self._failed_translation_ids = set()
        self._session_id=None
        self._history_dirty={}
        self._history_saved_draft = None
        self._preserve_session_for_restart=False
        self._export_jobs=[]
        self._export_dialog=None
        self._history_dialog=None
        self.history=None
        self._history_error=''
        try:
            self.history=HistoryStore(self.cfg.get('history.directory') or self.cfg.path.parent/'history')
            with self.history.db:
                self.history.db.execute("UPDATE sessions SET state='interrupted' WHERE state='recording'")
        except Exception as exc:
            self._history_error=str(exc)
            logging.exception('History could not be opened')
        self._history_timer=QtCore.QTimer(self)
        self._history_timer.setInterval(650)
        self._history_timer.timeout.connect(self._flush_history)
        self._history_timer.start()

        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint
                            | QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(880, 560)
        icon_path = ASSETS / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        self.setAcceptDrops(True)  # 支持把图片拖进窗口换背景

        # 背景层
        self.bg = BackgroundLayer(self)
        self.bg.setGeometry(self.rect())
        # 内容
        self._build_ui()
        self._apply_theme()
        self._restore_geometry()
        self._build_tray()
        self._install_shortcuts()
        self.bg.lower()
        self.btn_ontop.setChecked(bool(self.cfg.get("ui.window.on_top", False)))
        self.btn_overlay.setChecked(bool(self.cfg.get("overlay.enabled", False)))
        self._update_trans_label_from_cfg()
        self._ui_ready = True
        # 新启动只展示本次会话；上次定稿和草稿仍可从历史记录查看。
        if self._history_error:
            self.toast_msg('自动保存暂不可用：'+self._history_error)

    # ==================================================================
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- 标题栏
        self.titlebar = QtWidgets.QFrame()
        self.titlebar.setObjectName("titlebar")
        self.titlebar.setFixedHeight(52)
        tb = QtWidgets.QHBoxLayout(self.titlebar)
        tb.setContentsMargins(22, 8, 14, 4)
        tb.setSpacing(6)
        self.lbl_logo = QtWidgets.QLabel()
        logo_pm = QtGui.QPixmap(str(ASSETS / "icon.png"))
        if logo_pm.isNull():
            self.lbl_logo.setText("🌸")
        else:
            self.lbl_logo.setPixmap(logo_pm.scaled(
                20, 20, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
            self.lbl_logo.setFixedWidth(24)
        self.lbl_title = QtWidgets.QLabel(f"{APP_NAME}  {__version__}")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 15px;")
        tb.addWidget(self.lbl_logo)
        tb.addWidget(self.lbl_title)
        tb.addStretch(1)
        self.btn_overlay = self._win_btn("悬浮字幕", "悬浮字幕窗（F10）", checkable=True)
        self.btn_ontop = self._win_btn("置顶", "窗口置顶", checkable=True)
        self.btn_settings = self._win_btn("设置", "展开/收起设置面板", checkable=True)
        self.btn_min = self._win_btn("─", "最小化")
        self.btn_close = self._win_btn("×", "退出", close=True)
        for b in (self.btn_overlay, self.btn_ontop, self.btn_settings, self.btn_min, self.btn_close):
            tb.addWidget(b)
        outer.addWidget(self.titlebar)

        # 提示气泡（须在按钮状态初始化之前创建）
        self.toast = QtWidgets.QLabel("", self)
        self.toast.setObjectName("card")
        self.toast.setWordWrap(True)          # 提示里有「哪个服务不可用」这类长句，换行才不会超出窗口
        self.toast.setMaximumWidth(620)
        self.toast.setStyleSheet(
            "background: rgba(20,22,30,0.92); border: 1px solid rgba(255,255,255,0.12);"
            "border-radius: 10px; padding: 8px 14px; color: white;"
        )
        self.toast.hide()
        self._toast_timer = QtCore.QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.toast.hide)

        # ---- 主体
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(22, 10, 22, 0)
        body.setSpacing(12)
        outer.addLayout(body, 1)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(16)
        body.addLayout(left, 1)

        heading = QtWidgets.QHBoxLayout()
        heading_text = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("实时字幕")
        title.setObjectName("pageTitle")
        subtitle = QtWidgets.QLabel("播放视频或直播，让字幕跟上每一句。")
        subtitle.setObjectName("muted")
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading.addLayout(heading_text)
        heading.addStretch(1)
        self.btn_background = QtWidgets.QPushButton("自定义背景")
        self.btn_background.setObjectName("ghost")
        self.btn_background.clicked.connect(self._open_background_settings)
        heading.addWidget(self.btn_background)
        left.addLayout(heading)

        # 两行工具栏：打开设置抽屉、缩小窗口时，设备和操作按钮仍可点。
        bar = QtWidgets.QVBoxLayout()
        bar.setSpacing(10)
        primary = QtWidgets.QHBoxLayout()
        primary.setSpacing(8)
        secondary = QtWidgets.QHBoxLayout()
        secondary.setSpacing(6)
        self.btn_start = QtWidgets.QPushButton("开始聆听")
        self.btn_start.setObjectName("accent")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setMinimumWidth(150)
        self.btn_start.clicked.connect(self.toggle_start)
        primary.addWidget(self.btn_start)
        self.lbl_state = QtWidgets.QLabel("● 未开始")
        self.lbl_state.setStyleSheet("color: #657087;")
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setMinimumWidth(0)
        primary.addWidget(self.lbl_state, 1)
        self.lbl_translate_queue = QtWidgets.QLabel("翻译待启动")
        self.lbl_translate_queue.setObjectName("muted")
        self.lbl_translate_queue.setWordWrap(True)
        primary.addWidget(self.lbl_translate_queue)
        primary.addStretch(1)
        self.btn_text_translate = QtWidgets.QPushButton("文字翻译 ▾")
        self.btn_text_translate.setObjectName("ghost")
        self.btn_text_translate.setCheckable(True)
        self.btn_text_translate.setToolTip("输入或粘贴日文，使用设置中的文字翻译模型")
        self.btn_text_translate.toggled.connect(self._toggle_text_translation)
        primary.addWidget(self.btn_text_translate)
        self.cmb_device = QtWidgets.QComboBox()
        self.cmb_device.setMinimumWidth(180)
        self.cmb_device.setMaximumWidth(220)
        self.cmb_device.setToolTip("快速切换要监听的声音设备")
        self.cmb_device.addItem("默认输出设备", "")
        secondary.addWidget(self.cmb_device)
        secondary.addStretch(1)
        self.btn_clear = QtWidgets.QPushButton("清空记录")
        self.btn_clear.setObjectName("ghost")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_history = QtWidgets.QPushButton("搜索 / 回看")
        self.btn_history.setObjectName("ghost")
        self.btn_history.setToolTip("查看本次会话的全部字幕；显示行数不会删除旧记录")
        self.btn_history.clicked.connect(self._show_history)
        self.btn_sessions=QtWidgets.QPushButton('历史记录')
        self.btn_sessions.setObjectName('ghost')
        self.btn_sessions.clicked.connect(self._show_history)
        self.btn_export = QtWidgets.QPushButton("导出字幕")
        self.btn_export.setObjectName("ghost")
        self.btn_export.clicked.connect(self._export)
        self.btn_retry = QtWidgets.QPushButton("补译")
        self.btn_retry.setObjectName("ghost")
        self.btn_retry.setToolTip("聆听时重试本次会话中失败或未完成的翻译，不重新识别音频")
        self.btn_retry.setEnabled(False)
        self.btn_retry.clicked.connect(self._retry_missing_translations)
        secondary.addWidget(self.btn_retry)
        secondary.addWidget(self.btn_clear)
        secondary.addWidget(self.btn_history)
        secondary.addWidget(self.btn_sessions)
        secondary.addWidget(self.btn_export)
        bar.addLayout(primary)
        bar.addLayout(secondary)
        control_card = QtWidgets.QFrame()
        control_card.setObjectName("controlCard")
        card_layout = QtWidgets.QVBoxLayout(control_card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.addLayout(bar)
        left.addWidget(control_card)

        self.text_translation = TextTranslationPanel(self.cfg, self)
        self.text_translation.collapse_requested.connect(
            lambda: self.btn_text_translate.setChecked(False))
        self.text_translation.copied.connect(self.toast_msg)
        self.text_translation.settings_requested.connect(self._show_text_translation_settings)
        self.text_translation.busy_changed.connect(self._sync_text_translation_state)
        self.text_translation.hide()
        left.addWidget(self.text_translation)

        # 字幕区
        self.transcript = TranscriptView(self.cfg)
        self.transcript.copied.connect(self.toast_msg)
        reading_card = QtWidgets.QFrame()
        reading_card.setObjectName("readingCard")
        reading_layout = QtWidgets.QVBoxLayout(reading_card)
        reading_layout.setContentsMargins(18, 14, 14, 14)
        session_label = QtWidgets.QLabel("本次会话")
        session_label.setObjectName("muted")
        self.lbl_count = QtWidgets.QLabel("0 条字幕")
        self.lbl_count.setObjectName("faint")
        self.copy_actions = QtWidgets.QWidget()
        copy_layout = QtWidgets.QHBoxLayout(self.copy_actions)
        copy_layout.setContentsMargins(0, 0, 0, 0)
        copy_layout.setSpacing(6)
        copy_layout.addStretch(1)
        self.copy_buttons = {}
        for language, title in (("jp", "复制全部日文"), ("zh", "复制全部中文"),
                                ("both", "复制全部双语")):
            button = QtWidgets.QPushButton(title)
            button.setObjectName("ghost")
            button.setToolTip("复制本次会话的全部字幕，包含当前实时字幕，不受显示条数限制")
            button.setEnabled(False)
            button.clicked.connect(lambda checked=False, language=language:
                                   self.transcript.copy_all(language))
            self.copy_buttons[language] = button
            copy_layout.addWidget(button)
        self.transcript.content_changed.connect(self._update_copy_buttons)
        self.btn_latest=QtWidgets.QPushButton('↓ 回到最新')
        self.btn_latest.setObjectName('ghost')
        self.btn_latest.clicked.connect(self.transcript.jump_latest)
        self.transcript.following_changed.connect(lambda following:self.btn_latest.setVisible(not following))
        self.btn_latest.hide()
        self.session_header = SessionHeader(session_label, self.copy_actions,
                                             self.btn_latest, self.lbl_count)
        reading_layout.addWidget(self.session_header)
        reading_layout.addWidget(self.transcript, 1)
        left.addWidget(reading_card, 1)

        # 底部状态
        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(14)
        self.lbl_engine = QtWidgets.QLabel("引擎：未加载")
        self.lbl_engine.setObjectName("faint")
        self.lbl_trans = QtWidgets.QLabel("翻译：-")
        self.lbl_trans.setObjectName("faint")
        self.lbl_trans.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.lbl_trans.setMinimumWidth(180)
        self.lbl_trans.setMaximumWidth(330)
        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setObjectName("faint")
        self.level = LevelBar()
        foot.addWidget(self.lbl_engine)
        foot.addWidget(self.lbl_trans, 1)
        foot.addWidget(self.lbl_stats)
        foot.addStretch(1)
        foot.addWidget(QtWidgets.QLabel("音量"))
        foot.addWidget(self.level)
        left.addLayout(foot)

        # ---- 设置抽屉
        self.drawer_frame = QtWidgets.QFrame()
        self.drawer_frame.setObjectName("drawerCard")
        self._drawer_layout = QtWidgets.QVBoxLayout(self.drawer_frame)
        self._drawer_layout.setContentsMargins(4, 6, 4, 6)
        self.drawer_frame.setMaximumWidth(0)
        self.drawer_frame.setMinimumWidth(0)
        self.drawer_frame.hide()

        outer.addSpacing(12)

        # 连接按钮
        self.btn_settings.toggled.connect(self.toggle_drawer)
        self.btn_ontop.toggled.connect(self._toggle_ontop)
        self.btn_overlay.toggled.connect(self._toggle_overlay)
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_close.clicked.connect(self.close)

    def _update_copy_buttons(self) -> None:
        for language, button in self.copy_buttons.items():
            button.setEnabled(self.transcript.has_content(language))

    def _toggle_text_translation(self, on: bool) -> None:
        self.text_translation.refresh_model_label()
        self.text_translation.setVisible(on)
        self.btn_text_translate.setText("文字翻译 ▴" if on else "文字翻译 ▾")

    def _show_text_translation_settings(self) -> None:
        self.settings.nav.setCurrentRow(1)
        self.btn_settings.setChecked(True)
        QtCore.QTimer.singleShot(0, self, lambda: self.settings.pages.currentWidget()
                                 .ensureWidgetVisible(self.settings.text_settings_card))

    def _sync_text_translation_state(self) -> None:
        audio_active = self.pipeline is not None or bool(self._retiring_pipelines)
        if (self._pending_translation_engine_reset
                and not audio_active and not self.text_translation.is_busy()):
            from ..core.llama_engine import stop_translation

            self._pending_translation_engine_reset = False
            stop_translation()
        self.text_translation.set_audio_active(audio_active)
        self.btn_start.setEnabled(audio_active or not self.text_translation.is_busy())
        if (self._restart_requested and not self._closing and not audio_active
                and not self.text_translation.is_busy()):
            self._restart_requested = False
            QtCore.QTimer.singleShot(0, self._start_pipeline)

    def _win_btn(self, text: str, tip: str, checkable: bool = False, close: bool = False) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(text)
        b.setObjectName("closeBtn" if close else "winBtn")
        b.setToolTip(tip)
        b.setCheckable(checkable)
        if close or text in ("─",):
            b.setFixedWidth(38)
        return b

    # ==================================================================
    def _apply_theme(self) -> None:
        from .theme import build_qss

        theme = str(self.cfg.get("ui.theme", "light"))
        accent = str(self.cfg.get("ui.accent", "#ff86b3"))
        font = str(self.cfg.get("ui.font", "系统默认"))
        app = QtWidgets.QApplication.instance()
        f = app.font()
        if font and font != "系统默认":
            f.setFamily(font)
        app.setFont(f)
        photo = self.cfg.get("ui.bg", "none") not in ("", "none")
        app.setStyleSheet(build_qss(theme, accent, f.family(), photo))
        from .theme import PALETTES
        colors = PALETTES.get(theme, PALETTES["light"])
        palette = app.palette()
        for role, key in ((QtGui.QPalette.ColorRole.Window, "bg"),
                          (QtGui.QPalette.ColorRole.Base, "panel"),
                          (QtGui.QPalette.ColorRole.AlternateBase, "panel2"),
                          (QtGui.QPalette.ColorRole.Button, "panel"),
                          (QtGui.QPalette.ColorRole.Text, "text"),
                          (QtGui.QPalette.ColorRole.WindowText, "text"),
                          (QtGui.QPalette.ColorRole.ButtonText, "text")):
            palette.setColor(role, QtGui.QColor(colors[key]))
        app.setPalette(palette)
        from .theme import install_title_palette
        install_title_palette(app,self.cfg)
        if self._settings is not None:self._settings.apply_palette()
        self.bg._theme = theme
        self.bg.update()
        self.toast.setStyleSheet(f"background: {colors['panel']}; color: {colors['text']}; border: 1px solid {colors['border']}; border-radius: 10px; padding: 10px 16px;")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 15px; color: " +
                                     ("white" if theme == "dark" else "#1b2030") + ";")
        status=self.lbl_state.text().removeprefix('● ')
        self.lbl_state.setStyleSheet(f'color: {self._status_color(status)};')

    def _apply_bg(self) -> None:
        self._apply_theme()
        self.bg.set_background(
            str(self.cfg.get("ui.bg", "none")),
            int(self.cfg.get("ui.bg_blur", 16)),
            int(self.cfg.get("ui.bg_dim", 45)),
            int(self.cfg.get("ui.bg_opacity", 100)),
            animate=False,
        )

    def _schedule_bg_apply(self, delay_ms: int = 180) -> None:
        """背景模糊/暗度这类滑条一拖就是几十次改动，等手停下来再重算背景图。

        否则每动一格都要重新模糊一张全屏照片，界面就一顿一顿的。
        """
        if not hasattr(self, "_bg_timer"):
            self._bg_timer = QtCore.QTimer(self)
            self._bg_timer.setSingleShot(True)
            self._bg_timer.timeout.connect(self._apply_bg)
        self._bg_timer.start(max(0, int(delay_ms)))

    def _restore_geometry(self) -> None:
        w = int(self.cfg.get("ui.window.w", 1100))
        h = int(self.cfg.get("ui.window.h", 700))
        x, y = self.cfg.get("ui.window.x"), self.cfg.get("ui.window.y")
        self.resize(w, h)
        screens = QtWidgets.QApplication.screens()
        rect = QtCore.QRect(int(x), int(y), w, h) if x is not None and y is not None else QtCore.QRect()
        if x is not None and y is not None and any(s.availableGeometry().intersects(rect) for s in screens):
            self.move(int(x), int(y))
        else:
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            self.move(screen.center() - QtCore.QPoint(w // 2, h // 2 + 40))

    # ==================================================================
    def _refresh_devices(self) -> None:
        from ..core import audio as audio_mod

        self.cmb_device.blockSignals(True)
        self.cmb_device.clear()
        self.cmb_device.addItem("默认输出设备", "")
        devs = audio_mod.list_devices()
        src = str(self.cfg.get("audio.source", "loopback"))
        for name in devs.get("mic" if src == "mic" else "loopback", []):
            self.cmb_device.addItem(name, name)
        cur = str(self.cfg.get("audio.device", "") or "")
        idx = self.cmb_device.findData(cur)
        self.cmb_device.setCurrentIndex(max(0, idx))
        self.cmb_device.blockSignals(False)
        if not getattr(self, "_device_signal_connected", False):
            self.cmb_device.currentIndexChanged.connect(self._on_device_quick)
            self._device_signal_connected = True

    def _on_device_quick(self, _i: int) -> None:
        self.cfg.set("audio.device", self.cmb_device.currentData())
        self._on_setting_changed("audio.device", None)

    # ==================================================================
    @property
    def settings(self):
        """设置面板按需创建：只有真正展开抽屉时才构建，别拖慢启动。"""
        if self._settings is None:
            from .settings import SettingsPanel

            self._settings = SettingsPanel(self.cfg)
            self._settings.changed.connect(self._on_setting_changed)
            self._settings_dialog = QtWidgets.QDialog(self)
            self._settings_dialog.setWindowTitle("设置 · 译声")
            self._settings_dialog.resize(800, 700)
            self._settings_dialog.setMinimumSize(680, 520)
            layout = QtWidgets.QVBoxLayout(self._settings_dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._settings)
            self._settings_dialog.finished.connect(lambda _: self.btn_settings.setChecked(False))
        return self._settings

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        if not self._boot_done:
            self._boot_done = True
            # 先把窗口显示出来，再做背景照片、设备列表这些稍费时的收尾
            QtCore.QTimer.singleShot(0, self._deferred_init)

    def _deferred_init(self) -> None:
        self._apply_bg()
        self._refresh_devices()

    def toggle_drawer(self, on: bool) -> None:
        if on:
            self.settings
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
        elif hasattr(self, "_settings_dialog"):
            self._settings_dialog.hide()

    def _open_background_settings(self):
        self.btn_settings.setChecked(True)
        self.settings.nav.setCurrentRow(2)

    def _get_drawer_width(self) -> int:
        return self.drawer_frame.maximumWidth()

    def _set_drawer_width(self, w: int) -> None:
        w = max(0, int(w))
        self.drawer_frame.setMinimumWidth(w)
        self.drawer_frame.setMaximumWidth(w)

    drawerWidth = QtCore.Property(int, _get_drawer_width, _set_drawer_width)

    def drawer_width(self) -> int:
        """抽屉该多宽：按设置面板内容实际需要的宽度算，别让右边的内容被裁掉。"""
        need = 402
        panel = self._settings
        if panel is not None:
            inner = panel.widget()
            if inner is not None:
                need = max(need, inner.minimumSizeHint().width() + 40)
        limit = max(402, self.width() - 420)   # 给左边的字幕区留出合理宽度
        return max(402, min(need, limit))

    def _toggle_ontop(self, on: bool) -> None:
        self.cfg.set("ui.window.on_top", bool(on))
        flags = self.windowFlags()
        if on:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
        if self._ui_ready:
            self.toast_msg("窗口已置顶（再次点「置顶」可取消）" if on else "已取消窗口置顶")

    def _ensure_overlay(self) -> OverlayWindow:
        if self.overlay is None:
            self.overlay = OverlayWindow(self.cfg)
            self.overlay.interaction_changed.connect(self._on_overlay_interaction)
        return self.overlay

    def _on_overlay_interaction(self):
        if self._settings:
            self._settings.sync_overlay_controls()
        blocker = QtCore.QSignalBlocker(self.btn_overlay)
        self.btn_overlay.setChecked(bool(self.cfg.get("overlay.enabled", False)))
        del blocker

    def _restore_overlay_controls(self):
        self.cfg.set("overlay.click_through", False)
        self._ensure_overlay().refresh()
        self.btn_overlay.setChecked(True)
        self._on_overlay_interaction()

    def _toggle_overlay(self, on: bool) -> None:
        self.cfg.set("overlay.enabled", bool(on))
        ov = self._ensure_overlay()
        ov.refresh()
        ov.setVisible(bool(on))
        if self._settings:
            self._settings.sync_overlay_controls()
        if on:
            self._refresh_overlay()
            if self._ui_ready:
                self.toast_msg("悬浮字幕：拖动顶部移动，滚轮回看历史，A− / A＋调字号；设置中可调整底色与透明度")

    def _refresh_overlay(self) -> None:
        if self.overlay is None or not self.overlay.isVisible():
            return
        self.overlay.set_segments(self.transcript.session_segments(), self._live_jp, self._live_zh)

    # ==================================================================
    def toggle_start(self) -> None:
        if self.pipeline is not None and self.pipeline.isRunning():
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        from ..core.pipeline import Pipeline

        if self._closing:
            return
        if self.text_translation.is_busy():
            self._restart_requested = True
            self._on_status("文字翻译完成后开始聆听…")
            return
        if self._retiring_pipelines:
            self._restart_requested = True
            self._on_status("正在释放上一轮模型…")
            return
        if self.pipeline is not None and self.pipeline.isRunning():
            return
        if not self._preserve_session_for_restart:
            if not self._flush_history(state='stopped'):
                self.toast_msg('上一场字幕尚未保存，已保留当前会话。请检查历史保存位置后再开始。')
                return
            self._session_id=None
            self._session_timeline = _SessionTimeline()
            self.transcript.clear_all()
            self._failed_translation_ids.clear()
            self._live_jp=self._live_zh=''
            self.lbl_count.setText('0 条字幕')
            if self.overlay:self.overlay._reset()
        else:
            self._session_timeline.offset = None
        self._preserve_session_for_restart=False
        self._ensure_session()
        self._flush_history(state='recording')
        self.transcript.placeholder.setVisible(False)
        self.pipeline = Pipeline(self.cfg)
        self.pipeline.sig_status.connect(self._on_status)
        self.pipeline.sig_error.connect(self._on_error)
        self.pipeline.sig_partial.connect(self._on_partial)
        self.pipeline.sig_segment.connect(self._on_segment)
        self.pipeline.sig_live_zh.connect(self._on_live_zh)
        self.pipeline.sig_final_zh.connect(self._on_final_zh)
        self.pipeline.sig_level.connect(self.level.set_level)
        self.pipeline.sig_stats.connect(self._on_stats)
        self.pipeline.sig_translate_info.connect(self._on_translate_info)
        self.pipeline.sig_translate_queue.connect(self._on_translation_queue)
        self.pipeline.sig_translate_failed.connect(self._on_translation_failed)
        self.pipeline.finished.connect(self._pipeline_finished)
        self.pipeline.start()
        self._sync_text_translation_state()
        self.btn_retry.setEnabled(True)
        self.btn_start.setText("停止聆听")
        self._on_status("正在启动…")

    def _stop_pipeline(self) -> None:
        self._flush_history(state='stopped')
        self._restart_requested = False
        if self.pipeline is not None:
            old = self.pipeline
            for name in ("sig_status", "sig_error", "sig_partial", "sig_segment", "sig_live_zh",
                         "sig_final_zh", "sig_level", "sig_stats", "sig_translate_info",
                         "sig_translate_queue", "sig_translate_failed"):
                try:
                    getattr(old, name).disconnect()
                except RuntimeError:
                    pass
            self._retiring_pipelines.append(old)
            old.stop(max_wait=0)
            self.pipeline = None
        self.btn_start.setText("开始聆听")
        self.btn_retry.setEnabled(False)
        self._on_status("已停止")
        self.lbl_translate_queue.setText('翻译已停止')
        self.lbl_translate_queue.setStyleSheet('')
        self.level.set_level(0.0)
        self._sync_text_translation_state()

    def _restart_if_running(self) -> None:
        if self.pipeline is None or not self.pipeline.isRunning():
            return
        # Even a silent first epoch fixes the recording origin. The retiring
        # capture object will no longer be available when its successor starts.
        self._session_timeline.bind(self.pipeline)
        self._preserve_session_for_restart=True
        self._stop_pipeline()
        self._restart_requested = True

    def _pipeline_finished(self) -> None:
        old = self.sender()
        if old in self._retiring_pipelines:
            self._retiring_pipelines.remove(old)
        if self.pipeline is old:
            self.pipeline = None
            self.btn_start.setText("开始聆听")
        old.deleteLater()
        self._sync_text_translation_state()
        if self._restart_requested and not self._closing and not self._retiring_pipelines:
            self._restart_requested = False
            QtCore.QTimer.singleShot(0, self._start_pipeline)

    # ---- 管线事件 ------------------------------------------------------
    def _status_color(self,text):
        dark=self.cfg.get('ui.theme','light')=='dark'
        colors = {
            "正在启动…": "#f6bc65" if dark else "#a86b06",
            "正在聆听…": "#55d891" if dark else "#15803d",
            "已停止": "#9aa3b5" if dark else "#657087",
            "启动失败": "#ff6b6b",
        }
        color = colors.get(text, "#8db7ff" if dark else "#3563b7")
        if "正在加载" in text or "正在打开" in text:
            color = "#f6bc65" if dark else "#a86b06"
        return color

    def _on_status(self, text: str) -> None:
        color=self._status_color(text)
        self.lbl_state.setText(f"● {text}")
        self.lbl_state.setStyleSheet(f"color: {color};")
        if "失败" in text or "出错" in text:
            self.toast_msg(text)
        elif "降级" in text or "不可用" in text:
            self.toast_msg(text + "（在 设置 → 翻译服务 里可换回或修复）")

    def _on_error(self, text: str) -> None:
        self.toast_msg(text)
        from ..paths import LOG_PATH
        try:
            with LOG_PATH.open('a', encoding='utf-8') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {text}\n')
        except OSError:
            pass
        self.lbl_state.setToolTip(text)

    def _on_partial(self, text: str) -> None:
        previous = self._live_jp
        self._live_jp = text
        if text:
            self._ensure_session()
            if self._live_zh and previous and not text.startswith(previous):
                # A rewrite invalidates even a translation already painted in
                # the UI. Its revision-tagged clear may itself be stale.
                self._live_zh = ""
                self.transcript.set_live_zh("")
            self.transcript.set_live_jp(text)
        else:
            # A completed/cancelled draft must not lend its Chinese translation
            # to the next utterance while that utterance is still being decoded.
            self._live_zh = ""
            self.transcript.clear_live()
        self._refresh_overlay()

    def _on_live_zh(self, sid: str, zh: str) -> None:
        if (self.pipeline is None or not self.pipeline._active
                or self.pipeline._live['sid'] != sid):
            return
        self._live_zh = zh
        self.transcript.set_live_zh(zh)
        self._refresh_overlay()

    def _on_segment(self, seg: Segment) -> None:
        sender = self.sender()
        if self.pipeline is None or (sender is not None and sender is not self.pipeline):
            return
        clock = self._session_timeline
        offset = clock.bind(self.pipeline)
        if offset < 0 and seg.t1 + offset <= 0:
            return  # Recognition completed after Clear for audio entirely before it.
        seg=replace(seg, t0=max(0.0, seg.t0 + offset), t1=max(0.0, seg.t1 + offset))
        clock.last_end = max(clock.last_end, seg.t1)
        # The producer clears this utterance's preview before queueing ASR.
        # A live row still present now is newer, even if the words repeat.
        self.transcript.commit_segment(seg, adopt_live=False)
        current_finished=self.transcript._live is None
        self.lbl_count.setText(f"{len(self.transcript.session_segments())} 条字幕")
        if current_finished:self._live_jp=self._live_zh=''
        self._refresh_overlay()
        self._ensure_session()
        self._history_dirty[seg.sid]=self.transcript._history_by_sid[seg.sid]
        self._flush_history(clear_draft=current_finished)

    def _on_final_zh(self, sid: str, zh: str) -> None:
        self.transcript.set_zh_for(sid, zh)
        saved=self.transcript._history_by_sid.get(sid)
        if saved is not None:self._history_dirty[sid]=saved
        if self.overlay is not None:
            segment = self.transcript._history_by_sid.get(sid)
            if segment is not None:
                self.overlay.update_segment(sid, segment.jp, segment.zh)

    def _on_stats(self, stats: dict) -> None:
        self.lbl_stats.setText(
            f"校对 {stats.get('asr_ms', 0)} ms · RTF {stats.get('rtf', 0)} · 语句 {stats.get('audio_s', 0)} s")

    def _on_translation_queue(self, state):
        pending = int(state.get("pending", 0))
        failed = int(state.get("failed", 0))
        age = int(state.get("oldest_s", 0))
        audio_pending = int(state.get("audio_pending", 0))
        asr_pending = int(state.get("asr_pending", 0))
        if self.pipeline is None or self.pipeline._stop.is_set():
            self.lbl_translate_queue.setText("翻译已停止")
            return
        if pending:
            text = f"翻译中 · 待完成 {pending} 条"
            if age >= 8:
                text += f" · 最早等待 {age}s"
            if failed:
                text += f" · {failed} 条待补译"
        elif asr_pending:
            text = f"识别中 · 待完成 {asr_pending} 条"
            if failed:
                text += f" · {failed} 条待补译"
        else:
            text = "暂无排队 · 正在聆听" if not failed else f"有 {failed} 条待补译 · 原文已保留"
        if audio_pending > 1:
            text += f" · 音频缓冲 {audio_pending} 块"
        if pending and asr_pending:
            text += f" · 识别待完成 {asr_pending} 条"
        self.lbl_translate_queue.setText(text)
        warning='#f6bc65' if self.cfg.get('ui.theme','light')=='dark' else '#b45309'
        self.lbl_translate_queue.setStyleSheet(f"color: {warning};" if age >= 8 or failed else "")
        self.lbl_translate_queue.setToolTip("显示音频、整句识别与翻译排队情况；暂无排队不代表零延迟，断句和推理仍需要时间。超时译文保留原文，可在空闲时补译。")

    def _on_translation_failed(self, sid, error):
        self._failed_translation_ids.add(sid)
        segment = self.transcript._history_by_sid.get(sid)
        if segment is not None:
            segment.done = False
            self._history_dirty[sid] = segment
        line = self.transcript._map.get(sid)
        if line is not None:
            line.zh.setToolTip(error)
            line.zh.snap("译文未完成 · 可点补译重试")
        if self.overlay is not None:
            if segment is not None:
                self.overlay.update_segment(sid, segment.jp,
                    (segment.zh + "\n" if segment.zh else "") + "译文未完成 · 可在主窗口补译")

    def _retry_missing_translations(self):
        if self.pipeline is None or not self.pipeline.isRunning():
            self.toast_msg("开始聆听后可补译本次会话中未完成的字幕")
            return
        count = 0
        for segment in self.transcript.session_segments():
            if not segment.done or segment.sid in self._failed_translation_ids:
                if self.pipeline.enqueue_translation(segment.sid, segment.jp, [], retry=True):
                    count += 1
        if count:
            self.toast_msg(f"已安排 {count} 条补译；其余未完成字幕可在队列空闲后再次补译")
        elif any(not segment.done or segment.sid in self._failed_translation_ids
                 for segment in self.transcript.session_segments()):
            self.toast_msg("未完成字幕正在排队或队列已满，请稍后再补译")
        else:
            self.toast_msg("本次会话的译文均已完成")

    def _on_translate_info(self, info: dict) -> None:
        sid = info.get("sid")
        if sid:
            segment = self.transcript._history_by_sid.get(sid)
            if segment is not None:
                segment.done = True
                self._history_dirty[sid]=segment
                self._flush_history()
            self._failed_translation_ids.discard(sid)
            line = self.transcript._map.get(sid)
            if line is not None:
                line.zh.setToolTip("")
        if info.get('engine'):
            self.lbl_engine.setText('识别：日语专用 CPU · 翻译：' + info['engine'].upper())
            reason = info.get('engine_reason', '')
            self.lbl_engine.setToolTip(reason)
            if reason and reason != getattr(self, '_last_engine_warning', ''):
                self._last_engine_warning = reason
                self.toast_msg(f"首选引擎暂不可用，当前使用 {info['engine'].upper()}。详情见底部引擎提示。")
        elif info.get('provider') == 'llamacpp':
            from ..core import llama_engine
            running = [s.variant.upper() for s in list(llama_engine._SERVERS.values())
                        if s.mmproj is None and s._proc is not None and s._proc.poll() is None]
            if running:
                self.lbl_engine.setText('识别：日语专用 CPU · 翻译：' + running[-1])
        if info.get("warmup"):
            self.lbl_trans.setText("翻译模型已就绪")
            return
        kind = "实时" if info.get("live") else "整句"
        self.lbl_trans.setText(
            f"翻译[{kind}]：{info.get('label', '')} {info.get('seconds', '')}s")

    def _update_trans_label_from_cfg(self) -> None:
        from ..core.config import TRANSLATE_PRESETS

        pid = str(self.cfg.get("translate.provider", ""))
        label = TRANSLATE_PRESETS.get(pid, {}).get("label", pid or "-")
        model = str(self.cfg.service(pid).get("model") or "")
        if pid == "llamacpp":
            from ..core.llama_engine import resolve_model
            from ..core.readiness import MODEL_NEXT_STEP
            selected = resolve_model(model)
            if selected is None and self.cfg.get("translate.enabled", True):
                self.lbl_trans.setText("尚无可用的本地大模型。请在设置中下载或改用本地轻量翻译。")
                self.lbl_trans.setToolTip(MODEL_NEXT_STEP)
                return
            model = selected.name if selected is not None else Path(model).name
        self.lbl_trans.setToolTip("")
        suffix = ""
        if (str(self.cfg.get("translate.live_provider", "auto")) == "auto"
                and TRANSLATE_PRESETS.get(pid, {}).get("kind") == "ollama"):
            suffix = " · 草稿用极速模型"
        self.lbl_trans.setText(f"翻译：{label}" + (f" · {model}" if model else "") + suffix)

    # ---- 拖拽图片换背景 -------------------------------------------------
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                p = Path(u.toLocalFile())
                if p.suffix.lower() in self._IMG_EXTS:
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in self._IMG_EXTS:
                self.settings.add_background_file(p)
                self.toast_msg(f"已设为背景照片：{p.name}（可在设置里调整模糊与暗度）")
                e.acceptProposedAction()
                return
        e.ignore()

    # ---- 设置联动 ------------------------------------------------------
    def _on_setting_changed(self, path: str, value) -> None:
        if path.startswith(("translate.", "text_translate.")):
            self.text_translation.refresh_model_label()
        if (path in ("translate.provider", "translate.services")
                or path.startswith("translate.services.llamacpp")):
            # 先退休旧语音任务，防止它读到新配置后另开模型抢占显存。
            # 等待工作线程退出后再重置引擎，避免 UI 等待冷加载所持的锁。
            self._pending_translation_engine_reset = True
            self._restart_if_running()
            self._sync_text_translation_state()
        if path == 'history.directory':
            if self.history is None:
                try:
                    self.history=HistoryStore(value);self._history_error=''
                    self._ensure_session()
                    self._history_dirty.update({s.sid:s for s in self.transcript.session_segments()})
                    if self._flush_history():
                        self.toast_msg('自动保存已恢复')
                except Exception as exc:self.toast_msg('无法打开历史保存位置：'+str(exc))
                return
            if self.history:
                old=self.history
                try:
                    self._flush_history()
                    self.history=old.migrate_to(value)
                    if self.history is not old:old.close()
                    self.toast_msg('历史保存位置已更新；原目录保留备份')
                except Exception as exc:
                    self.history=old;self.cfg.set('history.directory',str(old.directory))
                    self.toast_msg('更换历史保存位置失败：'+str(exc))
        elif path in ("ui.theme", "ui.accent", "ui.font"):
            self._apply_theme()
        elif path.startswith("ui.bg"):
            self._schedule_bg_apply()
        elif path in ("translate.provider", "translate.services", "translate.live_provider"):
            self._update_trans_label_from_cfg()
        elif path.startswith("translate.services.llamacpp"):
            self._update_trans_label_from_cfg()
        elif path in ("ui.jp_size", "ui.zh_size", "ui.display", "ui.max_lines"):
            self.transcript.set_sizes(int(self.cfg.get("ui.jp_size", 13)), int(self.cfg.get("ui.zh_size", 19)))
            self.transcript.set_display_mode(str(self.cfg.get("ui.display", "both")))
            if path == "ui.max_lines":
                self.transcript.set_max_lines()
        elif path == "overlay.enabled":
            on = bool(self.cfg.get("overlay.enabled", False))
            if self.btn_overlay.isChecked() != on:
                self.btn_overlay.setChecked(on)
            self._toggle_overlay(on)
        elif path.startswith("overlay."):
            if self.overlay is not None:
                self.overlay.refresh()
                self.overlay.setVisible(bool(self.cfg.get("overlay.enabled", False)))
            if self._settings:
                self._settings.sync_overlay_controls()
        elif path in ("audio.source",):
            self._refresh_devices()
            self._restart_if_running()
        elif path in ("audio.device", "asr.partial", "asr.draft_mode", "asr.final_engine", "vad.threshold",
                      "vad.min_silence_ms", "asr.max_segment_s", "asr.context_prompt",
                      "asr.cloud.base_url", "asr.cloud.model", "asr.cloud.api_key", "asr.cloud.preset"):
            if self.pipeline is not None and self.pipeline.isRunning():
                self._restart_if_running()
                self.toast_msg("设置已应用，识别已自动重启")

    # ==================================================================
    def _clear(self) -> None:
        if not self._flush_history(state='stopped'):
            self.toast_msg('字幕尚未保存，没有清空。请检查历史保存位置，或先导出字幕。')
            return
        self._session_id=None
        self._session_timeline = _SessionTimeline()
        self._session_timeline.clear_at(self.pipeline)
        self.transcript.clear_all()
        self.lbl_count.setText("0 条字幕")
        self._failed_translation_ids.clear()
        self._live_jp = self._live_zh = ""
        self._refresh_overlay()

    def _ensure_session(self):
        if self.history and self._session_id is None:
            try:self._session_id=self.history.begin()
            except Exception as exc:
                if str(exc)!=self._history_error:
                    self._history_error=str(exc);self.toast_msg('创建历史场次失败：'+str(exc))
        return self._session_id

    def _flush_history(self,state=None,clear_draft=False) -> bool:
        # A failed commit must not let callers attach _history_dirty to a new session.
        if (state is None and getattr(self, '_history_busy_store', None) is self.history
                and time.monotonic() < getattr(self, '_history_retry_at', 0.0)):
            return False
        if self._session_id is None or self.history is None:
            if not (self._history_dirty or self._live_jp or self._live_zh):
                return True
            if self.history is None or self._ensure_session() is None:
                return False
        try:
            draft=None
            draft_key = (self._session_id, self._live_jp, self._live_zh)
            saved_draft = getattr(self, '_history_saved_draft', None)
            if (self._live_jp or self._live_zh) and draft_key != saved_draft and not clear_draft:
                clock = getattr(self, '_session_timeline', None)
                t = clock.position(self.pipeline) if clock is not None else 0.0
                draft=Segment(sid='__draft__',jp=self._live_jp,zh=self._live_zh,t0=t,t1=t)
            elif (not self._live_jp and not self._live_zh and saved_draft is not None
                  and saved_draft[0] == self._session_id and saved_draft != draft_key):
                clear_draft = True
            if not self._history_dirty and draft is None and state is None and not clear_draft:return True
            self.history.write(self._session_id,snapshot(self._history_dirty.values()),draft=draft,
                               clear_draft=clear_draft,state=state)
            self._history_dirty.clear()
            self._history_saved_draft = ((self._session_id, '', '') if clear_draft else draft_key)
            self._history_error = ''
            self._history_busy_store = None
            return True
        except Exception as exc:
            code = getattr(exc, 'sqlite_errorcode', 0) or 0
            if (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                self._history_busy_store = self.history
                self._history_retry_at = time.monotonic() + 0.6
            if str(exc)!=self._history_error:
                logging.exception('Automatic history save failed')
                self._history_error=str(exc);self.toast_msg('自动保存失败，请检查保存位置：'+str(exc))
            return False

    def _show_history(self):
        if self.history is None:
            self.toast_msg('历史记录暂不可用：'+self._history_error);return
        self._flush_history()
        from .history import HistoryDialog
        if self._history_dialog is None:self._history_dialog=HistoryDialog(self)
        self._history_dialog.show();self._history_dialog.raise_();self._history_dialog.activateWindow()

    def _export(self):
        self._export_segments(self.transcript.all_segments())

    def _export_segments(self,segs):
        frozen=snapshot(segs)
        if not frozen:
            self.toast_msg('还没有可导出的字幕');return
        self._flush_history()
        if self._export_dialog is not None:
            self._export_dialog.raise_();return
        directory=self.history.directory if self.history else HISTORY
        dialog=QtWidgets.QFileDialog(self,'导出字幕',str(directory/f"字幕_{time.strftime('%Y%m%d_%H%M%S')}.srt"))
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(['SRT 字幕 (*.srt)','纯文本 (*.txt)'])
        dialog.setDefaultSuffix('srt')
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog,True)
        dialog.filterSelected.connect(lambda selected:dialog.setDefaultSuffix('txt' if '*.txt' in selected else 'srt'))
        dialog.fileSelected.connect(lambda path:self._write_export_async(path,frozen))
        dialog.finished.connect(self._export_dialog_finished)
        self._export_dialog=dialog
        dialog.open()

    def _export_dialog_finished(self,_result):
        dialog=self._export_dialog;self._export_dialog=None
        if dialog is not None:dialog.deleteLater()

    def _write_export_async(self,path,segs):
        job=ExportJob(path,snapshot(segs),self)
        self._export_jobs.append(job)
        job.result.connect(self._export_result)
        job.finished.connect(self._export_finished)
        job.start()

    @QtCore.Slot(str,str)
    def _export_result(self,path,error):
        if not self._closing:self.toast_msg('导出失败：'+error if error else '已导出：'+path)

    def _export_finished(self):
        job=self.sender()
        if job in self._export_jobs:self._export_jobs.remove(job)
        job.deleteLater()

    def _to_srt(self,segs):
        return to_srt(snapshot(segs))

    # ==================================================================
    def toast_msg(self, text: str, msec: int = 4200) -> None:
        self.toast.setText(text)
        self.toast.adjustSize()
        self._place_toast()
        self.toast.show()
        self.toast.raise_()
        self._toast_timer.start(msec)

    def _place_toast(self) -> None:
        x = (self.width() - self.toast.width()) // 2
        self.toast.move(max(10, x), self.titlebar.height() + 8)

    # ---- 托盘 ----------------------------------------------------------
    def _build_tray(self) -> None:
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = None
            return
        icon_path = ASSETS / "icon.png"
        icon = QtGui.QIcon(str(icon_path)) if icon_path.exists() else self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self.tray = QtWidgets.QSystemTrayIcon(icon, self)
        menu = QtWidgets.QMenu()
        act_show = menu.addAction("显示主窗口")
        act_show.triggered.connect(lambda: (self.showNormal(), self.activateWindow()))
        act_toggle = menu.addAction("开始 / 停止聆听")
        act_toggle.triggered.connect(self.toggle_start)
        act_ov = menu.addAction("显示/隐藏悬浮字幕")
        act_ov.triggered.connect(lambda: self.btn_overlay.toggle())
        restore = menu.addAction("恢复浮窗操作（解除穿透）")
        restore.triggered.connect(self._restore_overlay_controls)
        menu.addSeparator()
        act_quit = menu.addAction("退出")
        act_quit.triggered.connect(self.close)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip(APP_NAME)
        self.tray.activated.connect(
            lambda r: self.showNormal() if r == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.show()

    # ---- 快捷键 --------------------------------------------------------
    def _install_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("F9"), self, activated=self.toggle_start)
        QtGui.QShortcut(QtGui.QKeySequence("F10"), self, activated=lambda: self.btn_overlay.toggle())
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=lambda: self.btn_settings.setChecked(False))

    # ---- 窗口移动/缩放 --------------------------------------------------
    _EDGE = 8

    def _edge_at(self, pos: QtCore.QPoint) -> str:
        edges = []
        if pos.x() < self._EDGE:
            edges.append("l")
        if pos.x() > self.width() - self._EDGE:
            edges.append("r")
        if pos.y() < self._EDGE:
            edges.append("t")
        if pos.y() > self.height() - self._EDGE:
            edges.append("b")
        return "".join(edges)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = e.position().toPoint()
        edge = self._edge_at(pos)
        if edge:
            self._resize_edges = edge
            self._resize_start = (e.globalPosition().toPoint(), self.geometry())
            e.accept()
            return
        # 标题栏空白处按下 = 拖动整个窗口（无边框窗口得自己实现，否则只能拉边缘改大小）
        if self.titlebar.geometry().contains(pos):
            self._start_move(e.globalPosition().toPoint())
            e.accept()

    def _start_move(self, global_pos: QtCore.QPoint) -> None:
        """开始拖动窗口：优先交给系统（有吸附、跨屏更顺），不行就自己算坐标移动。"""
        handle = self.windowHandle()
        if handle is not None:
            try:
                if handle.startSystemMove():
                    return
            except Exception:  # noqa: BLE001
                pass
        self._drag_pos = global_pos - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        pos = e.position().toPoint()
        if self._drag_pos is not None:                 # 拖动窗口
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()
            return
        if self._resize_start and self._resize_edges:
            gp, geo = self._resize_start
            d = e.globalPosition().toPoint() - gp
            new = QtCore.QRect(geo)
            if "l" in self._resize_edges:
                new.setLeft(min(geo.left() + d.x(), geo.right() - self.minimumWidth()))
            if "r" in self._resize_edges:
                new.setRight(max(geo.right() + d.x(), geo.left() + self.minimumWidth()))
            if "t" in self._resize_edges:
                new.setTop(min(geo.top() + d.y(), geo.bottom() - self.minimumHeight()))
            if "b" in self._resize_edges:
                new.setBottom(max(geo.bottom() + d.y(), geo.top() + self.minimumHeight()))
            self.setGeometry(new)
            e.accept()
            return
        edge = self._edge_at(pos)
        cursors = {
            "l": QtCore.Qt.CursorShape.SizeHorCursor, "r": QtCore.Qt.CursorShape.SizeHorCursor,
            "t": QtCore.Qt.CursorShape.SizeVerCursor, "b": QtCore.Qt.CursorShape.SizeVerCursor,
            "lt": QtCore.Qt.CursorShape.SizeFDiagCursor, "rb": QtCore.Qt.CursorShape.SizeFDiagCursor,
            "rt": QtCore.Qt.CursorShape.SizeBDiagCursor, "lb": QtCore.Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(edge, QtCore.Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._drag_pos is not None:
            self._drag_pos = None
            self._save_geometry()
            e.accept()
            return
        if self._resize_start:
            self._resize_start = None
            self._resize_edges = ""
            self._save_geometry()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        self.bg.setGeometry(self.rect())
        QtCore.QTimer.singleShot(180, self._reflow_bg)
        btn = getattr(self, "btn_settings", None)
        anim = getattr(self, "_drawer_anim", None)
        anim_running = False
        if anim is not None:
            try:
                anim_running = anim.state() == QtCore.QAbstractAnimation.State.Running
            except RuntimeError:
                anim_running = False  # C++ 对象已析构，当作不在运行
        if btn is not None and btn.isChecked() and not anim_running:
            # 窗口变窄时重新算一次抽屉宽度，尽量不让设置内容被裁掉
            self._set_drawer_width(self.drawer_width())
        super().resizeEvent(e)

    def _reflow_bg(self) -> None:
        if self.isVisible():
            self.bg.reflow()

    def moveEvent(self, e: QtGui.QMoveEvent) -> None:
        self._save_geometry_debounced()
        super().moveEvent(e)

    def _save_geometry_debounced(self) -> None:
        if not hasattr(self, "_geo_timer"):
            self._geo_timer = QtCore.QTimer(self)
            self._geo_timer.setSingleShot(True)
            self._geo_timer.timeout.connect(self._save_geometry)
        self._geo_timer.start(600)

    def _save_geometry(self) -> None:
        self.cfg.update_many({
            "ui.window.w": self.width(), "ui.window.h": self.height(),
            "ui.window.x": self.x(), "ui.window.y": self.y(),
        })

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._closing = True          # 关窗口时别再被“自动重启”之类的定时器拉起来
        self.text_translation.shutdown()
        if self.pipeline is not None:
            self._stop_pipeline()
        if self._retiring_pipelines:
            self._on_status("正在退出，等待模型完成当前任务…")
            QtCore.QTimer.singleShot(200, self.close)
            e.ignore()
            return
        if self._export_jobs:
            QtCore.QTimer.singleShot(100,self.close)
            e.ignore();return
        if self.text_translation.is_busy():
            self._on_status("正在退出，等待文字翻译结束…")
            QtCore.QTimer.singleShot(100, self.close)
            e.ignore()
            return
        if not self._flush_history(state='closed'):
            box = QtWidgets.QMessageBox
            answer = box.warning(self, '字幕未能保存',
                '最后一次保存失败。取消退出可返回窗口，更换保存位置或导出字幕。'
                '选择丢弃并退出可能丢失尚未保存的字幕。',
                box.StandardButton.Cancel | box.StandardButton.Discard, box.StandardButton.Cancel)
            if answer != box.StandardButton.Discard:
                self._closing = False
                self.text_translation.resume_after_cancelled_shutdown()
                self._on_status('已取消退出，当前字幕仍保留')
                e.ignore()
                return
        self._history_timer.stop()
        if self._history_dialog:self._history_dialog.close()
        if self._export_dialog:self._export_dialog.reject()
        self._save_geometry()
        try:
            try:
                from ..core.llama_engine import stop_all

                stop_all()
            except Exception:  # noqa: BLE001
                pass
            if self.overlay is not None:
                self.overlay.close()
            if self.tray is not None:
                self.tray.hide()
            if self.pipeline is not None and self.pipeline.isRunning():
                self.pipeline.stop(max_wait=6.0)   # 退出时等它把模型放干净，避免留下占内存的残留进程
        finally:
            if self.history:self.history.close();self.history=None
            e.accept()
