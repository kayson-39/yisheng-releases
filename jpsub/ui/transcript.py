# -*- coding: utf-8 -*-
"""连续字幕文档：跨句选择、增量更新和稳定的历史阅读位置。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.segment import Segment
from ..core.export import snapshot
from .theme import PALETTES


def _utf16_length(text: str) -> int:
    """QTextCursor 的位置使用 UTF-16 单元，而不是 Python 字符数。"""
    return len(text.encode("utf-16-le")) // 2


def _document_text(text: str) -> str:
    # 一条字幕始终占一个 block，句内换行使用软换行。
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\u2028")


class _TranslationDisplay:
    """兼容主窗口的错误提示接口，不把错误提示保存成真实译文。"""

    def __init__(self, line: _CaptionLine):
        self.line = line
        self.tooltip = ""

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def snap(self, text: str) -> None:
        self.line.error_text = text
        self.line.owner._update_line(self.line)


class _CaptionLine:
    def __init__(self, seg: Segment, owner: TranscriptView):
        self.seg = seg
        self.owner = owner
        self.block = QtGui.QTextBlock()
        self.rendered = ("", "")
        self.copy_values = ("", "")
        self.error_text = ""
        self.block_key = -1
        self.zh = _TranslationDisplay(self)

    def set_zh(self, text: str, typing: bool = True) -> None:
        if text:
            self.seg.zh = text
            self.error_text = ""
            self.owner._update_line(self)

    def set_jp(self, text: str) -> None:
        self.seg.jp = text
        self.owner._update_line(self)


@dataclass
class _ReadingAnchor:
    cursor: QtGui.QTextCursor
    y: int
    line: _CaptionLine | None
    offset: int


@dataclass
class _TextPosition:
    line: _CaptionLine
    offset: int


class TranscriptView(QtWidgets.QTextBrowser):
    """每句对应一个稳定文本块，选择和滚动不依赖独立标签控件。"""

    following_changed = QtCore.Signal(bool)
    copied = QtCore.Signal(str)
    content_changed = QtCore.Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._history: list[Segment] = []
        self._history_by_sid: dict[str, Segment] = {}
        self._content_by_sid: dict[str, tuple[bool, bool]] = {}
        self._content_counts = [0, 0]
        self.lines: list[_CaptionLine] = []
        self._map: dict[str, _CaptionLine] = {}
        self._block_lines: dict[int, _CaptionLine] = {}
        self._next_block_key = 0
        self._live: _CaptionLine | None = None
        self._selected_line: _CaptionLine | None = None
        self._following = True
        self._held_tail = False  # New committed rows only in history while reading a full page.
        self._mutating = False
        self._menu_active = False
        self._pending_lines: set[str] = set()
        self._pending_anchor: _ReadingAnchor | None = None
        self._sizes = (int(cfg.get("ui.jp_size", 13)), int(cfg.get("ui.zh_size", 19)))
        self._display = str(cfg.get("ui.display", "both"))
        self._reflow = QtCore.QTimer(self)
        self._reflow.setSingleShot(True)
        self._reflow.timeout.connect(self._finish_reflow)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QTextBrowser { background: transparent; border: none; padding: 0px; }")
        self.document().setDocumentMargin(8)
        self.setMouseTracking(True)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                                     | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
        bar = self.verticalScrollBar()
        bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        bar.actionTriggered.connect(self._scroll_action)
        bar.sliderMoved.connect(self._slider_moved)
        bar.installEventFilter(self)
        self.selectionChanged.connect(self._selection_changed)
        self.placeholder = QtWidgets.QLabel(
            "点击「开始聆听」后，这里会实时显示日语原文与中文翻译。\n"
            "先播放在看的直播，让声音从扬声器/耳机正常出声即可。\n"
            "默认使用随附的本地模型。需要调整字幕或使用云端翻译时，可打开「设置」。",
            self.viewport())
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setObjectName("muted")
        self.placeholder.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placeholder.setGeometry(self.viewport().rect())

    def _all_lines(self) -> list[_CaptionLine]:
        return self.lines + ([self._live] if self._live is not None else [])

    def _line_for_block(self, block: QtGui.QTextBlock) -> _CaptionLine | None:
        if not block.isValid():
            return None
        line = self._block_lines.get(block.userState())
        return line if line is not None and line.block == block else None

    def _bind_block(self, line: _CaptionLine) -> None:
        # Keep Python metadata outside the document's native ownership graph.
        if line.block_key < 0:
            line.block_key = self._next_block_key
            self._next_block_key += 1
        self._block_lines[line.block_key] = line
        line.block.setUserState(line.block_key)

    def _line_at(self, point: QtCore.QPoint) -> _CaptionLine | None:
        cursor = self.cursorForPosition(point)
        block = cursor.block()
        if not block.isValid():
            return None
        bounds = self.document().documentLayout().blockBoundingRect(block)
        document_y = point.y() + self.verticalScrollBar().value()
        if not bounds.top() <= document_y <= bounds.bottom():
            return None
        return self._line_for_block(block)

    def _capture_anchor(self) -> _ReadingAnchor:
        cursor = self.cursorForPosition(QtCore.QPoint(8, 1))
        block = cursor.block()
        # 顶部有上一句的残行时，优先固定下一条完整可见的字幕。
        # 否则上一句补译变长，会把原来的短句末尾误当成新长句的开头。
        if self.cursorRect(QtGui.QTextCursor(block)).top() < 0 and block.next().isValid():
            next_cursor = QtGui.QTextCursor(block.next())
            if self.cursorRect(next_cursor).top() < self.viewport().height():
                cursor = next_cursor
        line = self._line_for_block(cursor.block())
        return _ReadingAnchor(cursor, self.cursorRect(cursor).top(), line,
                              cursor.position() - cursor.block().position())

    def _text_position(self, position: int) -> _TextPosition | None:
        block = self.document().findBlock(position)
        line = self._line_for_block(block)
        return _TextPosition(line, position - block.position()) if line is not None else None

    def _restore_selection(self, saved: tuple[_TextPosition, _TextPosition] | None) -> None:
        if saved is None:
            return
        anchor, position = saved
        if not anchor.line.block.isValid() or not position.line.block.isValid():
            return
        cursor = QtGui.QTextCursor(self.document())
        cursor.setPosition(anchor.line.block.position() + min(anchor.offset, anchor.line.block.length() - 1))
        cursor.setPosition(position.line.block.position() + min(position.offset, position.line.block.length() - 1),
                           QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def _restore_anchor(self, anchor: _ReadingAnchor | None) -> None:
        self.document().documentLayout().documentSize()
        bar = self.verticalScrollBar()
        if self._following and not self._menu_active and not self.textCursor().hasSelection():
            self._set_bottom_margin(8)
            bar.setValue(bar.maximum())
        elif anchor is not None and not anchor.cursor.isNull():
            if anchor.line is not None and anchor.line.block.isValid():
                anchor.cursor = QtGui.QTextCursor(anchor.line.block)
                anchor.cursor.setPosition(anchor.line.block.position() + min(anchor.offset, anchor.line.block.length() - 1))
            target = bar.value() + self.cursorRect(anchor.cursor).top() - anchor.y
            # 原来不足一屏的记录前补历史时，需允许底部留白才能固定阅读行。
            # 每次按当前视口重新计算，缩窗或新字幕填满后不保留多余空白。
            frame = self.document().rootFrame().frameFormat()
            height = self.document().documentLayout().documentSize().height()
            margin = max(8, target + self.viewport().height() - height + frame.bottomMargin() + 1) if target > 0 else 8
            self._set_bottom_margin(margin)
            bar.setValue(target)

    def _set_bottom_margin(self, margin: float) -> None:
        root = self.document().rootFrame()
        fmt = root.frameFormat()
        if abs(fmt.bottomMargin() - margin) > 0.5:
            fmt.setBottomMargin(margin)
            root.setFrameFormat(fmt)
            self.document().documentLayout().documentSize()

    @contextmanager
    def _change(self, *, preserve_selection: bool = True) -> Iterator[None]:
        if self._mutating:
            yield
            return
        anchor = self._pending_anchor or self._capture_anchor()
        cursor = self.textCursor()
        start, end = self._text_position(cursor.anchor()), self._text_position(cursor.position())
        saved_selection = (start, end) if preserve_selection and cursor.hasSelection() and start is not None and end is not None else None
        self._mutating = True
        try:
            editing = QtGui.QTextCursor(self.document())
            editing.beginEditBlock()
            try:
                yield
            finally:
                editing.endEditBlock()
            self._restore_selection(saved_selection)
            self._restore_anchor(anchor)
        finally:
            self._mutating = False
        self._pending_anchor = anchor
        self._reflow.start(0)
        self._highlight_line()
        self.content_changed.emit()

    def _finish_reflow(self) -> None:
        anchor, self._pending_anchor = self._pending_anchor, None
        self._mutating = True
        try:
            self._restore_anchor(anchor)
        finally:
            self._mutating = False
        self.viewport().update()

    def _cancel_pending_scroll(self) -> None:
        self._reflow.stop()
        self._pending_anchor = None

    def _text_format(self, language: str) -> QtGui.QTextCharFormat:
        colors = PALETTES.get(str(self.cfg.get("ui.theme", "light")), PALETTES["light"])
        fmt = QtGui.QTextCharFormat()
        font = str(self.cfg.get("ui.font", "Microsoft YaHei UI"))
        if font and font != "系统默认":
            fmt.setFontFamilies([font])
        fmt.setFontPointSize(max(9 if language == "jp" else 10,
                                self._sizes[0 if language == "jp" else 1]))
        fmt.setFontWeight(QtGui.QFont.Weight.Normal if language == "jp"
                          else QtGui.QFont.Weight.DemiBold)
        fmt.setForeground(QtGui.QColor(colors["muted" if language == "jp" else "text"]))
        return fmt

    def _visible_values(self, line: _CaptionLine) -> tuple[str, str]:
        zh = line.seg.zh
        if line.error_text:
            zh = (zh + "\n" if zh else "") + line.error_text
        return (line.seg.jp if self._display in ("both", "jp") else "",
                zh if self._display in ("both", "zh") else "")

    def _format_line(self, line: _CaptionLine) -> None:
        if not line.block.isValid():
            return
        cursor = QtGui.QTextCursor(line.block)
        fmt = QtGui.QTextBlockFormat()
        fmt.setTopMargin(4)
        fmt.setBottomMargin(14)
        fmt.setLineHeight(110, QtGui.QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cursor.setBlockFormat(fmt)
        jp, zh = line.rendered
        if jp:
            cursor.setPosition(line.block.position())
            cursor.setPosition(cursor.position() + _utf16_length(_document_text(jp)),
                               QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(self._text_format("jp"))
        if zh:
            start = line.block.position() + (_utf16_length(_document_text(jp)) + 1 if jp else 0)
            cursor.setPosition(start)
            cursor.setPosition(start + _utf16_length(_document_text(zh)),
                               QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(self._text_format("zh"))

    def _selected_lines(self) -> list[_CaptionLine]:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return [self._selected_line] if self._selected_line is not None else []
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        return [line for line in self._all_lines() if line.block.isValid()
                and line.block.position() < end
                and line.block.position() + line.block.length() - 1 >= start]

    def _render_line(self, line: _CaptionLine, *, force: bool = False) -> None:
        self._remember_content(line.seg)
        values = self._visible_values(line)
        if line is self._live and not line.block.isValid() and self._hold_new_rows():
            self._held_tail = True
            return
        if line.block.isValid() and not force:
            cursor = self.textCursor()
            if cursor.hasSelection():
                pos = line.block.position()
                protected = pos < cursor.selectionEnd() and pos + line.block.length() - 1 >= cursor.selectionStart()
            else:
                protected = self._menu_active and self._selected_line is line
            if protected:
                self._pending_lines.add(line.seg.sid)
                return
        if not line.block.isValid():
            cursor = QtGui.QTextCursor(self.document())
            if self._live is not None and line is not self._live and self._live.block.isValid():
                cursor.setPosition(self._live.block.position())
                cursor.insertBlock()
                self._live.block = cursor.block()
                self._bind_block(self._live)
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousBlock)
            else:
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                if self.document().firstBlock().userState() >= 0:
                    cursor.insertBlock()
            line.block = cursor.block()
            self._bind_block(line)
        elif values == line.rendered:
            line.copy_values = (line.seg.jp, line.seg.zh)
            self._format_line(line)
            return
        old_jp, old_zh = line.rendered
        jp, zh = values
        if old_jp == jp and old_zh and zh.startswith(old_zh):
            cursor = QtGui.QTextCursor(line.block)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock)
            cursor.insertText(_document_text(zh[len(old_zh):]), self._text_format("zh"))
            line.rendered = values
            line.copy_values = (line.seg.jp, line.seg.zh)
            return
        cursor = QtGui.QTextCursor(line.block)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock,
                            QtGui.QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        jp, zh = values
        if jp:
            cursor.insertText(_document_text(jp), self._text_format("jp"))
            if zh:
                cursor.insertText("\u2028")
        if zh:
            cursor.insertText(_document_text(zh), self._text_format("zh"))
        line.rendered = values
        line.copy_values = (line.seg.jp, line.seg.zh)
        self._format_line(line)

    def _update_line(self, line: _CaptionLine) -> None:
        with self._change():
            self._render_line(line)

    def _remove_line(self, line: _CaptionLine) -> None:
        cursor = QtGui.QTextCursor(line.block)
        if line.block.next().isValid():
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)
        elif line.block.previous().isValid():
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousCharacter,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)
        else:
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)
            line.block.setUserState(-1)
        cursor.removeSelectedText()
        self._block_lines.pop(line.block_key, None)
        line.block = QtGui.QTextBlock()
        self._map.pop(line.seg.sid, None)
        self._pending_lines.discard(line.seg.sid)
        if self._selected_line is line:
            self._selected_line = None

    def _hold_new_rows(self) -> bool:
        if self._held_tail and not self._following:
            return True
        keep = int(self.cfg.get("ui.max_lines", 100))
        return (keep > 0 and len(self.lines) >= keep and
                (not self._following or self._menu_active or self.textCursor().hasSelection()))

    def _trim(self) -> None:
        # 阅读旧记录、选择文字和菜单操作时，绝不能删除当前阅读依赖的文本块。
        if not self._following or self.textCursor().hasSelection() or self._menu_active:
            return
        keep = int(self.cfg.get("ui.max_lines", 100))
        if keep > 0:
            while len(self.lines) > keep:
                self._remove_line(self.lines.pop(0))

    def _ensure_live(self) -> _CaptionLine:
        if self._live is None:
            self._live = _CaptionLine(Segment(sid="__live__", jp=""), self)
        self.live_placeholder_hide()
        return self._live

    def live_placeholder_hide(self) -> None:
        self.placeholder.setVisible(False)

    def set_live_jp(self, text: str) -> None:
        if text:
            with self._change():
                line = self._ensure_live()
                line.seg.jp = text
                self._render_line(line)

    def set_live_zh(self, text: str) -> None:
        if text or self._live is not None:
            with self._change():
                line = self._ensure_live()
                line.seg.zh = text
                self._render_line(line)

    def clear_live(self) -> None:
        if self._live is not None:
            with self._change():
                if self._live.block.isValid():
                    self._remove_line(self._live)
                self._live = None

    def commit_segment(self, seg: Segment, *, adopt_live: bool = True) -> _CaptionLine:
        with self._change():
            self.live_placeholder_hide()
            if seg.sid in self._history_by_sid:
                line = self._map.get(seg.sid)
                if line is not None:
                    line.seg.jp, line.seg.zh = seg.jp, seg.zh
                    self._render_line(line)
                    return line
                saved = self._history_by_sid[seg.sid]
                saved.jp, saved.zh = seg.jp, seg.zh
                self._remember_content(saved)
                return _CaptionLine(saved, self)
            if adopt_live and self._live is not None and self._live.seg.jp in (seg.live_jp, seg.jp):
                line = self._live
                old_sid = line.seg.sid
                line.seg.sid = seg.sid
                line.seg.jp = seg.jp
                line.seg.zh = seg.zh or line.seg.zh
                line.seg.t0, line.seg.t1 = seg.t0, seg.t1
                line.seg.rtf, line.seg.asr_seconds = seg.rtf, seg.asr_seconds
                line.seg.live_jp, line.seg.provider, line.seg.done = seg.live_jp, seg.provider, seg.done
                if line.block.isValid():
                    self._bind_block(line)
                self._pending_lines.discard(old_sid)
                self._live = None
            else:
                line = _CaptionLine(seg, self)
            self._history.append(line.seg)
            self._history_by_sid[seg.sid] = line.seg
            if not line.block.isValid() and self._hold_new_rows():
                # Keep the actual session for copying/saving and for catch-up.
                # Don't evict the reader's selected/visible blocks to render
                # a new row that they cannot currently see.
                self._held_tail = True
                self._remember_content(line.seg)
                return line
            self.lines.append(line)
            self._map[seg.sid] = line
            self._render_line(line)
            self._trim()
            return line

    def set_zh_for(self, sid: str, text: str) -> None:
        if not text:
            return
        with self._change():
            saved = self._history_by_sid.get(sid)
            if saved is not None:
                saved.zh = text
                self._remember_content(saved)
            line = self._map.get(sid)
            if line is not None:
                line.error_text = ""
                self._render_line(line)

    def set_sizes(self, jp: int, zh: int) -> None:
        with self._change():
            self._sizes = (jp, zh)
            for line in self._all_lines():
                self._format_line(line)

    def set_display_mode(self, mode: str) -> None:
        if mode == self._display:
            return
        with self._change(preserve_selection=False):
            # 主动切换显示语言会改变文本范围，不能继续保留旧语言的选区。
            cursor = self.textCursor()
            cursor.clearSelection()
            self.setTextCursor(cursor)
            self._display = mode
            self._pending_lines.clear()
            for line in self._all_lines():
                self._render_line(line, force=True)

    def set_max_lines(self) -> None:
        if self._held_tail and not self._following:
            # The current reading page is frozen. Thaw using the new limit.
            return
        with self._change():
            # 扩大显示范围时仅在前面补齐，不重建已有文档和选区。
            keep = int(self.cfg.get("ui.max_lines", 100))
            wanted = self._history if keep <= 0 else self._history[-keep:]
            missing = [seg for seg in wanted if seg.sid not in self._map]
            for seg in reversed(missing):
                line = _CaptionLine(seg, self)
                cursor = QtGui.QTextCursor(self.document())
                if self.lines or self._live is not None:
                    first = self.lines[0] if self.lines else self._live
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
                    cursor.insertBlock()
                    first.block = cursor.block()
                    self._bind_block(first)
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousBlock)
                    line.block = cursor.block()
                    self._bind_block(line)
                self.lines.insert(0, line)
                self._map[seg.sid] = line
                self._render_line(line)
            self._trim()

    def clear_all(self) -> None:
        self._cancel_pending_scroll()
        self._mutating = True
        try:
            self.clear()
            self._selected_line = None
            self._pending_lines.clear()
            self.lines.clear()
            self._map.clear()
            self._block_lines.clear()
            self._history.clear()
            self._history_by_sid.clear()
            self._content_by_sid.clear()
            self._content_counts = [0, 0]
            self._live = None
            self._held_tail = False
            self._set_following(True)
            self.placeholder.setVisible(True)
        finally:
            self._mutating = False
        self.setExtraSelections([])
        self.content_changed.emit()

    def load_segments(self, segments) -> None:
        self.clear_all()
        with self._change():
            self._history = list(snapshot(segments))
            self._history_by_sid = {seg.sid: seg for seg in self._history}
            for seg in self._history:
                self._remember_content(seg)
            keep = int(self.cfg.get("ui.max_lines", 100))
            wanted = self._history if keep <= 0 else self._history[-keep:]
            for seg in wanted:
                line = _CaptionLine(seg, self)
                self.lines.append(line)
                self._map[seg.sid] = line
                self._render_line(line)
            self.placeholder.setVisible(not bool(self._history))

    def _set_following(self, following: bool) -> None:
        if self._following != following:
            self._following = following
            if following and self._held_tail:
                self._restore_recent_rows()
            self.following_changed.emit(following)

    def _restore_recent_rows(self) -> None:
        # Resuming is an explicit end to the frozen reading page. Rebuild only
        # its replacement (normally the last 100), never the whole archive.
        self._held_tail = False
        keep = int(self.cfg.get("ui.max_lines", 100))
        wanted = self._history[-keep:] if keep > 0 else self._history
        with self._change(preserve_selection=False):
            live, old = self._live, self._map
            self._live = self._selected_line = None
            self.clear()
            self.lines = []
            self._map = {}
            self._block_lines.clear()
            self._pending_lines.clear()
            for seg in wanted:
                line = old.get(seg.sid) or _CaptionLine(seg, self)
                line.block, line.rendered = QtGui.QTextBlock(), ("", "")
                self.lines.append(line)
                self._map[seg.sid] = line
                self._render_line(line)
            if live is not None:
                self._live = live
                live.block, live.rendered = QtGui.QTextBlock(), ("", "")
                self._render_line(live)

    def _user_scroll(self) -> None:
        if not self._mutating:
            self._set_following(self.verticalScrollBar().maximum()
                                - self.verticalScrollBar().value() < 8
                                and not self.textCursor().hasSelection() and not self._menu_active)

    def _scroll_action(self, action: int) -> None:
        if self._mutating:
            return
        self._cancel_pending_scroll()
        bar = self.verticalScrollBar()
        self._set_following(bar.maximum() - bar.sliderPosition() < 8
                            and not self.textCursor().hasSelection() and not self._menu_active)

    def _slider_moved(self, value: int) -> None:
        self._cancel_pending_scroll()
        self._set_following(self.verticalScrollBar().maximum() - value < 8
                            and not self.textCursor().hasSelection() and not self._menu_active)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self._cancel_pending_scroll()
        delta = event.pixelDelta().y() or event.angleDelta().y()
        if delta > 0:
            self._set_following(False)
        super().wheelEvent(event)
        if delta < 0:
            self._user_scroll()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_current()
            event.accept()
            return
        self._cancel_pending_scroll()
        if event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_PageUp, QtCore.Qt.Key.Key_Home):
            self._set_following(False)
        super().keyPressEvent(event)
        if event.matches(QtGui.QKeySequence.StandardKey.MoveToEndOfDocument):
            # Qt 只保证文末光标可见，末段下边距仍可能留在视口之外。
            # Ctrl+End 明确表示回到最新；带 Shift 的选到文末仍走正常选区逻辑。
            self.jump_latest()
            return
        if event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_PageDown, QtCore.Qt.Key.Key_End):
            self._user_scroll()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._cancel_pending_scroll()
        line = self._line_at(event.position().toPoint())
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if line is not None and line not in self._selected_lines():
                cursor = self.cursorForPosition(event.position().toPoint())
                self.setTextCursor(cursor)
                self._selected_line = line
                self._set_following(False)
            self._highlight_line()
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._selected_line = line
            if line is not None:
                self._set_following(False)
        super().mousePressEvent(event)
        self._highlight_line()

    def _selection_changed(self) -> None:
        if self._mutating:
            return
        if self.textCursor().hasSelection():
            self._cancel_pending_scroll()
            self._set_following(False)
        else:
            self._flush_pending_lines()
        self._highlight_line()

    def _flush_pending_lines(self) -> None:
        if not self._pending_lines or self._menu_active:
            return
        pending, self._pending_lines = self._pending_lines, set()
        with self._change():
            for line in self._all_lines():
                if line.seg.sid in pending:
                    self._render_line(line)

    def _highlight_line(self) -> None:
        selected = []
        if self._selected_line is not None and self._selected_line.block.isValid() and not self.textCursor().hasSelection():
            extra = QtWidgets.QTextEdit.ExtraSelection()
            extra.cursor = QtGui.QTextCursor(self._selected_line.block)
            extra.cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock,
                                      QtGui.QTextCursor.MoveMode.KeepAnchor)
            color = QtGui.QColor(str(self.cfg.get("ui.accent", "#d95482")))
            color.setAlpha(32)
            extra.format.setBackground(color)
            extra.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            selected.append(extra)
        self.setExtraSelections(selected)

    @staticmethod
    def _copy_values(values: list[tuple[str, str]], language: str) -> str:
        if language not in ("jp", "zh", "both"):
            raise ValueError("不支持的复制语言：" + language)
        groups = []
        for jp, zh in values:
            texts = (jp,) if language == "jp" else (zh,) if language == "zh" else (jp, zh)
            group = "\n".join(text for text in texts if text)
            if group:
                groups.append(group)
        return "\n\n".join(groups)

    def _copy_text(self, text: str, description: str) -> None:
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.copied.emit("已复制" + description)

    def copy_current(self) -> None:
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._copy_text(cursor.selectedText().replace("\u2029", "\n").replace("\u2028", "\n"), "选中文字")
        elif self._selected_line is not None:
            self._copy_text(self._copy_values([self._selected_line.copy_values], "both"), "选中行双语")

    def _remember_content(self, seg: Segment) -> None:
        """Maintain copy-button availability without scanning the whole session.

        Live drafts are checked directly; only committed records enter the counts.
        These counts also include records trimmed out of the visible document.
        """
        if seg.sid not in self._history_by_sid:
            return
        old = self._content_by_sid.get(seg.sid, (False, False))
        new = (bool(seg.jp), bool(seg.zh))
        for index in (0, 1):
            self._content_counts[index] += int(new[index]) - int(old[index])
        self._content_by_sid[seg.sid] = new

    def has_content(self, language: str = "both") -> bool:
        if language not in ("jp", "zh", "both"):
            raise ValueError("不支持的复制语言：" + language)
        jp, zh = self._content_counts
        if self._live is not None:
            jp = jp or bool(self._live.seg.jp)
            zh = zh or bool(self._live.seg.zh)
        return bool(jp if language == "jp" else zh if language == "zh" else jp or zh)

    def copy_all(self, language: str) -> None:
        segments = self._history + ([self._live.seg] if self._live is not None else [])
        text = self._copy_values([(seg.jp, seg.zh) for seg in segments], language)
        description = {"jp": "本次会话全部日文", "zh": "本次会话全部中文", "both": "本次会话全部双语"}[language]
        self._copy_text(text, description)

    def build_context_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        # 固定打开菜单时看到的内容。选中字幕后收到的修订仍留在数据层。
        values = [line.copy_values for line in self._selected_lines()]
        for title, language in (("复制选中行日文", "jp"), ("复制选中行中文", "zh"), ("复制选中行双语", "both")):
            text = self._copy_values(values, language)
            action = menu.addAction(title)
            action.setEnabled(bool(text))
            action.triggered.connect(lambda checked=False, text=text, title=title:
                                     self._copy_text(text, title.removeprefix("复制")))
        return menu

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if event.reason() == QtGui.QContextMenuEvent.Reason.Mouse and self._line_at(event.pos()) is None:
            event.accept()
            return
        if not self._selected_lines():
            event.accept()
            return
        self._set_following(False)
        self._menu_active = True
        menu = self.build_context_menu()
        def closed() -> None:
            self._menu_active = False
            self._flush_pending_lines()
            menu.deleteLater()
        menu.aboutToHide.connect(closed)
        menu.popup(event.globalPos())
        event.accept()

    def eventFilter(self, obj, event):
        if obj is self.verticalScrollBar():
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self._cancel_pending_scroll()
        return super().eventFilter(obj, event)

    def viewportEvent(self, event):
        if event.type() == QtCore.QEvent.Type.ToolTip:
            line = self._line_at(event.pos())
            if line is not None and line.zh.tooltip:
                QtWidgets.QToolTip.showText(event.globalPos(), line.zh.tooltip, self)
                return True
        return super().viewportEvent(event)

    def jump_latest(self) -> None:
        self._cancel_pending_scroll()
        with self._change(preserve_selection=False):
            cursor = self.textCursor()
            cursor.clearSelection()
            self.setTextCursor(cursor)
            self._selected_line = None
            self._flush_pending_lines()
            self._set_following(True)
            self._trim()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        anchor = self._capture_anchor() if hasattr(self, "_history") else None
        super().resizeEvent(event)
        if hasattr(self, "placeholder"):
            self.placeholder.setGeometry(self.viewport().rect())
        if hasattr(self, "_reflow"):
            self._restore_anchor(anchor)
            self._pending_anchor = anchor
            self._reflow.start(0)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if hasattr(self, "_history") and event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.FontChange):
            with self._change():
                for line in self._all_lines():
                    self._format_line(line)

    def session_segments(self) -> list[Segment]:
        """供主窗口及悬浮窗增量读取；调用方不修改列表本身。"""
        return self._history

    def all_segments(self) -> list[Segment]:
        return list(snapshot(self._history))

    def recent_segments(self, count: int) -> list[Segment]:
        return self._history[-max(1, count):]

