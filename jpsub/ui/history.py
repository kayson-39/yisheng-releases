from PySide6 import QtCore,QtGui,QtWidgets


class HistoryDialog(QtWidgets.QDialog):
    def __init__(self,owner):
        super().__init__(owner)
        self.owner=owner
        self.setWindowTitle('历史记录 · 译声')
        self.resize(1050,700)
        layout=QtWidgets.QVBoxLayout(self)
        hint=QtWidgets.QLabel('按场次自动保存；选择左侧场次可搜索、回看或手动导出。')
        layout.addWidget(hint)
        row=QtWidgets.QHBoxLayout()
        self.query=QtWidgets.QLineEdit();self.query.setPlaceholderText('输入关键词，按回车查找所选场次中的原文或译文')
        self.scope=QtWidgets.QComboBox();self.scope.addItems(['所选场次','全部场次'])
        row.addWidget(self.query,1);row.addWidget(self.scope)
        layout.addLayout(row)
        splitter=QtWidgets.QSplitter()
        self.list=QtWidgets.QListWidget();self.list.setMinimumWidth(280)
        self.text=QtWidgets.QPlainTextEdit();self.text.setReadOnly(True)
        splitter.addWidget(self.list);splitter.addWidget(self.text);splitter.setStretchFactor(1,1)
        layout.addWidget(splitter,1)
        buttons=QtWidgets.QHBoxLayout()
        self.rename_button=QtWidgets.QPushButton('重命名场次')
        self.export_button=QtWidgets.QPushButton('导出所选场次')
        self.find_button=QtWidgets.QPushButton('查找下一个')
        folder=QtWidgets.QPushButton('打开保存文件夹')
        for b in (self.rename_button,self.export_button,self.find_button,folder):
            b.setAutoDefault(False);b.setDefault(False);buttons.addWidget(b)
        layout.addLayout(buttons)
        self.list.currentItemChanged.connect(self.refresh_text)
        self.query.textChanged.connect(self.refresh_text);self.scope.currentIndexChanged.connect(self.refresh_text)
        self.find_button.clicked.connect(self.find_next);self.query.returnPressed.connect(self.find_next)
        self.rename_button.clicked.connect(self.rename)
        self.export_button.clicked.connect(lambda:owner._export_segments(self.selected_segments()))
        folder.clicked.connect(lambda:QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(owner.history.directory))))
        self.timer=QtCore.QTimer(self);self.timer.setInterval(1000);self.timer.timeout.connect(self.refresh)
        self._signature=None;self._text_signature=None
        self._records=[];self._text_source_signature=None;self._live_revision=0
        self._records_store=None;self._records_token=None
        self.owner.transcript.content_changed.connect(self._live_changed)
        self.refresh();self.timer.start()

    def _live_changed(self):
        # Coalesce streaming updates until the next refresh. Do not clone an
        # entire long session once per token, or once per idle timer tick.
        self._live_revision+=1

    def selected_id(self):
        item=self.list.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def selected_segments(self):
        sid=self.selected_id()
        return self.owner.history.load(sid) if sid else self.owner.transcript.all_segments()

    def refresh(self):
        store=self.owner.history
        token=store.change_token()
        if store is self._records_store and token==self._records_token:
            self.refresh_text()  # On-screen subtitles can change before autosave.
            return
        if store is not self._records_store or token[1]!=self._records_token[1]:
            # External writers need not update our per-session timestamps.
            self._text_source_signature=None
        records=store.sessions()
        self._records_store,self._records_token=store,token
        self._records=records
        signature=[(r['id'],r['title'],r['count'],r['state']) for r in records]
        if signature!=self._signature:
            selected=self.selected_id();self.list.blockSignals(True);self.list.clear()
            item=QtWidgets.QListWidgetItem('本次会话\n正在显示的字幕');item.setData(QtCore.Qt.ItemDataRole.UserRole,None);self.list.addItem(item)
            target=item
            for r in records:
                status={'recording':'进行中','stopped':'已停止','closed':'已保存','interrupted':'上次意外结束'}.get(r['state'],r['state'])
                entry=QtWidgets.QListWidgetItem(f'{r["title"]}\n{r["count"]} 条字幕 · {status}')
                entry.setData(QtCore.Qt.ItemDataRole.UserRole,r['id']);self.list.addItem(entry)
                if r['id']==selected:target=entry
            self.list.setCurrentItem(target);self.list.blockSignals(False);self._signature=signature
        self.refresh_text()

    def refresh_text(self,*_):
        needle=self.query.text().strip();sid=self.selected_id()
        self.rename_button.setEnabled(sid is not None)
        global_search=self.scope.currentIndex()==1 and bool(needle)
        if global_search:
            revision=tuple((r['id'],r['title'],r['updated'],r['count'],r['state']) for r in self._records)
            source_signature=('all',id(self.owner.history),needle.casefold(),revision)
        elif sid is not None:
            record=next((r for r in self._records if r['id']==sid),None)
            revision=(record['updated'],record['count'],record['state']) if record else None
            source_signature=('saved',id(self.owner.history),sid,revision)
        else:
            source_signature=('live',id(self.owner.transcript),self._live_revision)
        if source_signature==self._text_source_signature:
            return
        if global_search:
            # Format one archive at a time; matching text is the only part
            # which must survive into the final multi-session document.
            groups=((r['title'],self.owner.history.load(r['id'])) for r in self._records)
        else:groups=[('',self.selected_segments())]
        folded=needle.casefold()
        parts=[]
        for title,segs in groups:
            for s in segs:
                if global_search and folded not in (s.jp+'\n'+s.zh).casefold():continue
                stamp=f'{int(s.t0//60):02}:{int(s.t0%60):02}'
                parts.append((f'【{title}】\n' if title else '')+f'[{stamp}] {s.jp}\n{s.zh or "（译文尚未完成）"}')
        content='\n\n'.join(parts) or '没有匹配的字幕记录。'
        if content!=self._text_signature:
            bar=self.text.verticalScrollBar();old=bar.value();at_end=bar.maximum()-old<8
            self.text.setPlainText(content)
            bar.setValue(bar.maximum() if at_end else old);self._text_signature=content
        self._text_source_signature=source_signature

    def find_next(self):
        needle=self.query.text().strip()
        if needle and not self.text.find(needle):
            cursor=self.text.textCursor();cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start);self.text.setTextCursor(cursor);self.text.find(needle)

    def rename(self):
        sid=self.selected_id()
        if not sid:return
        title,ok=QtWidgets.QInputDialog.getText(self,'重命名场次','场次名称：',text=self.list.currentItem().text().split('\n')[0])
        if ok and title.strip():self.owner.history.rename(sid,title);self.refresh()

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)

    def showEvent(self,event):
        super().showEvent(event)
        if hasattr(self,'timer'):self.timer.start();self.refresh()
