"""Durable, incremental session records; independent of live display limits."""
from __future__ import annotations
import json, sqlite3, time, uuid, threading
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from .segment import Segment


class HistoryStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'history.sqlite3'
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY, title TEXT NOT NULL, started REAL NOT NULL,
            updated REAL NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items(session TEXT NOT NULL, sid TEXT NOT NULL,
            position INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(session,sid));
            CREATE INDEX IF NOT EXISTS item_order ON items(session,position);''')

    @contextmanager
    def _live_write(self):
        # These two transactions run from the subtitle UI. Let the existing
        # autosave timer retry contention rather than waiting in sqlite for 5s.
        # Restore the connection policy for explicit archival operations.
        with self._lock:
            timeout = int(self.db.execute('PRAGMA busy_timeout').fetchone()[0])
            self.db.execute('PRAGMA busy_timeout=0')
            try:
                with self.db:
                    if not self.db.in_transaction:
                        self.db.execute('BEGIN IMMEDIATE')
                    yield
            finally:
                self.db.execute(f'PRAGMA busy_timeout={timeout}')

    def begin(self):
        sid = uuid.uuid4().hex
        now = time.time()
        with self._live_write():
            self.db.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',
                (sid, time.strftime('会话 %Y-%m-%d %H:%M:%S'), now, now, 'recording'))
        return sid

    def write(self, session, segments=(), draft=None, clear_draft=False, state=None):
        with self._live_write():
            for seg in segments:
                existing = self.db.execute('SELECT position FROM items WHERE session=? AND sid=?', (session, seg.sid)).fetchone()
                pos = existing[0] if existing else self.db.execute('SELECT COALESCE(MAX(position),-1)+1 FROM items WHERE session=? AND sid<>?',(session,'__draft__')).fetchone()[0]
                self.db.execute('INSERT INTO items VALUES(?,?,?,?) ON CONFLICT(session,sid) DO UPDATE SET payload=excluded.payload',
                    (session,seg.sid,pos,json.dumps(asdict(seg),ensure_ascii=False)))
            if clear_draft:
                self.db.execute('DELETE FROM items WHERE session=? AND sid=?',(session,'__draft__'))
            elif draft and (draft.jp or draft.zh):
                self.db.execute('INSERT INTO items VALUES(?,?,?,?) ON CONFLICT(session,sid) DO UPDATE SET payload=excluded.payload',
                    (session,'__draft__',2147483647,json.dumps(asdict(draft),ensure_ascii=False)))
            self.db.execute('UPDATE sessions SET updated=?,state=COALESCE(?,state) WHERE id=?',(time.time(),state,session))

    def change_token(self):
        """Cheap per-connection token for polling. Neither field alone suffices:
        total_changes covers our writes; data_version covers external commits.
        Compare only tokens obtained from the same open connection.
        """
        return self.db.total_changes, self.db.execute('PRAGMA data_version').fetchone()[0]

    def sessions(self):
        return [dict(r) for r in self.db.execute('''SELECT s.*,COUNT(i.sid) AS count FROM sessions s
            JOIN items i ON i.session=s.id GROUP BY s.id ORDER BY s.started DESC''')]

    def load(self, session):
        fields = Segment.__dataclass_fields__
        return [Segment(**{k:v for k,v in json.loads(r[0]).items() if k in fields})
            for r in self.db.execute('SELECT payload FROM items WHERE session=? ORDER BY position',(session,))]

    def rename(self, session, title):
        title = title.strip()
        if not title:raise ValueError('场次名称不能为空')
        with self.db:self.db.execute('UPDATE sessions SET title=? WHERE id=?',(title,session))

    def migrate_to(self, directory):
        other = HistoryStore(directory)
        if other.path.resolve() == self.path.resolve():
            other.close();return self
        try:
            with other.db:
                for row in self.db.execute('SELECT * FROM sessions'):
                    target=other.db.execute('SELECT updated FROM sessions WHERE id=?',(row['id'],)).fetchone()
                    source_newer=target is None or row['updated']>=target[0]
                    if source_newer:
                        other.db.execute('INSERT INTO sessions VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,started=excluded.started,updated=excluded.updated,state=excluded.state',tuple(row))
                        # A completed sentence must not resurrect an obsolete draft
                        # when the user switches back to a previous save location.
                        other.db.execute("DELETE FROM items WHERE session=? AND sid='__draft__'",(row['id'],))
                    for item in self.db.execute('SELECT * FROM items WHERE session=?',(row['id'],)):
                        if source_newer:
                            other.db.execute('INSERT INTO items VALUES(?,?,?,?) ON CONFLICT(session,sid) DO UPDATE SET position=excluded.position,payload=excluded.payload',tuple(item))
                        elif item['sid']!='__draft__':
                            other.db.execute('INSERT OR IGNORE INTO items VALUES(?,?,?,?)',tuple(item))
        except Exception:
            other.close()
            raise
        return other

    def close(self):
        if self.db is not None:
            self.db.close();self.db=None
