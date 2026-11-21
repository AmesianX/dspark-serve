#!/usr/bin/env python3
"""websearch-index 0층 — 로컬 색인.

설계 요지
---------
검색 서버를 "질의 → 외부엔진 중계기"로 보지 않는다. 그렇게 보면 질의를 던질
때마다 차단 확률이 올라가고, 쓸수록 더 막힌다. 여기서는 뒤집는다:

  - 이미 수집해 둔 것은 로컬에서 꺼낸다 (0층, 차단이라는 개념 자체가 없다)
  - 모르는 것만 밖에 묻는다 (1층 이상)
  - 두 번 물은 주제는 구독으로 승격해서 영구히 0층으로 내린다

⛔ 한글 주의: FTS5 unicode61 은 교착어의 조사를 못 떼므로 ('추론' 이 '추론을' 에
   매치 안 됨) 질의 토큰에 prefix '*' 를 붙여야 한다. make_match() 가 그 역할.
"""
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit

DB_PATH = Path(__file__).resolve().parent / "state" / "index.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS docs (
  id        INTEGER PRIMARY KEY,
  url       TEXT UNIQUE NOT NULL,
  title     TEXT NOT NULL DEFAULT '',
  body      TEXT NOT NULL DEFAULT '',
  host      TEXT NOT NULL DEFAULT '',
  source    TEXT NOT NULL DEFAULT '',   -- 어느 창구에서 왔는가
  kind      TEXT NOT NULL DEFAULT '',   -- feed | sub | live | page
  published REAL,                       -- 원문 발행시각(epoch)
  fetched   REAL NOT NULL               -- 우리가 넣은 시각
);
CREATE INDEX IF NOT EXISTS docs_pub  ON docs(published DESC);
CREATE INDEX IF NOT EXISTS docs_kind ON docs(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
  title, body, content='docs', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
  INSERT INTO docs_fts(rowid,title,body) VALUES (new.id,new.title,new.body);
END;
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts,rowid,title,body) VALUES('delete',old.id,old.title,old.body);
END;
CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts,rowid,title,body) VALUES('delete',old.id,old.title,old.body);
  INSERT INTO docs_fts(rowid,title,body) VALUES (new.id,new.title,new.body);
END;

-- 질의 → 구독 승격 대장
CREATE TABLE IF NOT EXISTS subs (
  q         TEXT PRIMARY KEY,
  lang      TEXT NOT NULL DEFAULT 'ko',
  created   REAL NOT NULL,
  last_run  REAL NOT NULL DEFAULT 0,
  runs      INTEGER NOT NULL DEFAULT 0,
  gained    INTEGER NOT NULL DEFAULT 0    -- 이 구독이 가져온 신규 문서 수
);

-- 질의 로그: 승격 판단 + 3층 사용률(KPI) 계산용
CREATE TABLE IF NOT EXISTS qlog (
  id    INTEGER PRIMARY KEY,
  q     TEXT NOT NULL,
  norm  TEXT NOT NULL,
  ts    REAL NOT NULL,
  layer TEXT NOT NULL,
  n     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS qlog_norm ON qlog(norm);
CREATE INDEX IF NOT EXISTS qlog_ts   ON qlog(ts DESC);

-- 수집 대상 피드
CREATE TABLE IF NOT EXISTS feeds (
  url      TEXT PRIMARY KEY,
  name     TEXT NOT NULL DEFAULT '',
  last_run REAL NOT NULL DEFAULT 0,
  ok       INTEGER NOT NULL DEFAULT 0,
  fail     INTEGER NOT NULL DEFAULT 0,
  note     TEXT NOT NULL DEFAULT ''
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(str(path), timeout=30.0)
    cx.row_factory = sqlite3.Row
    cx.executescript(SCHEMA)
    return cx


# ── 질의 정규화 / FTS 질의 생성 ────────────────────────────────────────────
_WORD = re.compile(r"[0-9A-Za-z_]+|[가-힣]+|[぀-ヿ]+|[一-鿿]+")
# FTS5 에서 특별한 의미를 갖는 글자들. 사용자 질의에 섞여 들어오면 구문오류가 난다.
_FTS_BAD = re.compile(r'["*():^-]')


def norm_query(q: str) -> str:
    """승격 판단용 정규화 — 대소문자/어순/중복 무시."""
    toks = sorted({t.lower() for t in _WORD.findall(q) if len(t) > 1})
    return " ".join(toks)


def tokens(q: str) -> list[str]:
    return [t for t in _WORD.findall(_FTS_BAD.sub(" ", q)) if len(t) > 1]


def make_match(q: str, mode: str = "and") -> str:
    """사용자 질의 → FTS5 MATCH 식.

    ⛔ 한글 조사 때문에 모든 토큰에 prefix '*' 를 붙인다.
       '추론' 은 '추론을' 에 매치되지 않지만 '추론*' 은 된다.
    ⛔ 전 토큰 AND 만 쓰면 색인에 있는데도 0건이 나와 불필요하게 밖으로 나간다.
       search() 가 AND → OR 순으로 두 번 시도한다.
    """
    toks = tokens(q)
    if not toks:
        return ""
    parts = [f'"{t}"*' for t in toks]
    return (" OR " if mode == "or" else " ").join(parts)


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


# ── 적재 ──────────────────────────────────────────────────────────────────
def upsert(cx: sqlite3.Connection, rows) -> int:
    """문서 적재. 이미 있는 URL 은 본문이 길어질 때만 갱신한다.

    반환: 새로 들어온 문서 수 (구독의 '수확량' 측정에 쓴다)
    """
    now = time.time()
    new = 0
    for r in rows:
        url = (r.get("url") or "").strip()
        if not url or len(url) > 2000:
            continue
        title = (r.get("title") or "").strip()[:500]
        body = (r.get("body") or "").strip()[:20000]
        if not title and not body:
            continue
        cur = cx.execute("SELECT id, length(body) AS lb FROM docs WHERE url=?", (url,)).fetchone()
        if cur is None:
            cx.execute(
                "INSERT INTO docs(url,title,body,host,source,kind,published,fetched)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (url, title, body, host_of(url), r.get("source", ""),
                 r.get("kind", ""), r.get("published"), now),
            )
            new += 1
        elif len(body) > (cur["lb"] or 0):
            cx.execute("UPDATE docs SET title=?, body=?, fetched=? WHERE id=?",
                       (title or "", body, now, cur["id"]))
    cx.commit()
    return new


# ── 검색 ──────────────────────────────────────────────────────────────────
# ⛔⛔ 적합도 관문. 이게 없으면 OR 폴백이 흔한 토큰 하나만 맞아도 통과시킨다.
#     실측 사고: '국내 반도체 수출 규제 영향' → '수능 연기'(영향), '샤오미18 국내
#     출시'(국내) 가 0층 상위로 나왔다. 0층이 쓰레기를 자신있게 내놓으면 1층으로
#     넘어가지도 않고 에이전트가 틀린 답을 한다 — 차단당하는 것보다 나쁘다.
COVER_MIN = 0.6     # 질의 토큰의 60% 이상이 문서에 실제로 나타나야 한다
COVER_MIN_TOKENS = 2


def coverage(text: str, toks: list[str]) -> float:
    if not toks:
        return 0.0
    low = (text or "").lower()
    return sum(1 for t in toks if t.lower() in low) / len(toks)


def search(cx: sqlite3.Connection, q: str, n: int = 20, fresh_days: float = 0.0):
    """AND 로 먼저 찾고, 모자라면 OR 로 넓히되 **적합도 관문을 통과한 것만** 쓴다."""
    toks = tokens(q)
    strict = _search_one(cx, q, n, fresh_days, "and")   # AND 는 정의상 커버리지 1.0
    if len(strict) >= n or not toks:
        return strict[:n]

    need = max(COVER_MIN_TOKENS, round(len(toks) * COVER_MIN))
    need = min(need, len(toks))
    seen = {r["url"] for r in strict}
    for r in _search_one(cx, q, n * 5, fresh_days, "or"):
        if r["url"] in seen:
            continue
        hit = sum(1 for t in toks if t.lower() in f'{r["title"]} {r["content"]}'.lower())
        if hit < need:
            continue
        r["partial"] = f"{hit}/{len(toks)}"
        strict.append(r)
        seen.add(r["url"])
        if len(strict) >= n:
            break
    return strict


def _search_one(cx: sqlite3.Connection, q: str, n: int, fresh_days: float, mode: str):
    m = make_match(q, mode)
    if not m:
        return []
    sql = """
      SELECT d.url, d.title, d.body, d.host, d.source, d.kind, d.published,
             bm25(docs_fts, 4.0, 1.0) AS score
      FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
      WHERE docs_fts MATCH ?
    """
    args = [m]
    if fresh_days > 0:
        sql += " AND d.published IS NOT NULL AND d.published > ?"
        args.append(time.time() - fresh_days * 86400)
    # bm25 는 작을수록 좋다. 신선도를 약하게 섞는다(하루당 0.15점 가산).
    sql += " ORDER BY score - COALESCE((d.published - ?) / 86400.0, 0) * 0.15 LIMIT ?"
    args += [time.time(), n]
    try:
        rows = cx.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        out.append({
            "title": r["title"],
            "url": r["url"],
            "content": (r["body"] or "")[:400],
            "host": r["host"],
            "source": r["source"],
            "published": r["published"],
            "layer": 0,
        })
    return out


def log_query(cx, q: str, layer: str, n: int) -> None:
    cx.execute("INSERT INTO qlog(q,norm,ts,layer,n) VALUES(?,?,?,?,?)",
               (q[:300], norm_query(q), time.time(), layer, n))
    cx.commit()
