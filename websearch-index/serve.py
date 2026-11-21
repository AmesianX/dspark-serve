#!/usr/bin/env python3
"""websearch-index 서버 — 0층 + 1층, 그리고 '승격'.

질의 처리
---------
  1. 0층(로컬 FTS5)을 먼저 본다. 충분하면 거기서 끝. 수 ms, 차단 개념 없음.
  2. 모자라면 1층(기계용 공개 창구)을 병렬로 때린다. 결과는 **색인에 적재**되므로
     같은 질의는 다음부터 0층에서 나온다.
  3. 같은 주제를 두 번 이상 물으면 **구독으로 승격**한다. 그때부터 그 주제는
     수집기가 알아서 계속 긁어오고, 영원히 0층이다.

→ 기존 검색서버는 쓸수록 레이트리밋이 쌓여 **더** 막힌다.
  이 구조는 쓸수록 0층이 넓어져 **덜** 막힌다. 그게 설계의 전부다.

⛔ 바인딩은 127.0.0.1 + 172.18.0.1(nginx-network 게이트웨이)뿐. 0.0.0.0 금지.
⛔ X-Index-Token 필수(/health 제외).
"""
import json
import os
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import channels
import core

ROOT = Path(__file__).resolve().parent
TOKEN = os.environ.get("INDEX_TOKEN", "")
PORT = int(os.environ.get("INDEX_PORT", "8097"))
BINDS = ["127.0.0.1", "172.18.0.1"]

MIN_LOCAL = int(os.environ.get("INDEX_MIN_LOCAL", "6"))   # 0층이 이만큼 주면 밖에 안 나간다
PROMOTE_AT = int(os.environ.get("INDEX_PROMOTE_AT", "2")) # 같은 주제 N번 → 구독 승격
LIVE_PARALLEL = 7

_cx_local = threading.local()
_write_lock = threading.Lock()


def cx():
    if getattr(_cx_local, "cx", None) is None:
        _cx_local.cx = core.connect()
    return _cx_local.cx


def is_korean(s: str) -> bool:
    return any("가" <= ch <= "힣" for ch in s)


# ── 1층 ───────────────────────────────────────────────────────────────────
def live_fanout(q: str) -> tuple[list, dict]:
    """기계용 공개 창구를 병렬로 때린다. 한 곳이 죽어도 나머지는 산다."""
    names = list(channels.LIVE)
    out, report = [], {}

    def job(name):
        t0 = time.time()
        try:
            rows = channels.LIVE[name](q)
            return name, rows, f"{len(rows)}건 {time.time()-t0:.2f}s"
        except Exception as e:
            return name, [], f"✗ {type(e).__name__} {time.time()-t0:.2f}s"

    with ThreadPoolExecutor(max_workers=LIVE_PARALLEL) as ex:
        for name, rows, note in ex.map(job, names):
            report[name] = note
            out.extend(rows)
    return out, report


def promote(q: str) -> str:
    """같은 주제를 PROMOTE_AT 번 이상 물었으면 구독으로 올린다."""
    norm = core.norm_query(q)
    if not norm:
        return ""
    c = cx()
    n = c.execute("SELECT COUNT(*) n FROM qlog WHERE norm=?", (norm,)).fetchone()["n"]
    if n < PROMOTE_AT:
        return ""
    if c.execute("SELECT q FROM subs WHERE q=?", (q,)).fetchone():
        return ""
    # 정규화가 같은 다른 표현이 이미 구독 중이면 중복 등록하지 않는다
    for r in c.execute("SELECT q FROM subs").fetchall():
        if core.norm_query(r["q"]) == norm:
            return ""
    with _write_lock:
        c.execute("INSERT OR IGNORE INTO subs(q,lang,created) VALUES(?,?,?)",
                  (q, "ko" if is_korean(q) else "en", time.time()))
        c.commit()
    threading.Thread(target=_sub_first_run, args=(q,), daemon=True).start()
    return q


def _sub_first_run(q: str):
    """승격 직후 한 번 당겨온다 — 다음 질의부터 바로 0층이 되도록."""
    try:
        c = core.connect()
        rows = channels.google_news(q, "ko" if is_korean(q) else "en", 50)
        new = core.upsert(c, rows)
        c.execute("UPDATE subs SET last_run=?, runs=runs+1, gained=gained+? WHERE q=?",
                  (time.time(), new, q))
        c.commit()
        print(f"[승격] '{q}' → 구독 등록, 즉시 {len(rows)}건 수확(신규 {new})", flush=True)
    except Exception as e:
        print(f"[승격] '{q}' 첫 수확 실패: {type(e).__name__}: {e}", flush=True)


def dedupe(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        u = (r.get("url") or "").split("#")[0].rstrip("/")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def do_search(q: str, n: int, live: bool = True) -> dict:
    """live=False 면 0층만 본다 (SearXNG 기본셋용, 2ms).

    ⛔ live=False 여도 질의는 반드시 기록하고 승격을 돌린다. 그래야 평소의
       일반 검색 트래픽이 공짜로 승격 엔진을 먹여살린다 — 에이전트가 자주 묻는
       주제가 저절로 구독이 되고 0층이 그쪽으로 자란다. 이 구조의 핵심.
    """
    t0 = time.time()
    c = cx()
    local = core.search(c, q, n)
    if not live:
        with _write_lock:
            core.log_query(c, q, "0L", len(local))
        promoted = promote(q)
        return {"query": q, "layer": 0, "results": local[:n],
                "took_ms": round((time.time() - t0) * 1000, 1),
                "promoted": promoted, "note": "0층 전용(local) — 외부 요청 0회"}
    if len(local) >= MIN_LOCAL:
        with _write_lock:
            core.log_query(c, q, "0", len(local))
        promoted = promote(q)
        return {"query": q, "layer": 0, "results": local[:n],
                "took_ms": round((time.time() - t0) * 1000, 1),
                "promoted": promoted, "note": "로컬 색인 적중 — 외부 요청 0회"}

    rows, report = live_fanout(q)
    with _write_lock:
        new = core.upsert(c, rows)
    merged = dedupe(core.search(c, q, n) + [
        {"title": r["title"], "url": r["url"], "content": (r["body"] or "")[:400],
         "host": core.host_of(r["url"]), "source": r["source"],
         "published": r["published"], "layer": 1} for r in rows])
    with _write_lock:
        core.log_query(c, q, "1", len(merged))
    promoted = promote(q)
    return {"query": q, "layer": 1, "results": merged[:n],
            "took_ms": round((time.time() - t0) * 1000, 1),
            "indexed_new": new, "channels": report, "promoted": promoted,
            "note": "0층 부족 → 기계용 공개창구 질의. 결과는 색인에 적재됨"}


def stats() -> dict:
    c = cx()
    g = lambda s, *a: c.execute(s, a).fetchone()
    total = g("SELECT COUNT(*) n FROM docs")["n"]
    by_kind = {r["kind"]: r["n"] for r in
               c.execute("SELECT kind, COUNT(*) n FROM docs GROUP BY kind")}
    q24 = c.execute("SELECT layer, COUNT(*) n FROM qlog WHERE ts>? GROUP BY layer",
                    (time.time() - 86400,)).fetchall()
    mix = {r["layer"]: r["n"] for r in q24}
    tot = sum(mix.values()) or 1
    subs = c.execute("SELECT q, runs, gained FROM subs ORDER BY gained DESC LIMIT 10").fetchall()
    feeds_bad = c.execute("SELECT name, note FROM feeds WHERE fail>0").fetchall()
    return {
        "docs": total,
        "docs_by_kind": by_kind,
        "subs": g("SELECT COUNT(*) n FROM subs")["n"],
        "queries_24h": sum(mix.values()),
        "layer_mix_24h": {k: f"{v} ({v*100//tot}%)" for k, v in sorted(mix.items())},
        "layer0_hit_rate_24h": f"{mix.get('0',0)*100//tot}%",
        "top_subs": [dict(r) for r in subs],
        "failing_feeds": [dict(r) for r in feeds_bad],
        "db_mb": round(core.DB_PATH.stat().st_size / 1e6, 1) if core.DB_PATH.exists() else 0,
    }


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        qs = urllib.parse.parse_qs(u.query)
        path = u.path.rstrip("/") or "/"
        if path == "/health":
            return self._send({"ok": True, "port": PORT, "docs": stats()["docs"]})
        if TOKEN and self.headers.get("X-Index-Token") != TOKEN:
            return self._send({"error": "unauthorized"}, 401)
        try:
            if path == "/search":
                q = (qs.get("q") or [""])[0].strip()
                if not q:
                    return self._send({"error": "q required"}, 400)
                n = min(int((qs.get("n") or ["20"])[0]), 100)
                live = (qs.get("live") or ["1"])[0] not in ("0", "false", "no")
                return self._send(do_search(q, n, live))
            if path == "/stats":
                return self._send(stats())
            if path == "/subs":
                c = cx()
                return self._send({"subs": [dict(r) for r in
                                            c.execute("SELECT * FROM subs ORDER BY created")]})
            return self._send({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._send({"error": f"{type(e).__name__}: {e}"}, 500)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    core.connect()  # 스키마 보장
    servers = []
    for b in BINDS:
        try:
            s = Server((b, PORT), H)
            servers.append(s)
            threading.Thread(target=s.serve_forever, daemon=True).start()
            print(f"[index] listen {b}:{PORT}", flush=True)
        except OSError as e:
            print(f"[index] {b}:{PORT} 바인드 실패: {e}", flush=True)
    if not servers:
        raise SystemExit("바인드 전부 실패")
    print(f"[index] 0층 {stats()['docs']:,}건 적재됨. 토큰 {'설정됨' if TOKEN else '없음(⛔)'}",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
