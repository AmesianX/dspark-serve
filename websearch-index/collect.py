#!/usr/bin/env python3
"""websearch-index 수집기 — 0층을 채운다.

두 가지를 받아온다:
  1. 고정 피드 (sources.json)
  2. **승격된 구독** — 에이전트가 두 번 이상 물은 주제. 그 주제는 이때부터
     영구히 0층에서 나오므로, 쓸수록 무차단 영역이 넓어진다.

⛔ 실패한 피드를 영원히 재시도하지 않는다. 연속 실패가 쌓이면 간격을 벌린다.
   공짜로 열어준 문을 우리가 두드려 망가뜨리면 그 문도 닫힌다.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import channels
import core

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources.json"
PARALLEL = 6          # 동시 요청 수. 메모리/대역폭 모두 작게 유지.
FEED_GAP = 900.0      # 같은 피드를 다시 받기까지 최소 15분
SUB_GAP = 1800.0      # 같은 구독을 다시 받기까지 최소 30분
FAIL_BACKOFF = 1800.0 # 실패 1회당 추가로 벌리는 간격


def load_sources(cx) -> int:
    data = json.loads(SOURCES.read_text(encoding="utf-8"))
    n = 0
    for name, url in data["feeds"]:
        cur = cx.execute("SELECT url FROM feeds WHERE url=?", (url,)).fetchone()
        if cur is None:
            cx.execute("INSERT INTO feeds(url,name) VALUES(?,?)", (url, name))
            n += 1
        else:
            cx.execute("UPDATE feeds SET name=? WHERE url=?", (name, url))
    cx.commit()
    return n


def due_feeds(cx, force: bool):
    rows = cx.execute("SELECT * FROM feeds").fetchall()
    now = time.time()
    out = []
    for r in rows:
        gap = FEED_GAP + min(r["fail"], 8) * FAIL_BACKOFF
        if force or now - r["last_run"] >= gap:
            out.append(r)
    return out


def run_feeds(cx, force=False, verbose=True):
    todo = due_feeds(cx, force)
    if not todo:
        return 0, 0
    results = {}

    def job(r):
        try:
            return r["url"], channels.rss(r["url"], source=r["name"], kind="feed"), ""
        except Exception as e:
            return r["url"], None, f"{type(e).__name__}: {str(e)[:90]}"

    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for url, rows, err in ex.map(job, todo):
            results[url] = (rows, err)

    total_new = 0
    ok_n = 0
    now = time.time()
    for r in todo:
        rows, err = results[r["url"]]
        if rows is None:
            cx.execute("UPDATE feeds SET last_run=?, fail=fail+1, note=? WHERE url=?",
                       (now, err, r["url"]))
            if verbose:
                print(f"  ✗ {r['name']:18s} {err}")
            continue
        new = core.upsert(cx, rows)
        total_new += new
        ok_n += 1
        cx.execute("UPDATE feeds SET last_run=?, ok=ok+1, fail=0, note='' WHERE url=?",
                   (now, r["url"]))
        if verbose:
            print(f"  ✓ {r['name']:18s} {len(rows):4d}건 수신 / 신규 {new:4d}")
    cx.commit()
    return ok_n, total_new


def run_subs(cx, force=False, verbose=True):
    now = time.time()
    rows = cx.execute("SELECT * FROM subs").fetchall()
    todo = [r for r in rows if force or now - r["last_run"] >= SUB_GAP]
    if not todo:
        return 0, 0

    def job(r):
        try:
            return r["q"], channels.google_news(r["q"], r["lang"], 50), ""
        except Exception as e:
            return r["q"], None, f"{type(e).__name__}: {str(e)[:70]}"

    total_new = 0
    ok_n = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for q, got, err in ex.map(job, todo):
            if got is None:
                if verbose:
                    print(f"  ✗ 구독 {q[:34]:34s} {err}")
                continue
            new = core.upsert(cx, got)
            total_new += new
            ok_n += 1
            cx.execute("UPDATE subs SET last_run=?, runs=runs+1, gained=gained+? WHERE q=?",
                       (time.time(), new, q))
            if verbose:
                print(f"  ✓ 구독 {q[:34]:34s} {len(got):3d}건 / 신규 {new:3d}")
    cx.commit()
    return ok_n, total_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="한 번만 돌고 끝")
    ap.add_argument("--force", action="store_true", help="간격 무시하고 전부")
    ap.add_argument("--interval", type=float, default=600.0)
    a = ap.parse_args()

    cx = core.connect()
    added = load_sources(cx)
    if added:
        print(f"[collect] 신규 피드 {added}개 등록")

    while True:
        t0 = time.time()
        print(f"[collect] {time.strftime('%H:%M:%S')} 수집 시작")
        f_ok, f_new = run_feeds(cx, force=a.force)
        s_ok, s_new = run_subs(cx, force=a.force)
        total = cx.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
        print(f"[collect] 피드 {f_ok}개/신규 {f_new} · 구독 {s_ok}개/신규 {s_new} "
              f"· 색인 총 {total:,}건 · {time.time()-t0:.1f}s")
        if a.once:
            return 0
        time.sleep(max(60.0, a.interval))


if __name__ == "__main__":
    sys.exit(main())
