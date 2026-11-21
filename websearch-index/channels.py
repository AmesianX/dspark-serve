#!/usr/bin/env python3
"""websearch-index 1층 — 기계용 공개 창구.

여기 있는 것들의 공통점: **봇을 위해 만들어진 문**이다. 키가 없고, 차단이 없고,
robots.txt 가 막지 않는다. 검색엔진의 사람용 문(/search)이 막히는 것과 무관하다.

실측 2026-11-18 (.66 에서):
  GoogleNews RSS  'vllm site:github.com'  → 100건 전부 GitHub, 0.65s
  GoogleNews RSS  'AI when:1h'            → 100건, 수 분 전 기사, 0.65s
  GoogleNews RSS  난수질의                → 0건  ← 정직성 통과
  HN Algolia / arXiv / GitHub / StackExchange / Wikipedia → 전부 200
  ⛔ Reddit .json 403, CommonCrawl CDX 504, Bing News RSS 빈 응답 → 쓰지 않는다
"""
import json
import time
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime

import feedparser

UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/159.0.0.0 Safari/537.36")
TIMEOUT = 30

# ⛔ 호스트별 최소 간격. 이 창구들은 우리를 차단하지 않지만, 연속으로 때리면
#    일시적으로 거절한다(arXiv 는 공식적으로 3초 간격을 요구). 공짜로 열어둔
#    문을 우리가 망가뜨리지 않기 위한 것이기도 하다.
_MIN_GAP = {
    "export.arxiv.org": 3.0,
    "api.stackexchange.com": 1.0,   # 무인증 일일 할당량 300건 — 아껴 쓴다
    "api.github.com": 1.0,          # 무인증 분당 10회
}
_last_hit: dict[str, float] = {}


def _throttle(url: str) -> None:
    host = urllib.parse.urlsplit(url).hostname or ""
    gap = _MIN_GAP.get(host)
    if not gap:
        return
    wait = gap - (time.time() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.time()


def _get(url: str, accept: str = "*/*", tries: int = 2) -> bytes:
    last = None
    for i in range(tries):
        _throttle(url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:          # 과부하/일시거절은 한 번 더 기다렸다 재시도
            last = e
            if i + 1 < tries:
                time.sleep(2.0 + 2.0 * i)
    raise last


def _epoch(entry) -> float | None:
    for k in ("published", "updated", "created"):
        v = entry.get(k)
        if not v:
            continue
        try:
            return parsedate_to_datetime(v).timestamp()
        except Exception:
            pass
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                return time.mktime(v)
            except Exception:
                pass
    return None


def _strip(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", html or "").replace("&nbsp;", " ").strip()


# ── 범용 RSS/Atom ─────────────────────────────────────────────────────────
def rss(url: str, source: str = "", kind: str = "feed", limit: int = 200):
    raw = _get(url, "application/rss+xml, application/atom+xml, application/xml;q=0.9")
    d = feedparser.parse(raw)
    out = []
    for e in d.entries[:limit]:
        link = e.get("link") or ""
        if not link:
            continue
        body = _strip(e.get("summary") or
                      (e.get("content", [{}])[0].get("value") if e.get("content") else ""))
        out.append({
            "url": link,
            "title": _strip(e.get("title") or "")[:500],
            "body": body[:4000],
            "source": source or (d.feed.get("title") or url)[:120],
            "kind": kind,
            "published": _epoch(e),
        })
    return out


# ── Google News RSS: 임의 질의를 받는 '검색' 창구 ─────────────────────────
#   연산자 지원 확인됨: site:  when:1h/1d  intitle:  OR  "구문"
def google_news(q: str, lang: str = "ko", limit: int = 100):
    if lang == "ko":
        qs = "hl=ko&gl=KR&ceid=KR:ko"
    else:
        qs = "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&{qs}"
    rows = rss(url, source="GoogleNews", kind="sub", limit=limit)
    for r in rows:
        # 제목 꼬리의 ' - 매체명' 을 출처로 올린다
        if " - " in r["title"]:
            head, _, tail = r["title"].rpartition(" - ")
            if 0 < len(tail) <= 40:
                r["title"], r["source"] = head, tail
    return rows


# ── Hacker News (Algolia) ─────────────────────────────────────────────────
def hackernews(q: str, limit: int = 30):
    url = ("https://hn.algolia.com/api/v1/search?query="
           f"{urllib.parse.quote(q)}&tags=story&hitsPerPage={limit}")
    d = json.loads(_get(url))
    out = []
    for h in d.get("hits", []):
        link = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        out.append({
            "url": link,
            "title": (h.get("title") or "")[:500],
            "body": (h.get("story_text") or h.get("comment_text") or "")[:2000],
            "source": f"HN({h.get('points', 0)}p)",
            "kind": "live",
            "published": h.get("created_at_i"),
        })
    return out


# ── arXiv ─────────────────────────────────────────────────────────────────
def arxiv(q: str, limit: int = 20):
    url = ("https://export.arxiv.org/api/query?search_query=all:"
           f"{urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=descending"
           f"&max_results={limit}")
    rows = rss(url, source="arXiv", kind="live", limit=limit)
    return rows


# ── GitHub (무인증도 분당 10회, 토큰 있으면 30회) ─────────────────────────
def github(q: str, limit: int = 20, token: str = ""):
    url = ("https://api.github.com/search/repositories?q="
           f"{urllib.parse.quote(q)}&sort=updated&per_page={limit}")
    _throttle(url)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    out = []
    for it in d.get("items", []):
        ts = None
        try:
            ts = parsedate_to_datetime(it["pushed_at"].replace("Z", "+0000")).timestamp()
        except Exception:
            pass
        out.append({
            "url": it["html_url"],
            "title": f"{it['full_name']} ★{it.get('stargazers_count', 0)}",
            "body": (it.get("description") or "")[:1000],
            "source": "GitHub", "kind": "live", "published": ts,
        })
    return out


# ── StackExchange ─────────────────────────────────────────────────────────
def stackexchange(q: str, limit: int = 20, site: str = "stackoverflow"):
    url = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
           f"&q={urllib.parse.quote(q)}&site={site}&pagesize={limit}&filter=withbody")
    import gzip
    import io
    raw = _get(url)
    try:
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    except Exception:
        pass
    d = json.loads(raw)
    return [{
        "url": it["link"],
        "title": _strip(it.get("title") or "")[:500],
        "body": _strip(it.get("body") or "")[:2000],
        "source": f"SO({it.get('score', 0)})",
        "kind": "live",
        "published": it.get("creation_date"),
    } for it in d.get("items", [])]


# ── Wikipedia ─────────────────────────────────────────────────────────────
def wikipedia(q: str, limit: int = 10, lang: str = "ko"):
    url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
           f"&srsearch={urllib.parse.quote(q)}&srlimit={limit}&format=json")
    d = json.loads(_get(url))
    out = []
    for it in d.get("query", {}).get("search", []):
        t = it["title"]
        out.append({
            "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(t.replace(' ', '_'))}",
            "title": t,
            "body": _strip(it.get("snippet") or "")[:1000],
            "source": f"Wikipedia({lang})", "kind": "live", "published": None,
        })
    return out


# 1층에서 실제로 때릴 창구 (질의당 병렬 실행)
LIVE = {
    "googlenews_ko": lambda q: google_news(q, "ko", 50),
    "googlenews_en": lambda q: google_news(q, "en", 50),
    "hackernews":    lambda q: hackernews(q, 30),
    "github":        lambda q: github(q, 15),
    "stackexchange": lambda q: stackexchange(q, 15),
    "wikipedia_ko":  lambda q: wikipedia(q, 8, "ko"),
    # ⛔ arxiv() 는 여기 넣지 않는다. 실측: 실시간 API 12.9초/5건 vs
    #    RSS 피드 0.1초/200건. 그래서 arXiv 는 0층 수집기로 내렸다.
    #    이 프로젝트의 논지가 그대로 드러나는 자리.
}
