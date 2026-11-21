#!/usr/bin/env python3
"""browser-lane — 신원 있는 검색 레인.

설계 근거는 DESIGN.md. 요약: SearXNG 는 쿠키를 하드코딩으로 버려서(client.py:93)
봇탐지에 "신원 없는 방문자"로 찍힌다. 이 레인은 영속 프로필(=신원)을 들고
google/brave/qwant 를 치고, SearXNG 는 json_engine 으로 이걸 또 하나의 엔진처럼 쓴다.

자원 규칙 (이 박스는 vLLM 이 117GB, 가용 ~4GB, 통합메모리라 OOM=박스 사망):
  - 브라우저 상주 금지. 유휴 IDLE_EXIT 초면 종료, 다음 질의에 재기동.
    프로필은 디스크에 남으니 신원은 유지된다.
  - 네비게이션 동시 1개 (워커 스레드 1개로 자연 직렬화). 레이트 보호도 됨.
  - 반드시 메모리 하드캡으로 띄울 것 — serve.sh 참조.
"""
from __future__ import annotations

import json
import os
import queue
import random
import re
import string
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IDENTITIES = ROOT / "identities"
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)

BIND_HOSTS = os.environ.get("LANE_BIND", "127.0.0.1,172.18.0.1").split(",")
PORT = int(os.environ.get("LANE_PORT", "8099"))
TOKEN = os.environ.get("LANE_TOKEN", "")
IDLE_EXIT = float(os.environ.get("LANE_IDLE_EXIT", "180"))
NAV_TIMEOUT = float(os.environ.get("LANE_NAV_TIMEOUT", "30000"))
UA = os.environ.get(
    "LANE_UA",
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/159.0.0.0 Safari/537.36",
)

# 챌린지 표식.
#
# ⛔ 2026-11-18 버그 교훈: 처음엔 본문 전체에서 이 문자열들을 찾았다. 그랬더니
#    **정상 구글 결과 페이지**(man7.org io_uring(7) 등 유기적 결과 9건이 멀쩡히
#    렌더된 화면)가 본문 어딘가의 '/sorry/index' 문자열 때문에 챌린지로 오판됐고,
#    멀쩡한 엔진이 챌린지로 오판돼 차단됐다. 'recaptcha/api2' 'enablejs' 도
#    정상 페이지에 흔히 들어있다.
#
# 그래서 판정 순서를 뒤집는다: **먼저 추출하고, 결과가 나오면 그걸로 끝.**
# 결과가 0건일 때만 아래 표식을 본다. 산출물로 판단하는 게 문자열 매칭보다 정확하다.
#
# 표식도 좁혔다 — 최종 URL / 제목 / 본문 앞부분에만 적용한다.
CHALLENGE_BACKOFF = 3600.0        # 첫 챌린지 후 재우는 시간
CHALLENGE_BACKOFF_MAX = 86400.0   # 반복되면 최대 하루
CHALLENGE_URL_MARKS = ["/sorry/", "/challenge", "captcha"]
CHALLENGE_TEXT_MARKS = [
    "봇이 아닌지 확인",            # brave (ko)
    "verifying you are human",
    "just a moment",              # Cloudflare
    "unusual traffic",            # google (en)
    "비정상적인 트래픽",            # google (ko)
    "systems have detected unusual",
]

STEALTH = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
Object.defineProperty(navigator,'languages',{get:()=>['ko-KR','ko','en-US','en']});
window.chrome = window.chrome || {runtime:{}};
"""


# ── 엔진별 추출기 ──────────────────────────────────────────────────────────
# ⚠️ 2026-11-18: google/brave 가 테스트로 잠겨 있어 셀렉터 미검증. 식은 뒤 검증 필요.
#    검증 방법: /selftest (난수 질의 → 0건이어야 함) + 정상 질의 유기적 결과 육안 확인.
ENGINES = {
    "google": {
        "url": "https://www.google.com/search?q={q}",
        "home": "https://www.google.com/",
        # Google 은 결과 href 를 /goto?url=<불투명> 로 감싼다. 그대로 두면 링크는
        # 동작하지만 최종 URL 이 가려진다. 해석하려면 결과마다 요청 1개가 더
        # 필요해서(20개면 20요청) 레이트 예산이 터진다 → wrapped 플래그만 달아 보낸다.
        "extract": """() => {
          const out=[], seen=new Set();
          for (const h of document.querySelectorAll('#search h3, #rso h3')) {
            const a=h.closest('a'); if(!a||!a.href) continue;
            const t=(h.innerText||'').trim(); if(!t||seen.has(t)) continue; seen.add(t);
            const card=h.closest('div[data-hveid]')||h.parentElement;
            const sn=card?card.querySelector('div[data-sncf], .VwiC3b'):null;
            const cite=card?card.querySelector('cite'):null;
            out.push({title:t, url:a.href,
                      wrapped:a.href.includes('/goto?') || a.href.includes('/url?'),
                      cite:cite?(cite.innerText||'').trim():'',
                      content:sn?(sn.innerText||'').trim().slice(0,300):''});
          }
          return out;
        }""",
    },
    "brave": {
        "url": "https://search.brave.com/search?q={q}&source=web",
        "home": "https://search.brave.com/",
        "extract": """() => {
          const out=[], seen=new Set();
          const cards=document.querySelectorAll('#results .snippet, #results [data-type=web]');
          for (const c of cards) {
            const a=c.querySelector('a[href^="http"]'); if(!a) continue;
            if (a.href.includes('search.brave.com')) continue;
            const tEl=c.querySelector('.title, .snippet-title, h3, h4');
            const t=(tEl?tEl.innerText:a.innerText||'').trim();
            if(!t||seen.has(a.href)) continue; seen.add(a.href);
            const sn=c.querySelector('.snippet-description, .snippet-content, p');
            out.push({title:t, url:a.href, wrapped:false, cite:'',
                      content:sn?(sn.innerText||'').trim().slice(0,300):''});
          }
          return out;
        }""",
    },
    "qwant": {
        "url": "https://www.qwant.com/?q={q}&t=web",
        "home": "https://www.qwant.com/",
        "extract": """() => {
          const out=[], seen=new Set();
          for (const a of document.querySelectorAll('a[href^="http"]')) {
            if (a.href.includes('qwant.com')) continue;
            const t=(a.innerText||'').trim();
            if (t.length < 12 || seen.has(a.href)) continue; seen.add(a.href);
            out.push({title:t.slice(0,160), url:a.href, wrapped:false, cite:'', content:''});
          }
          return out;
        }""",
    },
}


# ── 엔진 상태 ──────────────────────────────────────────────────────────────
class EngineState:
    """ok | suspended(일시 유예 후 자동 복귀)

    ⛔ 사람을 부르는 상태는 두지 않는다. 챌린지를 띄우는 엔진은 '지금 우리를
    원하지 않는 엔진'일 뿐이고, 사람이 캡차를 눌러주는 건 해법이 아니라
    기술부채다. 길게 재우고 자동 복귀시킨 뒤, 안 되면 엔진을 버린다.
    """

    def __init__(self):
        self.status = "ok"
        self.until = 0.0
        self.note = ""
        self.last_shot = ""
        self.last_ok = 0.0
        self.queries = 0
        self.challenges = 0

    def available(self) -> bool:
        if self.status == "suspended" and time.time() < self.until:
            return False
        if self.status == "suspended":
            self.status, self.note = "ok", ""
        return True

    def as_dict(self):
        return {
            "status": self.status,
            "until": round(max(0.0, self.until - time.time()), 1),
            "note": self.note,
            "last_challenge_shot": self.last_shot,
            "queries": self.queries,
            "challenges": self.challenges,
            "last_ok_ago": round(time.time() - self.last_ok, 1) if self.last_ok else None,
        }


STATES = {name: EngineState() for name in ENGINES}
STATES_LOCK = threading.Lock()


def notify(msg: str) -> None:
    """엔진이 내려갔다는 기록. 사람에게 조치를 요구하지 않는다 — 통보일 뿐."""
    line = f"[browser-lane] {msg}"
    print(line, flush=True)
    (STATE / "notify.log").open("a", encoding="utf-8").write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    )
    for cli in ("/home/user/.local/bin/codemason", "/usr/local/bin/cm2"):
        if os.path.exists(cli):
            os.system(f"{cli} notify {json.dumps(line)} >/dev/null 2>&1 &")
            return


# ── 브라우저 워커 (playwright sync API 는 한 스레드에서만 써야 한다) ─────────
class BrowserWorker(threading.Thread):
    def __init__(self, identity: str):
        super().__init__(daemon=True)
        self.identity = identity
        self.jobs: queue.Queue = queue.Queue()
        self.last_used = time.time()
        self._pw = None
        self._ctx = None

    # --- 브라우저 수명 ---
    def _ensure(self):
        if self._ctx is not None:
            return
        from playwright.sync_api import sync_playwright

        prof = IDENTITIES / self.identity
        prof.mkdir(parents=True, exist_ok=True)
        os.chmod(prof, 0o700)
        self._pw = sync_playwright().start()
        self._ctx = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(prof),
            headless=False,  # xvfb-run 아래에서 돈다. headless-shell 은 UA 에 흔적이 남는다.
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--password-store=basic",
            ],
            user_agent=UA,
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 1366, "height": 900},
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"},
        )
        self._ctx.add_init_script(STEALTH)
        print(f"[lane] 브라우저 기동 (신원={self.identity})", flush=True)

    def _shutdown(self):
        if self._ctx is None:
            return
        try:
            self._ctx.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass
        self._ctx = self._pw = None
        print("[lane] 유휴 → 브라우저 종료 (프로필=신원은 디스크에 남음)", flush=True)

    # --- 실제 검색 ---
    def _search(self, engine: str, q: str, want: int):
        cfg = ENGINES[engine]
        self._ensure()
        pg = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()

        # 홈 선방문: 쿠키/동의를 받고 레퍼러 사슬을 만든다. 신원이 이미 있으면 싸다.
        try:
            if time.time() - STATES[engine].last_ok > 600:
                pg.goto(cfg["home"], wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
                pg.wait_for_timeout(random.randint(900, 1800))
        except Exception:
            pass

        url = cfg["url"].format(q=urllib.parse.quote_plus(q))
        resp = pg.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        pg.wait_for_timeout(random.randint(1800, 3000))
        status = resp.status if resp else 0

        # ── 1단계: 먼저 뽑는다. 뽑히면 성공이다. ──────────────────────────
        # 문자열 매칭보다 산출물이 정확하다 (위 CHALLENGE_* 주석의 오판 사고 참조).
        try:
            items = pg.evaluate(cfg["extract"]) or []
        except Exception as e:
            items = []
            print(f"[lane] {engine} 추출기 실패: {type(e).__name__}: {e}", flush=True)

        if items:
            with STATES_LOCK:
                st = STATES[engine]
                st.status, st.note, st.last_ok = "ok", "", time.time()
                st.queries += 1
            return {"status": status, "results": items[:want]}

        # ── 2단계: 0건일 때만 챌린지인지 따진다 ──────────────────────────
        final_url = (pg.url or "").lower()
        try:
            head = (pg.inner_text("body") or "")[:1500].lower()
        except Exception:
            head = ""
        title = (pg.title() or "").lower()

        hit = next((m for m in CHALLENGE_URL_MARKS if m in final_url), None) \
            or next((m for m in CHALLENGE_TEXT_MARKS if m in head or m in title), None)

        if hit or status in (403, 429):
            shot = STATE / f"challenge-{engine}-{time.strftime('%Y%m%d-%H%M%S')}.png"
            try:
                pg.screenshot(path=str(shot), full_page=False)
            except Exception:
                shot = ""
            with STATES_LOCK:
                st = STATES[engine]
                st.challenges += 1
                # 챌린지마다 유예를 2배로 늘린다 (1h → 2h → 4h … 최대 24h).
                # 계속 두드려서 IP 평판을 태우지 않기 위한 것이고, 동시에
                # 사람을 호출하지 않기 위한 것이다. 시간이 지나면 알아서 복귀.
                back = min(CHALLENGE_BACKOFF * (2 ** (st.challenges - 1)),
                           CHALLENGE_BACKOFF_MAX)
                st.status = "suspended"
                st.until = time.time() + back
                st.note = (f"챌린지 (HTTP {status}, 표식 '{hit}') "
                           f"→ {back/3600:.0f}h 후 자동 재시도")
                st.last_shot = str(shot)
            notify(f"{engine}: 챌린지 감지 (HTTP {status}, '{hit}'). "
                   f"{back/3600:.0f}시간 재움. 스크린샷 {shot}")
            return {"error": "challenged", "status": status, "mark": hit,
                    "retry_in_s": round(back), "screenshot": str(shot), "results": []}

        # 0건이지만 챌린지는 아니다 = 진짜로 결과가 없는 질의(난수 등)이거나
        # 추출기 셀렉터가 상했다. 엔진을 내리지는 않되 구분해서 돌려준다.
        with STATES_LOCK:
            STATES[engine].last_ok = time.time()
        return {"status": status, "results": [], "empty_reason": "no_results_no_challenge",
                "page_title": pg.title()[:80]}

    # --- 워커 루프 ---
    def run(self):
        while True:
            try:
                job = self.jobs.get(timeout=5.0)
            except queue.Empty:
                if self._ctx is not None and time.time() - self.last_used > IDLE_EXIT:
                    self._shutdown()
                continue
            fn, args, box = job
            self.last_used = time.time()
            try:
                box["result"] = fn(*args)
            except Exception as e:
                box["result"] = {"error": f"{type(e).__name__}: {e}", "results": []}
                print("[lane] 작업 실패:\n" + traceback.format_exc(), flush=True)
            finally:
                self.last_used = time.time()
                box["done"].set()

    def submit(self, fn, *args, timeout=120.0):
        box = {"done": threading.Event(), "result": None}
        self.jobs.put((fn, args, box))
        if not box["done"].wait(timeout):
            return {"error": "timeout", "results": []}
        return box["result"]


WORKER: BrowserWorker | None = None


# ── HTTP ───────────────────────────────────────────────────────────────────
def _json(handler, code, obj):
    raw = json.dumps(obj, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class Handler(BaseHTTPRequestHandler):
    server_version = "browser-lane/1.0"

    def log_message(self, fmt, *a):
        print(f"[lane] {self.address_string()} {fmt % a}", flush=True)

    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)

        if u.path == "/health":
            with STATES_LOCK:
                eng = {k: v.as_dict() for k, v in STATES.items()}
            return _json(self, 200, {
                "ok": True,
                "identity": WORKER.identity,
                "browser_up": WORKER._ctx is not None,
                "idle_exit_s": IDLE_EXIT,
                "engines": eng,
                "identity_cookies": _cookie_summary(WORKER.identity),
            })

        if TOKEN and self.headers.get("X-Lane-Token") != TOKEN:
            return _json(self, 403, {"error": "bad or missing X-Lane-Token"})

        if u.path == "/selftest":
            # 정직성 가드: 난수 질의는 0건이어야 한다 (bing 교훈).
            rnd = "".join(random.choice(string.ascii_lowercase) for _ in range(18))
            eng = qs.get("engine", ["google"])[0]
            if eng not in ENGINES:
                return _json(self, 400, {"error": f"unknown engine {eng}"})
            r = WORKER.submit(WORKER._search, eng, rnd, 10)
            n = len(r.get("results", []))
            return _json(self, 200, {
                "engine": eng, "random_query": rnd, "results": n,
                "verdict": "정직" if n == 0 else "⚠️ 난수에 결과를 반환 — 이 엔진 믿지 말 것",
                "raw_error": r.get("error"),
            })

        if u.path == "/search":
            q = (qs.get("q") or [""])[0].strip()
            eng = (qs.get("engine") or ["google"])[0]
            want = max(1, min(30, int((qs.get("n") or ["10"])[0])))
            if not q:
                return _json(self, 400, {"error": "q required", "results": []})
            if eng not in ENGINES:
                return _json(self, 400, {"error": f"unknown engine {eng}", "results": []})
            with STATES_LOCK:
                st = STATES[eng]
                if not st.available():
                    return _json(self, 200, {
                        "engine": eng, "results": [],
                        "skipped": st.status, "note": st.note,
                    })
            r = WORKER.submit(WORKER._search, eng, q, want)
            r["engine"] = eng
            r["query"] = q
            return _json(self, 200, r)

        return _json(self, 404, {"error": "not found"})


def _cookie_summary(identity: str):
    """신원은 감사 가능해야 한다 — 쿠키 이름·호스트·만료를 그대로 보여준다."""
    import shutil
    import sqlite3
    import tempfile

    src = IDENTITIES / identity / "Default" / "Cookies"
    if not src.exists():
        return {"count": 0, "note": "아직 신원 없음 (쿠키 파일 미생성)"}
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "c.sqlite"
        shutil.copy2(src, cp)
        try:
            rows = sqlite3.connect(f"file:{cp}?mode=ro", uri=True).execute(
                "select host_key,name,expires_utc from cookies order by host_key"
            ).fetchall()
        except Exception as e:
            return {"count": -1, "note": f"읽기 실패: {e}"}
    import datetime
    out = []
    for h, n, e in rows:
        exp = "세션"
        if e:
            try:
                exp = (datetime.datetime(1601, 1, 1)
                       + datetime.timedelta(microseconds=e)).strftime("%Y-%m-%d")
            except Exception:
                exp = "?"
        out.append({"host": h, "name": n, "expires": exp})
    return {"count": len(out), "cookies": out}


def main():
    global WORKER
    identity = os.environ.get("LANE_IDENTITY", "default")
    WORKER = BrowserWorker(identity)
    WORKER.start()

    if not TOKEN:
        print("[lane] ⚠️ LANE_TOKEN 미설정 — 인증 없이 떠 있다. serve.sh 를 쓸 것.", flush=True)

    servers = []
    for host in [h.strip() for h in BIND_HOSTS if h.strip()]:
        try:
            srv = ThreadingHTTPServer((host, PORT), Handler)
        except OSError as e:
            print(f"[lane] {host}:{PORT} 바인딩 실패: {e}", flush=True)
            continue
        servers.append(srv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[lane] listen http://{host}:{PORT}  신원={identity}", flush=True)
    if not servers:
        sys.exit("[lane] 바인딩할 주소가 없다")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
