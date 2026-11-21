"""사람(실브라우저) vs 기계(HTTP 클라이언트) — 같은 IP에서 같은 엔진을 쳐서
막힘의 원인이 'IP 평판'인지 '요청 모양'인지 가른다."""
import re, sys, time

TARGETS = {
    "brave":  "https://search.brave.com/search?q=io_uring+documentation&source=web",
    "google": "https://www.google.com/search?q=io_uring+documentation",
    "qwant":  "https://www.qwant.com/?q=io_uring+documentation&t=web",
    "ddg":    "https://html.duckduckgo.com/html/?q=io_uring+documentation",
}

def verdict(status, html):
    low = html.lower()
    marks = []
    if status in (403, 429): marks.append(f"HTTP{status}")
    for pat, tag in [("captcha","CAPTCHA"), ("unusual traffic","구글 봇경고"),
                     ("cf-challenge","CF챌린지"), ("just a moment","CF챌린지"),
                     ("rate limit","레이트리밋"), ("too many requests","레이트리밋"),
                     ("enablejs","JS요구"), ("consent.google","동의페이지")]:
        if pat in low: marks.append(tag)
    # 결과 링크 개수 대략 추정
    hits = len(re.findall(r'href="https?://(?!(?:www\.)?(?:brave|google|qwant|duckduckgo)\.)', html))
    return f"HTTP {status} / {len(html):>7,}B / 외부링크~{hits:>3} / {' '.join(marks) or '표식없음'}"

def machine():
    from curl_cffi import requests
    print("── 기계: curl_cffi impersonate=chrome (= 지금 SearXNG 가 하는 것) ──")
    for name, url in TARGETS.items():
        try:
            r = requests.get(url, impersonate="chrome", timeout=25)
            print(f"  {name:7s} {verdict(r.status_code, r.text)}")
        except Exception as e:
            print(f"  {name:7s} ERR {type(e).__name__}: {str(e)[:70]}")
        time.sleep(2)

def human():
    from playwright.sync_api import sync_playwright
    print("── 사람: 실제 Chromium (JS 실행 + 쿠키 + 서브리소스) ──")
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
        ctx = br.new_context(locale="ko-KR", timezone_id="Asia/Seoul",
                             viewport={"width":1366,"height":900})
        pg = ctx.new_page()
        for name, url in TARGETS.items():
            try:
                resp = pg.goto(url, wait_until="domcontentloaded", timeout=30000)
                pg.wait_for_timeout(2500)
                print(f"  {name:7s} {verdict(resp.status if resp else 0, pg.content())}")
            except Exception as e:
                print(f"  {name:7s} ERR {type(e).__name__}: {str(e)[:70]}")
        ctx.close(); br.close()

if __name__ == "__main__":
    if sys.argv[1:] == ["machine"]: machine()
    elif sys.argv[1:] == ["human"]: human()
    else: machine(); print(); human()
