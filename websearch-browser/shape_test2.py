"""사람 흉내를 제대로 낸 브라우저 vs curl_cffi — 같은 IP.
headful chromium + 진짜 UA + 영속 프로필(쿠키 누적) + webdriver 은닉 + 홈페이지 선방문."""
import re, sys, time, pathlib

TARGETS = {
    "brave":  ("https://search.brave.com/", "https://search.brave.com/search?q=io_uring+documentation&source=web"),
    "google": ("https://www.google.com/",   "https://www.google.com/search?q=io_uring+documentation"),
    "ddg":    ("https://duckduckgo.com/",   "https://html.duckduckgo.com/html/?q=io_uring+documentation"),
}
UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/159.0.0.0 Safari/537.36")

def verdict(status, html):
    low = html.lower(); marks = []
    if status in (403, 429): marks.append(f"HTTP{status}")
    for pat, tag in [("captcha","CAPTCHA"), ("unusual traffic","구글봇경고"),
                     ("just a moment","CF챌린지"), ("too many requests","레이트리밋"),
                     ("enablejs","JS요구"), ("sorry/index","구글sorry")]:
        if pat in low and tag not in marks: marks.append(tag)
    hits = len(re.findall(r'href="(?:https?:)?//(?!(?:www\.|html\.|search\.)?(?:brave|google|qwant|duckduckgo|gstatic|googleapis)\.)', html))
    return f"HTTP {status} / {len(html):>7,}B / 외부링크~{hits:>3} / {' '.join(marks) or '표식없음'}"

STEALTH = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
Object.defineProperty(navigator,'languages',{get:()=>['ko-KR','ko','en-US','en']});
Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});
window.chrome = {runtime:{}, loadTimes:function(){}, csi:function(){}};
const q = navigator.permissions.query;
navigator.permissions.query = (p) => p.name==='notifications'
  ? Promise.resolve({state:Notification.permission}) : q(p);
"""

def run():
    from playwright.sync_api import sync_playwright
    prof = pathlib.Path("/home/user/work/websearch-browser/profile"); prof.mkdir(exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(prof), headless=False, channel=None,
            args=["--no-sandbox","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled",
                  "--window-size=1366,900"],
            user_agent=UA, locale="ko-KR", timezone_id="Asia/Seoul",
            viewport={"width":1366,"height":900},
            extra_http_headers={"Accept-Language":"ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"})
        ctx.add_init_script(STEALTH)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        ua_real = pg.evaluate("navigator.userAgent") if True else ""
        print(f"  (실제 UA: {ua_real[:72]})")
        print(f"  (webdriver: {pg.evaluate('navigator.webdriver')})")
        for name,(home,url) in TARGETS.items():
            try:
                pg.goto(home, wait_until="domcontentloaded", timeout=30000)  # 쿠키 받기
                pg.wait_for_timeout(1800)
                pg.mouse.move(400,300); pg.mouse.wheel(0,250); pg.wait_for_timeout(600)
                resp = pg.goto(url, wait_until="domcontentloaded", timeout=30000)
                pg.wait_for_timeout(3000)
                print(f"  {name:7s} {verdict(resp.status if resp else 0, pg.content())}")
            except Exception as e:
                print(f"  {name:7s} ERR {type(e).__name__}: {str(e)[:70]}")
        ctx.close()

if __name__ == "__main__":
    print("── 사람 흉내(headful + 영속프로필 + 은닉 + 홈방문) ──")
    run()
