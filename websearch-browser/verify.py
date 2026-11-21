"""실제 유기적 결과를 뽑아 검증. 난수 질의로 환각(쓰레기 결과) 여부까지 본다."""
import pathlib, re, sys
UA=("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/159.0.0.0 Safari/537.36")
STEALTH="""Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
window.chrome={runtime:{}};"""
QUERIES=[("정상","io_uring documentation"),("난수","zxqwvbnmklkjhgfd qpwoeirutyalskdjfh")]

def scrape(pg, engine, q):
    if engine=="brave":
        pg.goto("https://search.brave.com/search?q="+q.replace(" ","+")+"&source=web",
                wait_until="domcontentloaded", timeout=30000)
        pg.wait_for_timeout(2500)
        items=pg.eval_on_selector_all(
            "#results .snippet[data-type='web'], #results div.snippet",
            "els=>els.map(e=>{const a=e.querySelector('a[href^=\\\"http\\\"]');"
            "return a?{t:(a.innerText||'').trim().slice(0,60),u:a.href.slice(0,60)}:null}).filter(Boolean)")
    else:
        pg.goto("https://www.google.com/search?q="+q.replace(" ","+"),
                wait_until="domcontentloaded", timeout=30000)
        pg.wait_for_timeout(2500)
        items=pg.eval_on_selector_all(
            "#search a h3, #rso a h3",
            "els=>els.map(h=>({t:h.innerText.trim().slice(0,60),u:(h.closest('a')||{}).href||''}))")
    title=pg.title()[:50]
    return title, items

from playwright.sync_api import sync_playwright
prof=pathlib.Path("/home/user/work/websearch-browser/profile")
with sync_playwright() as p:
    ctx=p.chromium.launch_persistent_context(user_data_dir=str(prof), headless=False,
        args=["--no-sandbox","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled"],
        user_agent=UA, locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width":1366,"height":900})
    ctx.add_init_script(STEALTH)
    pg=ctx.pages[0] if ctx.pages else ctx.new_page()
    for engine in ("brave","google"):
        for label,q in QUERIES:
            try:
                title,items=scrape(pg,engine,q)
                print(f"[{engine}/{label}] page='{title}'  결과 {len(items)}건")
                for it in items[:4]: print(f"     - {it['t']}  ->  {it['u']}")
                if not items: print("     (0건)")
            except Exception as e:
                print(f"[{engine}/{label}] ERR {type(e).__name__}: {str(e)[:80]}")
    ctx.close()
