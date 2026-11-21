import pathlib
UA=("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/159.0.0.0 Safari/537.36")
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    ctx=p.chromium.launch_persistent_context(
        user_data_dir="/home/user/work/websearch-browser/profile", headless=False,
        args=["--no-sandbox","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled"],
        user_agent=UA, locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width":1366,"height":900})
    ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
    pg=ctx.pages[0] if ctx.pages else ctx.new_page()
    pg.goto("https://search.brave.com/search?q=io_uring+documentation&source=web",
            wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_timeout(3500)
    print("title:", pg.title())
    print("body 앞 300자:", pg.inner_text("body")[:300].replace("\n"," | "))
    print("--- 후보 셀렉터별 개수:")
    for sel in ["#results","#results > *","[data-type=web]",".snippet","div.snippet",
                "a[href^='http']","main a[href^='http']","[data-pos]","noscript"]:
        try: print(f"   {sel:26s} {len(pg.query_selector_all(sel))}")
        except Exception as e: print(f"   {sel:26s} ERR")
    print("--- 외부 링크 상위 12:")
    links=pg.eval_on_selector_all("a[href^='http']",
      "els=>els.map(a=>({t:(a.innerText||'').trim().slice(0,55),u:a.href})).filter(x=>x.u&&!x.u.includes('brave.com'))")
    for l in links[:12]: print(f"   {l['t'][:52]:54s} {l['u'][:58]}")
    ctx.close()
