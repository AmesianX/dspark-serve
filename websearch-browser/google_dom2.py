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
    pg.goto("https://www.google.com/search?q=io_uring+documentation",
            wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_timeout=getattr(pg,'wait_for_timeout'); pg.wait_for_timeout(2500)
    print("h3 개수:", len(pg.query_selector_all("h3")))
    html=pg.evaluate("""() => {
      const h=document.querySelector('#search h3, #rso h3'); if(!h) return 'NO H3';
      let n=h; for(let i=0;i<5 && n.parentElement;i++) n=n.parentElement;
      return n.outerHTML.slice(0,2600);
    }""")
    print("--- 결과카드 outerHTML(2.6KB) ---")
    print(html)
    ctx.close()
