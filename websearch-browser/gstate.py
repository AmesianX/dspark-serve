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
    r=pg.goto("https://www.google.com/search?q=io_uring+documentation", wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_timeout(3000)
    print("status:", r.status if r else "?", "| url:", pg.url[:90])
    print("title:", pg.title())
    print("body:", pg.inner_text("body")[:400].replace("\n"," | "))
    ctx.close()
