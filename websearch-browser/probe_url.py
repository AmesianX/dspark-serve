import os
from playwright.sync_api import sync_playwright
UA=os.environ.get("LANE_UA","Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/159.0.0.0 Safari/537.36")
with sync_playwright() as p:
    ctx=p.chromium.launch_persistent_context(
        user_data_dir="/home/user/work/websearch-browser/identities/default", headless=False,
        args=["--no-sandbox","--disable-dev-shm-usage","--disable-blink-features=AutomationControlled","--password-store=basic"],
        user_agent=UA, locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width":1366,"height":900})
    ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
    pg=ctx.pages[0] if ctx.pages else ctx.new_page()
    for url,label in [("https://www.google.com/search?q=io_uring+documentation","기본"),
                      ("https://www.google.com/search?q=io_uring+documentation&udm=14","udm=14(웹)")]:
        pg.goto(url, wait_until="domcontentloaded", timeout=30000); pg.wait_for_timeout(2500)
        info=pg.evaluate("""() => {
          const h=document.querySelector('#search h3, #rso h3'); if(!h) return null;
          const a=h.closest('a'); const card=h.closest('div[data-hveid]')||h.parentElement;
          const attrs={}; if(a) for(const at of a.attributes) attrs[at.name]=String(at.value).slice(0,90);
          const cite=card?card.querySelector('cite'):null;
          return {title:h.innerText.slice(0,40), attrs, cite:cite?cite.innerText:'',
                  cardAttrs: card? Object.fromEntries([...card.attributes].map(x=>[x.name,String(x.value).slice(0,60)])):{}};
        }""")
        print(f"--- {label}: {pg.url[:70]}")
        if not info: print("    (h3 없음)"); continue
        print("    title:", info["title"]); print("    cite :", info["cite"][:70])
        for k,v in info["attrs"].items(): print(f"    a[{k}] = {v}")
    ctx.close()
