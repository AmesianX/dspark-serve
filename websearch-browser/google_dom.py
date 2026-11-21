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
    pg.goto("https://www.google.com/search?q=io_uring+documentation&num=20",
            wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_timeout(2500)
    out=pg.evaluate("""() => {
      const cards=[...document.querySelectorAll('#search div.g, #rso div[data-hveid]')];
      const seen=new Set(); const res=[];
      for (const c of cards){
        const h=c.querySelector('h3'); if(!h) continue;
        const a=h.closest('a'); if(!a) continue;
        // 실제 URL 후보: cite 텍스트, a의 data-* , ping 파라미터
        const cite=c.querySelector('cite');
        const sn=c.querySelector('div[data-sncf], .VwiC3b, [data-content-feature]');
        const key=h.innerText;
        if(seen.has(key)) continue; seen.add(key);
        res.push({title:h.innerText, href:a.href, ping:a.getAttribute('ping')||'',
                  cite:cite?cite.innerText:'', snip:sn?sn.innerText.slice(0,80):'' });
      }
      return res;
    }""")
    print("추출", len(out), "건")
    for r in out[:5]:
        print("-", r["title"][:58])
        print("   cite :", r["cite"][:70])
        print("   ping :", r["ping"][:90])
        print("   href :", r["href"][:70])
        print("   snip :", r["snip"][:70])
    ctx.close()
