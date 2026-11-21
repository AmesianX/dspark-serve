import json, sys
d = json.load(sys.stdin)
if "error" in d:
    print("   ERROR:", d["error"]); sys.exit()
print(f'   {d["layer"]}층 | {d["took_ms"]}ms | {len(d["results"])}건 | '
      f'승격:{d.get("promoted") or "-"} | 신규색인:{d.get("indexed_new","-")}')
for r in d["results"][:3]:
    print(f'     [{r["layer"]}] {str(r["source"])[:16]:16s} {r["title"][:56]}')
ch = d.get("channels")
if ch: print("     창구:", " ".join(f"{k}={v}" for k, v in ch.items()))
