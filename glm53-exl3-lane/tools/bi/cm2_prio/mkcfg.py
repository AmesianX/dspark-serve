import json, os, sys
src = json.load(open(os.path.expanduser("~/.codemason/config.json")))
d, backend = sys.argv[1], sys.argv[2]
os.makedirs(d, exist_ok=True)
src["model_base_url"] = "http://127.0.0.1:18000"
src["model_api_key"] = "proxy-injects-real-key"
src["model_aes_key"] = ""
src["backend"] = backend
src["remote_enabled"] = False
src["max_concurrent_subagents"] = 3
json.dump(src, open(os.path.join(d, "config.json"), "w"), ensure_ascii=False, indent=1)
print("cfg", d, backend)
