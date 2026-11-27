"""Decision + budget for browser_escalate, without SearXNG.

    python3 edge/searxng/plugins/test_browser_escalate.py
"""
import importlib.util, os, sys, types

# Stub the two SearXNG imports so the module loads standalone.
searx = types.ModuleType("searx"); rt = types.ModuleType("searx.result_types")
rt.EngineResults = list; sys.modules.update({"searx": searx, "searx.result_types": rt})
pkg = types.ModuleType("plg"); pkg.__path__ = []; pkg.Plugin = object; pkg.PluginInfo = object
sys.modules["plg"] = pkg
os.environ["BROWSER_ESCALATE_BUDGET_N"] = "2"; os.environ["BROWSER_ESCALATE_BUDGET_WINDOW"] = "10"
spec = importlib.util.spec_from_file_location("plg.browser_escalate",
                                              os.path.join(os.path.dirname(__file__), "browser_escalate.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

bad = []
def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name); ok or bad.append(name)

check("0 results -> escalate", m.should_escalate(0, 6, 0))
check("4 results -> escalate", m.should_escalate(4, 6, 0))
check("20 results, all fine -> no", not m.should_escalate(20, 6, 0))
check("7 results, 3/6 blocked -> escalate", m.should_escalate(7, 6, 3))
check("7 results, 2/6 blocked -> no", not m.should_escalate(7, 6, 2))
check("12 results, 6/6 blocked -> no (enough anyway)", not m.should_escalate(12, 6, 6))
check("budget 1", m._take_budget(100.0))
check("budget 2", m._take_budget(101.0))
check("budget spent -> denied", not m._take_budget(102.0))
check("window passes -> allowed again", m._take_budget(112.0))
m._store("q", 0.0, [{"url": "u", "title": "t"}])
check("cache hit within ttl", m._cached("q", 10.0) == [{"url": "u", "title": "t"}])
check("cache expired", m._cached("q", 10.0 + m.CACHE_TTL + 1) is None)
print("ALL PASS" if not bad else f"FAILED {bad}"); sys.exit(1 if bad else 0)
