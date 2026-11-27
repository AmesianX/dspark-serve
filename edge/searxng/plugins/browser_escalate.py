# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-side escalation to the identity-bearing browser lane (layer 3).

The decision to spend the owner's browser identity belongs to the SERVER, not
to whoever sends the query. The first version wired the lane in as SearXNG
engines reachable only by bang (``!gbr``) -- so the caller decided: the model
did not know the syntax and never used it, and anyone holding the SearXNG key
could search as the owner (a bang reaches even a ``disabled: true`` engine).
Those engines are gone (2026-11-27). This plugin is now the only caller of the
lane, and nothing in a request can make it call: it looks at what the normal
layers returned and escalates on its own when they came back thin or blocked.

When it escalates (all must hold):
  * first page, general category, the default engine set (>= 3 engines; a
    query a bang pinned to one engine is the caller's choice and stays so);
  * fewer than MIN_RESULTS results, OR at least half the queried engines came
    back unresponsive (blocked / timed out) with fewer than 2*MIN_RESULTS;
  * the lane budget is not spent (BUDGET_N calls per BUDGET_WINDOW seconds).
    The identity is a finite resource -- the lane exists to be used rarely --
    and thin results can be produced on purpose, so the budget is the guard,
    not the query.

Results are cached per query for CACHE_TTL. SearXNG runs a single worker
process here, so the module-level budget is the server's budget.

The lane: ~/work/dspark-serve/websearch-browser/ (127.0.0.1 + 172.18.0.1 only,
``X-Lane-Token``). Only google is used: brave and qwant were walled (11-18).
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
import typing
import urllib.parse
import urllib.request

from searx.result_types import EngineResults

from . import Plugin, PluginInfo

if typing.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request
    from searx.search import SearchWithPlugins

    from . import PluginCfg

LANE_URL = os.environ.get("BROWSER_LANE_URL", "http://172.18.0.1:8099/search")
TOKEN_FILE = os.environ.get("BROWSER_LANE_TOKEN_FILE", "/etc/searxng/browser_lane_token")
MIN_RESULTS = int(os.environ.get("BROWSER_ESCALATE_MIN_RESULTS", "5"))
BUDGET_N = int(os.environ.get("BROWSER_ESCALATE_BUDGET_N", "6"))
BUDGET_WINDOW = float(os.environ.get("BROWSER_ESCALATE_BUDGET_WINDOW", "600"))
CACHE_TTL = float(os.environ.get("BROWSER_ESCALATE_CACHE_TTL", "1800"))
# Cold start of the lane's Chromium is 3-8 s; the old 3 s SearXNG default is
# what made every first !gbr time out.
TIMEOUT = float(os.environ.get("BROWSER_ESCALATE_TIMEOUT", "15"))

_lock = threading.Lock()
_calls: collections.deque[float] = collections.deque()
_cache: "collections.OrderedDict[str, tuple[float, list[dict]]]" = collections.OrderedDict()
STATS = {"checked": 0, "escalated": 0, "cache_hits": 0, "budget_denied": 0, "lane_errors": 0}


def should_escalate(n_results: int, n_queried: int, n_unresponsive: int) -> bool:
    """Pure decision, so it can be tested without SearXNG."""
    if n_results < MIN_RESULTS:
        return True
    blocked = n_queried > 0 and n_unresponsive * 2 >= n_queried
    return blocked and n_results < 2 * MIN_RESULTS


def _take_budget(now: float) -> bool:
    with _lock:
        while _calls and now - _calls[0] > BUDGET_WINDOW:
            _calls.popleft()
        if len(_calls) >= BUDGET_N:
            return False
        _calls.append(now)
        return True


def _cached(q: str, now: float) -> list[dict] | None:
    with _lock:
        hit = _cache.get(q)
        if hit and now - hit[0] <= CACHE_TTL:
            _cache.move_to_end(q)
            return hit[1]
        return None


def _store(q: str, now: float, items: list[dict]) -> None:
    with _lock:
        _cache[q] = (now, items)
        _cache.move_to_end(q)
        while len(_cache) > 256:
            _cache.popitem(last=False)


def _ensure_score_counter(plugin_id: str) -> None:
    """Register the score counter SearXNG will add to for our results.

    ``ResultContainer.close()`` adds every main result's score to a counter
    keyed by engine name, and the counters exist only for the engines
    registered at startup. Results a plugin returns carry the engine name
    ``plugin: <id>``, which was never registered, so ``close()`` raised
    KeyError and the WHOLE search failed -- exactly on the queries we escalate.
    Upstream plugins only return answers, never main results, which is why
    nothing else trips on it. Reproduced in-container (merge_test) before this.
    """
    try:
        from searx import metrics

        st = metrics.counter_storage
        key = ("engine", f"plugin: {plugin_id}", "score")
        if st is not None and key not in st.counters:
            st.configure(*key)
    except Exception:  # pylint: disable=broad-except
        pass


def _lane(q: str) -> list[dict]:
    with open(TOKEN_FILE, encoding="utf-8") as f:
        token = f.read().strip()
    url = LANE_URL + "?" + urllib.parse.urlencode({"q": q, "engine": "google", "n": 10})
    req = urllib.request.Request(url, headers={"X-Lane-Token": token})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = json.loads(r.read())
    if body.get("error") or body.get("skipped"):
        raise RuntimeError(f"lane: {body.get('error') or body.get('skipped')}")
    return [x for x in body.get("results", []) if x.get("url") and x.get("title")]


class SXNGPlugin(Plugin):
    id = "browser_escalate"

    def __init__(self, plg_cfg: "PluginCfg"):
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="Browser escalation",
            description="When the normal engines come back thin or blocked, the server asks "
            "its own browser lane once (budgeted). Not selectable from a query.",
            preference_section="general",
        )

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> EngineResults:
        results = EngineResults()
        sq = search.search_query
        refs = sq.engineref_list or []
        if sq.pageno > 1 or len(refs) < 3 or not any(e.category == "general" for e in refs):
            return results
        q = (sq.query or "").strip()
        if len(q) < 2:
            return results
        rc = search.result_container
        n = len(rc.main_results_map)
        queried = len({e.name for e in refs})
        unresp = len({u.engine for u in rc.unresponsive_engines})
        STATS["checked"] += 1
        if not should_escalate(n, queried, unresp):
            return results
        now = time.time()
        items = _cached(q, now)
        if items is not None:
            STATS["cache_hits"] += 1
        else:
            if not _take_budget(now):
                STATS["budget_denied"] += 1
                self.log.warning("browser_escalate: budget spent (%d/%ss), not escalating %r",
                                 BUDGET_N, int(BUDGET_WINDOW), q[:80])
                return results
            try:
                items = _lane(q)
            except Exception as e:  # pylint: disable=broad-except
                STATS["lane_errors"] += 1
                self.log.warning("browser_escalate: lane failed for %r: %s", q[:80], e)
                return results
            _store(q, now, items)
            STATS["escalated"] += 1
            self.log.info("browser_escalate: %r had %d results, %d/%d engines unresponsive -> lane +%d",
                          q[:80], n, unresp, queried, len(items))
        if items:
            _ensure_score_counter(self.id)
        for x in items:
            results.add(results.types.MainResult(url=x["url"], title=x["title"], content=x.get("content", "")))
        return results
