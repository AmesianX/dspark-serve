"""Shared settings for the lane tools: lane directory, server URL and API key.

The key comes from $API_KEY if set, otherwise from VLLM_API_KEY in the lane's
glm53-serve.env — the same file the server is launched from.
"""
import os
import re

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("URL", "http://localhost:8000")


def api_key() -> str:
    key = os.environ.get("API_KEY")
    if key:
        return key
    with open(os.path.join(LANE, "glm53-serve.env"), encoding="utf-8") as f:
        m = re.search(r'VLLM_API_KEY:=([^}]+)\}', f.read())
    if not m:
        raise SystemExit("API key not found: set API_KEY or VLLM_API_KEY in glm53-serve.env")
    return m.group(1)
