"""Gateway forces a per-session cache_salt (encrypt_gateway.py). CPU, no network.

    python3 glm53-exl3-lane/tools/test_gateway_session_salt.py   (from the repo root)
"""
import base64, json, os, sys, types

os.environ.update(LLAMACPP_AES_KEY=base64.b64encode(b"k" * 32).decode(),
                  VLLM_API_KEY="test", GATEWAY_CACHE_SALT_SECRET="x" * 64)
sys.path.insert(0, os.getcwd())
import encrypt_gateway as g  # noqa: E402

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        FAILED.append(name)


def req(peer="172.18.0.5", real="203.0.113.7"):
    return types.SimpleNamespace(client=types.SimpleNamespace(host=peer),
                                 headers={"x-real-ip": real} if real else {})


def apply(body, path="v1/chat/completions", r=None):
    out = g._apply_session_salt(path, r or req(), json.dumps(body).encode())
    return json.loads(out)


A = "a" * 32
B = "b" * 32
a1 = apply({"messages": [], "cache_session": A})
a2 = apply({"messages": [], "cache_session": A, "cache_salt": "attacker-chosen"})
b1 = apply({"messages": [], "cache_session": B})
check("세션 id 는 전달 전에 제거", "cache_session" not in a1)
check("같은 세션 = 같은 salt", a1["cache_salt"] == a2["cache_salt"])
check("클라이언트 cache_salt 는 덮어씀", a2["cache_salt"] != "attacker-chosen")
check("다른 세션 = 다른 salt", a1["cache_salt"] != b1["cache_salt"])
n1 = apply({"messages": []}, r=req(real="203.0.113.7"))
n2 = apply({"messages": []}, r=req(real="198.51.100.9"))
check("세션 없으면 주소별 salt", n1["cache_salt"] != n2["cache_salt"] and n1["cache_salt"])
bad = apply({"messages": [], "cache_session": "short"})
check("형식이 틀린 세션 id 는 주소 salt 로", bad["cache_salt"] == n1["cache_salt"])
check("생성 경로가 아니면 손대지 않음",
      g._apply_session_salt("v1/models", req(), b'{"cache_salt":"x"}') == b'{"cache_salt":"x"}')
os.environ["GATEWAY_CACHE_SALT_SECRET"] = "y" * 64
import importlib  # noqa: E402
g2 = importlib.reload(g)
check("salt 는 서버 비밀에 묶임(비밀이 바뀌면 다른 salt)",
      json.loads(g2._apply_session_salt("v1/chat/completions", req(),
                 json.dumps({"cache_session": A}).encode()))["cache_salt"] != a1["cache_salt"])

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
