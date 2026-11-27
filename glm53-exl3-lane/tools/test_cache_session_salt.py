"""Session cache salt starts at the private system block (overlay kv_cache_utils.py).

CPU only, loads the overlay by path:
    docker cp tools/test_cache_session_salt.py vllm_glm53:/tmp/ && \
    docker cp overlay/v1core/kv_cache_utils.py vllm_glm53:/tmp/ovl_kv_cache_utils.py && \
    docker exec -w /tmp -e CUDA_VISIBLE_DEVICES= -e GLM53_CACHE_SHARE_TOKENIZER=/models/glm53-exl3 \
        vllm_glm53 python3 /tmp/test_cache_session_salt.py
"""
import importlib.util
import sys
import types

FAILED = []


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + str(extra)) if extra else ''}")
    if not cond:
        FAILED.append(name)


spec = importlib.util.spec_from_file_location("ovl_kv_cache_utils", "/tmp/ovl_kv_cache_utils.py")
ku = importlib.util.module_from_spec(spec)
ku.__package__ = "vllm.v1.core"
sys.modules["ovl_kv_cache_utils"] = ku  # dataclasses look their module up here
spec.loader.exec_module(ku)
tok = ku._resolve_share_tokens()
check("특수 토큰 해석", tok is not None, tok)
S, U = tok
HB = 16


def req(prompt, opt=True, salt="s1"):
    return types.SimpleNamespace(
        prompt_token_ids=prompt, cache_salt=salt,
        sampling_params=types.SimpleNamespace(extra_args={"cache_share_system_prefix": True} if opt else None),
    )


# [gMASK]<sop>|S eff | S tools.... | S static.... | S private.. | U user...
p = [1, 2, S] + [5] * 10 + [S] + [6] * 40 + [S] + [7] * 30 + [S] + [8] * 20 + [U] + [9] * 30 + [S] + [9] * 5
last_sys = max(i for i in range(p.index(U)) if p[i] == S)
check("동의 시 salt 시작 = 개인 <|system|> 를 블록 내림", ku._cache_salt_start(req(p), HB) == last_sys // HB * HB,
      (ku._cache_salt_start(req(p), HB), last_sys))
check("시작점 이후 토큰엔 개인 블록 시작이 포함", ku._cache_salt_start(req(p), HB) <= last_sys)
check("동의 없으면 0 (upstream)", ku._cache_salt_start(req(p, opt=False), HB) == 0)
def with_opt(v):
    return types.SimpleNamespace(prompt_token_ids=p, cache_salt="s",
                                 sampling_params=types.SimpleNamespace(extra_args={"cache_share_system_prefix": v}))


# vLLM's vllm_xargs schema coerces JSON true to int 1 -- that is what arrives.
check("동의값 1(정수, 실제 전송형) 인정", ku._cache_salt_start(with_opt(1), HB) == last_sys // HB * HB)
for bad in ("true", "1", 2, 1.0, 0, None):
    check(f"동의값 {bad!r} 는 0", ku._cache_salt_start(with_opt(bad), HB) == 0)
check("사용자 턴 안의 <|system|> 는 경계를 못 옮김", ku._cache_salt_start(req(p), HB) < p.index(U))
check("<|user|> 없으면 0", ku._cache_salt_start(req([1, 2, S, 5, 5, S, 6]), HB) == 0)

# extra keys: salt only on the block starting at the boundary
r = req(p)
r._glm53_salt_start = 64
r.lora_request = None
r.mm_features = []
r.prompt_embeds = None
r._prompt_embeds_per_block_hashes = None
k0, _ = ku.generate_block_hash_extra_keys(r, 0, 16, 0)
k64, _ = ku.generate_block_hash_extra_keys(r, 64, 80, 0)
check("공개 블록엔 salt 없음", not k0 or "s1" not in k0, k0)
check("경계 블록엔 salt", k64 is not None and "s1" in k64, k64)

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {FAILED}")
    sys.exit(1)
print("전부 통과")
