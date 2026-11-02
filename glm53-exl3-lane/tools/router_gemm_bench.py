"""Router GEMM [M,4096]x[4096,288] bf16->fp32 on GB10: cuBLAS (current tier 5)
vs vLLM's cuteDSL ll_bf16 (dotprod / clustered split-K). Correctness + graph timing.
Run ONLY with the serving stack stopped (second GPU process on GB10 = OOM risk):
  docker run --rm --gpus all --network none -v <lane>:/lane:ro -v <weights>:/w:ro <image> python /lane/tools/router_gemm_bench.py
"""
import json, os, sys, time, traceback
import torch
import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as L

torch.manual_seed(0)
dev = "cuda"
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "cutedsl", L.is_available(), "pdl", L._use_pdl())
K, N = 4096, 288

# real gate weight (layer 3 = first sparse layer) if available
W = None
try:
    from safetensors import safe_open
    idx = json.load(open("/w/model.safetensors.index.json"))["weight_map"]
    key = next(k for k in idx if k.endswith("layers.3.mlp.gate.weight"))
    with safe_open(os.path.join("/w", idx[key]), "pt") as f:
        W = f.get_tensor(key).to(dev, torch.bfloat16).contiguous()
    print("gate weight", key, tuple(W.shape), W.dtype)
except Exception as e:
    print("real gate weight unavailable:", e)
if W is None:
    W = (torch.randn(N, K, device=dev) * 0.02).bfloat16()

def cublas(x, w=None): return torch.mm(x, (W if w is None else w).T, out_dtype=torch.float32)
WS = [W.clone() for _ in range(43)]   # 43 layers x 2.36MB = 101MB >> L2: each call reads DRAM, as in serving

def make_kernel(backend, M, bs=128, split=(6, 4)):
    g = L.LLBf16Gemm()
    ck = (g.CompileKey(backend="dotprod", M=M, K=K, bs=bs) if backend == "dotprod"
          else g.CompileKey(backend="splitk", split_k=split[0], num_stages=split[1]))
    g.compile(ck)
    kern = g._compiled_cache[(M, K, bs)] if backend == "dotprod" else g._splitk_cache[split]
    def run(x, out, w=None):
        w = W if w is None else w
        if backend == "dotprod": kern(x, w, out, N, L._stream())
        else: kern(x, w, out, L._stream(), 1.0)
        return out
    return run

def graph_time(fn, x, out, calls=43, reps=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in WS[:3]: fn(x, out, w)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in WS[:calls]: fn(x, out, w)
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / calls * 1e6  # us per call

rows = []
for M in (1, 2, 4, 5, 8, 10, 16, 20):
    x = torch.randn(M, K, device=dev).bfloat16()
    ref64 = (x.double() @ W.double().T)
    base = cublas(x)
    res = {"M": M, "cublas_us": graph_time(lambda a, o, w=None: o.copy_(cublas(a, w)), x, torch.empty(M, N, device=dev)),
           "cublas_err": (base.double() - ref64).abs().max().item()}
    VARIANTS = (("dot128", dict(backend="dotprod", bs=128)), ("dot256", dict(backend="dotprod", bs=256)),
                ("splitk64", dict(backend="splitk", split=(6, 4))), ("splitk82", dict(backend="splitk", split=(8, 2))))
    only = os.environ.get("ONLY")  # one variant per process: a launch failure poisons the context
    for name, kw in VARIANTS:
        if only and name != only: continue
        try:
            fn = make_kernel(M=M, **kw); out = torch.empty(M, N, device=dev)
            fn(x, out); torch.cuda.synchronize()
            err = (out.double() - ref64).abs().max().item()
            top_same = (out.topk(8, dim=-1).indices.sort(-1).values == base.topk(8, dim=-1).indices.sort(-1).values).all().item()
            res[name] = {"us": graph_time(fn, x, out), "err": err, "top8_same_as_cublas": top_same}
        except Exception as e:
            res[name] = {"FAIL": f"{type(e).__name__}: {str(e)[:160]}"}
            print(json.dumps(res), flush=True); sys.exit(1)
    rows.append(res); print(json.dumps(res), flush=True)

if os.environ.get("ONLY", "dot128") != "dot128": sys.exit(0)
# top-8 agreement on many rows (routing flips), best dotprod vs cublas
x = torch.randn(4096, K, device=dev).bfloat16() * 1.5
agree = tot = 0
try:
    fn = make_kernel("dotprod", M=4, bs=128)
    for i in range(0, x.shape[0], 4):
        xb = x[i:i+4].contiguous(); o = torch.empty(4, N, device=dev); fn(xb, o)
        a = o.topk(8, -1).indices.sort(-1).values; b = cublas(xb).topk(8, -1).indices.sort(-1).values
        agree += (a == b).all(-1).sum().item(); tot += 4
    print(f"top8 set agreement dotprod vs cuBLAS over {tot} random rows: {agree}/{tot}")
except Exception as e:
    print("agreement test failed:", e)
