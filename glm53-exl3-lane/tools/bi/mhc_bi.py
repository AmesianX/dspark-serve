import torch, time
import vllm.model_executor.kernels.mhc.tilelang as T
torch.manual_seed(0); dev = "cuda"
H, hc = 4096, 4; hc3 = hc * 2 + hc * hc
Mmax = 4613
x = (torch.randn(Mmax, H, device=dev) * 0.5).to(torch.bfloat16)
res = (torch.randn(Mmax, hc, H, device=dev) * 0.5).to(torch.bfloat16)
post = torch.rand(Mmax, hc, device=dev, dtype=torch.float32)
comb = torch.rand(Mmax, hc, hc, device=dev, dtype=torch.float32)
fn = torch.randn(hc3, hc * H, device=dev, dtype=torch.float32) * 0.01
sc = torch.rand(3, device=dev); base = torch.randn(hc3, device=dev) * 0.1
nw = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
def run(M, force):
    T._BI_FORCE[0] = force
    o = T._mhc_fused_post_pre_orig(x[:M], res[:M], post[:M], comb[:M], fn, sc, base, 1e-6, 1e-6, 1e-6, 2.0, 20, 1, 1, nw, 1e-5)
    T._BI_FORCE[0] = None
    return [t.clone() for t in o]
for force, Ms in (("small", [1, 5, 12, 20, 994]), ("big", [5, 64, 994, 1540, 4613])):
    ref = run(Ms[-1], force)
    for M in Ms[:-1]:
        o = run(M, force)
        eq = all(torch.equal(a[:M], b[:M]) for a, b in zip(o, ref))
        print(f"{force:5s} M={M:5d} rows == M={Ms[-1]} rows: {eq}")
def tm(f, it=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
for M in (5, 994, 4613):
    print(f"M={M}: upstream {tm(lambda: run(M, None)):.3f}ms  small {tm(lambda: run(M, 'small')):.3f}ms  big {tm(lambda: run(M, 'big')):.3f}ms")
