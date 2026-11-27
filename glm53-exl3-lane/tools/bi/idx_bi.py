import torch, time
import torch.nn.functional as F
torch.manual_seed(0); dev = "cuda"
X = torch.randn(9236, 4096, device=dev).to(torch.bfloat16)
cases = {
  "wk_proj bf16 4096->144": (lambda W: (lambda x: F.linear(x, W)))((torch.randn(144, 4096, device=dev) * .02).to(torch.bfloat16)),
  "kpool_gate bf16 4096->128": (lambda W: (lambda x: F.linear(x, W)))((torch.randn(128, 4096, device=dev) * .02).to(torch.bfloat16)),
  "weights fp32 16x4096 @ x^T": (lambda Wnk: (lambda x: (Wnk @ x.float().t()).t().clone(memory_format=torch.contiguous_format)))(torch.randn(16, 4096, device=dev) * .02),
}
def pad_to(x, n):
    return x if x.shape[0] == n else torch.cat([x, x.new_zeros(n - x.shape[0], x.shape[1])])
for name, f in cases.items():
    single = [f(pad_to(X[r:r+1], 2))[0] for r in range(16)]
    bad_small = [M for M in range(2, 17) if not all(torch.equal(f(X[:M])[r], single[r]) for r in range(M))]
    ref1024 = f(X[:1024])
    pos_ok = torch.equal(f(torch.roll(X[:1024], 300, 0))[300], ref1024[0])
    bad_big = [M for M in [17, 64, 512, 1023, 1025, 4613] if not torch.equal(f(X[:M])[:16], ref1024[:16])]
    print(f"{name:28s} 2..16 differ from M=2-single at {bad_small}; fixed-1024 position-invariant {pos_ok}; M!=1024 differ at {bad_big}")
def tm(fn, it=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for name, f in cases.items():
    print(f"{name:28s} M=5 {tm(lambda: f(X[:5])):.1f}us  M=16 {tm(lambda: f(X[:16])):.1f}us  M=1024 {tm(lambda: f(X[:1024])):.1f}us  M=9236 {tm(lambda: f(X[:9236])):.1f}us")
