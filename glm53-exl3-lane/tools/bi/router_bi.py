import torch, time
import torch.nn.functional as F
from vllm.model_executor.layers.batch_invariant import matmul_persistent
torch.manual_seed(0); dev = "cuda"
K, N = 4096, 288
W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
X = torch.randn(9236, K, device=dev).to(torch.bfloat16)
lin = lambda x: F.linear(x, W)
def pad16(x):
    out = []
    for i in range(0, x.shape[0], 16):
        c = x[i:i+16]
        if c.shape[0] < 16:
            c = torch.cat([c, c.new_zeros(16 - c.shape[0], K)])
        out.append(F.linear(c, W))
    return torch.cat(out)[: x.shape[0]]
pers = lambda x: matmul_persistent(x, W.t())
ref1 = lin(X[:1])[0]
for name, f in [("cublas", lin), ("cublas_pad16", pad16), ("persistent", pers)]:
    base = f(X[:16])
    bad = [M for M in [1,2,3,5,8,12,16,17,20,32,64,100,512,1540,4613,9236] if not torch.equal(f(X[:M])[:min(M,16)], base[:min(M,16)])]
    rerun = torch.equal(f(X[:37]), f(X[:37]))
    pos = all(torch.equal(f(torch.roll(X[:16], s, 0))[s], base[0]) for s in range(16))
    print(f"{name:13s} rows differ from M=16 at M={bad}  rerun_same={rerun}  pos_invariant={pos}")
def tm(f, it=200):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
for M in [5, 12, 20, 1540, 4613, 9236]:
    x = X[:M]
    print(f"M={M:5d}  cublas {tm(lambda: lin(x)):8.1f}us  pad16 {tm(lambda: pad16(x), 20 if M>100 else 200):9.1f}us  persistent {tm(lambda: pers(x)):8.1f}us")
