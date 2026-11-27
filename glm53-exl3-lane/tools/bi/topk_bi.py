import torch
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
E, K = 288, 8
L = torch.randn(9236, E, device=dev, dtype=torch.float32) * 2
bias = torch.randn(E, device=dev, dtype=torch.float32) * 0.1
def f(x):
    return ops.grouped_topk(x, 1, 1, K, True, 2.5, bias, 1)
w16, i16 = f(L[:16])
bad = []
for M in [1, 2, 3, 5, 8, 12, 16, 17, 20, 32, 33, 64, 65, 128, 256, 512, 1024, 1540, 4613, 9236]:
    w, i = f(L[:M]); m = min(M, 16)
    if not (torch.equal(w[:m], w16[:m]) and torch.equal(i[:m], i16[:m])): bad.append(M)
print("grouped_topk rows differ from M=16 at M =", bad)
w, i = f(L[:37]); w2, i2 = f(L[:37]); print("rerun same:", torch.equal(w, w2) and torch.equal(i, i2))
first = None
for M in range(1000, 1541):
    w, i = f(L[:M])
    if not (torch.equal(w[:16], w16) and torch.equal(i[:16], i16)): first = M; break
print("first differing M:", first)
# chunked at 1024: identical to M=16 for every row?
def chunked(x, c=1024):
    ws, is_ = zip(*[f(x[s:s+c]) for s in range(0, x.shape[0], c)]); return torch.cat(ws), torch.cat(is_)
wa, ia = chunked(L[:9236]); wb, ib = chunked(L[:4613])
ref = [f(L[s:s+16]) for s in range(0, 4613, 16)]
rw = torch.cat([r[0] for r in ref]); ri = torch.cat([r[1] for r in ref])
print("chunked(1024) rows == 16-row calls:", torch.equal(wb, rw) and torch.equal(ib, ri), torch.equal(wa[:4613], rw))
