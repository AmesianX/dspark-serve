import torch
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
L = torch.randn(4613, 288, device=dev) * 2
bias = torch.randn(288, device=dev) * 0.1
f = lambda x: ops.grouped_topk(x, 1, 1, 8, True, 2.5, bias, 1)
W = []; I = []
for s in range(0, 4613, 1024):
    w, i = f(L[s:s+1024]); W.append(w.clone()); I.append(i.clone())
W = torch.cat(W); I = torch.cat(I)
rows = list(range(0, 4613, 7))
bw = [r for r in rows if not torch.equal(W[r], f(L[r:r+1])[0][0])]
bi = [r for r in rows if not torch.equal(I[r], f(L[r:r+1])[1][0])]
print("chunk1024 vs M=1: w mismatch", len(bw), "idx mismatch", len(bi), "of", len(rows), bw[:5])
