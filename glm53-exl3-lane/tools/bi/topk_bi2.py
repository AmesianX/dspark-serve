import torch
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
E, K = 288, 8
L = torch.randn(4613, E, device=dev, dtype=torch.float32) * 2
bias = torch.randn(E, device=dev, dtype=torch.float32) * 0.1
f = lambda x: ops.grouped_topk(x, 1, 1, K, True, 2.5, bias, 1)
single = [f(L[r:r+1]) for r in range(0, 300)]
for M in [2, 16, 17, 64, 300, 1024]:
    w, i = f(L[:M])
    nw = sum(not torch.equal(w[r], single[r][0][0]) for r in range(min(M, 300)))
    ni = sum(not torch.equal(i[r], single[r][1][0]) for r in range(min(M, 300)))
    # same set of experts, maybe different order?
    ns = sum(not torch.equal(i[r].sort()[0], single[r][1][0].sort()[0]) for r in range(min(M, 300)))
    print(f"M={M:5d}: rows w!=M1 {nw}, idx!=M1 {ni}, idx-set!=M1 {ns}")
