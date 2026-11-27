import torch, torch.nn.functional as F
torch.manual_seed(0); dev = "cuda"
@torch.compile(dynamic=True)
def ln(x, w, b, dim: int, eps: float):
    return F.layer_norm(x.float(), (dim,), w, b, eps).type_as(x)
D = 128
w = torch.randn(D, device=dev) * 0.1 + 1; b = torch.randn(D, device=dev) * 0.1
X = (torch.randn(4000, D, device=dev) * 3).to(torch.bfloat16)
ref = {M: ln(X[:M], w, b, D, 1e-6) for M in (5, 16, 989, 1024, 2975)}
for M in (16, 989, 1024, 2975):
    n = sum((~(ref[M][:5] == ref[5])).any(-1).tolist()) if M >= 5 else 0
    full = (~(ref[M][:989] == ref[989][:989])).any(-1).sum().item() if M >= 989 else -1
    print(f"LN M={M:5d}: rows(0..4) differing vs M=5: {n}   rows(0..988) differing vs M=989: {full}")
# cuBLAS fixed 1024 full sweep
Wb = (torch.randn(144, 4096, device=dev) * .02).to(torch.bfloat16)
Y = torch.randn(1024, 4096, device=dev).to(torch.bfloat16)
r0 = F.linear(Y, Wb)
bad = 0
for off in range(1024):
    c = torch.roll(Y, off, 0)
    if not torch.equal(F.linear(c, Wb)[off], r0[0]): bad += 1
print("cuBLAS fixed-1024 full position sweep: rows differing", bad, "/ 1024")
