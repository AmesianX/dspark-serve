import torch
import vllm._flashkda_C  # noqa
torch.manual_seed(0); dev = "cuda"
T, H, D = 989, 32, 128
q = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
k = torch.nn.functional.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).to(torch.bfloat16)
v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
A_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.1
dt_bias = torch.randn(H * D, device=dev, dtype=torch.float32) * 0.1
init = torch.randn(1, H, D, D, device=dev, dtype=torch.float32) * 0.1
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
def run(fill=None):
    out = torch.empty(v.shape, dtype=v.dtype, device=dev); fin = torch.empty_like(init)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T * H, D, 1), dtype=torch.uint8, device=dev)
    if fill is not None: ws.fill_(fill)
    torch.ops._flashkda_C.fwd(q, k, v, g, beta, D ** -0.5, out, ws, A_log, dt_bias.view(-1, D), -5.0, init, fin, cu)
    return out.clone(), fin.clone()
ref = run(0)
same = sum(torch.equal(run(0)[0], ref[0]) for _ in range(30))
print("zero-workspace repeats identical:", same, "/ 30")
for f in (1, 77, 255):
    o, s = run(f); print(f"workspace filled {f}: out equal {torch.equal(o, ref[0])} state equal {torch.equal(s, ref[1])}  maxdiff {(o.float()-ref[0].float()).abs().max().item():.2e}")
same = sum(torch.equal(run()[0], ref[0]) for _ in range(30))
print("uninit-workspace repeats identical:", same, "/ 30")
