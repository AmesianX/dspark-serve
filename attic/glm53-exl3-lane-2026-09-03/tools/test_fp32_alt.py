import torch, time
torch.manual_seed(0); dev="cuda"; COPIES=48
def bench(f, n=192):
    for i in range(8): f(i)
    torch.cuda.synchronize(); t=time.perf_counter()
    for i in range(n): f(i)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
for M in (1,5,8,16,20):
    x=torch.randn(M,4096,device=dev); xb=x.to(torch.bfloat16)
    W_nk=[torch.randn(64,4096,device=dev)*0.05 for _ in range(COPIES)]        # [N,K] fp32 (F.linear layout)
    W_kn=[w.t().contiguous() for w in W_nk]                                   # [K,N] fp32 (current torch.mm layout)
    ref=x@W_kn[0]
    cands={
      "mm(x,W_kn) [현재]": lambda i: torch.mm(x, W_kn[i%COPIES]),
      "F.linear(x,W_nk)": lambda i: torch.nn.functional.linear(x, W_nk[i%COPIES]),
      "(W_nk@x.T).T": lambda i: (W_nk[i%COPIES] @ x.t()).t(),
      "einsum": lambda i: torch.einsum("mk,nk->mn", x, W_nk[i%COPIES]),
      "matmul x@W_nk.T(view)": lambda i: torch.matmul(x, W_nk[i%COPIES].t()),
      "mv loop M": lambda i: torch.stack([torch.mv(W_nk[i%COPIES], x[m]) for m in range(M)]),
    }
    print(f"== fp32 M={M} N=64 K=4096 ==")
    for name,f in cands.items():
        err=((f(0)-ref).abs().max()/ref.abs().max()).item(); print(f"  {name:26s} {bench(f):7.1f}us  rel-err={err:.1e}")
    # bf16 skinny variants (N=192 like wk_weights_proj, N=288 router->fp32)
    for N in (192,288):
        Wb=[torch.randn(N,4096,device=dev,dtype=torch.bfloat16)*0.05 for _ in range(COPIES)]
        cb={
          "F.linear bf16": lambda i: torch.nn.functional.linear(xb, Wb[i%COPIES]),
          "mm out_dtype=fp32": lambda i: torch.mm(xb, Wb[i%COPIES].t(), out_dtype=torch.float32),
          "F.linear then .float()": lambda i: torch.nn.functional.linear(xb, Wb[i%COPIES]).float(),
        }
        print(f"== bf16 M={M} N={N} ==")
        for name,f in cb.items():
            try: print(f"  {name:26s} {bench(f):7.1f}us")
            except Exception as e: print(f"  {name:26s} ERR {str(e)[:60]}")
