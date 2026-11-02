import torch, time
torch.manual_seed(0); dev="cuda"; COPIES=48
def bench(f, n=192):
    for i in range(8): f(i)
    torch.cuda.synchronize(); t=time.perf_counter()
    for i in range(n): f(i)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
print("defaults: bf16_rpr=",torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,"fp16_rpr=",torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,"tf32=",torch.backends.cuda.matmul.allow_tf32)
xb=torch.randn(5,4096,device=dev,dtype=torch.bfloat16)
for flag in (True, False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
    for N in (128,192,288):
        Wb=[torch.randn(N,4096,device=dev,dtype=torch.bfloat16)*0.05 for _ in range(COPIES)]
        t1=bench(lambda i: torch.nn.functional.linear(xb, Wb[i%COPIES]))
        t2=bench(lambda i: torch.mm(xb, Wb[i%COPIES].t(), out_dtype=torch.float32))
        t3=bench(lambda i: (Wb[i%COPIES] @ xb.t()).t())
        t4=bench(lambda i: torch.mm(Wb[i%COPIES], xb.t(), out_dtype=torch.float32).t())
        print(f"bf16_rpr={flag!s:5s} M=5 N={N:3d}: F.linear={t1:5.1f}us  mm(out fp32)={t2:5.1f}us  (W@x.T).T={t3:5.1f}us  (W@x.T out fp32).T={t4:5.1f}us")
