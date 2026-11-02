"""Gate the AOT-installed package (not the JIT build): the artifact that ships.

Run with the install target on PYTHONPATH, e.g.
    PYTHONPATH=/tmp/glm53site python tests/run_installed.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Import the package FIRST, while the repo root is not yet on sys.path, so the
# installed copy wins over the source tree -- this gate exists to test the
# shipped artifact.
import torch  # noqa: E402
import glm53_sparse_mla as G  # noqa: E402

# Only now add the repo root, at the front, so `tests.*` resolves to these files
# and not to some other `tests` package that happens to be importable in the
# container.
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "_stubs"))

from tests.cases import DIM, cases  # noqa: E402
from tests.reference import sparse_mla_fp32  # noqa: E402

REL_TOL, COS_TOL = 4e-3, 0.9999


def main():
    print("package:", G.__file__)
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor}")
    ok = True
    for name, q, kv, idx in cases():
        out = G.sparse_fwd(q, kv, idx, DIM ** -0.5)
        torch.cuda.synchronize()
        ref, _ = sparse_mla_fp32(q, kv, idx, DIM ** -0.5)
        got, ref = out.float(), ref.float()
        finite = bool(torch.isfinite(got).all())
        err = (got - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-6)
        fg, fr = got.reshape(-1, DIM), ref.reshape(-1, DIM)
        nz = fr.abs().sum(-1) > 0
        cos = torch.nn.functional.cosine_similarity(fg, fr, dim=-1)
        cmin = cos[nz].min().item() if nz.any() else 1.0
        good = finite and rel <= REL_TOL and cmin >= COS_TOL
        if name == "row_fully_masked":
            good = good and bool((out[2] == 0).all())
        ok &= good
        print(f"  {name:20s} rel={rel:.3e} cos_min={cmin:.6f} {'OK' if good else 'FAIL'}")
    print()
    print("INSTALLED-PACKAGE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
