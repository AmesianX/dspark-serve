"""Extract the abliteration edit as a low-rank adapter (o_proj only).

The lovesenko abliteration is an output-space projection applied to
``self_attn.o_proj`` on 45 of the 46 decoder layers (the MTP layer is byte
identical). Those tensors are plain BF16 even in the 4bpw EXL3 checkpoint --
only the MoE experts are trellis-quantized -- so the difference between the
original and the abliterated checkpoint can be reproduced exactly at serving
time instead of being baked into the weights:

    o_proj(x) + scale * (x @ A.T) @ B.T        scale in {0, 1}

with ``W_abliterated = W_original + B @ A``. scale 0 is the original model bit
for bit; scale 1 is the abliterated model bit for bit (up to the rank cut).

Rank: measured 19-20 per layer for 99.9% of the delta energy, so the default
keeps 24 and reports the residual. At rank 24 the adapter is ~29MB in BF16,
which is nothing next to the 8-9GB a second copy of the o_proj stack costs.

Usage:
    python3 build_ab_lora.py [--rank 24] [--base <original snapshot>] \
        [--abl <abliterated snapshot dir | o_proj file from tools/fetch_oproj.py>] \
        [--out ~/Models/glm53-ab-lora]

Runs on CPU, one layer at a time (each delta is at most 128MB), so it is safe
to run while the server is up.
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ORIGINAL = os.path.expanduser("~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw")
ABLITERATED = os.path.expanduser(
    "~/Models/hfcache/hub/"
    "models--lovesenko--GLM-5.3-Flash-tr3-4bpw-Abliterated/"
    "snapshots/c8f58e6aa9117c73607d692978b22f091d80450c"
)


def weight_map(root: str) -> dict[str, str]:
    """Snapshot dir -> its index; a single safetensors file -> every key in it."""
    if os.path.isfile(root):
        with safe_open(root, "pt") as f:
            return {k: root for k in f.keys()}
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        return {k: os.path.join(root, v) for k, v in json.load(f)["weight_map"].items()}


def load(root: str, wmap: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(wmap[key], "pt") as f:
        return f.get_tensor(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--base", default=ORIGINAL)
    ap.add_argument("--abl", default=ABLITERATED)
    ap.add_argument("--out", default=os.path.expanduser("~/Models/glm53-ab-lora"))
    args = ap.parse_args()

    wa, wb = weight_map(args.base), weight_map(args.abl)
    keys = sorted(
        (k for k in wa if k.endswith("self_attn.o_proj.weight")),
        key=lambda k: int(k.split("layers.")[1].split(".")[0]),
    )
    print(f"o_proj 텐서 {len(keys)}개, rank {args.rank}\n")

    out: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {"rank": args.rank, "layers": {}}
    print(f"{'층':>3} {'shape':>15} {'잔차/ΔW':>9} {'ΔW/W':>8}")
    for key in keys:
        layer = int(key.split("layers.")[1].split(".")[0])
        w_base = load(args.base, wa, key).to(torch.float32)
        delta = load(args.abl, wb, key).to(torch.float32) - w_base
        if not delta.any():
            print(f"{layer:3d} {str(tuple(w_base.shape)):>15} {'편집 없음':>9}")
            meta["layers"][str(layer)] = {"edited": False}
            continue

        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(args.rank, s.numel())
        # delta ~= B @ A with B = U[:, :r] * S[:r], A = Vh[:r]
        b = (u[:, :r] * s[:r]).contiguous()
        a = vh[:r].contiguous()
        residual = (delta - b @ a).norm() / delta.norm()

        out[f"layers.{layer}.A"] = a.to(torch.bfloat16)
        out[f"layers.{layer}.B"] = b.to(torch.bfloat16)
        meta["layers"][str(layer)] = {
            "edited": True,
            "residual": round(residual.item(), 6),
            "delta_over_w": round((delta.norm() / w_base.norm()).item(), 6),
            "in_features": int(delta.shape[1]),
        }
        print(
            f"{layer:3d} {str(tuple(w_base.shape)):>15} "
            f"{residual:9.5f} {delta.norm() / w_base.norm():8.4f}"
        )

    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "ab_lora.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    nbytes = sum(t.numel() * t.element_size() for t in out.values())
    edited = sum(1 for v in meta["layers"].values() if v.get("edited"))
    print(f"\n편집된 층 {edited}개, 어댑터 {nbytes / 2**20:.0f}MB → {args.out}")


if __name__ == "__main__":
    main()
