"""Turn an abliterated install into the ORIGINAL model (brandonmusic/GLM-5.3-Flash-tr3-4bpw).

For anyone who already has lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated on disk:
the two checkpoints differ only in the 46 BF16 o_proj tensors, so the original is
rebuilt from the abliterated snapshot plus the 3.89 GB original o_proj file
(tools/fetch_oproj.py --source original). The 44 shards that hold an o_proj are
copied and patched in place; the other 76 shards and all config/tokenizer files are
HARD-linked (symlinks break inside the container). Extra disk ~72 GB.
The result is byte identical to the original repository's shards.
Run on EACH node — weights must be local to each box.

    python3 tools/build_original_from_abliterated.py \
        [--src <abliterated snapshot>] [--base <o_proj_original.safetensors>] [--dst <out dir>]
"""
import argparse, json, os, struct, shutil, sys, hashlib
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=os.path.expanduser("~/Models/hfcache/hub/models--lovesenko--GLM-5.3-Flash-tr3-4bpw-Abliterated/snapshots/c8f58e6aa9117c73607d692978b22f091d80450c"))
_ap.add_argument("--base", default=os.path.expanduser("~/Models/glm53-oproj/o_proj_original.safetensors"))
_ap.add_argument("--dst", default=os.path.expanduser("~/Models/brandonmusic--GLM-5.3-Flash-tr3-4bpw"))
_a = _ap.parse_args()
SRC, BASE, DST = _a.src, _a.base, _a.dst
def hdr(p):
    with open(p,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        return 8+n, json.loads(f.read(n))
boff,bh=hdr(BASE)
idx=json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
o={k:v for k,v in idx.items() if "o_proj" in k}
shards={}
for k,v in o.items(): shards.setdefault(v,[]).append(k)
print(f"대상 샤드 {len(shards)}개, 텐서 {len(o)}개 → {DST}")
os.makedirs(DST,exist_ok=True)
# 1) 비-샤드 파일과 영향 없는 샤드는 하드링크
for f in sorted(os.listdir(SRC)):
    d=f"{DST}/{f}"; srcf=f"{SRC}/{f}"
    if os.path.lexists(d) or f in shards: continue
    if os.path.isdir(srcf): continue          # 로딩에 불필요
    os.link(srcf, d)                          # 하드링크(컨테이너서 심링크는 깨짐)
# 2) 영향받는 샤드: 복사 후 바이트 패치
for i,(sh,keys) in enumerate(sorted(shards.items()),1):
    dst=f"{DST}/{sh}"
    if not os.path.exists(dst) or os.path.getsize(dst)!=os.path.getsize(f"{SRC}/{sh}"):
        shutil.copyfile(f"{SRC}/{sh}", dst)
    loff,lh=hdr(dst)
    with open(BASE,"rb") as bf, open(dst,"r+b") as df:
        for k in keys:
            lm, bm = lh[k], bh[k]
            assert lm["shape"]==bm["shape"] and lm["dtype"]==bm["dtype"], f"불일치 {k}"
            ls,le=lm["data_offsets"]; bs,be=bm["data_offsets"]
            assert le-ls == be-bs, f"길이 불일치 {k}"
            bf.seek(boff+bs); data=bf.read(be-bs)
            assert len(data)==le-ls
            df.seek(loff+ls); df.write(data)
    print(f"  [{i}/{len(shards)}] {sh} ← {len(keys)}개", flush=True)
print("완료")
