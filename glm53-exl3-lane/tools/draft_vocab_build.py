"""Build the draft-only vocabulary (draft_vocab_32k.json) and report held-out coverage.

Inputs in the working dir: gen.json (draft_vocab_corpus_gen.py, needs the server up) and
cm.json (draft_vocab_extract_codemason.py). Tokenizer/model are read from /w, so run it in
a CPU-only container of the serving image:
  W=~/Models/hfcache/hub/models--lovesenko--GLM-5.3-Flash-tr3-4bpw-Abliterated
  docker run --rm --network none --entrypoint python -w /d -v <workdir>:/d \
    -v $W/snapshots/c8f58e6aa9117c73607d692978b22f091d80450c:/w:ro -v $W/blobs:$W/blobs:ro \
    ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1 /d/draft_vocab_build.py
Set = mandatory (specials, byte tokens, UTF-8 fragments, single Hangul, single KS X 1001 hanja)
+ corpus frequency + BPE merge-rank fill, with zh-only han / ja-only han / kana /
Cyrillic / Arabic / Thai / Devanagari excluded. The corpus is sampled at temperature 1.0, so a
rebuild is not bit-identical; the tracked JSON is the artifact of record.
"""
import json, collections
from tokenizers import Tokenizer
tk=Tokenizer.from_file("/w/tokenizer.json")
tj=json.load(open("/w/tokenizer.json")); vocab=tj["model"]["vocab"]; added={t["id"] for t in tj.get("added_tokens",[])}
V=max(max(vocab.values()),max(added))+1
bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n+=1
dec={chr(c):b for b,c in zip(bs,cs)}
def is_han(o): return 0x4E00<=o<=0x9FFF or 0x3400<=o<=0x4DBF or 0xF900<=o<=0xFAFF or 0x20000<=o<=0x2FA1F
def ks(ch):
    try: ch.encode("euc_kr"); return True
    except UnicodeEncodeError: return False
def drop_script(o): return (0x0400<=o<=0x04FF or 0x0600<=o<=0x06FF or 0x0E00<=o<=0x0E7F or 0x0900<=o<=0x097F
                           or 0x3040<=o<=0x30FF or 0x31F0<=o<=0x31FF or 0xFF65<=o<=0xFF9F)
allowed=set(); mandatory=set(); kind={}
for tok,i in vocab.items():
    try: raw=bytes(dec[c] for c in tok)
    except KeyError: allowed.add(i); mandatory.add(i); kind[i]="unspelled"; continue
    if len(raw)==1: allowed.add(i); mandatory.add(i); kind[i]="byte"; continue
    try: s=raw.decode("utf-8")
    except UnicodeDecodeError: allowed.add(i); mandatory.add(i); kind[i]="fragment"; continue
    if any(drop_script(ord(ch)) for ch in s): continue
    hs=[ch for ch in s if is_han(ord(ch))]
    if hs and not all(ks(ch) for ch in hs): continue           # zh-only / ja-only han
    allowed.add(i)
    if len(hs)==1 and len(s.strip())==1: mandatory.add(i); kind[i]="hanja1"
    elif any(0xAC00<=ord(ch)<=0xD7A3 for ch in s) and len(s.strip())==1: mandatory.add(i); kind[i]="hangul1"
for i in added: allowed.add(i); mandatory.add(i); kind[i]="special"
print(f"allowed by script rule: {len(allowed):,}  mandatory: {len(mandatory):,} {collections.Counter(kind.values())}")

docs=json.load(open("gen.json"))+json.load(open("cm.json"))
freq=collections.Counter()
for d in docs:
    if d["split"]=="train": freq.update(tk.encode(d["text"],add_special_tokens=False).ids)
print(f"train tokens: {sum(freq.values()):,}  distinct: {len(freq):,}")
ranked=[t for t,_ in freq.most_common() if t in allowed and t not in mandatory]
sets={}
for K in (32768,49152,65536):
    s=set(mandatory)
    for t in ranked:
        if len(s)>=K: break
        s.add(t)
    nfreq=len(s)
    for t in sorted(allowed):                 # BPE merge-rank prior: lower id = more frequent merge
        if len(s)>=K: break
        s.add(t)
    sets[K]=s
    print(f"   K={K}: +{len(s)-nfreq:,} filled by merge rank")
    print(f"K={K}: mandatory {len(mandatory):,} + by-frequency {len(s)-len(mandatory):,} (distinct train tokens available: {len(ranked):,})")
sets["script-only"]=allowed
# evaluation on held-out docs
evald=collections.defaultdict(list)
def lang(d):
    if d["domain"]=="codemason_real": return "real(codemason)"
    return "ko-prompt" if any(0xAC00<=ord(c)<=0xD7A3 for c in d["prompt"]) else "en-prompt"
for d in docs:
    if d["split"]=="eval":
        ids=tk.encode(d["text"],add_special_tokens=False).ids
        evald[d["domain"]].append(ids); evald["LANG:"+lang(d)].append(ids)
print("\ncoverage of held-out model output (fraction of generated tokens the drafter can propose)")
hdr="domain".ljust(16)+"tokens".rjust(9)+"".join(str(k).rjust(13) for k in sets)
print(hdr)
tot=collections.Counter()
for dom,lst in sorted(evald.items()):
    ids=[t for x in lst for t in x]; row=dom.ljust(16)+f"{len(ids):9d}"
    for k,s in sets.items():
        c=sum(1 for t in ids if t in s); row+=f"{100*c/len(ids):12.2f}%"; tot[(k,'c')]+=c
    tot['n']+=len(ids); print(row)
print("ALL".ljust(16)+f"{tot['n']:9d}"+"".join(f"{100*tot[(k,'c')]/tot['n']:12.2f}%" for k in sets))
# ASCII tokens needed for English coverage targets (English-prompt eval text)
en=[t for x in evald["LANG:en-prompt"] for t in x]
id2tok={v:k for k,v in vocab.items()}
def ascii_tok(t):
    tok=id2tok.get(t)
    if tok is None: return False
    try: return all(dec[c]<128 for c in tok)
    except KeyError: return False
ascii_rank=[t for t,_ in freq.most_common() if ascii_tok(t)]
base=set(mandatory); en_cnt=collections.Counter(en)
for target in (0.99,0.995,0.999):
    s2=set(base); k=0
    need=len(en)
    covered=sum(1 for t in en if t in s2)
    for t in ascii_rank:
        if covered/need>=target: break
        if t not in s2:
            s2.add(t); k+=1; covered+=en_cnt.get(t,0)
    print(f"English eval coverage {100*target:.1f}%: needs {k:,} ASCII tokens by frequency (reached {100*covered/need:.2f}%) -> set size {len(s2):,}")
# script-excluded tokens actually emitted (eval, all domains)
ex=collections.Counter()
for d in docs:
    if d["split"]!="eval": continue
    for t in tk.encode(d["text"],add_special_tokens=False).ids:
        if t not in allowed: ex[t]+=1
tot_eval=sum(len(x) for k,v in evald.items() if not k.startswith("LANG:") for x in v)
print(f"\nscript-excluded tokens in eval output: {sum(ex.values()):,} / {tot_eval:,} ({100*sum(ex.values())/tot_eval:.2f}%)  top:", [(tk.decode([t]),c) for t,c in ex.most_common(25)])
# misses sample for 32K
miss=collections.Counter()
for lst in evald.values():
    for x in lst:
        for t in x:
            if t not in sets[32768]: miss[t]+=1
print("\ntop misses at 32K:", [ (tk.decode([t]),c) for t,c in miss.most_common(40)])
json.dump(sorted(sets[32768]),open("draft_vocab_32k.json","w"))
print("saved draft_vocab_32k.json", len(sets[32768]))
