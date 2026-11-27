import uuid
from tokenizers import Tokenizer
t = Tokenizer.from_file("/m/tokenizer.json")
def hl(tag): return len(t.encode(f"[세션 {tag} {uuid.uuid5(uuid.NAMESPACE_DNS, tag)}]\n", add_special_tokens=False).ids)
base = {n: hl(f"E5-{n}") for n in "ABCD"}
print("ABCD header tokens:", base)
used = set(base.values())
for i in range(2000):
    tag = f"E5-C{'z' * (i % 9)}{i}"; h = hl(tag)
    if all(abs(h - u) >= 3 for u in used):
        print("pick", tag, h); break
