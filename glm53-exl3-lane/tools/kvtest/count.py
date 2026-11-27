import sys, glob, collections
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload")
import importlib.util
spec = importlib.util.spec_from_file_location("g", "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/glm53_nvme.py")
p = glob.glob("/kv-offload/*/r*.kv")[0]; fp = p.split("/")[2]
import struct, zlib, os
H = struct.Struct("<8sII32sHIQIQd64s")
fd = os.open(p, os.O_RDONLY); groups = collections.Counter(); n = 0
for slot in range(6445):
    b = os.pread(fd, H.size, slot * 31027200)
    if b[:8] == b"GLMKVNV1":
        f = H.unpack(b); groups[f[5]] += 1; n += 1
print(p, "valid-magic slots:", n, "by group:", dict(groups))
