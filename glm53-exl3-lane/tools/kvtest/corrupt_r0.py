import glob, json, os, struct, sys
H = struct.Struct("<8sII32sHIQIQd64s"); SB = 31027200
win = json.loads(sys.stdin.readline())          # {"B": [t0,t1], "C": [t0,t1]}
p = glob.glob("/kv-offload/*/r0.kv")[0]; fd = os.open(p, os.O_RDWR)
by = {k: [] for k in win}
for slot in range(6445):
    b = os.pread(fd, H.size, slot * SB)
    if b[:8] != b"GLMKVNV1": continue
    f = H.unpack(b)
    for k, (t0, t1) in win.items():
        if t0 <= f[9] <= t1 and f[5] == 0: by[k].append((f[8], slot))
for k in by: by[k].sort()
sb = by["B"][len(by["B"]) // 2][1]; sc = by["C"][len(by["C"]) // 2][1]
off = sb * SB + 4096 + 777777; old = os.pread(fd, 1, off); os.pwrite(fd, bytes([old[0] ^ 0xFF]), off)
os.pwrite(fd, b"\0" * 4096, sc * SB)
os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
print(json.dumps({"B_chunks": len(by["B"]), "B_payload_flipped_slot": sb, "C_chunks": len(by["C"]), "C_header_zeroed_slot": sc,
                  "B_rank_of_slot": [s for _, s in by["B"]].index(sb), "C_rank_of_slot": [s for _, s in by["C"]].index(sc)}))
