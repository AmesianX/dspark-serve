import glob, os, struct
H = struct.Struct("<8sII32sHIQIQd64s"); SB = 31027200
p = glob.glob("/kv-offload/*/r1.kv")[0]
fd = os.open(p, os.O_RDWR)
rows = []
for slot in range(6445):
    b = os.pread(fd, H.size, slot * SB)
    if b[:8] == b"GLMKVNV1":
        f = H.unpack(b); rows.append((f[8], slot, f[5]))
rows.sort()
recent = [r for r in rows[-30:] if r[2] == 0]          # newest group-0 chunks (fixed-seed run)
seq, slot, g = recent[len(recent) // 2]
off = slot * SB + 4096 + 123457
old = os.pread(fd, 1, off)
os.pwrite(fd, bytes([old[0] ^ 0xFF]), off); os.fsync(fd)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
print("corrupted", p, "slot", slot, "group", g, "byte", old.hex(), "->", bytes([old[0] ^ 0xFF]).hex())
