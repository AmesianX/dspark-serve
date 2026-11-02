#!/usr/bin/env bash
# ============================================================================
#  Push the per-box files from THIS box (.66 master) to the slave (.67).
#
#  vllm-dspark.sh mounts serve-dspark.sh and reasoning-patch/*.py into the
#  container from LOCAL disk on EACH node, and the launcher itself (vllm-dspark.sh)
#  is executed on both nodes (master SSHes `bash $SELF START` to the slave). So
#  after editing any of these on .66 they MUST be pushed here, or the slave runs
#  STALE code (e.g. served-model-name mismatch → rank0/rank1 disagree).
#
#  Gateway files (encrypt_gateway.py / run-gateway.sh) run ONLY on .66 → excluded.
#  Model weights already live on both boxes → excluded.
#
#  Usage:  ./sync-67.sh            (rsync + md5 verify each item)
#          SLAVE=192.168.1.67 ./sync-67.sh   (override host)
# ============================================================================
set -uo pipefail

SLAVE="${SLAVE:-10.0.1.2}"
DIR=/home/user/work/dspark-serve
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8"
FILES=(serve-dspark.sh vllm-dspark.sh)
DIRS=(reasoning-patch)

$SSH "$SLAVE" "mkdir -p $DIR" || { echo "ERR: slave unreachable ($SLAVE)"; exit 1; }
echo "== sync $DIR  ->  $SLAVE =="
fail=0

for f in "${FILES[@]}"; do
    [ -f "$DIR/$f" ] || { echo "skip $f (missing locally)"; continue; }
    rsync -a "$DIR/$f" "$SLAVE:$DIR/$f"
    l=$(md5sum "$DIR/$f" | awk '{print $1}')
    r=$($SSH "$SLAVE" "md5sum '$DIR/$f' 2>/dev/null" | awk '{print $1}')
    if [ "$l" = "$r" ]; then echo "  OK   $f    ($l)"; else echo "  FAIL $f  .66=$l  .67=${r:-none}"; fail=1; fi
done

for d in "${DIRS[@]}"; do
    [ -d "$DIR/$d" ] || { echo "skip $d/ (missing locally)"; continue; }
    rsync -a --delete "$DIR/$d/" "$SLAVE:$DIR/$d/"
    l=$(find "$DIR/$d" -type f -exec md5sum {} \; | sort -k2 | md5sum | awk '{print $1}')
    r=$($SSH "$SLAVE" "find '$DIR/$d' -type f -exec md5sum {} \; | sort -k2 | md5sum" | awk '{print $1}')
    if [ "$l" = "$r" ]; then echo "  OK   $d/   ($l)"; else echo "  FAIL $d/  .66=$l  .67=${r:-none}"; fail=1; fi
done

if [ "$fail" -eq 0 ]; then echo "== all synced (md5 verified) =="; else echo "== MISMATCH — check above =="; exit 1; fi
