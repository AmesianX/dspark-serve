#!/usr/bin/env bash
# MemAvailable + 상위 RSS 소비자를 0.2s 간격 샘플링. 가드가 트립하기 전에
# "누가 먹었나"를 남기는 게 목적. 사용: ./memtrace.sh [출력파일]
OUT="${1:-/tmp/memtrace_$(date +%Y%m%d_%H%M%S).log}"
echo "# ts  MemAvail_MB  top5(rss_MB:comm)" > "$OUT"
prev=""
while true; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    top=$(ps -eo rss,comm --sort=-rss --no-headers | head -5 | awk '{printf "%d:%s ", $1/1024, $2}')
    line="$(date +%T.%2N)  ${avail}  ${top}"
    echo "$line" >> "$OUT"
    # 여유가 6GB 밑으로 떨어지면 그 순간 전체 스냅샷을 따로 덤프
    if [ "$avail" -lt 6144 ]; then
        {
          echo "=== $(date +%T.%2N) MemAvailable=${avail}MB SNAPSHOT ==="
          ps -eo pid,rss,etime,args --sort=-rss --no-headers | head -20
          echo "--- docker:"; docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null
        } >> "${OUT}.snap"
    fi
    sleep 0.2
done
