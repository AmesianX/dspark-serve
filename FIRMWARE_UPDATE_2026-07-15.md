# 펌웨어 업데이트 2026-07-15 (진행 중 → 재접속 후 마무리)

## 한 일
- LVFS 펌웨어 2건, **양 노드 모두 적용**:
  | 항목 | 이전 | 신규 |
  |---|---|---|
  | Embedded Controller | 0x03000302 | 0x03000508 |
  | SoC FW (UEFI+GPU) | 0x0200980f | 0x02009b0b |
- `.67`: 스테이징 → 리부트 → **검증 완료** (두 버전 다 반영, 대기 업데이트 0)
- `.66`: 스테이징 완료 → 리부트 발령 (capsule-on-disk, 부팅 시 플래시)
- 서빙은 `vllm-dspark.sh STOPALL`로 양 노드 정상 종료함. `restart=no`라 자동 복귀 안 됨.

## 남은 일 (재접속 후)
1. `.66` 펌웨어 버전 검증:
   `fwupdmgr get-devices --show-all | grep -A4 -E "(Embedded Controller|UEFI Device Firmware):"`
   → EC `0x03000508`, SoC `0x02009b0b` 확인. `fwupdmgr get-updates` = 대기 0.
2. 서빙 재기동: `bash /home/user/work/dspark-serve/vllm-dspark.sh RESTARTALL`
3. 회귀 확인 — **기준선 (펌웨어 이전, 2026-07-14 실측)**:
   - decode **46.4 t/s** (17k까지 무열화)
   - prefill **1480~1685** (정상상태)
   - 함정: SSE 청크 ≠ 토큰(spec decode → 3배 저평가), prefix-caching 캐시히트 → **nonce 필수**
   - SoC FW에 **GPU 펌웨어 포함**이라 성능이 위아래로 움직일 수 있음. 벤치 꼭 돌릴 것.

## 안 건드린 것
- ConnectX-7 (28.45.4028), NVMe (NXHB202Q) = 이미 최신, 업데이트 없음
- open-webui / nginx-proxy-manager / esscode-db = .66 리부트로 같이 내려갔다 자동 복귀 (restart 정책 있음)
