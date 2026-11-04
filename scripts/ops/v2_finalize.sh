#!/bin/bash
# Wait for the v2 PD12M expansion, gate the UHD training tiles, then merge
# both into the final finetune tier and upload.
set -uo pipefail
REPO_DIR=/home/phhofm/Documents/GitHub/lucid-sisr
PY="$REPO_DIR/venv/bin/python"
LOG=/media/phhofm/SSD/pd12m_v2/merge.log
cd "$REPO_DIR" || exit 1
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

log "waiting for PD12M v2 expansion"
while pgrep -f "pd12m_expand.py" >/dev/null 2>&1; do sleep 300; done
log "v2 expansion done"

log "gating UHD-IQA training tiles"
"$PY" -m lucid --input /media/phhofm/SSD/uhd_ws_train/input \
      --output /media/phhofm/SSD/uhd_ws_train/out \
      --workspace /media/phhofm/SSD/uhd_ws_train/out/ws \
      --stage all --threshold 0.30 --dedupe-threshold 0.96 \
      --iqa-gate topiq_nr:0.665 --iqa-gate clipiqa+:0.45 \
      --iqa-gate niqe:6.5 --iqa-gate qrealign-mini:0.78 \
      --no-multiscale --simple-names >> "$LOG" 2>&1 || log "UHD gating FAILED"

log "merging v2 + UHD"
"$PY" /tmp/kilo/merge_v2.py /media/phhofm/SSD/pd12m_v1/v1_pool.npy \
      Phips/lucid-pd12m-512-v2 >> "$LOG" 2>&1 || { log "MERGE FAILED"; exit 1; }

touch /media/phhofm/SSD/lucid_v2_final/MERGE_DONE
log "ALL DONE"
