#!/bin/bash
# PD12M finalize + publish orchestrator.
# Waits for the ingestion stream, builds the sets, visualizations, card,
# and uploads everything to a private HuggingFace repo. Resumable: safe to
# rerun; postprocess and uploads overwrite.
set -uo pipefail
BASE=/media/phhofm/SSD/pd12m_v1
REPO=/home/phhofm/Documents/GitHub/lucid-sisr
PY="$REPO/venv/bin/python"
HF=/home/phhofm/.local/bin/hf
TARGET=Phips/lucid-pd12m-512
LOG=$BASE/finalize.log
cd "$REPO" || exit 1
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

log "waiting for stream to finish"
while pgrep -f "pd12m_stream.py" >/dev/null 2>&1; do sleep 120; done
log "stream finished"

if [ ! -s "$BASE/master_lineage.csv" ] || [ -z "$(ls -A "$BASE/tiles" 2>/dev/null)" ]; then
  log "driver finalize missing; running dedupe + finalize"
  "$PY" -m lucid --input "$BASE/workspace/scored" --output "$BASE" \
        --workspace "$BASE/workspace" --stage dedupe >> "$LOG" 2>&1
  "$PY" -m lucid --input "$BASE/workspace/scored" --output "$BASE" \
        --workspace "$BASE/workspace" --stage finalize --simple-names >> "$LOG" 2>&1
fi

log "postprocessing sets"
"$PY" /tmp/kilo/pd12m_postprocess.py >> "$LOG" 2>&1 || { log "POSTPROCESS FAILED"; exit 1; }

log "contact sheets + histogram"
"$PY" /tmp/kilo/pd12m_contact_sheets.py >> "$LOG" 2>&1

log "dataset card"
"$PY" /tmp/kilo/pd12m_write_card.py >> "$LOG" 2>&1

log "creating repo $TARGET"
"$HF" repo create "$TARGET" --repo-type dataset --private >> "$LOG" 2>&1 || log "repo exists / create skipped"

for SET in pd12m_train_full pd12m_train_small pd12m_train_mini pd12m_bench pd12m_trainval; do
  log "pack + upload $SET"
  "$PY" /tmp/kilo/pd12m_pack_upload.py "$SET" "$TARGET" "$SET" >> "$LOG" 2>&1 || log "UPLOAD FAILED $SET"
done

log "uploading score sheets + visuals + card"
for f in pd12m_train_full_scores.csv pd12m_train_small_scores.csv \
         pd12m_train_mini_scores.csv pd12m_bench_scores.csv \
         pd12m_trainval_scores.csv sets_summary.json \
         complexity_histogram.png contact_pd12m_train_full.png \
         contact_pd12m_train_small.png contact_pd12m_train_mini.png \
         contact_pd12m_bench.png contact_pd12m_trainval.png README.md; do
  [ -f "$BASE/$f" ] && "$HF" upload "$TARGET" "$BASE/$f" "$f" --repo-type dataset >> "$LOG" 2>&1
done

touch "$BASE/UPLOAD_DONE"
log "ALL DONE — repo: https://huggingface.co/datasets/$TARGET"
