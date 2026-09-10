#!/usr/bin/env bash
set -euo pipefail

# Resumes only the incomplete Llama run.  The SFT entry point infers the
# completed phase/epoch from trajectory-epoch-1 and therefore does not rerun
# warm-up or overwrite any existing checkpoint.
ROOT="/home1/liujianjian/2-paper-Coling-v2"
PY="/home1/liujianjian/anaconda3/envs/ljj/bin/python"
CONFIG="$ROOT/train/TrajWeaver-v1/configs/llama31_8b.yaml"
WEAVER_ROOT="$ROOT/train/TrajWeaver-v1/checkpoints/llama31-8b-terra"
TRIGGER_ROOT="$ROOT/train/TrajWeaver-v1/checkpoints/llama31-8b-terra-trigger"
LOG_DIR="$ROOT/train/TrajWeaver-v1/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/llama31-8b-terra.resume-pipeline.log") 2>&1
cd "$ROOT"

echo "pipeline_start=$(date '+%F %T %Z')"
echo "gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -1)"

CUDA_VISIBLE_DEVICES=0 "$PY" -u train/TrajWeaver-v1/train_sft.py \
  --config "$CONFIG" \
  --annotations train/TrajWeaver-v1/data/terra/annotations.jsonl \
  --output-dir "$WEAVER_ROOT" \
  --resume-components "$WEAVER_ROOT/trajectory-epoch-1"

WEAVER_CHECKPOINT="$("$PY" - <<'PY'
import json
from pathlib import Path
path = Path("train/TrajWeaver-v1/checkpoints/llama31-8b-terra/selected_weaver_checkpoint.json")
value = json.loads(path.read_text(encoding="utf-8"))
selected = Path(value["selected_checkpoint"])
if not selected.exists():
    raise SystemExit(f"selected Weaver checkpoint does not exist: {selected}")
print(selected)
PY
)"
echo "selected_weaver=$WEAVER_CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 "$PY" -u train/TrajWeaver-v1/train_trigger.py \
  --config "$CONFIG" \
  --weaver-checkpoint "$WEAVER_CHECKPOINT" \
  --annotations train/TrajWeaver-v1/data/terra/annotations.jsonl \
  --output-dir "$TRIGGER_ROOT"

bash train/TrajWeaver-v1/run_full_eval.sh
echo "pipeline_complete=$(date '+%F %T %Z')"
