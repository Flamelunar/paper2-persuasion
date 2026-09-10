#!/usr/bin/env bash
set -euo pipefail

ROOT="/home1/liujianjian/2-paper-Coling-v2"
PY="/home1/liujianjian/anaconda3/envs/ljj/bin/python"
CONFIG="$ROOT/train/TrajWeaver-v1/configs/llama31_8b.yaml"
TRIGGER_ROOT="$ROOT/train/TrajWeaver-v1/checkpoints/llama31-8b-terra-trigger"
RESULTS="$ROOT/results"
MODEL="Meta-Llama-3.1-8B-Instruct"
LOG_DIR="$ROOT/train/TrajWeaver-v1/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/llama31-8b-terra.eval.log") 2>&1

cd "$ROOT"
TRIGGER_CHECKPOINT="$($PY - <<'PY'
import json
print(json.load(open('train/TrajWeaver-v1/checkpoints/llama31-8b-terra-trigger/selected_trigger_checkpoint.json'))['selected_checkpoint'])
PY
)"
echo "selected_trigger=$TRIGGER_CHECKPOINT"

# Never silently append a different checkpoint's records to this method's
# append-only trace.  A partial trace produced by the same selected checkpoint
# may be resumed; any other checkpoint is a hard failure and must be handled
# explicitly instead of contaminating the 525-pair comparison.
RAW_PATH="$RESULTS/$MODEL/raw/TrajWeaver-v1.jsonl"
if [[ -s "$RAW_PATH" ]]; then
  "$PY" - "$RAW_PATH" "$TRIGGER_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = str(Path(sys.argv[2]).resolve())
seen = set()
for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    checkpoint = row.get("run_config", {}).get("checkpoint")
    if checkpoint is None:
        raise SystemExit(
            f"Refusing to resume an unproven TrajWeaver trace: "
            f"missing run_config.checkpoint at line {line_number} in {path}"
        )
    seen.add(str(Path(checkpoint).resolve()))
if seen and seen != {expected}:
    raise SystemExit(
        f"Refusing to mix TrajWeaver checkpoints in {path}: "
        f"found={sorted(seen)} expected={expected}"
    )
print(f"raw_checkpoint_guard=ok records={sum(1 for x in path.read_text(encoding='utf-8').splitlines() if x.strip())}")
PY
fi

# The raw file is append-only and can resume if an API request is interrupted.
# Retry incomplete/error rows at the episode level, but never proceed to
# alignment until the selected checkpoint has exactly 525 successful records.
RAW_READY=0
for attempt in 1 2 3; do
  echo "raw_eval_attempt=$attempt"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u train/TrajWeaver-v1/run_ctompersu.py \
    --config "$CONFIG" \
    --checkpoint "$TRIGGER_CHECKPOINT" \
    --model-name "$MODEL" \
    --output-dir "$RESULTS" \
    --raw-subdir raw \
    --max-turns 4 \
    --simulator-model gpt-4o-mini \
    --judge-model gpt-5.6-luna \
    --workers 2
  if "$PY" - "$RAW_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
latest = {}
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        latest[int(row["index"])] = row
ok = set(range(525))
if set(latest) == ok and all(row.get("status") == "ok" for row in latest.values()):
    print("raw_complete=525")
    raise SystemExit(0)
print(
    f"raw_incomplete records={len(latest)}/525 "
    f"errors={sum(row.get('status') != 'ok' for row in latest.values())}"
)
raise SystemExit(1)
PY
  then
    RAW_READY=1
    break
  fi
  [[ "$attempt" -lt 3 ]] || break
done
[[ "$RAW_READY" -eq 1 ]] || { echo "raw evaluation did not reach 525 successful records" >&2; exit 1; }

"$PY" - <<'PY'
import json
from pathlib import Path
from eval.ctompersu_eval import materialize, validate_aligned_artifact

root = Path('results')
model = 'Meta-Llama-3.1-8B-Instruct'
dataset = Path('data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json')
materialize(
    root,
    dataset,
    root,
    models={model},
    methods={'TrajWeaver-v1'},
    raw_subdir='raw',
    aligned_subdir='aligned',
)
validate_aligned_artifact(root / model / 'aligned/TrajWeaver-v1.json', json.loads(dataset.read_text()))
print('aligned=525')
PY

QUALITY_READY=0
for attempt in 1 2 3; do
  echo "quality_eval_attempt=$attempt"
  "$PY" eval/evaluate_zero_shot_quality.py \
    --targets llama \
    --method trajweaver \
    --workers 4 \
    --judge-model gpt-5.6-luna
  if "$PY" - <<'PY'
import json
from pathlib import Path

path = Path('results/Meta-Llama-3.1-8B-Instruct/quality/TrajWeaver-v1-summary.json')
value = json.loads(path.read_text(encoding='utf-8'))
if int(value.get('n', -1)) == 525 and int(value.get('requested_n', -1)) == 525:
    print('quality_complete=525')
    raise SystemExit(0)
print(f"quality_incomplete n={value.get('n')} requested={value.get('requested_n')}")
raise SystemExit(1)
PY
  then
    QUALITY_READY=1
    break
  fi
  [[ "$attempt" -lt 3 ]] || break
done
[[ "$QUALITY_READY" -eq 1 ]] || { echo "quality evaluation did not reach 525 successful records" >&2; exit 1; }

"$PY" eval/paired_comparison.py --model "$MODEL"
"$PY" eval/update_metric_allocation.py --model "$MODEL" --write
echo "TRAJWEAVER_EVAL_COMPLETE"
