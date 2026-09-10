#!/usr/bin/env bash
# Passive status monitor for the long-running TrajWeaver pipeline.
# It never starts, stops, resumes, or recomputes any experiment.
set -u

ROOT="/home1/liujianjian/2-paper-Coling-v2"
LOG_DIR="$ROOT/train/TrajWeaver-v1/logs"
LOG_FILE="$LOG_DIR/llama31-8b-terra.monitor.log"
INTERVAL_SECONDS=1800

mkdir -p "$LOG_DIR"

snapshot() {
  {
    date '+%F %T %Z'
    ps -eo pid,ppid,lstart,etime,stat,pcpu,pmem,args | \
      grep -E 'train_sft.py|train_trigger.py|run_full_eval.sh|run_ctompersu.py|evaluate_zero_shot_quality.py|paired_comparison.py|update_metric_allocation.py' | \
      grep -v -E 'grep -E|monitor_training.sh' || true
    find "$ROOT/train/TrajWeaver-v1/checkpoints/llama31-8b-terra" \
         "$ROOT/train/TrajWeaver-v1/checkpoints/llama31-8b-terra-trigger" \
         "$ROOT/results/Meta-Llama-3.1-8B-Instruct" \
         -maxdepth 3 -type f \
         \( -name 'metadata.json' -o -name 'selected_*checkpoint.json' \
            -o -name 'TrajWeaver-v1.jsonl' -o -name 'TrajWeaver-v1.json' \
            -o -name 'TrajWeaver-v1-summary.json' -o -name '*paired*' \) \
         -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort -r
    printf '%s\n' '---'
  } >> "$LOG_FILE"
}

while true; do
  sleep "$INTERVAL_SECONDS"
  snapshot
  if ! ps -eo args | grep -E 'train_sft.py|train_trigger.py|run_full_eval.sh|run_ctompersu.py|evaluate_zero_shot_quality.py|paired_comparison.py|update_metric_allocation.py' | grep -v -E 'grep -E|monitor_training.sh' >/dev/null; then
    break
  fi
done
