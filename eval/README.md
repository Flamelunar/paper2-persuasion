# CToMPersu evaluation layout

`ctompersu_eval.py` is the single evaluator for the reproduction. It owns
completion checks, the paper-aligned metrics, aggregation, canonical JSON
snapshots, and source-aligned exports. The contract tests live beside it in
`eval/test_ctompersu_contract.py`; there is no duplicate top-level `tests/`
implementation for this experiment. The tests import this evaluator and the
train-time protocol directly, so metric and alignment logic has one owner.

The unrelated `train/TrajWeaver-v1/tests/` directory remains scoped to that
separate training package and is intentionally not merged into CToMPersu.

The active training entry points are `train/ctompersu_zero_shot.py`,
`train/ctompersu_ma2p.py`, and `train/TrajWeaver-v1/run_ctompersu.py`; direct Python commands are documented in
`train/README_CToMPersu_reproduction.md`. Raw training traces are append-only
JSONL. After all requested model/method
jobs reach 525 successful indexed records, `eval/finalize_ctompersu.py`
materializes readable arrays under `aligned/` and `canonical/` and writes the
CSV/Markdown/JSON tables under `tables/`. Each aligned row has its
`source_index`, the source `scenario`, and a `generated_dialogue` using a
consistent role/content turn schema. The golden/reference dialogue is not
duplicated: it remains in the immutable CToMPersu source dataset at the same
`source_index`. `evaluation` contains the retained run metadata and judge
outcome.

## Zero-shot quality metrics

`evaluate_zero_shot_quality.py` performs the blind offline evaluation for ACR,
UNF, Contextual Groundedness, and Goal Drift Rate. It reads only the public
scenario fields and observable dialogue, appends versioned judgments under
`results/<model>/quality/`, and resumes from completed indices.

```bash
python eval/evaluate_zero_shot_quality.py --targets llama gemma --workers 8
python eval/evaluate_zero_shot_quality.py --targets llama --method ma2p --workers 6
python eval/evaluate_zero_shot_quality.py --targets llama --method trajweaver --workers 2
```

The corresponding `zero-shot-summary.json` contains the 525-dialogue
aggregates. Goal Drift Rate counts a dialogue once when any persuader turn has
a substantive goal dilution, substitution, or reversal; a bounded sub-step
that remains connected to the original goal is not counted.

After the TrajWeaver quality file is complete, `paired_comparison.py` writes a
separate per-index comparison and deterministic paired-bootstrap intervals; it
never rewrites baseline raw or aligned artifacts.
