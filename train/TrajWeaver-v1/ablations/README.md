# TrajWeaver-v1 mechanism ablations

This directory owns the fast, diagnostic 100-example experiment. It is
deliberately separate from the production 525-example artifacts and never
writes `results/tables/metric_allocation.md`.

## Fixed subset

The manifest is
`manifests/eval100_seed20260910.json`. It contains 100 sorted original
`source_index` values sampled with seed `20260910`, together with the full
dataset digest and one public-scenario digest per row. Every variant must use
this manifest; do not replace it with `--limit`, which would renumber rows.

The completed canonical Zero-shot and TrajWeaver-v1 traces are filtered onto
the manifest with:

```bash
python train/TrajWeaver-v1/ablations/make_manifest.py
python train/TrajWeaver-v1/ablations/materialize_existing.py \
  --variant public-zero-shot legacy-v1
```

## New model variants

Run one variant at a time on the available GPU:

```bash
python train/TrajWeaver-v1/ablations/run_ablation.py --variant r0-no-memory
python train/TrajWeaver-v1/ablations/evaluate_ablation.py --variant r0-no-memory
```

Replace the variant with `g8-static-goal`, `g8bd8-always`, or
`g8bd8-trigger-strict`. The runner uses local Llama-3.1-8B weights and the
existing API simulator (`gpt-4o-mini`) and success judge (`gpt-5.6-luna`).
The quality evaluator uses the same blind `gpt-5.6-luna` rubric as the main
evaluation. Raw and quality jobs are append-only and can be resumed.

Finally create the isolated comparison report:

```bash
python train/TrajWeaver-v1/ablations/report_ablation.py
```

Outputs are under
`results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100/`:
each variant has `raw.jsonl`, `aligned.json`, `quality.jsonl`, and (after
quality evaluation) `quality-summary.json`; the root has
`ablation_report.md` and `ablation_report.json`.

`r0-no-memory` inserts zero latent tokens, `g8-static-goal` inserts eight
cached goal tokens, `g8bd8-always` inserts all sixteen every turn, and
`g8bd8-trigger-strict` inserts all sixteen only for INVOKE and inserts zero
for SKIP. The latter is the intended true no-intervention fallback.
