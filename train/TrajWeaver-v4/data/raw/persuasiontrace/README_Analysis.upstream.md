# Analysis Guide

This guide covers the analysis steps that feed the current paper figures,
tables, and exported manuscript assets.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Persuasiveness](#persuasiveness)
- [Human Trajectory Clustering](#human-trajectory-clustering)
- [Rhetoric Regression](#rhetoric-regression)
- [Counterfactual Replay](#counterfactual-replay)
- [LLM-Judge Human-Likeness](#llm-judge-human-likeness)
- [Stance Bias + Naive Penalty](#stance-bias--naive-penalty)
- [Model Sweep Policy Rank](#model-sweep-policy-rank)
- [Human-vs-Simulator Round Figure](#human-vs-simulator-round-figure)
- [Paper Asset Export](#paper-asset-export)

## Pipeline Overview

The paper export works in two layers. First,
`scripts/paper/regenerate_results_assets.sh` reruns the paper-used analysis
steps and refreshes the corresponding CSVs and figures. Then
`analysis.latex.export_paper_assets` syncs the prompt/table/figure artifacts
that the manuscript consumes.

If you want everything refreshed, run:

```bash
bash scripts/paper/regenerate_results_assets.sh
```

If you only want one slice, run one or more named steps:

```bash
bash scripts/paper/regenerate_results_assets.sh --only persuasiveness
bash scripts/paper/regenerate_results_assets.sh --only human_clusters
bash scripts/paper/regenerate_results_assets.sh --only rhetoric_regression
bash scripts/paper/regenerate_results_assets.sh --only counterfactual_replay
bash scripts/paper/regenerate_results_assets.sh --only llm_judge
bash scripts/paper/regenerate_results_assets.sh --only stance_bias
bash scripts/paper/regenerate_results_assets.sh --only model_sweep_policy_rank
bash scripts/paper/regenerate_results_assets.sh --only human_sim_round_figure
bash scripts/paper/regenerate_results_assets.sh --only export_assets
```

## Persuasiveness

This step rebuilds the condition-level persuasion summary and the control
comparison used in the paper.

Artifacts:

- `analysis/data/persuasiveness_bucket_summary.csv`
- `analysis/figures/results_persuasiveness_bucket_summary.pdf`
- `analysis/data/persuasiveness_welch_vs_control.csv`

## Human Trajectory Clustering

This is the clustering pass used for the human belief-trajectory figures in the
main text and appendix.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only human_clusters
```

Key artifacts:

- `analysis/data/human_clusters_k2_summary.csv`
- `analysis/data/human_clusters_k2_by_init_bin.csv`
- `analysis/data/human_clusters_k2_by_init_bin_heatmap.pdf`
- `analysis/data/human_clusters_k2_shapes.pdf`
- `analysis/data/human_clusters_k2_pca_scatter.pdf`

## Rhetoric Regression

This step fits the rhetoric-outcome analysis used for the paper coefficient
plots and summary tables.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only rhetoric_regression
```

Key artifacts:

- `analysis/data/results_rhetoric_llm_nonppt_coefficients.csv`
- `analysis/figures/results_rhetoric_llm_nonppt_coefficients.pdf`

## Counterfactual Replay

This is the main simulator-fidelity pass used in the paper. It replays human
rounds against simulator targets and regenerates the replay-based comparison
figures.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only counterfactual_replay
```

In the paper pipeline this step is post-processing only: it reads cached
`*_round_errors.jsonl` files and rebuilds the figures without rerunning replay
model calls.

Key artifacts:

- `analysis/data/simulator_counterfactual_replay_social_bn_summary.csv`
- `analysis/figures/results_counterfactual_target_error.pdf`
- `analysis/figures/results_counterfactual_node_error.pdf`
- `analysis/figures/results_counterfactual_replay_conditional_score_divergence.pdf`
- `analysis/figures/results_counterfactual_human_loo.pdf`

## LLM-Judge Human-Likeness

This step scores simulator and human outputs with an LLM judge and regenerates
the comparison figure used in the paper.

Cache-first run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only llm_judge
```

Refresh judge calls:

```bash
bash scripts/paper/regenerate_results_assets.sh --only llm_judge --refresh-llm-judge
```

Key artifacts:

- `analysis/data/rl_human_match_sim_compare_simulator_llm_judge_summary.csv`
- `analysis/figures/results_llm_judge_human_likeness.pdf`

## Stance Bias + Naive Penalty

This stage measures stance asymmetry from mirrored for-vs-against rollouts and
then derives the naive-policy penalty figure from the same export.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only stance_bias
```

Input:

- `STANCE_BIAS_EPISODES_JSONL` (default:
  `results/stance_mirror_dryrun_3sim_allprops_gpt5_fixed/episodes.jsonl`)

Key artifacts:

- `analysis/data/rl_human_match_sim_compare_stance_bias_summary.csv`
- `analysis/figures/results_stance_bias.pdf`
- `analysis/data/rl_human_match_sim_compare_naive_penalty.csv`
- `analysis/figures/results_naive_penalty.pdf`

## Model Sweep Policy Rank

This step ranks policy variants from the existing sweep rollouts using the
simulator-based metrics reported in the appendix.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only model_sweep_policy_rank
```

Key artifacts:

- `analysis/data/model_sweep_policy_rank_by_simulator.csv`
- `analysis/figures/model_sweep_policy_rank_by_simulator.pdf`

## Human-vs-Simulator Round Figure

This regenerates the side-by-side methods figure showing one human round next
to a matched simulator replay round.

Run:

```bash
bash scripts/paper/regenerate_results_assets.sh --only human_sim_round_figure
```

Key artifacts:

- `analysis/figures/round_human_vs_simulator.html`
- `../persuasiontrace-overleaf/figures/round_human_vs_simulator.pdf`
  (when Overleaf repo is present)

## Paper Asset Export

Here we write prompt/table exports and manifest-enabled
figures into the persuasiontrace paper repo when that repo is available
locally.

Run:

```bash
python -m analysis.latex.export_paper_assets
```

or

```bash
make paper-assets
```

Outputs:

- Generated prompt/table TeX files in
  `../persuasiontrace-overleaf/include/generated` when available
  (otherwise `analysis/latex/generated`)
- Manifest-enabled figure sync to
  `../persuasiontrace-overleaf/figures` when available
  (otherwise `analysis/latex/generated/figures`)
