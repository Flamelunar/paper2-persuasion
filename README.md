# paper2-persuasion

Code and experiment artifacts for the CToMPersu persuasion study.

This repository contains:

- the public-only Zero-shot and MA²P evaluation runners;
- the TrajWeaver-v1 implementation (frozen Reasoner, independent Weaver and Trigger LoRA adapters, static Goal memory and dynamic Belief/Desire memory);
- leakage-filtered CToMPersu train/dev data, 2k/300 pilot LoRA data, and Codex Terra teacher annotations;
- Llama-3.1-8B main raw/aligned/quality results and the fixed-100 mechanism ablations;
- the complete experiment and failure-analysis report in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

## Reproducibility notes

The local backbone is not included. Set `model.name_or_path` in `train/TrajWeaver-v1/configs/llama31_8b.yaml` to a local copy of `Meta-Llama-3.1-8B-Instruct`. Checkpoints and model weights are deliberately excluded from this repository.

The evaluation protocol uses a fixed `gpt-4o-mini` persuadee simulator and `gpt-5.6-luna` independent judge. Running the evaluation requires configuring the project API endpoint and credentials locally; no credentials are stored here.

```bash
python -m unittest discover -s train/TrajWeaver-v1/tests -v
python eval/finalize_ctompersu.py --help
python train/TrajWeaver-v1/run_ctompersu.py --help
```

The Terra teacher annotations in `train/TrajWeaver-v1/data/terra/annotations.jsonl` were created through the local Codex plug-in, not by committing an API key or a remote credential.

## Scope and license

The source code and generated experiment artifacts are provided for research discussion and reproducibility. The CToMPersu data and any base-model weights remain subject to their original licenses and terms; this repository does not redistribute model weights.
