# TrajWeaver-v1

TrajWeaver-v1 is a goal-preserving extension of the CToMPersu Zero-shot
persuader. It targets the observed failure mode in which a persuader responds
pleasantly to the latest objection but silently turns an original persuasion
goal into a different, easier-to-accept goal. The fixed original target and
the persuadee's changing stance must therefore be represented separately.

## Implemented architecture

One physical causal-LM backbone is used, with two independent named PEFT
adapters. The base Reasoner is always frozen and always decodes with all LoRA
adapters disabled.

```text
original public goal ── Weaver LoRA ──> G: 8 static goal tokens
                                             │ cache once per dialogue
public dialogue history ─ Weaver LoRA + G ─> B: 4 belief tokens
                                      └────> D: 4 desire/intention tokens
                                             │
                    Trigger LoRA ──> SKIP / INVOKE dynamic B+D memory
                                             │
                  Frozen Reasoner <── [ G ; B ; D ] (16 tokens) ──> response
```

- `G` is produced only from the public goal specification. It never reads
  dialogue history, and is encoded once then reused across all turns of a
  dialogue.
- `B,D` are rebuilt from the full public history on every turn. They model the
  persuadee's belief and intention toward the *same* original goal, rather
  than a new target.
- If Trigger chooses `SKIP`, the eight dynamic positions use learned null-state
  tokens; the Reasoner still receives a fixed 16-token layout.
- Weaver trainables: `weaver` LoRA, 8+4+4 latent queries, projections,
  controller, route/action/state/alignment heads, and null-state tokens.
- Trigger trainables: its own `trigger` LoRA and its two-class output head
  only. It cannot update Weaver or Reasoner.

The current version uses supervised distillation only. There is **no GRPO**
and no background GPU process.

## Data and labels

`data/train.jsonl` is the canonical source: 5,279 Full scenarios and 19,209
turn samples, after removing all 525 Eval rows by public-scenario identity.
`data/dev.jsonl` has 587 scenarios / 2,135 turns and has the same exclusion.

`build_lora_datasets.py` materializes the initial 2k scenario pilot and retains
only scenario-ID manifests for the 1k, 2k, and full 5,279 scales. The files
currently present are:

| Component | Train | Dev |
| --- | ---: | ---: |
| Weaver SFT | 2,000 scenarios / 7,269 turns | 300 scenarios / 1,087 turns |
| Trigger SFT | 2,000 scenarios / 7,269 turns | 300 scenarios / 1,087 turns |

The initial generated copies have bootstrap trigger labels (`SKIP` on turn 1,
`INVOKE` thereafter) only to validate data plumbing. They are not accepted by
the normal training entrypoints. The formal teacher pass is performed directly
by the Codex CLI with GPT-5.6 Terra; it does not call ChatAnywhere/OpenAI and
does not use an API key. `annotate_terra.py` only validates and appends those
direct Codex outputs. The main train commands require at least 95% full
teacher-label coverage in both splits.

```bash
source /home1/liujianjian/anaconda3/etc/profile.d/conda.sh
conda activate ljj
cd /home1/liujianjian/2-paper-Coling-v2

# Already run once in this workspace; it materializes the fixed 2k/300 source.
python train/TrajWeaver-v1/build_lora_datasets.py

# Direct Codex Terra pass; safe to interrupt and rerun. No API call is made.
# The compatibility entrypoint below invokes the local `codex exec` plug-in.
python train/TrajWeaver-v1/annotate_teacher.py \
  --model gpt-5.6-terra --reasoning-effort max --batch-size 64 --workers 2
python train/TrajWeaver-v1/annotate_terra.py --check
```

The teacher prompt sees only the public scenario, observable history, and gold
next response. It classifies `BELIEF_STATES`, `DESIRE_STATES`, five goal
alignment states, and `SKIP/INVOKE`; private CToMPersu fields are excluded.
Gold turns labeled as goal drift have language-SFT weight zero, while their
alignment head still receives supervision.

## Run order

Use the Python entry points directly. The examples below use the initial
2,000/300 scenario split and do not start automatically.

```bash
# optional two-example plumbing test only; permits bootstrap labels
CUDA_VISIBLE_DEVICES=0 python train/TrajWeaver-v1/train_sft.py \
  --config train/TrajWeaver-v1/configs/qwen25_7b.yaml \
  --output-dir train/TrajWeaver-v1/checkpoints/smoke-qwen \
  --allow-bootstrap-only --limit 2

# main Weaver SFT; teacher labels are required
CUDA_VISIBLE_DEVICES=0 python train/TrajWeaver-v1/train_sft.py \
  --config train/TrajWeaver-v1/configs/qwen25_7b.yaml \
  --train-data train/TrajWeaver-v1/data/lora/weaver_sft/train.scenarios-2000.jsonl \
  --dev-data train/TrajWeaver-v1/data/lora/weaver_sft/dev.scenarios-300.jsonl \
  --annotations train/TrajWeaver-v1/data/terra/annotations.jsonl \
  --output-dir train/TrajWeaver-v1/checkpoints/qwen25-7b
CUDA_VISIBLE_DEVICES=1 python train/TrajWeaver-v1/train_sft.py \
  --config train/TrajWeaver-v1/configs/llama31_8b.yaml \
  --train-data train/TrajWeaver-v1/data/lora/weaver_sft/train.scenarios-2000.jsonl \
  --dev-data train/TrajWeaver-v1/data/lora/weaver_sft/dev.scenarios-300.jsonl \
  --annotations train/TrajWeaver-v1/data/terra/annotations.jsonl \
  --output-dir train/TrajWeaver-v1/checkpoints/llama31-8b

# then train the separate Trigger from the final Weaver checkpoint
CUDA_VISIBLE_DEVICES=0 python train/TrajWeaver-v1/train_trigger.py \
  --config train/TrajWeaver-v1/configs/qwen25_7b.yaml \
  --weaver-checkpoint train/TrajWeaver-v1/checkpoints/qwen25-7b/trajectory-epoch-3 \
  --train-data train/TrajWeaver-v1/data/lora/trigger_sft/train.scenarios-2000.jsonl \
  --dev-data train/TrajWeaver-v1/data/lora/trigger_sft/dev.scenarios-300.jsonl \
  --output-dir train/TrajWeaver-v1/checkpoints/qwen25-7b-trigger
CUDA_VISIBLE_DEVICES=1 python train/TrajWeaver-v1/train_trigger.py \
  --config train/TrajWeaver-v1/configs/llama31_8b.yaml \
  --weaver-checkpoint train/TrajWeaver-v1/checkpoints/llama31-8b/trajectory-epoch-3 \
  --train-data train/TrajWeaver-v1/data/lora/trigger_sft/train.scenarios-2000.jsonl \
  --dev-data train/TrajWeaver-v1/data/lora/trigger_sft/dev.scenarios-300.jsonl \
  --output-dir train/TrajWeaver-v1/checkpoints/llama31-8b-trigger
```

Use the final `trigger-epoch-*` checkpoint for evaluation, because it contains
both the trained Weaver and Trigger adapters:

```bash
python train/TrajWeaver-v1/run_ctompersu.py \
  --config train/TrajWeaver-v1/configs/llama31_8b.yaml \
  --checkpoint train/TrajWeaver-v1/checkpoints/llama31-8b-trigger/trigger-epoch-2 \
  --model-name Meta-Llama-3.1-8B-Instruct \
  --output-dir results --raw-subdir raw
```

## Optional later preference stage

`build_preferences.py` can make DPO pairs only from externally generated and
evaluated candidates. Its ranking order is: no pressure violation, preserve
the goal, groundedness/UNF, ACR/success, then scalar score. When supplied, a
drifted candidate is used as a rejected hard negative. The script never
manufactures candidates, scores, or judge outputs. DPO is intentionally not
part of the first SFT+Trigger run, and it is not GRPO.

## Verification

```bash
python -m unittest discover -s train/TrajWeaver-v1/tests -v
```

The unit tests cover Eval-exclusion data integrity, goal-only encoding,
16-token shapes, frozen Reasoner behavior, Weaver/Trigger gradient isolation,
and dual-adapter checkpoint round trips.
