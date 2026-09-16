# Reuse assessment

## `train/ctompersu_zero_shot.py` and `train/ctompersu_common.py`

Reuse without changing the evaluation contract:

- `load_dataset`, public scenario projection, and private-field isolation;
- `FixedPersuadee` as the controlled simulator;
- `OutcomeJudge`, four-turn budget, success stopping, and result schema;
- `LocalPersuader` model-loading behavior; and
- resume/error recording and per-run metadata.

Replace the direct prompt policy with a stateful local persuader object. The v4
object should own goal memory, state memory, concern ledger, trajectory, Trigger
decision, and response audit for one dialogue. The simulator and judge remain
outside this object.

## TrajWeaver-v3

Directly reusable ideas and contracts:

- public-only scenario/history conversion and leakage checks;
- immutable Goal tokens and dynamic State tokens;
- shared backbone with isolated Weaver and Trigger trainable components;
- paired counterfactual scoring, provenance hashes, uncertainty filtering, and
  utility-regret model selection;
- checkpoint/config compatibility checks; and
- concern-focused prefix selection from the 500 planner annotations.

Required changes:

- v3 State tokens have no supervised belief/desire semantics;
- v3 Trigger predicts whether State tokens improve reference-response NLL, not
  whether the persuadee state invalidates the current plan;
- v3 has no persistent concern ledger or explicit response audit;
- v3 does not separate `addressed` from `resolved`; and
- v3 does not maintain and selectively revise a structured trajectory.

The v3 paired-NLL label remains a useful utility signal, but it cannot be the
only definition of a valid v4 Trigger event.

## Official MemGen implementation

Reuse or adapt:

- learned query latents in `memgen/model/weaver.py`;
- reasoner-to-Weaver and Weaver-to-reasoner projection pattern;
- LoRA adapter separation between Weaver and Trigger;
- two-stage training order: Weaver first, Trigger second;
- explicit augmentation budget and Trigger decision logging; and
- save/load separation for auxiliary heads and adapters.

Do not copy unchanged:

- comma/period/newline or token-level augmentation points;
- a generic two-class memory-use interpretation with no state semantics;
- multiple full backbone copies for an 8B deployment without a memory budget;
- task-specific GSM8K/KodCode/TriviaQA processors and rewards; or
- Trigger GRPO before the dialogue reward and state labels are validated.

V4 should port the mechanism rather than vendor the entire MemGen repository.
The initial implementation should share one local backbone where possible and
switch adapters/components, following v3's lower-memory pattern.

## Attached design document

Useful proposals:

- dual explicit and latent memory;
- belief/desire-aware Trigger reasons;
- turn-level trajectory adaptation;
- evidence spans and uncertainty; and
- PersuasionForGood, CaSiNo, and PersuasionTrace as complementary sources.

Revisions made in this scaffold:

- continuous tracking is separated from sparse memory invocation;
- Trigger decisions occur only after persuadee turns;
- concern handling is audited separately from later resolution;
- the first Trigger is deliberately smaller than the proposed many-head model;
- ToMELP is not treated as an available dataset because its public repository
  currently contains no benchmark files; and
- the unverified DailyPersuasion download claim is not used.
