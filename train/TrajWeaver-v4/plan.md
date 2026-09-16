# TrajWeaver-v4 plan

## Research question

Can a persuasion model improve goal achievement and concern targeting by
maintaining an evidence-grounded persuadee state, detecting belief/desire
deviation at each turn, and selectively replanning instead of always reusing or
always regenerating memory?

The contribution should be framed as **trigger-guided persuadee-state
trajectory adaptation**, not as adding generic memory to TrajWeaver-v3.

## Core causal chain

```text
persuadee utterance
  -> evidence-grounded belief/desire/concern update
  -> plan-state mismatch and unresolved-concern detection
  -> selective replan + latent memory refresh
  -> response that explicitly addresses the active concern
  -> next-turn persuadee evidence confirms persistence, change, or resolution
```

The intervention is selective state-conditioned replanning. The intermediate
effects are better state tracking, more useful Trigger decisions, and higher
concern coverage. The task-level outcomes are persuasion success, acceptance,
lower goal drift, and lower unnecessary memory invocation.

## Non-negotiable design decisions

1. **State tracking is continuous.** The tracker runs after every persuadee
   turn. A sparse Trigger must not skip observation of the persuadee.
2. **The Trigger is turn-level.** It runs once after a completed persuadee
   utterance, rather than at commas, periods, or generation tokens.
3. **Addressed is not resolved.** A persuader response may address a concern;
   only a later persuadee utterance can support `resolved`.
4. **The public goal is immutable.** Replanning changes the route, subgoal, or
   strategy, never the requested target action.
5. **State claims require evidence spans.** Unsupported inferred motives are
   represented as uncertain hypotheses, not facts.
6. **Private simulator fields remain isolated.** They may drive the fixed
   persuadee simulator but never enter the local persuader, tracker, planner,
   Trigger, or evaluator input.

## Runtime loop

At dialogue start, encode the public goal once and initialize an empty state and
trajectory. For each turn:

1. Generate the persuader reply from the public scenario, observable history,
   immutable goal memory, and the current trajectory memory.
2. Audit which active concerns the response explicitly addressed. Mark them
   `addressed_pending`; do not mark them resolved.
3. Obtain the next persuadee utterance from the existing fixed simulator.
4. Update beliefs, desires, constraints, concern ledger, uncertainty, and
   evidence references from that utterance.
5. Confirm whether prior concerns remain open, were resolved, or recurred using
   only the new persuadee evidence.
6. Compute state delta and mismatch with the active trajectory.
7. Run the Trigger. If it fires, regenerate the structured trajectory and
   refresh latent state/trajectory tokens. Otherwise retain the trajectory.
8. Continue until success or the existing four-turn budget is exhausted.

## Trigger contract

The minimal Trigger output should remain compact:

```json
{
  "fire": true,
  "reasons": ["new_concern", "desire_shift"],
  "severity": 0.81,
  "confidence": 0.87,
  "evidence_refs": ["persuadee_turn_2"]
}
```

`reasons` is multi-label and may contain `belief_shift`, `desire_shift`,
`new_concern`, `unresolved_concern`, `uncertainty`, or `trajectory_conflict`.
The first implementation should use one binary fire head plus one reason head;
separate regression heads for every state dimension are deferred until the
base mechanism is shown to work.

## Training stages

### Stage 0: protocol-compatible baseline

Keep the dataset loader, public/private boundary, fixed persuadee, final judge,
four-turn budget, resume behavior, and record format from
`train/ctompersu_zero_shot.py`. Replace only the local persuader policy.

### Stage 1: explicit state and concern supervision

Train or distill a structured tracker before introducing latent memory. Use:

- existing CToMPersu public histories and the 500 planner annotations;
- PersuasionForGood for human persuasion language and strategy labels;
- CaSiNo preference ranks/reasons as desire supervision; and
- PersuasionTrace serial measurements as belief-trajectory supervision and an
  external evaluation set.

The state tracker must be evaluated independently. If it cannot recover active
concerns and state changes, latent memory cannot rescue the claimed mechanism.

### Stage 2: response targeting and trajectory supervision

Construct examples linking `(state, concern, current trajectory)` to a compact
next-step plan and the next persuader response. Add a response audit label that
distinguishes direct acknowledgement, substantive handling, unsupported claim,
and omission. The current 500 planner records can seed concern and guidance
labels, but they do not by themselves label whether a generated response solved
the concern.

### Stage 3: Weaver training

Adapt MemGen's learned latent queries and projection path to compress the
explicit state plus structured trajectory into a small latent block. Train with
the same reference response under controlled conditions:

```text
goal only
goal + stale trajectory
goal + updated state/trajectory latent memory
```

This makes the value of state-conditioned replanning measurable rather than
conflating it with additional context length.

### Stage 4: Trigger training

Freeze the tracker and Weaver first. Derive Trigger supervision from two
signals:

1. semantic need: material state delta, new/recurrent concern, uncertainty, or
   conflict between state and current trajectory;
2. counterfactual utility: quality/targeting gain from replan versus reuse,
   minus invocation cost.

Ambiguous samples stay outside Trigger training. The first version should use
offline counterfactual labels; MemGen's GRPO trainer is a later option after the
turn-level loop and rewards are validated.

## Minimum evidence package

- Baselines: zero-shot, TrajWeaver-v3, goal-only, always-replan, random-trigger,
  NLL-only Trigger, and the full semantic-plus-utility Trigger.
- State metrics: belief stance error/calibration, desire Recall@K or rank
  correlation, concern category/status F1, evidence-span validity.
- Response metrics: active-concern coverage, groundedness, unsupported-claim
  rate, goal drift, autonomy violation, and pairwise targeting preference.
- Trigger metrics: precision/recall/F1, invocation rate, useful-trigger rate,
  utility regret, and latency/token overhead.
- Dialogue outcomes: success, acceptance, turns to success, belief-trajectory
  area/change, backslide rate, and human/independent-judge quality.
- Required ablations: no explicit state, no latent memory, no response audit,
  always trigger, no semantic Trigger labels, and no utility labels.

## Stop/go gates before implementation expansion

1. State schema can represent CToMPersu, CaSiNo, and PersuasionTrace without
   leaking private fields.
2. Concern resolution is scored from subsequent persuadee evidence.
3. The explicit-state baseline improves concern targeting before latent memory
   is added.
4. The learned Trigger beats always-replan on utility/compute tradeoff.
5. Full system improves both persuasion outcome and targeting without increasing
   unsupported claims or goal drift.

## Implementation approval boundary

This scaffold intentionally stops before Python implementation. After the
architecture is approved, implementation should begin with the schema/data
validators and a protocol-compatible explicit-state baseline, then add Weaver
and Trigger training in separate changes.
