# TrajWeaver-v4

TrajWeaver-v4 is currently a design and data scaffold. Model and training code
will be added only after the architecture is approved.

The central objective is a closed-loop persuasion system that:

1. updates an explicit persuadee state after every persuadee turn;
2. tracks whether the persuader addressed the active concern;
3. confirms concern resolution only from later persuadee evidence;
4. invokes latent memory and trajectory replanning only when the current state
   materially invalidates the active plan; and
5. generates a grounded, autonomy-supportive response for the unchanged public
   persuasion goal.

The design starts from `train/ctompersu_zero_shot.py`, reuses selected
TrajWeaver-v3 data and training contracts, and adapts the official MemGen
Trigger/Weaver mechanism at a dialogue-turn boundary.

## Current status

- Architecture and reuse analysis: `docs/`
- Public raw datasets and provenance: `data/`
- Implementation package: reserved at `trajweaver_v4/`
- No v4 model or training implementation has been written yet.

See `plan.md` for the proposed stages and approval boundary.

## Reserved package layout

```text
trajweaver_v4/
  data/        # normalization and leakage-safe input contracts
  state/       # belief, desire, concern, and constraint tracking
  planning/    # structured trajectory creation and revision
  memory/      # explicit-state to latent-memory Weaver
  trigger/     # turn-level replan decision and reason heads
  generation/  # response generation conditioned on goal and trajectory
  evaluation/  # state, targeting, Trigger, safety, and outcome metrics
  runtime/     # one-dialogue closed-loop orchestration
```

These directories are placeholders only. Their APIs will be defined after the
current plan is approved.
