# Persuadee state and closed-loop schema

This is a design contract, not an implemented serialization format.

## State snapshot

```json
{
  "dialogue_id": "...",
  "turn_index": 2,
  "goal": "immutable public goal",
  "beliefs": [
    {
      "proposition": "the proposed action is affordable",
      "stance": "against",
      "confidence": 0.82,
      "evidence_refs": ["persuadee_turn_2"]
    }
  ],
  "desires": [
    {
      "object": "keep total cost within budget",
      "polarity": "seek",
      "importance": 0.91,
      "status": "unsatisfied",
      "evidence_refs": ["persuadee_turn_2"]
    }
  ],
  "constraints": [
    {
      "text": "limited budget",
      "status": "active",
      "evidence_refs": ["persuadee_turn_2"]
    }
  ],
  "concerns": [
    {
      "concern_id": "c1",
      "category": "cost_effort",
      "summary": "the option may be too expensive",
      "status": "open",
      "introduced_at": 2,
      "last_evidence_ref": "persuadee_turn_2",
      "addressed_by_turn": null,
      "resolution_evidence_ref": null
    }
  ],
  "uncertainty": 0.18
}
```

Allowed concern states are:

```text
open -> addressed_pending -> resolved
  |            |               |
  +------------+---------------+-> recurred
```

- `addressed_pending` requires explicit evidence in the persuader response.
- `resolved` requires later persuadee evidence showing acceptance, reduced
  objection, or a concrete feasible next step.
- Silence does not resolve a concern.
- A repeated or strengthened concern becomes `recurred`.

## Trajectory

```json
{
  "trajectory_version": 3,
  "primary_concern_id": "c1",
  "subgoal": "establish an affordable path without weakening the main goal",
  "strategy": "acknowledge_then_compare_feasible_options",
  "response_requirements": [
    "acknowledge the cost concern",
    "avoid inventing prices",
    "ask for the relevant budget constraint"
  ],
  "advance_when": "the persuadee identifies a feasible option",
  "repair_when": "cost remains unacceptable or another concern appears"
}
```

## Response audit

```json
{
  "persuader_turn": 3,
  "active_concern_ids": ["c1"],
  "addressed_concern_ids": ["c1"],
  "coverage": "substantive",
  "response_evidence": "...",
  "unsupported_claim": false,
  "goal_drift": false
}
```

Coverage values are `none`, `acknowledgement_only`, `substantive`, and
`actionable`. This audit describes the persuader response; it does not determine
whether the persuadee's concern was actually resolved.

## Trigger event

```json
{
  "fire": true,
  "reasons": ["unresolved_concern", "trajectory_conflict"],
  "severity": 0.74,
  "confidence": 0.86,
  "evidence_refs": ["persuadee_turn_3"],
  "old_trajectory_version": 3,
  "new_trajectory_version": 4
}
```

The Trigger input may include the frozen hidden state of the public prompt plus
explicit delta features. It must not include the reference response, private
simulator fields, or future persuadee turns.
