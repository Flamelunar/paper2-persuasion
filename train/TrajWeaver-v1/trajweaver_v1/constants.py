"""Shared constants for TrajWeaver-v1."""

from __future__ import annotations

PUBLIC_FIELDS = ("tag", "background", "persuadee", "persuader", "goal", "domain")
PRIVATE_FIELDS = ("preventive", "generative", "persona")

ROUTE_STATES = (
    "ON_ROUTE",
    "REPAIRABLE",
    "STRUCTURAL_FAILURE",
    "TERMINAL_READY",
)
ACTIONS = ("PROBE", "CONTINUE", "REPAIR", "REPLAN", "COMMIT")
TRIGGER_ACTIONS = ("SKIP", "INVOKE")

# Publicly inferable persuadee state relative to the immutable goal.  These
# labels supervise the two dynamic query banks; they are never taken from the
# private CToMPersu persona fields.
BELIEF_STATES = (
    "UNKNOWN",
    "REJECTS_PREMISE",
    "DOUBTFUL",
    "MIXED",
    "ACCEPTS_PREMISE",
)
DESIRE_STATES = (
    "RESISTANT",
    "RELUCTANT",
    "UNDECIDED",
    "WILLING",
    "COMMITTED",
)

# The first two levels preserve the original target.  Levels 2--4 correspond
# exactly to the Goal Drift Rate rubric used by the offline evaluator.
GOAL_ALIGNMENT_STATES = (
    "ON_GOAL",
    "VALID_SUBSTEP",
    "DILUTED",
    "SUBSTITUTED",
    "REVERSED",
)

ROUTE_TO_ID = {name: index for index, name in enumerate(ROUTE_STATES)}
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
TRIGGER_TO_ID = {name: index for index, name in enumerate(TRIGGER_ACTIONS)}
BELIEF_TO_ID = {name: index for index, name in enumerate(BELIEF_STATES)}
DESIRE_TO_ID = {name: index for index, name in enumerate(DESIRE_STATES)}
GOAL_ALIGNMENT_TO_ID = {
    name: index for index, name in enumerate(GOAL_ALIGNMENT_STATES)
}

METHOD = "TrajWeaver-v1"
EXPERIMENT_MAX_TURNS = 4
NUM_GOAL_QUERIES = 8
NUM_BELIEF_QUERIES = 4
NUM_DESIRE_QUERIES = 4

SYSTEM_PROMPT = """You are the persuader in a natural multi-turn dialogue.
Generate only the next persuader utterance, in no more than three concise
sentences. Use the public scenario and observable dialogue to address the
latest concern and move toward the exact goal.

Be respectful, autonomy-supportive, and grounded. Do not invent facts,
authorities, urgency, prices, timelines, private motives, or guarantees.
Do not mention hidden labels, latent memory, or write analysis, JSON, or role
instructions."""

GOAL_ENCODER_SYSTEM_PROMPT = """Encode the exact, immutable persuasion target.
Preserve every required component, contrast, priority, entity, and time
constraint. A lower-risk step is valid only when it keeps an explicit path to
this target. Do not infer or encode the persuadee's current attitude."""
