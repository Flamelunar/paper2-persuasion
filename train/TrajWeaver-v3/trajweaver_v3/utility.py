"""Offline memory utility labels in mean NLL units (nats per target token)."""

import math
import random


def select_prefixes(rows, limit, seed):
    """At most one prefix per scenario before expanding to more data."""
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    rng = random.Random(seed)
    groups = list(by_scenario.values())
    rng.shuffle(groups)
    if limit > 0:
        groups = groups[:limit]
    return [rng.choice(group) for group in groups]


def label_utility(skip_nll, invoke_nll, cost=0.01, margin=0.002):
    if not all(math.isfinite(value) for value in (skip_nll, invoke_nll, cost, margin)):
        raise ValueError("Non-finite utility value")
    if min(skip_nll, invoke_nll, cost, margin) < 0:
        raise ValueError("NLL, cost, and margin must be nonnegative")
    gain = skip_nll - invoke_nll
    net = gain - cost
    uncertain = abs(net) <= margin
    return {"skip_nll": skip_nll, "invoke_nll": invoke_nll, "gain": gain,
            "net_gain": net, "invocation_cost": cost, "uncertainty_margin": margin,
            "trigger_id": int(net > 0), "trigger_action": "INVOKE" if net > 0 else "SKIP",
            "use_for_training": not uncertain, "label_source": "paired_reference_nll"}
