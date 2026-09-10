from __future__ import annotations

import unittest

from eval.evaluate_zero_shot_quality import aggregate


class ZeroShotQualityAggregationTests(unittest.TestCase):
    def test_goal_drift_rate_counts_dialogues_not_turns(self) -> None:
        records = [
            {
                "acr": {"label": 1},
                "unf": {"score": 4},
                "groundedness": {"score": 3},
                "goal_drift": {"turn_scores": [{"score": 0}, {"score": 1}]},
            },
            {
                "acr": {"label": 0},
                "unf": {"score": 2},
                "groundedness": {"score": 1},
                "goal_drift": {"turn_scores": [{"score": 2}, {"score": 4}]},
            },
        ]

        result = aggregate(records)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["goal_drift_count"], 1)
        self.assertEqual(result["goal_drift_rate_percent"], 50.0)
        self.assertEqual(result["acr_percent"], 50.0)
        self.assertEqual(result["unf_percent"], 75.0)
        self.assertEqual(result["contextual_groundedness_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
