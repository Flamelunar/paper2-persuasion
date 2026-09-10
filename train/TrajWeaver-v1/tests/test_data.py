from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajweaver_v1.constants import PRIVATE_FIELDS
from trajweaver_v1.data import (
    goal_messages,
    goal_specification,
    iter_turn_samples,
    load_json_list,
    repair_dialogue,
    scenario_key,
    split_full_excluding_eval,
    student_messages,
)

PROJECT = Path(__file__).resolve().parents[3]
FULL = PROJECT / "data/CToMPersu/dataset/CToMPersu_Full/CToMPersu.json"
EVAL = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"


class DataTests(unittest.TestCase):
    def test_multiline_repair(self) -> None:
        turns = repair_dialogue(
            [
                "persuader:",
                "Would a small trial work?",
                "persuadee: Maybe,",
                "if it is reversible.",
                "persuader: What would make it safe?",
                "persuadee: A clear limit.",
                "persuader: Then start with one week.",
                "persuadee: I can try that.",
            ]
        )
        self.assertEqual(len(turns), 6)
        self.assertEqual(turns[0]["content"], "Would a small trial work?")
        self.assertEqual(turns[1]["content"], "Maybe, if it is reversible.")

    def test_exact_split_counts_and_no_eval_leakage(self) -> None:
        full, evaluation = load_json_list(FULL), load_json_list(EVAL)
        train, dev, manifest = split_full_excluding_eval(full, evaluation, seed=42)
        self.assertEqual(manifest["full_unique_records"], 6257)
        self.assertEqual(manifest["eval_unique_records"], 391)
        self.assertEqual(manifest["eval_rows_present_exactly_in_full"], 525)
        self.assertEqual(manifest["eval_unique_public_scenarios"], 391)
        self.assertEqual(manifest["retained_scenarios"], 5866)
        self.assertEqual((len(train), len(dev)), (5279, 587))
        eval_keys = {scenario_key(row["scenario"]) for row in evaluation}
        self.assertFalse({scenario_key(row["scenario"]) for row in train + dev} & eval_keys)
        self.assertEqual(
            sum(1 for row in train for _ in iter_turn_samples(row, "train")), 19209
        )
        self.assertEqual(
            sum(1 for row in dev for _ in iter_turn_samples(row, "dev")), 2135
        )
        self.assertEqual(
            {
                rounds: sum(len(repair_dialogue(row["dialog"])) // 2 == rounds for row in train)
                for rounds in (3, 4)
            },
            {3: 1907, 4: 3372},
        )

    def test_student_rows_have_no_private_fields(self) -> None:
        row = load_json_list(FULL)[-1]
        sample = next(iter_turn_samples(row, "train"))
        serialized = json.dumps(sample, ensure_ascii=False)
        self.assertFalse(set(PRIVATE_FIELDS).intersection(sample["scenario"]))
        self.assertTrue(all(f'"{field}"' not in serialized for field in PRIVATE_FIELDS))
        prompt = student_messages(sample)[1]["content"]
        self.assertIn("turn 1 of 4", prompt)
        self.assertIn("4 persuader response(s) remain", prompt)

    def test_goal_encoder_prompt_is_goal_only_and_history_invariant(self) -> None:
        row = load_json_list(FULL)[-1]
        sample = next(iter_turn_samples(row, "train"))
        sample["history"] = [
            {"role": "persuadee", "content": "I reject that; talk about a different topic."}
        ]
        specification = goal_specification(sample)
        serialized = json.dumps(goal_messages(sample), ensure_ascii=False)
        self.assertEqual(specification["exact_goal"], sample["scenario"]["goal"])
        self.assertIn(sample["scenario"]["goal"], serialized)
        self.assertNotIn("I reject that", serialized)
        self.assertNotIn("Observable dialogue", serialized)
        self.assertFalse(set(PRIVATE_FIELDS).intersection(specification))


if __name__ == "__main__":
    unittest.main()
