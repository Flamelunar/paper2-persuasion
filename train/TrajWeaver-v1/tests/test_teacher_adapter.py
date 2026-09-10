from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajweaver_v1.teacher_adapter import overlay_annotation, official_run_annotations


class TeacherAdapterTests(unittest.TestCase):
    def test_supplied_batch_wrapper_and_bayesian_belief(self) -> None:
        run = {
            "index": 0,
            "status": "ok",
            "result": {
                "initial_anchor": {"question": "What matters most?"},
                "turns": [
                    {
                        "system_action": "A3",
                        "next_question": "Would a reversible trial address that?",
                        "bayesian": {
                            "posterior_belief": {
                                "on_route": 0.1,
                                "repairable": 0.7,
                                "structural_failure": 0.1,
                                "terminal_ready": 0.1,
                            }
                        },
                    }
                ],
            },
        }
        rows = official_run_annotations(run, scenario_id_override="fixture")
        self.assertEqual(rows[0]["sample_id"], "fixture:t1")
        self.assertEqual(rows[0]["action"], "PROBE")
        self.assertEqual(rows[1]["route_state"], "REPAIRABLE")
        self.assertEqual(rows[1]["action"], "REPAIR")

    def test_overlay_maps_dynamic_state_and_trigger_to_ids(self) -> None:
        sample = {
            "sample_id": "fixture:t1",
            "scenario": {"goal": "Choose the original option"},
            "history": [],
            "target": "Could we discuss it?",
            "route_state": "ON_ROUTE",
            "route_id": 0,
            "action": "PROBE",
            "action_id": 0,
        }
        row = overlay_annotation(
            sample,
            {
                "route_state": "repairable",
                "action": "A3",
                "belief_state": "doubtful",
                "desire_state": "reluctant",
                "goal_alignment": 1,
                "trigger_action": True,
                "anchor_summary": "must not enter model-facing data",
                "label_source": "proactive_teacher_gpt4o",
            },
        )
        self.assertEqual(row["belief_state"], "DOUBTFUL")
        self.assertEqual(row["belief_id"], 2)
        self.assertEqual(row["desire_state"], "RELUCTANT")
        self.assertEqual(row["desire_id"], 1)
        self.assertEqual(row["goal_alignment"], "VALID_SUBSTEP")
        self.assertEqual(row["goal_alignment_id"], 1)
        self.assertEqual(row["trigger_action"], "INVOKE")
        self.assertEqual(row["trigger_id"], 1)
        self.assertNotIn("anchor_summary", row)


if __name__ == "__main__":
    unittest.main()
