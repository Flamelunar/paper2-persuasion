"""Contract tests kept with the single CToMPersu evaluation package.

There is intentionally no second top-level ``tests`` copy for this
experiment. These tests cover the public/private boundary, the four-turn
protocol, the current MA²P prompt pipeline, and aligned result materialization.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from eval.ctompersu_eval import (  # noqa: E402
    aligned_rows,
    canonicalize,
    check_complete,
    summarize,
    validate_aligned_artifact,
)
from train.ctompersu_common import (  # noqa: E402
    EXPERIMENT_MAX_TURNS,
    Generation,
    PROTOCOL_VERSION,
    load_dataset,
)
from train.ctompersu_ma2p import KnowledgeBase, run_one as run_ma2p_episode  # noqa: E402
from train.ctompersu_zero_shot import (  # noqa: E402
    direct_prompt,
    parse_args as parse_zero_shot_args,
    run_one as run_zero_shot_episode,
)


class FakePersuader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, *, max_new_tokens: int = 180) -> Generation:
        self.calls.append((system, user))
        if "Task Interpreter" in system:
            text = json.dumps({
                "initial_stance": "uncertain", "likely_values": ["reliability"],
                "likely_resistance": ["risk"], "possible_latent_concerns": ["cost"],
                "safe_persuasion_space": ["small step"],
            })
        elif "Dual-Anchor Planner" in system:
            text = json.dumps({
                "a1_initial_anchor": {"question": "What matters most to you here?", "strategy_id": "q1", "used_theories": []},
                "a5_terminal_anchor": {"question": "If the concern were addressed, what small step would you consider?", "strategy_id": "q5", "used_theories": []},
            })
        elif "Route Planner" in system:
            text = json.dumps({"route_summary": "test", "nodes": [{"node_id": "r1", "subgoal": "clarify", "question_operator": "clarify", "terminal_bridge": "bridge"}]})
        elif "State and Evidence Tracker" in system:
            text = json.dumps({
                "dialogue_state": {"engagement": 0.7, "concern_clarity": 0.6, "value_alignment": 0.5, "goal_proximity": 0.4, "action_readiness": 0.3, "active_a5_fit": 0.6},
                "control_evidence": {"recoverability": {"score": 0.7}, "confidence": 0.8},
            })
        elif "safety and autonomy critic" in system:
            text = '{"approved":true,"revised_question":"","issues":[]}'
        else:
            text = "Would you consider one small, voluntary step?"
        return Generation(text, 1, len(text), 0.001)


class FakeSimulator:
    def respond(self, scenario: dict, history: list[dict[str, str]], *, final_turn: bool) -> dict:
        return {"utterance": "I will consider that small step.", "accepted": True, "acceptance_level": 4, "reactance": 1}


class FakeJudge:
    def evaluate_success(self, scenario: dict, history: list[dict[str, str]]) -> bool:
        return len(history) >= 2

    def evaluate_quality(self, scenario: dict, history: list[dict[str, str]]) -> dict:
        return {"acceptance_level": 4, "persuasive": 8, "logical_coherence": 8, "helpfulness": 8, "rationale": "fixture"}


class CToMPersuContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
        cls.data = load_dataset(cls.path)

    def test_dataset_and_protocol(self) -> None:
        self.assertEqual(len(self.data), 525)
        self.assertEqual(EXPERIMENT_MAX_TURNS, 4)
        self.assertEqual(PROTOCOL_VERSION, "ctompersu_reference_dual_anchor_bayes_v4_4turns")

    def test_five_local_models_remain_configured(self) -> None:
        config = json.loads((PROJECT / "configs/zero_shot_models.json").read_text())
        self.assertEqual(len(config["models"]), 5)
        self.assertEqual(config["protocol"]["max_turns"], 4)

    def test_zero_shot_prompt_is_public_only(self) -> None:
        scenario = self.data[0]["scenario"]
        system, user = direct_prompt(scenario, [], 4)
        prompt = system + user
        self.assertNotIn(json.dumps(scenario["preventive"], ensure_ascii=False), prompt)
        self.assertNotIn(json.dumps(scenario["generative"], ensure_ascii=False), prompt)
        self.assertIn("Remaining persuader turns (including this one): 4", prompt)

    def test_planning_prompts_are_public_only(self) -> None:
        fake = FakePersuader()
        record = run_ma2p_episode(
            0, self.data[0], fake, FakeSimulator(), FakeJudge(), KnowledgeBase(), 1  # type: ignore[arg-type]
        )
        self.assertEqual(record["status"], "ok")
        prompt = "\n".join(system + user for system, user in fake.calls)
        scenario = self.data[0]["scenario"]
        self.assertNotIn(json.dumps(scenario["preventive"], ensure_ascii=False), prompt)
        self.assertNotIn(json.dumps(scenario["generative"], ensure_ascii=False), prompt)

    def test_reproduction_episodes_respect_turn_cap(self) -> None:
        result = run_ma2p_episode(
            0, self.data[0], FakePersuader(), FakeSimulator(), FakeJudge(), KnowledgeBase(), 4  # type: ignore[arg-type]
        )
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(result["turn_count"], 4)
        self.assertEqual(result["method_config"]["configurator"], "frozen_KB")
        zero = run_zero_shot_episode(
            0, self.data[0], FakePersuader(), FakeSimulator(), FakeJudge(), 4  # type: ignore[arg-type]
        )
        self.assertEqual(zero["status"], "ok")
        self.assertLessEqual(zero["turn_count"], 4)

    def test_zero_shot_protocol_cli_default(self) -> None:
        original = sys.argv
        try:
            sys.argv = ["zero_shot.py", "--model-path", "/tmp/model", "--model-name", "fixture"]
            args = parse_zero_shot_args()
            self.assertEqual(args.raw_subdir, "raw")
            self.assertEqual(args.max_turns, 4)
        finally:
            sys.argv = original

    def test_partial_aggregation_withholds_full_dataset_success(self) -> None:
        record = {
            "index": 0, "status": "ok", "turn_count": 1, "stop_reason": "judge_success", "success_turn": 1,
            "run_config": {"max_turns": 4, "judge_model": "gpt-5.6-luna"},
            "outcome": {"success": True, "success_turn": 1, "acceptance_level": 4, "persuasive": 8, "logical_coherence": 8, "helpfulness": 8},
        }
        result = summarize([record], self.data, "fixture", "Zero-shot")
        self.assertEqual(result["N completed"], "1/525")
        self.assertIsNone(result["Success (%)"])

    def test_aligned_and_canonical_exports_are_index_ordered(self) -> None:
        source = self.data[:2]
        records = {0: {"index": 0, "status": "ok", "method": "Zero-shot", "outcome": {"success": True}}}
        aligned = aligned_rows(source, records, "fixture", "Zero-shot")
        self.assertEqual([row["scenario"] for row in aligned], [row["scenario"] for row in source])
        self.assertEqual([row["source_index"] for row in aligned], [0, 1])
        self.assertNotIn("golden_dialogue", aligned[0])
        self.assertEqual(aligned[0]["generated_dialogue"], [])
        self.assertNotIn("history", aligned[0]["evaluation"])
        self.assertEqual(aligned[1]["evaluation"]["status"], "pending")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw" / "fixture"
            raw.mkdir(parents=True)
            (raw / "zero-shot.jsonl").write_text('{"index": 0, "status": "error"}\n{"index": 0, "status": "ok", "attempt": 2}\n{"index": 1, "status": "ok"}\n')
            self.assertEqual(canonicalize(root / "raw", root / "canonical"), 1)
            output = json.loads((root / "canonical" / "fixture" / "zero-shot.json").read_text())
            self.assertEqual([row["index"] for row in output], [0, 1])

    def test_aligned_validation_rejects_duplicated_gold_dialogue(self) -> None:
        rows = aligned_rows(self.data[:1], {0: {"index": 0, "status": "ok", "history": []}}, "fixture", "Zero-shot")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "aligned.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            validate_aligned_artifact(path, self.data[:1])
            rows[0]["golden_dialogue"] = []
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated golden dialogue"):
                validate_aligned_artifact(path, self.data[:1])

    def test_completion_check_requires_all_indices(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "raw" / "fixture"
            root.mkdir(parents=True)
            (root / "zero-shot.jsonl").write_text('{"index": 0, "status": "ok"}\n{"index": 3, "status": "ok"}\n')
            result = check_complete(Path(temp) / "raw", "fixture", "Zero-shot", total=2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["out_of_range"], 1)


if __name__ == "__main__":
    unittest.main()
