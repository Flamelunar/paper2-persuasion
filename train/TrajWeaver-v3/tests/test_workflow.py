"""CPU integration test using a locally initialized tiny Llama and tokenizer."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from trajweaver_v3.data import read_jsonl, write_json, write_jsonl
from trajweaver_v3.model import TrajWeaverV3


def entrypoint(name, options):
    spec = importlib.util.spec_from_file_location("v3_test_" + name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    with patch.object(sys, "argv", [name, *map(str, options)]), \
            patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(output):
        module.main()
    return output.getvalue()


class WorkflowTests(unittest.TestCase):
    def test_train_score_resume_trigger_predict_and_reject_stale_labels(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            vocab = {token: i for i, token in enumerate([
                "[UNK]", "[PAD]", "[EOS]", "hello", "yes", "no", "goal", "public",
                "system", "user", "assistant", ":", "A", "B", "price", "okay"])}
            raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                                pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }} : "
                "{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant : {% endif %}")
            tokenizer.save_pretrained(root / "base")
            base = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=1024, eos_token_id=2))
            base.save_pretrained(root / "base")
            config = {"model": {"name_or_path": str(root / "base"), "load_in_4bit": False},
                "architecture": {"goal_tokens": 2, "state_tokens": 2, "trigger_hidden": 8},
                "lora": {"rank": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj"]},
                "data": {"max_goal_tokens": 128, "max_prompt_tokens": 512, "max_target_tokens": 32},
                "training": {"seed": 42,
                    "sft": {"epochs": 1, "gradient_accumulation": 2,
                            "learning_rate": 0.001, "goal_only_weight": 0.25},
                    "trigger": {"epochs": 1, "gradient_accumulation": 2, "learning_rate": 0.001}},
                "utility": {"cost": 0.01, "margin": 0.0}}
            write_json(root / "config.json", config)
            common = ["--config", root / "config.json"]
            for split in ("train", "dev"):
                rows = [{"sample_id": f"{split}{i}:t1", "scenario_id": f"{split}{i}",
                    "scenario": {"goal": "goal", "background": "public", "persuader": "A", "persuadee": "B"},
                    "history": [], "target": "hello yes", "split": split,
                    "turn_index": 0, "max_turns": 4} for i in range(3)]
                write_jsonl(root / f"sft/{split}.jsonl", rows)
            entrypoint("train", [*common, "--stage", "sft", "--data-dir", root / "sft",
                                 "--output", root / "weaver"])
            best = json.loads((root / "weaver/best.json").read_text())["checkpoint"]
            scoring = [*common, "--checkpoint", best, "--data-dir", root / "sft",
                       "--output", root / "utility", "--train-prefixes", 3, "--dev-prefixes", 3]
            entrypoint("build_trigger_data", scoring)
            rows_before = (root / "utility/train.jsonl").read_bytes()
            entrypoint("build_trigger_data", scoring)
            self.assertEqual(rows_before, (root / "utility/train.jsonl").read_bytes())
            utility_rows = read_jsonl(root / "utility/train.jsonl")
            self.assertEqual(len(utility_rows), 3)
            self.assertNotIn("target", utility_rows[0])
            entrypoint("train", [*common, "--stage", "trigger", "--checkpoint", best,
                "--data-dir", root / "utility", "--output", root / "trigger"])
            selected = json.loads((root / "trigger/best.json").read_text())
            self.assertTrue(selected["trigger_trained"])
            prediction = entrypoint("predict", [*common, "--checkpoint", selected["checkpoint"],
                "--input", root / "sft/dev.jsonl", "--max-new-tokens", 2])
            self.assertIn("controller", json.loads(prediction))
            with self.assertRaisesRegex(ValueError, "another Weaver"):
                entrypoint("train", [*common, "--stage", "trigger", "--checkpoint", selected["checkpoint"],
                    "--data-dir", root / "utility", "--output", root / "wrong"])

            # Only the external simulator/Judge are stubbed. Early acceptance
            # must not truncate the four-turn protocol or count duplicates twice.
            sys.path.insert(0, str(HERE.parents[1]))
            from train.ctompersu_common import FixedPersuadee, OutcomeJudge
            record = {"scenario": {"goal": "goal", "background": "public", "persuader": "A",
                "persuadee": "B", "generative": {"belief": "simulator only"}}}
            write_json(root / "eval.json", [record, record])
            cached_goals = []

            def generate(_model, _prompt, **kwargs):
                cached_goals.append(kwargs["goal"])
                return torch.tensor([[3]]), {"memory_mode": "trigger", "memory_tokens": 2}

            evaluation = [*common, "--checkpoint", selected["checkpoint"],
                          "--dataset", root / "eval.json", "--output", root / "eval.jsonl"]
            with patch.object(FixedPersuadee, "respond", return_value={"utterance": "yes", "accepted": True}) as sim, \
                    patch.object(OutcomeJudge, "evaluate_final", return_value={"success": True}) as judge, \
                    patch.object(TrajWeaverV3, "generate", generate):
                entrypoint("run_eval", evaluation)
                self.assertEqual(sim.call_count, 4)
                self.assertEqual(judge.call_count, 1)
                self.assertEqual([call.kwargs["final_turn"] for call in sim.call_args_list],
                                 [False, False, False, True])
                self.assertTrue(all(goal is cached_goals[0] for goal in cached_goals))
                entrypoint("run_eval", evaluation)
                self.assertEqual(sim.call_count, 4)
            evaluated = read_jsonl(root / "eval.jsonl")
            self.assertEqual(len(evaluated), 1)
            self.assertEqual(evaluated[0]["turn_count"], 4)
            self.assertNotIn("generative", evaluated[0]["scenario"])

            # The main table can opt into the original row-level denominator,
            # including duplicate public scenarios.
            preserved = [*common, "--checkpoint", selected["checkpoint"],
                         "--dataset", root / "eval.json", "--output", root / "eval_rows.jsonl",
                         "--preserve-rows"]
            with patch.object(FixedPersuadee, "respond", return_value={"utterance": "yes", "accepted": True}), \
                    patch.object(OutcomeJudge, "evaluate_final", return_value={"success": True}), \
                    patch.object(TrajWeaverV3, "generate", generate):
                entrypoint("run_eval", preserved)
            evaluated_rows = read_jsonl(root / "eval_rows.jsonl")
            self.assertEqual(len(evaluated_rows), 2)
            self.assertEqual(len({row["evaluation_id"] for row in evaluated_rows}), 2)
            self.assertEqual(len({row["scenario_id"] for row in evaluated_rows}), 1)


if __name__ == "__main__":
    unittest.main()
