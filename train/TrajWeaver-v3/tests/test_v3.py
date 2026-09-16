import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from trajweaver_v3.data import (assert_public, messages, prepare, repair_dialogue,
                               read_jsonl, write_json)
from trajweaver_v3.model import TrajWeaverV3, load_checkpoint, save_checkpoint
from trajweaver_v3.utility import label_utility, select_prefixes

torch.set_num_threads(1)


def make_model(kind="llama"):
    torch.manual_seed(7)
    settings = dict(vocab_size=67, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.0)
    base = (LlamaForCausalLM(LlamaConfig(**settings)) if kind == "llama" else
            Qwen2ForCausalLM(Qwen2Config(**settings)))
    backbone = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"), adapter_name="weaver")
    return TrajWeaverV3(backbone, goal_tokens=8, state_tokens=8, trigger_hidden=16)


def tensors():
    return (torch.tensor([[2, 3, 4, 5]]), torch.tensor([[6, 7, 8]]),
            torch.tensor([[9, 10, 11]]))


class ModelTests(unittest.TestCase):
    def test_weaver_gradient_isolation_for_supported_backbones(self):
        for kind in ("llama", "qwen"):
            with self.subTest(kind=kind):
                model = make_model(kind)
                prompt, goal, target = tensors()
                model.set_component("weaver")
                model.sft_loss(prompt, goal, target).backward()
                for name in ("goal_queries", "state_queries"):
                    grad = getattr(model, name).grad
                    self.assertIsNotNone(grad)
                    self.assertGreater(grad.abs().sum().item(), 0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if "lora_" in n))
                self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                    if n.startswith("trigger.") or (n.startswith("backbone.") and "lora_" not in n)))

    def test_trigger_cannot_update_weaver_or_base(self):
        model = make_model()
        model.set_component("trigger")
        prompt, _, _ = tensors()
        torch.nn.functional.cross_entropy(model.trigger_logits(model.hook(prompt)),
                                          torch.tensor([1])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if n.startswith("trigger.")))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                            if not n.startswith("trigger.")))

    def test_skip_retains_goal_and_avoids_state_computation(self):
        model = make_model().eval()
        prompt, goal, _ = tensors()
        cached = model.encode_goal(goal).detach()
        snapshot = cached.clone()
        with patch.object(model, "trigger_logits", return_value=torch.tensor([[50., -50.]])), \
                patch.object(model, "state", side_effect=AssertionError("SKIP computed state")):
            generated, info = model.generate(prompt, goal=cached, mode="trigger", max_new_tokens=2)
            self.assertEqual(info["memory_tokens"], 8)
            self.assertEqual(info["trigger_action"], "SKIP")
        expected, _ = model.generate(prompt, goal=cached, mode="goal_only", max_new_tokens=2)
        torch.testing.assert_close(expected, generated)
        torch.testing.assert_close(snapshot, cached)
        _, full = model.generate(prompt, goal=cached, mode="always", max_new_tokens=2)
        self.assertEqual(full["memory_tokens"], 16)

    def test_no_memory_matches_frozen_reasoner(self):
        model = make_model().eval()
        prompt, _, _ = tensors()
        with torch.no_grad(), model.backbone.disable_adapter():
            expected = model.backbone(input_ids=prompt).logits[:, -1].argmax(-1)
        actual, info = model.generate(prompt, mode="none", max_new_tokens=1)
        torch.testing.assert_close(expected, actual[:, 0])
        self.assertEqual(info["memory_tokens"], 0)

    def test_utility_has_no_grad_and_equals_explicit_pair(self):
        model = make_model().eval()
        prompt, goal_ids, target = tensors()
        measured = model.utility(prompt, goal_ids, target)
        with torch.no_grad():
            goal = model.encode_goal(goal_ids)
            state = model.state(model.hook(prompt), goal)
            skip = model.nll(prompt, target, goal).item()
            invoke = model.nll(prompt, target, torch.cat((goal, state), dim=1)).item()
        self.assertAlmostEqual(measured["gain"], skip - invoke)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(measured, model.utility(prompt, goal_ids, target))

    def test_checkpoint_roundtrip_and_version_rejection(self):
        model = make_model().eval()
        with torch.no_grad():
            model.goal_queries.add_(0.1)
            model.trigger[-1].bias.add_(0.2)
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.03)
        prompt, goal, target = tensors()
        expected = model.utility(prompt, goal, target)
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(root, model, {}, {"stage": "trigger", "trigger_trained": True})
            loaded = make_model().eval()
            metadata = load_checkpoint(root, loaded)
            self.assertTrue(metadata["trigger_trained"])
            self.assertEqual(loaded.utility(prompt, goal, target), expected)
            torch.testing.assert_close(loaded.trigger_logits(loaded.hook(prompt)),
                                       model.trigger_logits(model.hook(prompt)))
            payload = torch.load(Path(root) / "trajweaver.pt", weights_only=True)
            payload["format_version"] = 2
            torch.save(payload, Path(root) / "trajweaver.pt")
            with self.assertRaisesRegex(ValueError, "v3"):
                load_checkpoint(root, loaded)


class DataTests(unittest.TestCase):
    @staticmethod
    def record(goal):
        return {"scenario": {"goal": goal, "background": "public", "persuader": "A",
            "persuadee": "B", "tag": "demo", "domain": ["demo"],
            "generative": {"belief": "secret"}}, "dialog": [
            "persuader: hello", "continuation", "persuadee: no",
            "persuader: why", "persuadee: price", "persuader: okay", "persuadee: thanks"]}

    def test_conversion_repairs_fragments_deduplicates_and_checks_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for split in ("train", "dev", "test"):
                rows = [self.record(split)] * (2 if split == "test" else 1)
                write_json(root / f"{split}.json", rows)
            manifest = prepare(root, root / "out")
            self.assertEqual(manifest["splits"]["test"]["duplicates_removed"], 1)
            rows = read_jsonl(root / "out/train.jsonl")
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["target"], "hello continuation")
            self.assertEqual(rows[0]["history"], [])
            self.assertEqual(rows[1]["history"][-1]["content"], "no")
            assert_public(rows)
            write_json(root / "dev.json", [self.record("train")])
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare(root, root / "other")

    def test_target_not_in_model_messages_and_private_keys_rejected(self):
        row = {"scenario": {"goal": "original"}, "history": [], "turn_index": 0,
               "max_turns": 4, "target": "FUTURE RESPONSE"}
        self.assertNotIn("FUTURE RESPONSE", json.dumps(messages(row)))
        row["scenario"]["persona"] = "secret"
        with self.assertRaises(ValueError):
            messages(row)

    def test_utility_labels_have_cost_margin_and_no_fake_ties(self):
        self.assertEqual(label_utility(2.0, 1.9)["trigger_id"], 1)
        self.assertEqual(label_utility(2.0, 2.0)["trigger_id"], 0)
        self.assertFalse(label_utility(2.01, 2.0)["use_for_training"])
        self.assertTrue(label_utility(2.0, 2.0)["use_for_training"])
        with self.assertRaises(ValueError):
            label_utility(float("nan"), 2.0)

    def test_prefix_selection_is_scenario_balanced_and_reproducible(self):
        rows = [{"scenario_id": str(s), "turn_index": t} for s in range(20) for t in range(4)]
        selected = select_prefixes(rows, 10, 42)
        self.assertEqual(selected, select_prefixes(rows, 10, 42))
        self.assertEqual(len({row["scenario_id"] for row in selected}), 10)


if __name__ == "__main__":
    unittest.main()
