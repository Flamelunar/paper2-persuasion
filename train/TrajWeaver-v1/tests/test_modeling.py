from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from trajweaver_v1.checkpoint import load_checkpoint, save_checkpoint
from trajweaver_v1.modeling import TrajWeaverModel


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


def make_model(kind: str = "llama", seed: int = 7) -> TrajWeaverModel:
    torch.manual_seed(seed)
    common = dict(
        vocab_size=113,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    base = (
        LlamaForCausalLM(LlamaConfig(**common))
        if kind == "llama"
        else Qwen2ForCausalLM(Qwen2Config(**common))
    )
    peft_model = get_peft_model(base, lora_config(), adapter_name="weaver")
    peft_model.add_adapter("trigger", lora_config())
    return TrajWeaverModel(peft_model, controller_hidden=16)


class ModelingTests(unittest.TestCase):
    def test_dual_adapter_latent_shapes(self) -> None:
        for kind in ("llama", "qwen"):
            with self.subTest(kind=kind):
                model = make_model(kind)
                output = model.weave(
                    torch.randint(0, 112, (2, 7)),
                    goal_ids=torch.randint(0, 112, (2, 5)),
                    force_probe=torch.tensor([True, False]),
                    force_invoke=True,
                )
                self.assertEqual(output.memory_tokens.shape, (2, 16, 32))
                self.assertEqual(output.goal_tokens.shape, (2, 8, 32))
                self.assertEqual(output.belief_tokens.shape, (2, 4, 32))
                self.assertEqual(output.desire_tokens.shape, (2, 4, 32))
                self.assertEqual(output.route_logits.shape, (2, 4))
                self.assertEqual(output.action_logits.shape, (2, 5))
                self.assertEqual(output.trigger_logits.shape, (2, 2))
                self.assertEqual(int(output.action_ids[0]), 0)

    def test_ablation_memory_modes_have_exact_prefix_lengths(self) -> None:
        model = make_model().eval()
        prompt = torch.randint(0, 112, (1, 7))
        goal = torch.randint(0, 112, (1, 5))
        goal_only = model.weave(prompt, goal_ids=goal, memory_mode="goal_only")
        always = model.weave(prompt, goal_ids=goal, memory_mode="always")
        self.assertEqual(tuple(goal_only.memory_tokens.shape), (1, 8, 32))
        self.assertEqual(tuple(goal_only.belief_tokens.shape), (1, 0, 32))
        self.assertEqual(tuple(goal_only.desire_tokens.shape), (1, 0, 32))
        self.assertEqual(tuple(always.memory_tokens.shape), (1, 16, 32))

        original_trigger = model.predict_trigger
        model.predict_trigger = lambda *_args, **_kwargs: torch.tensor([[10.0, -10.0]])
        strict_skip = model.weave(prompt, goal_ids=goal, memory_mode="trigger_strict")
        self.assertEqual(tuple(strict_skip.memory_tokens.shape), (1, 0, 32))
        self.assertEqual(tuple(strict_skip.goal_tokens.shape), (1, 0, 32))
        model.predict_trigger = original_trigger

        generated, control = model.generate(
            prompt, memory_mode="none", max_new_tokens=1, eos_token_id=None
        )
        self.assertEqual(tuple(control.memory_tokens.shape), (1, 0, 32))
        self.assertEqual(int(control.memory_tokens.shape[1]), 0)
        self.assertEqual(generated.shape[0], 1)

    def test_weaver_gradients_are_isolated_from_reasoner_and_trigger(self) -> None:
        model = make_model()
        model.train()
        model.set_trainable_component("weaver")
        prompt = torch.randint(0, 112, (2, 8))
        target = torch.randint(0, 112, (2, 4))
        goal = torch.randint(0, 112, (2, 5))
        loss, _ = model.supervised_loss(
            prompt,
            target,
            goal_ids=goal,
            route_labels=torch.tensor([0, 1]),
            action_labels=torch.tensor([0, 2]),
            belief_labels=torch.tensor([1, 2]),
            desire_labels=torch.tensor([1, 3]),
            goal_alignment_labels=torch.tensor([0, 1]),
            force_probe=torch.tensor([True, False]),
            force_invoke=True,
        )
        loss.backward()
        for name in ("goal_queries", "belief_queries", "desire_queries"):
            gradient = getattr(model, name).grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                for name, parameter in model.named_parameters()
                if ".weaver." in name and "lora_" in name
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name.startswith("backbone.") and "lora_" not in name
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if ".trigger." in name and "lora_" in name
            )
        )
        self.assertIsNone(model.trigger_head.weight.grad)

    def test_trigger_gradients_are_isolated_from_weaver_and_reasoner(self) -> None:
        model = make_model()
        model.train()
        model.set_trainable_component("trigger")
        loss, _ = model.trigger_loss(
            torch.randint(0, 112, (2, 8)), torch.tensor([0, 1])
        )
        loss.backward()
        self.assertIsNotNone(model.trigger_head.weight.grad)
        self.assertGreater(float(model.trigger_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                for name, parameter in model.named_parameters()
                if ".trigger." in name and "lora_" in name
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if ".weaver." in name and "lora_" in name
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name.startswith("backbone.") and "lora_" not in name
            )
        )

    def test_memory_disabled_matches_base_reasoner_logits(self) -> None:
        model = make_model()
        model.eval()
        ids = torch.randint(0, 112, (1, 7))
        with model.backbone.disable_adapter():
            reference = model.backbone(input_ids=ids).logits[:, -1, :]
        actual = model.next_token_logits(ids, memory_tokens=None)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)

    def test_cached_goal_is_static_and_dual_adapter_checkpoint_round_trips(self) -> None:
        model = make_model(seed=11).eval()
        with torch.no_grad():
            model.goal_queries.add_(0.25)
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.add_(0.1)
        goal_ids = torch.randint(0, 112, (1, 5))
        first = torch.randint(0, 112, (1, 6))
        other = torch.randint(0, 112, (1, 9))
        cached_goal = model.encode_goal(goal_ids)
        expected = model.weave(first, goal_tokens=cached_goal, force_invoke=True)
        changed_history = model.weave(other, goal_tokens=cached_goal, force_invoke=True)
        repeated = model.weave(first, goal_tokens=cached_goal, force_invoke=True)
        torch.testing.assert_close(expected.goal_tokens, cached_goal)
        torch.testing.assert_close(changed_history.goal_tokens, cached_goal)
        torch.testing.assert_close(expected.memory_tokens, repeated.memory_tokens)

        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, {"round_trip": True})
            restored = make_model(seed=11).eval()
            metadata = load_checkpoint(directory, restored)
            actual = restored.weave(first, goal_ids=goal_ids, force_invoke=True)
        self.assertTrue(metadata["round_trip"])
        torch.testing.assert_close(actual.memory_tokens, expected.memory_tokens, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
