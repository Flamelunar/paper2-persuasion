from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAUSAL_MODEL_TYPES = {"llama", "qwen2"}
CONDITIONAL_MODEL_TYPES = {"gemma4", "mistral3", "qwen3_5"}


@dataclass
class GeneratedText:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    error: str | None = None


class LocalChatModel:
    """Small adapter over causal and multimodal-capable Transformers chat models."""

    def __init__(self, model_path: str | Path, device: str, dtype: str = "bf16") -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.torch = torch
        self.model_path = str(model_path)
        self.device = device
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        self.model_type = config.model_type
        if dtype != "bf16":
            raise ValueError("This diagnostic is configured for bf16 only")
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": device},
        }
        self.is_conditional = self.model_type in CONDITIONAL_MODEL_TYPES
        if self.model_type in CAUSAL_MODEL_TYPES:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs).eval()
            self.processor = None
        elif self.is_conditional:
            # The three checkpoints expose conditional-generation architectures even for text-only chats.
            from transformers import AutoModelForImageTextToText

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True, fix_mistral_regex=True
            )
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs).eval()
        else:
            raise ValueError(f"Unsupported local model_type {self.model_type!r} for {self.model_path}")

    def _render(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int]:
        if self.is_conditional:
            chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            # Qwen3.5 enables a reasoning channel by default.  This benchmark evaluates
            # the public assistant utterance, so prevent hidden reasoning from consuming
            # the entire response budget or appearing in the scored text.
            if self.model_type == "qwen3_5":
                chat_kwargs["enable_thinking"] = False
            inputs = self.processor.apply_chat_template(messages, **chat_kwargs)
        else:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )
            # Recent Transformers returns a BatchEncoding for some tokenizers and a Tensor for others.
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids["input_ids"]
            inputs = {"input_ids": ids, "attention_mask": self.torch.ones_like(ids)}
        input_tokens = int(inputs["input_ids"].shape[-1])
        return inputs, input_tokens

    def generate(self, messages: list[dict[str, str]], max_input_tokens: int, max_new_tokens: int) -> GeneratedText:
        started = time.perf_counter()
        try:
            inputs, input_tokens = self._render(messages)
            if input_tokens > max_input_tokens:
                raise ValueError(f"input_too_long:{input_tokens}>{max_input_tokens}")
            device_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **device_inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completion_ids = generated[:, input_tokens:]
            text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
            return GeneratedText(
                text=text,
                input_tokens=input_tokens,
                output_tokens=int(completion_ids.shape[-1]),
                latency_seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # Persist failures as data; never silently drop a fixture.
            return GeneratedText(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}:{exc}",
            )

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()
