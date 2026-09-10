"""Shared-backbone latent trajectory Weaver and frozen Reasoner.

The base causal LM exists only once.  Its ``weaver`` LoRA adapter first encodes
eight immutable goal tokens from the original goal alone, then encodes eight
dynamic persuadee-state tokens from the public dialogue.  The adapter is
disabled while the same frozen backbone realizes the next response.  This
keeps the MemGen latent insertion mechanism without loading three 7B models.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import (
    ACTIONS,
    ACTION_TO_ID,
    BELIEF_STATES,
    DESIRE_STATES,
    GOAL_ALIGNMENT_STATES,
    NUM_BELIEF_QUERIES,
    NUM_DESIRE_QUERIES,
    NUM_GOAL_QUERIES,
    ROUTE_STATES,
    TRIGGER_ACTIONS,
    TRIGGER_TO_ID,
)


@dataclass
class TrajWeaverOutput:
    memory_tokens: Tensor
    goal_tokens: Tensor
    belief_tokens: Tensor
    desire_tokens: Tensor
    route_logits: Tensor
    action_logits: Tensor
    belief_logits: Tensor
    desire_logits: Tensor
    goal_alignment_logits: Tensor
    trigger_logits: Tensor
    route_ids: Tensor
    action_ids: Tensor
    belief_ids: Tensor
    desire_ids: Tensor
    goal_alignment_ids: Tensor
    trigger_ids: Tensor


def _decoder(backbone: nn.Module) -> nn.Module:
    """Resolve the decoder below either a PEFT wrapper or a plain CausalLM."""

    get_base = getattr(backbone, "get_base_model", None)
    causal_lm = get_base() if callable(get_base) else backbone
    decoder = getattr(causal_lm, "base_model", None)
    if decoder is None or decoder is causal_lm:
        decoder = getattr(causal_lm, "model", None)
    if decoder is None or decoder is causal_lm:
        raise ValueError(f"Cannot resolve decoder from {type(backbone).__name__}")
    return decoder


def _causal_lm(backbone: nn.Module) -> nn.Module:
    get_base = getattr(backbone, "get_base_model", None)
    return get_base() if callable(get_base) else backbone


def _adapter_disabled(backbone: nn.Module) -> contextlib.AbstractContextManager[Any]:
    disable = getattr(backbone, "disable_adapter", None)
    return disable() if callable(disable) else contextlib.nullcontext()


def _activate_adapter(backbone: nn.Module, name: str) -> None:
    setter = getattr(backbone, "set_adapter", None)
    if not callable(setter):
        raise ValueError("TrajWeaver requires a PEFT backbone with named adapters")
    setter(name)


def _last_valid(states: Tensor, attention_mask: Tensor) -> Tensor:
    positions = torch.arange(states.shape[1], device=states.device).unsqueeze(0)
    indices = (attention_mask.long() * positions).max(dim=-1).values
    return states[torch.arange(states.shape[0], device=states.device), indices]


def _project_logits(output_head: nn.Module, states: Tensor) -> Tensor:
    """Apply a frozen lm_head even when k-bit preparation kept it in fp32."""

    parameter = next(output_head.parameters(), None)
    if parameter is not None:
        states = states.to(dtype=parameter.dtype)
    return output_head(states)


class TrajWeaverModel(nn.Module):
    """Immutable eight-token goal plus dynamic eight-token persuadee state."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_goal_queries: int = NUM_GOAL_QUERIES,
        num_belief_queries: int = NUM_BELIEF_QUERIES,
        num_desire_queries: int = NUM_DESIRE_QUERIES,
        controller_hidden: int = 512,
    ) -> None:
        super().__init__()
        if num_goal_queries <= 0 or num_belief_queries <= 0 or num_desire_queries <= 0:
            raise ValueError("Goal, belief, and desire query banks must be non-empty")
        self.backbone = backbone
        self.num_goal_queries = num_goal_queries
        self.num_belief_queries = num_belief_queries
        self.num_desire_queries = num_desire_queries
        hidden_size = int(_causal_lm(backbone).config.hidden_size)
        input_device = backbone.get_input_embeddings().weight.device
        self.hidden_size = hidden_size

        self.goal_queries = nn.Parameter(
            torch.randn(num_goal_queries, hidden_size, device=input_device) * 0.02
        )
        self.belief_queries = nn.Parameter(
            torch.randn(num_belief_queries, hidden_size, device=input_device) * 0.02
        )
        self.desire_queries = nn.Parameter(
            torch.randn(num_desire_queries, hidden_size, device=input_device) * 0.02
        )
        self.goal_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.controller = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, controller_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.route_head = nn.Linear(controller_hidden, len(ROUTE_STATES))
        self.action_head = nn.Linear(controller_hidden, len(ACTIONS))
        self.goal_alignment_head = nn.Linear(
            controller_hidden, len(GOAL_ALIGNMENT_STATES)
        )
        self.belief_head = nn.Linear(hidden_size, len(BELIEF_STATES))
        self.desire_head = nn.Linear(hidden_size, len(DESIRE_STATES))
        self.trigger_head = nn.Linear(hidden_size, len(TRIGGER_ACTIONS))
        self.action_embeddings = nn.Parameter(
            torch.randn(len(ACTIONS), hidden_size, device=input_device) * 0.02
        )
        self.goal_alignment_embeddings = nn.Parameter(
            torch.randn(len(GOAL_ALIGNMENT_STATES), hidden_size, device=input_device)
            * 0.02
        )
        self.null_state_tokens = nn.Parameter(
            torch.randn(
                num_belief_queries + num_desire_queries,
                hidden_size,
                device=input_device,
            )
            * 0.02
        )
        self.state_norm = nn.LayerNorm(hidden_size)
        # ``device_map=auto`` has already placed the (possibly quantized)
        # backbone. Moving the wrapper as a whole is illegal for 4-bit models,
        # so place only newly created fp32 modules beside its input embedding.
        for module in (
            self.goal_projection,
            self.state_projection,
            self.controller,
            self.route_head,
            self.action_head,
            self.goal_alignment_head,
            self.belief_head,
            self.desire_head,
            self.trigger_head,
            self.state_norm,
        ):
            module.to(input_device)

        # Base weights and lm_head are always frozen. PEFT LoRA parameters keep
        # their existing trainability and are used only by ``weave``.
        for name, parameter in self.backbone.named_parameters():
            if "lora_" not in name and "modules_to_save" not in name:
                parameter.requires_grad_(False)

        self.set_trainable_component("weaver")

    @property
    def device(self) -> torch.device:
        return self.backbone.get_input_embeddings().weight.device

    @property
    def num_memory_tokens(self) -> int:
        return self.num_goal_queries + self.num_state_tokens

    @property
    def num_state_tokens(self) -> int:
        return self.num_belief_queries + self.num_desire_queries

    def auxiliary_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone.") and parameter.requires_grad:
                yield parameter

    def set_trainable_component(self, component: str) -> None:
        """Expose either Weaver or Trigger parameters; never the Reasoner.

        Both components are independent PEFT adapters on one physical
        backbone. Weaver training additionally updates latent queries,
        projections, and controller heads. Trigger training updates only its
        LoRA adapter and binary output head.
        """

        if component not in {"weaver", "trigger"}:
            raise ValueError(f"Unknown trainable component: {component}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        adapter_marker = f".{component}."
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name and adapter_marker in name:
                parameter.requires_grad_(True)
        if component == "weaver":
            for name, parameter in self.named_parameters():
                if not name.startswith("backbone.") and not name.startswith("trigger_head."):
                    parameter.requires_grad_(True)
            _activate_adapter(self.backbone, "weaver")
        else:
            for parameter in self.trigger_head.parameters():
                parameter.requires_grad_(True)
            _activate_adapter(self.backbone, "trigger")

    def component_parameters(self, component: str) -> Iterator[nn.Parameter]:
        marker = f".{component}."
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone.") and marker not in name:
                continue
            yield parameter

    def _encode_queries(
        self,
        input_ids: Tensor,
        queries: Tensor,
        attention_mask: Tensor | None = None,
        prefix_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Contextualize one query bank with the active Weaver adapter."""

        _activate_adapter(self.backbone, "weaver")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        embedding = self.backbone.get_input_embeddings()
        input_embeddings = embedding(input_ids)
        parts = [input_embeddings]
        masks = [attention_mask]
        if prefix_tokens is not None:
            if prefix_tokens.shape[0] != input_ids.shape[0]:
                raise ValueError("Prefix-token batch size must match input batch size")
            prefix_tokens = prefix_tokens.to(self.device, dtype=input_embeddings.dtype)
            parts.append(prefix_tokens)
            masks.append(
                torch.ones(
                    prefix_tokens.shape[:2],
                    dtype=attention_mask.dtype,
                    device=self.device,
                )
            )
        contextual_queries = queries.to(dtype=input_embeddings.dtype).unsqueeze(0)
        contextual_queries = contextual_queries.expand(input_ids.shape[0], -1, -1)
        parts.append(contextual_queries)
        query_mask = torch.ones(
            (input_ids.shape[0], queries.shape[0]),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        masks.append(query_mask)
        inputs_embeds = torch.cat(parts, dim=1)
        full_mask = torch.cat(masks, dim=1)
        position_ids = full_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        outputs = _decoder(self.backbone)(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, -queries.shape[0] :, :], input_embeddings

    def encode_goal(
        self, goal_ids: Tensor, goal_attention_mask: Tensor | None = None
    ) -> Tensor:
        """Encode the original goal alone into dialogue-static latent tokens."""

        raw_goal, goal_embeddings = self._encode_queries(
            goal_ids, self.goal_queries, goal_attention_mask
        )
        return self.goal_projection(raw_goal.float()).to(goal_embeddings.dtype)

    @staticmethod
    def _empty_control(
        *, batch_size: int, hidden_size: int, dtype: torch.dtype, device: torch.device
    ) -> TrajWeaverOutput:
        """Return an auditable zero-memory control record.

        Ablations that remove the latent prefix must not accidentally pass the
        learned null-state tokens to the Reasoner.  Keeping a complete control
        object makes the existing trace schema usable while all token fields
        are genuinely empty.
        """

        def empty_tokens() -> Tensor:
            return torch.empty((batch_size, 0, hidden_size), dtype=dtype, device=device)

        def zero_logits(classes: int) -> Tensor:
            return torch.zeros((batch_size, classes), dtype=dtype, device=device)

        trigger_logits = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=device),
            num_classes=len(TRIGGER_ACTIONS),
        ).to(dtype)
        return TrajWeaverOutput(
            memory_tokens=empty_tokens(),
            goal_tokens=empty_tokens(),
            belief_tokens=empty_tokens(),
            desire_tokens=empty_tokens(),
            route_logits=zero_logits(len(ROUTE_STATES)),
            action_logits=zero_logits(len(ACTIONS)),
            belief_logits=zero_logits(len(BELIEF_STATES)),
            desire_logits=zero_logits(len(DESIRE_STATES)),
            goal_alignment_logits=zero_logits(len(GOAL_ALIGNMENT_STATES)),
            trigger_logits=trigger_logits,
            route_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            action_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            belief_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            desire_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            goal_alignment_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            trigger_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def predict_trigger(
        self, prompt_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Predict whether dynamic state memory should be invoked this turn."""

        _activate_adapter(self.backbone, "trigger")
        prompt_ids = prompt_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        position_ids = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        outputs = _decoder(self.backbone)(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        trigger_state = _last_valid(outputs.last_hidden_state, attention_mask)
        return self.trigger_head(trigger_state.float())

    def weave(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        goal_ids: Tensor | None = None,
        goal_attention_mask: Tensor | None = None,
        goal_tokens: Tensor | None = None,
        force_probe: Tensor | bool | None = None,
        force_invoke: Tensor | bool | None = True,
        memory_mode: Literal["legacy", "goal_only", "always", "trigger_strict"] = "legacy",
    ) -> TrajWeaverOutput:
        """Combine immutable goal memory with the current public user state.

        Exactly one of ``goal_ids`` and ``goal_tokens`` is required. Training
        passes ``goal_ids`` so the goal encoder receives gradients. Evaluation
        encodes the goal once and reuses ``goal_tokens`` for every turn.
        """

        if memory_mode not in {"legacy", "goal_only", "always", "trigger_strict"}:
            raise ValueError(f"Unknown memory mode: {memory_mode}")
        if (goal_ids is None) == (goal_tokens is None):
            raise ValueError("Pass exactly one of goal_ids or cached goal_tokens")
        if goal_tokens is None:
            goal_tokens = self.encode_goal(goal_ids, goal_attention_mask)
        else:
            goal_tokens = goal_tokens.to(self.device)
        if goal_tokens.shape[1] != self.num_goal_queries:
            raise ValueError(
                f"Expected {self.num_goal_queries} goal tokens, got {goal_tokens.shape[1]}"
            )

        batch_size = prompt_ids.shape[0]
        if memory_mode == "always":
            force_invoke = True
        elif memory_mode == "trigger_strict":
            force_invoke = None
        if force_invoke is None:
            trigger_logits = self.predict_trigger(prompt_ids, attention_mask)
            trigger_ids = trigger_logits.argmax(dim=-1)
        else:
            if isinstance(force_invoke, bool):
                invoke_mask = torch.full(
                    (batch_size,), force_invoke, dtype=torch.bool, device=self.device
                )
            else:
                invoke_mask = force_invoke.to(
                    device=self.device, dtype=torch.bool
                ).view(-1)
            trigger_ids = invoke_mask.long()
            trigger_logits = F.one_hot(
                trigger_ids, num_classes=len(TRIGGER_ACTIONS)
            ).float()
        invoke_mask = trigger_ids.eq(TRIGGER_TO_ID["INVOKE"])

        prompt_embeddings = self.backbone.get_input_embeddings()(prompt_ids.to(self.device))
        null_state = self.null_state_tokens.float().unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if bool(invoke_mask.any()):
            state_queries = torch.cat([self.belief_queries, self.desire_queries], dim=0)
            raw_state, prompt_embeddings = self._encode_queries(
                prompt_ids, state_queries, attention_mask, prefix_tokens=goal_tokens
            )
            woven_state = self.state_projection(raw_state.float())
            projected_state = torch.where(
                invoke_mask[:, None, None], woven_state, null_state
            )
        else:
            projected_state = null_state
        raw_belief = projected_state[:, : self.num_belief_queries]
        raw_desire = projected_state[:, self.num_belief_queries :]
        goal_summary = goal_tokens.float().mean(dim=1)
        state_summary = projected_state.mean(dim=1)
        goal_state_gap = goal_summary - state_summary
        controller_state = self.controller(
            torch.cat([goal_summary, state_summary, goal_state_gap], dim=-1)
        )
        route_logits = self.route_head(controller_state)
        action_logits = self.action_head(controller_state)
        belief_logits = self.belief_head(raw_belief.mean(dim=1))
        desire_logits = self.desire_head(raw_desire.mean(dim=1))
        goal_alignment_logits = self.goal_alignment_head(controller_state)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        if force_probe is not None:
            if isinstance(force_probe, bool):
                force_probe = torch.full(
                    (prompt_ids.shape[0],), force_probe, dtype=torch.bool, device=self.device
                )
            else:
                force_probe = force_probe.to(device=self.device, dtype=torch.bool).view(-1)
            probe = F.one_hot(
                torch.full(
                    (prompt_ids.shape[0],),
                    ACTION_TO_ID["PROBE"],
                    dtype=torch.long,
                    device=self.device,
                ),
                num_classes=len(ACTIONS),
            ).to(action_probabilities.dtype)
            action_probabilities = torch.where(
                force_probe.unsqueeze(-1), probe, action_probabilities
            )
        action_vector = action_probabilities.float() @ self.action_embeddings.float()
        alignment_probabilities = torch.softmax(goal_alignment_logits, dim=-1)
        alignment_vector = (
            alignment_probabilities.float() @ self.goal_alignment_embeddings.float()
        )
        state_tokens = self.state_norm(
            projected_state + action_vector.unsqueeze(1) + alignment_vector.unsqueeze(1)
        )
        state_tokens = state_tokens.to(prompt_embeddings.dtype)
        goal_tokens = goal_tokens.to(prompt_embeddings.dtype)
        if memory_mode == "goal_only":
            belief_tokens = state_tokens[:, :0]
            desire_tokens = state_tokens[:, :0]
            memory = goal_tokens
        else:
            belief_tokens = state_tokens[:, : self.num_belief_queries]
            desire_tokens = state_tokens[:, self.num_belief_queries :]
            memory = torch.cat([goal_tokens, state_tokens], dim=1)
        if memory_mode == "trigger_strict" and not bool(invoke_mask.any()):
            # SKIP is a true no-intervention fallback.  In particular, do not
            # expose the static goal or the learned null-state tokens here.
            empty = goal_tokens[:, :0]
            memory = empty
            goal_tokens = empty
            belief_tokens = empty
            desire_tokens = empty
        action_ids = action_probabilities.argmax(dim=-1)
        return TrajWeaverOutput(
            memory_tokens=memory,
            goal_tokens=goal_tokens,
            belief_tokens=belief_tokens,
            desire_tokens=desire_tokens,
            route_logits=route_logits,
            action_logits=action_logits,
            belief_logits=belief_logits,
            desire_logits=desire_logits,
            goal_alignment_logits=goal_alignment_logits,
            trigger_logits=trigger_logits,
            route_ids=route_logits.argmax(dim=-1),
            action_ids=action_ids,
            belief_ids=belief_logits.argmax(dim=-1),
            desire_ids=desire_logits.argmax(dim=-1),
            goal_alignment_ids=goal_alignment_logits.argmax(dim=-1),
            trigger_ids=trigger_ids,
        )

    def _reasoner_states(
        self,
        prompt_ids: Tensor,
        memory_tokens: Tensor | None,
        *,
        prompt_attention_mask: Tensor | None = None,
        target_ids: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Any, int, Tensor]:
        prompt_ids = prompt_ids.to(self.device)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids)
        else:
            prompt_attention_mask = prompt_attention_mask.to(self.device)
        embedding = self.backbone.get_input_embeddings()
        prompt_embeddings = embedding(prompt_ids)
        parts = [prompt_embeddings]
        masks = [prompt_attention_mask]
        if memory_tokens is not None:
            memory_tokens = memory_tokens.to(self.device, dtype=prompt_embeddings.dtype)
            parts.append(memory_tokens)
            masks.append(
                torch.ones(memory_tokens.shape[:2], dtype=prompt_attention_mask.dtype, device=self.device)
            )
        prefix_length = sum(part.shape[1] for part in parts)
        if target_ids is not None:
            target_ids = target_ids.to(self.device)
            parts.append(embedding(target_ids))
            if target_attention_mask is None:
                target_attention_mask = torch.ones_like(target_ids)
            else:
                target_attention_mask = target_attention_mask.to(self.device)
            masks.append(target_attention_mask)
        inputs_embeds = torch.cat(parts, dim=1)
        full_mask = torch.cat(masks, dim=1)
        position_ids = full_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        with _adapter_disabled(self.backbone):
            outputs = _decoder(self.backbone)(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                return_dict=True,
            )
        return outputs, prefix_length, full_mask

    def target_logprobs(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
        memory_tokens: Tensor | None,
        *,
        prompt_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Sequence log-probability without materializing prompt vocabulary logits."""

        target_ids = target_ids.to(self.device)
        if target_attention_mask is None:
            target_attention_mask = torch.ones_like(target_ids)
        else:
            target_attention_mask = target_attention_mask.to(self.device)
        outputs, prefix_length, _ = self._reasoner_states(
            prompt_ids,
            memory_tokens,
            prompt_attention_mask=prompt_attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_attention_mask,
        )
        predicting_states = outputs.last_hidden_state[
            :, prefix_length - 1 : prefix_length - 1 + target_ids.shape[1], :
        ]
        output_head = _causal_lm(self.backbone).get_output_embeddings()
        logits = _project_logits(output_head, predicting_states)
        token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return (token_logprobs * target_attention_mask).sum(dim=-1)

    def supervised_loss(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
        *,
        goal_ids: Tensor,
        goal_attention_mask: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        route_labels: Tensor | None = None,
        action_labels: Tensor | None = None,
        belief_labels: Tensor | None = None,
        desire_labels: Tensor | None = None,
        goal_alignment_labels: Tensor | None = None,
        language_sample_weights: Tensor | None = None,
        force_probe: Tensor | bool | None = None,
        force_invoke: Tensor | bool | None = True,
        route_weight: float = 0.25,
        action_weight: float = 0.25,
        belief_weight: float = 0.2,
        desire_weight: float = 0.2,
        goal_alignment_weight: float = 0.3,
        language_weight: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        woven = self.weave(
            prompt_ids,
            prompt_attention_mask,
            goal_ids=goal_ids,
            goal_attention_mask=goal_attention_mask,
            force_probe=force_probe,
            force_invoke=force_invoke,
        )
        sequence_logprobs = self.target_logprobs(
            prompt_ids,
            target_ids,
            woven.memory_tokens,
            prompt_attention_mask=prompt_attention_mask,
            target_attention_mask=target_attention_mask,
        )
        if target_attention_mask is not None:
            token_counts = target_attention_mask.to(sequence_logprobs.device).sum(dim=-1).clamp_min(1)
        else:
            token_counts = torch.full_like(sequence_logprobs, target_ids.shape[1])
        per_sample_language = -(sequence_logprobs / token_counts)
        if language_sample_weights is None:
            language_sample_weights = torch.ones_like(per_sample_language)
        else:
            language_sample_weights = language_sample_weights.to(
                self.device, dtype=per_sample_language.dtype
            )
        language_loss = (
            per_sample_language * language_sample_weights
        ).sum() / language_sample_weights.sum().clamp_min(1.0)
        total = language_weight * language_loss
        losses: dict[str, Tensor] = {"language": language_loss.detach()}

        def add_classification(
            name: str, logits: Tensor, labels: Tensor | None, weight: float
        ) -> None:
            nonlocal total
            if labels is None or weight <= 0:
                return
            labels = labels.to(self.device)
            valid = labels.ge(0)
            if not bool(valid.any()):
                return
            value = F.cross_entropy(logits.float()[valid], labels[valid])
            total = total + weight * value
            losses[name] = value.detach()

        add_classification("route", woven.route_logits, route_labels, route_weight)
        add_classification("action", woven.action_logits, action_labels, action_weight)
        add_classification("belief", woven.belief_logits, belief_labels, belief_weight)
        add_classification("desire", woven.desire_logits, desire_labels, desire_weight)
        add_classification(
            "goal_alignment",
            woven.goal_alignment_logits,
            goal_alignment_labels,
            goal_alignment_weight,
        )
        losses["total"] = total.detach()
        return total, losses

    def trigger_loss(
        self,
        prompt_ids: Tensor,
        trigger_labels: Tensor,
        *,
        prompt_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Supervised binary trigger loss; Weaver and Reasoner stay frozen."""

        logits = self.predict_trigger(prompt_ids, prompt_attention_mask)
        labels = trigger_labels.to(self.device)
        valid = labels.ge(0)
        if not bool(valid.any()):
            raise ValueError("Trigger batch has no labeled examples")
        return F.cross_entropy(logits.float()[valid], labels[valid]), logits

    @torch.no_grad()
    def next_token_logits(
        self,
        prompt_ids: Tensor,
        *,
        prompt_attention_mask: Tensor | None = None,
        memory_tokens: Tensor | None = None,
    ) -> Tensor:
        outputs, _, full_mask = self._reasoner_states(
            prompt_ids,
            memory_tokens,
            prompt_attention_mask=prompt_attention_mask,
        )
        final_state = _last_valid(outputs.last_hidden_state, full_mask)
        return _project_logits(
            _causal_lm(self.backbone).get_output_embeddings(), final_state
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        goal_ids: Tensor | None = None,
        goal_attention_mask: Tensor | None = None,
        goal_tokens: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        turn_index: int = 0,
        max_new_tokens: int = 192,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        memory_mode: Literal["none", "legacy", "goal_only", "always", "trigger_strict"] = "legacy",
    ) -> tuple[Tensor, TrajWeaverOutput]:
        """Generate once from latent-augmented prefix using a KV cache."""

        if prompt_ids.shape[0] != 1:
            raise ValueError("Inference currently requires batch size one")
        if memory_mode == "none":
            prompt_ids = prompt_ids.to(self.device)
            empty_control = self._empty_control(
                batch_size=prompt_ids.shape[0],
                hidden_size=self.hidden_size,
                dtype=self.backbone.get_input_embeddings().weight.dtype,
                device=self.device,
            )
            woven = empty_control
        else:
            woven = self.weave(
                prompt_ids,
                prompt_attention_mask,
                goal_ids=goal_ids,
                goal_attention_mask=goal_attention_mask,
                goal_tokens=goal_tokens,
                force_probe=(turn_index == 0),
                force_invoke=None,
                memory_mode=memory_mode,
            )
        prompt_ids = prompt_ids.to(self.device)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids)
        else:
            prompt_attention_mask = prompt_attention_mask.to(self.device)
        embedding = self.backbone.get_input_embeddings()
        prompt_embeddings = embedding(prompt_ids)
        memory = woven.memory_tokens.to(prompt_embeddings.dtype)
        inputs_embeds = torch.cat([prompt_embeddings, memory], dim=1)
        attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(memory.shape[:2], dtype=prompt_attention_mask.dtype, device=self.device),
            ],
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        decoder = _decoder(self.backbone)
        output_head = _causal_lm(self.backbone).get_output_embeddings()
        generated: list[Tensor] = []
        with _adapter_disabled(self.backbone):
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = _project_logits(output_head, outputs.last_hidden_state[:, -1, :]).float()
            for _ in range(max_new_tokens):
                if temperature <= 0:
                    next_token = logits.argmax(dim=-1)
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_p, sorted_i = torch.sort(probabilities, descending=True, dim=-1)
                        remove = torch.cumsum(sorted_p, dim=-1) > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_p = sorted_p.masked_fill(remove, 0)
                        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                        next_token = sorted_i.gather(-1, torch.multinomial(sorted_p, 1)).squeeze(-1)
                    else:
                        next_token = torch.multinomial(probabilities, 1).squeeze(-1)
                generated.append(next_token)
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
                    dim=1,
                )
                outputs = decoder(
                    input_ids=next_token.unsqueeze(1),
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                logits = _project_logits(output_head, outputs.last_hidden_state[:, -1, :]).float()
        tokens = (
            torch.stack(generated, dim=1)
            if generated
            else torch.empty((1, 0), dtype=torch.long, device=self.device)
        )
        return tokens, woven


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def load_model_and_tokenizer(
    config: dict[str, Any],
    *,
    for_training: bool,
    checkpoint: str | Path | None = None,
) -> tuple[TrajWeaverModel, Any]:
    """Load one QLoRA training backbone or one bf16 inference backbone."""

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    model_path = str(model_cfg["name_or_path"])
    local_only = bool(model_cfg.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_only,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_4bit = for_training and bool(model_cfg.get("load_in_4bit_training", True))
    kwargs: dict[str, Any] = {
        "local_files_only": local_only,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
    elif torch.cuda.is_available():
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
    backbone = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if load_4bit:
        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True))
        )
    lora_cfg = config["lora"]

    def adapter_config(name: str) -> LoraConfig:
        adapter = lora_cfg.get(name, lora_cfg)
        return LoraConfig(
            r=int(adapter.get("rank", 16)),
            lora_alpha=int(adapter.get("alpha", 32)),
            lora_dropout=float(adapter.get("dropout", 0.1)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(adapter["target_modules"]),
        )

    backbone = get_peft_model(
        backbone, adapter_config("weaver"), adapter_name="weaver"
    )
    backbone.add_adapter("trigger", adapter_config("trigger"))
    backbone.set_adapter("weaver")
    architecture = config.get("architecture", {})
    model = TrajWeaverModel(
        backbone,
        num_goal_queries=int(architecture.get("goal_queries", 8)),
        num_belief_queries=int(architecture.get("belief_queries", 4)),
        num_desire_queries=int(architecture.get("desire_queries", 4)),
        controller_hidden=int(architecture.get("controller_hidden", 512)),
    )
    if checkpoint is not None:
        from .checkpoint import load_checkpoint

        load_checkpoint(checkpoint, model)
    if for_training:
        model.train()
        if bool(config["training"].get("gradient_checkpointing", True)):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            backbone.config.use_cache = False
    else:
        model.eval()
    return model, tokenizer
