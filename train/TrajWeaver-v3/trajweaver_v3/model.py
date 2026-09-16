"""Frozen Reasoner, hook-conditioned Weaver, and an offline utility Trigger."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import write_json


class TrajWeaverV3(nn.Module):
    def __init__(self, backbone, goal_tokens=8, state_tokens=8, trigger_hidden=128):
        super().__init__()
        if min(goal_tokens, state_tokens, trigger_hidden) <= 0:
            raise ValueError("Memory and Trigger dimensions must be positive")
        self.backbone = backbone
        size = backbone.config.hidden_size
        self.goal_queries = nn.Parameter(torch.randn(goal_tokens, size) * 0.02)
        self.state_queries = nn.Parameter(torch.randn(state_tokens, size) * 0.02)
        self.goal_projection = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, size))
        self.hook_projection = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, size))
        self.state_projection = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, size))
        self.trigger = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, trigger_hidden),
                                     nn.GELU(), nn.Linear(trigger_hidden, 2))
        for name, module in self.named_children():
            if name != "backbone":
                module.to(self.device)
        self.goal_queries.data = self.goal_queries.data.to(self.device)
        self.state_queries.data = self.state_queries.data.to(self.device)
        self.set_component("weaver")

    @property
    def device(self):
        return self.backbone.get_input_embeddings().weight.device

    @property
    def decoder(self):
        return self.backbone.get_base_model().model

    def set_component(self, component):
        if component not in ("weaver", "trigger", "frozen"):
            raise ValueError(component)
        self.component = component
        self.backbone.set_adapter("weaver")
        for name, parameter in self.named_parameters():
            weaver = ("lora_" in name and ".weaver." in name) or (
                not name.startswith("backbone.") and not name.startswith("trigger."))
            parameter.requires_grad_((component == "weaver" and weaver) or
                                     (component == "trigger" and name.startswith("trigger.")))

    def _forward_embeddings(self, embeddings, *, cache=False):
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=self.device)
        return self.decoder(inputs_embeds=embeddings, attention_mask=mask,
                            use_cache=cache, return_dict=True)

    def _queries(self, context, queries, projection):
        # With ``device_map=auto`` the decoder output can land on a later GPU
        # than the input embedding and auxiliary modules.  Keep the query
        # context on the input device and explicitly move features to the
        # projection device before applying the small trainable head.
        anchor = self.device
        context = context.to(anchor)
        query = queries.to(device=anchor, dtype=context.dtype).unsqueeze(0).expand(context.shape[0], -1, -1)
        output = self._forward_embeddings(torch.cat((context, query), dim=1))
        projection_device = next(projection.parameters()).device
        features = output.last_hidden_state[:, -queries.shape[0]:].to(projection_device).float()
        return projection(features).to(device=anchor, dtype=context.dtype)

    def encode_goal(self, ids):
        embeddings = self.backbone.get_input_embeddings()(ids.to(self.device))
        return self._queries(embeddings, self.goal_queries, self.goal_projection)

    @torch.no_grad()
    def hook(self, prompt):
        """Only the observable prefix reaches the frozen Reasoner; never the target."""
        embeddings = self.backbone.get_input_embeddings()(prompt.to(self.device))
        with self.backbone.disable_adapter():
            return self._forward_embeddings(embeddings).last_hidden_state.detach()

    def state(self, hook, goal):
        hook = hook.to(goal.device)
        context = torch.cat((self.hook_projection(hook.float()).to(goal.dtype), goal), dim=1)
        return self._queries(context, self.state_queries, self.state_projection)

    def trigger_logits(self, hook):
        trigger_device = next(self.trigger.parameters()).device
        return self.trigger(hook[:, -1].detach().to(trigger_device).float())

    def logits(self, hidden):
        head = self.backbone.get_output_embeddings()
        return head(hidden.to(head.weight.dtype)).float()

    def nll(self, prompt, target, memory):
        """Mean next-token NLL; gradients flow through memory into Weaver only."""
        prompt, target = prompt.to(self.device), target.to(self.device)
        if target.shape[1] == 0:
            raise ValueError("Empty target")
        embedding = self.backbone.get_input_embeddings()
        prefix = torch.cat((embedding(prompt), memory), dim=1)
        inputs = torch.cat((prefix, embedding(target)), dim=1)
        with self.backbone.disable_adapter():
            output = self._forward_embeddings(inputs)
        states = output.last_hidden_state[:, prefix.shape[1] - 1:-1]
        return F.cross_entropy(self.logits(states).reshape(-1, self.backbone.config.vocab_size),
                               target.reshape(-1))

    def sft_loss(self, prompt, goal_ids, target, goal_only_weight=0.25):
        if goal_only_weight < 0:
            raise ValueError("Negative goal-only weight")
        hook = self.hook(prompt)
        goal = self.encode_goal(goal_ids)
        dynamic = self.state(hook, goal)
        full = self.nll(prompt, target, torch.cat((goal, dynamic), dim=1))
        skip = self.nll(prompt, target, goal) if goal_only_weight else full.detach()
        return (full + goal_only_weight * skip) / (1 + goal_only_weight)

    @torch.no_grad()
    def utility(self, prompt, goal_ids, target):
        goal = self.encode_goal(goal_ids)
        hook = self.hook(prompt)
        dynamic = self.state(hook, goal)
        skip = float(self.nll(prompt, target, goal))
        invoke = float(self.nll(prompt, target, torch.cat((goal, dynamic), dim=1)))
        return {"skip_nll": skip, "invoke_nll": invoke, "gain": skip - invoke}

    @torch.no_grad()
    def generate(self, prompt, *, goal_ids=None, goal=None, mode="trigger",
                 max_new_tokens=192, eos_token_id=None):
        if mode not in ("none", "goal_only", "always", "trigger"):
            raise ValueError(mode)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prompt = prompt.to(self.device)
        if prompt.shape[0] != 1:
            raise ValueError("Generation uses batch size one")
        embedding = self.backbone.get_input_embeddings()
        prompt_emb = embedding(prompt)
        memory, probability, invoked = prompt_emb[:, :0], None, False
        if mode != "none":
            if goal is None:
                if goal_ids is None:
                    raise ValueError("Goal input required")
                goal = self.encode_goal(goal_ids)
            memory = goal
            if mode != "goal_only":
                hook = self.hook(prompt)
                if mode == "trigger":
                    probability = float(self.trigger_logits(hook).softmax(-1)[0, 1])
                    invoked = probability > 0.5
                else:
                    invoked = True
                if invoked:
                    memory = torch.cat((goal, self.state(hook, goal)), dim=1)
        inputs = torch.cat((prompt_emb, memory), dim=1)
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else
                      ([] if eos_token_id is None else [eos_token_id]))
        generated = []
        with self.backbone.disable_adapter():
            output = self._forward_embeddings(inputs, cache=True)
            cache = output.past_key_values
            logits = self.logits(output.last_hidden_state[:, -1])
            length = inputs.shape[1]
            for _ in range(max_new_tokens):
                token = logits.argmax(-1)
                generated.append(token)
                if int(token.item()) in eos_ids:
                    break
                length += 1
                output = self.decoder(input_ids=token[:, None], past_key_values=cache,
                    attention_mask=torch.ones((1, length), dtype=torch.long, device=self.device),
                    use_cache=True, return_dict=True)
                cache = output.past_key_values
                logits = self.logits(output.last_hidden_state[:, -1])
        return torch.stack(generated, dim=1), {
            "memory_mode": mode, "trigger_action": "INVOKE" if invoked else "SKIP",
            "invoke_probability": probability, "memory_tokens": memory.shape[1],
            "goal_tokens": 0 if mode == "none" else self.goal_queries.shape[0],
        }


def load_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_checkpoint(path, model, config, metadata):
    from peft import get_peft_model_state_dict

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 3, "config": config, "metadata": metadata,
        "auxiliary": {key: value.detach().cpu() for key, value in model.state_dict().items()
                      if not key.startswith("backbone.")},
        "weaver": {key: value.detach().cpu() for key, value in
                   get_peft_model_state_dict(model.backbone, adapter_name="weaver").items()}}
    temporary = path / "trajweaver.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(path / "trajweaver.pt")
    write_json(path / "metadata.json", metadata)


def load_checkpoint(path, model, config=None):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    payload = torch.load(Path(path) / "trajweaver.pt", map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("This loader accepts only TrajWeaver-v3 checkpoints")
    if config is not None:
        for field in ("architecture", "lora"):
            if payload["config"][field] != config[field]:
                raise ValueError(f"Checkpoint/config mismatch: {field}")
        if payload["config"]["model"]["name_or_path"] != config["model"]["name_or_path"]:
            raise ValueError("Checkpoint/base model mismatch")
        if payload["config"]["data"] != config["data"]:
            raise ValueError("Checkpoint/tokenization configuration mismatch")
    expected = {key for key in model.state_dict() if not key.startswith("backbone.")}
    if set(payload["auxiliary"]) != expected:
        raise ValueError("Incomplete or incompatible auxiliary checkpoint")
    if set(payload["weaver"]) != set(get_peft_model_state_dict(model.backbone, adapter_name="weaver")):
        raise ValueError("Incomplete or incompatible Weaver adapter")
    model.load_state_dict(payload["auxiliary"], strict=False)
    set_peft_model_state_dict(model.backbone, payload["weaver"], adapter_name="weaver")
    return payload["metadata"]


def load_model(config, checkpoint=None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["name_or_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16}
    if cfg.get("load_in_4bit", True):
        if not torch.cuda.is_available():
            raise RuntimeError("Configured 4-bit model requires a CUDA GPU")
        kwargs.update(device_map=cfg.get("device_map", "auto"), quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    elif torch.cuda.is_available():
        kwargs["device_map"] = cfg.get("device_map", "auto")
    else:
        kwargs["torch_dtype"] = torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg["name_or_path"], **kwargs)
    if cfg.get("load_in_4bit", True):
        # Adapter/Reasoner switching shares one backbone; checkpoint recomputation
        # must not run under a different adapter state.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    lora = config["lora"]
    backbone = get_peft_model(base, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
        lora_dropout=0.0, target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        adapter_name="weaver")
    model = TrajWeaverV3(backbone, **config["architecture"])
    metadata = load_checkpoint(checkpoint, model, config) if checkpoint is not None else {}
    model.eval()
    return model, tokenizer, metadata
