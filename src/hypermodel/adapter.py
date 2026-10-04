"""Architecture-neutral view of a causal LM: decoder blocks and the matrices that write into the residual stream."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class _Layout:
    blocks: str  # dotted path from the model root to the ModuleList of decoder blocks
    mlp_out: str  # dotted path from a block to its MLP output projection
    final_norm: str  # dotted path from the model root to the norm before the unembedding


_LLAMA_LIKE = _Layout("model.layers", "mlp.down_proj", "model.norm")
_REGISTRY = {
    "gpt_neox": _Layout("gpt_neox.layers", "mlp.dense_4h_to_h", "gpt_neox.final_layer_norm"),
    "llama": _LLAMA_LIKE,
    "mistral": _LLAMA_LIKE,
    "qwen2": _LLAMA_LIKE,
    "qwen3": _LLAMA_LIKE,
}


class ModelAdapter:
    def __init__(self, model: nn.Module, tokenizer=None):
        model_type = model.config.model_type
        if model_type not in _REGISTRY:
            raise ValueError(f"unsupported architecture {model_type!r}; add it to the adapter registry")
        self.model = model
        self.tokenizer = tokenizer
        self._layout = _REGISTRY[model_type]

    @property
    def n_layers(self) -> int:
        return len(self.blocks())

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def blocks(self) -> nn.ModuleList:
        """A forward hook on block l sees the residual stream after layer l."""
        return self.model.get_submodule(self._layout.blocks)

    def residual_writers(self) -> list[nn.Linear]:
        return [b.get_submodule(self._layout.mlp_out) for b in self.blocks()]

    def final_norm(self) -> nn.Module:
        return self.model.get_submodule(self._layout.final_norm)

    def unembedding(self) -> torch.Tensor:
        """[vocab, d_model]: final_norm(h) @ unembedding().T are the logits."""
        return self.model.get_output_embeddings().weight


def load(name: str, device: str = "cuda", dtype: torch.dtype = torch.float32) -> ModelAdapter:
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # batched greedy generation continues from the true last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return ModelAdapter(model, tok)
