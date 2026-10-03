import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

D_MODEL = 32
N_LAYERS = 3


def _tiny_neox():
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=D_MODEL, num_hidden_layers=N_LAYERS,
                        num_attention_heads=4, intermediate_size=64)
    return GPTNeoXForCausalLM(cfg)


def _tiny_llama():
    cfg = LlamaConfig(vocab_size=64, hidden_size=D_MODEL, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=64)
    return LlamaForCausalLM(cfg)


@pytest.fixture(params=[_tiny_neox, _tiny_llama], ids=["gpt_neox", "llama"])
def tiny_model(request):
    torch.manual_seed(0)
    return request.param().eval()
