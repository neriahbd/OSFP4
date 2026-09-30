from types import SimpleNamespace

import torch

from llmcompressor.core import Event, State
from llmcompressor_osfp4.modifiers import OSFP4Modifier


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16)
        self.k_proj = torch.nn.Linear(16, 16)
        self.v_proj = torch.nn.Linear(16, 16)
        self.o_proj = torch.nn.Linear(16, 16)

    def forward(self, x, past_key_value=None):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        if past_key_value is not None:
            key, value = past_key_value.update(key, value)
        return self.o_proj(query + key + value)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(16, 32)
        self.up_proj = torch.nn.Linear(16, 32)
        self.down_proj = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(16)
        self.self_attn = TinyAttention()
        self.post_attention_layernorm = torch.nn.LayerNorm(16)
        self.mlp = TinyMLP()

    def forward(self, x):
        x = self.self_attn(self.input_layernorm(x))
        return self.mlp(self.post_attention_layernorm(x))


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            hidden_size=16,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyDecoder()])
        self.lm_head = torch.nn.Linear(16, 16)

    def save_pretrained(self, *_args, **_kwargs):
        raise NotImplementedError("the synthetic test model is not serializable")

    def forward(self, x):
        return self.lm_head(self.model.layers[0](x))


def make_osfp4_modifier(**kwargs):
    """Construct the paper-compatible OSFP4 configuration used in tests."""
    kwargs.setdefault("scheme", "NVFP4")
    kwargs.setdefault("ignore", ["lm_head"])
    return OSFP4Modifier(**kwargs)


def run_modifier(*, steps=0, **kwargs):
    torch.manual_seed(0)
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(steps=steps, **kwargs)
    state = State(model=model)
    event = Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    model(torch.randn(2, 3, 16))
    modifier.on_sequential_epoch_end(state, event, list(model.modules()))
    modifier.on_calibration_end(state, event)
    return model, modifier
