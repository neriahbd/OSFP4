from types import SimpleNamespace

import pytest
import torch
from transformers import (
    BloomConfig,
    BloomForCausalLM,
    CohereConfig,
    CohereForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    LlamaConfig,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    WhisperConfig,
    WhisperForConditionalGeneration,
)
from transformers import (
    LlamaForCausalLM as TransformersLlamaForCausalLM,
)

from llmcompressor.core import State
from llmcompressor_osfp4.modifiers import OSFP4Modifier as _OSFP4Modifier
from llmcompressor_osfp4.modifiers.base import OSFP4Mapping
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY,
)
from llmcompressor.modifiers.transform.smoothquant.utils import MAPPINGS_REGISTRY

from ._testing import LlamaForCausalLM, make_osfp4_modifier


def _linear(in_features=16, out_features=16):
    return torch.nn.Linear(in_features, out_features, bias=False)


class MappingFixtureModel(torch.nn.Module):
    def __init__(self, block, config=None, extra_linear=False):
        super().__init__()
        self.config = config or SimpleNamespace()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([block])
        if extra_linear:
            self.vision_projector = _linear()
        self.lm_head = _linear()


def _standard_mapping_block(*, shared_expert=False, attention_gate=False):
    block = torch.nn.Module()
    block.input_layernorm = torch.nn.LayerNorm(16)
    block.self_attn = torch.nn.Module()
    block.self_attn.q_proj = _linear()
    block.self_attn.k_proj = _linear()
    block.self_attn.v_proj = _linear()
    block.self_attn.o_proj = _linear()
    if attention_gate:
        block.self_attn.gate_proj = _linear()
    block.post_attention_layernorm = torch.nn.LayerNorm(16)
    block.pre_mlp_layernorm = torch.nn.LayerNorm(16)
    block.mlp = torch.nn.Module()
    mlp = block.mlp
    if shared_expert:
        mlp.shared_expert = torch.nn.Module()
        mlp = mlp.shared_expert
    mlp.gate_proj = _linear(16, 32)
    mlp.up_proj = _linear(16, 32)
    mlp.down_proj = _linear(32, 16)
    return block


def _phi_mapping_block():
    block = torch.nn.Module()
    block.input_layernorm = torch.nn.LayerNorm(16)
    block.self_attn = torch.nn.Module()
    block.self_attn.qkv_proj = _linear(16, 48)
    block.self_attn.o_proj = _linear()
    block.post_attention_layernorm = torch.nn.LayerNorm(16)
    block.mlp = torch.nn.Module()
    block.mlp.gate_up_proj = _linear(16, 64)
    block.mlp.down_proj = _linear(32, 16)
    return block


def _bloom_mapping_block():
    block = torch.nn.Module()
    block.input_layernorm = torch.nn.LayerNorm(16)
    block.self_attention = torch.nn.Module()
    block.self_attention.query_key_value = _linear(16, 48)
    block.self_attention.dense = _linear()
    block.post_attention_layernorm = torch.nn.LayerNorm(16)
    block.mlp = torch.nn.Module()
    block.mlp.dense_h_to_4h = _linear(16, 64)
    block.mlp.dense_4h_to_h = _linear(64, 16)
    return block


def _whisper_mapping_block():
    block = torch.nn.Module()
    block.self_attn_layer_norm = torch.nn.LayerNorm(16)
    block.self_attn = torch.nn.Module()
    block.self_attn.q_proj = _linear()
    block.self_attn.k_proj = _linear()
    block.self_attn.v_proj = _linear()
    block.self_attn.out_proj = _linear()
    block.final_layer_norm = torch.nn.LayerNorm(16)
    block.fc1 = _linear(16, 32)
    block.fc2 = _linear(32, 16)
    return block


def _deepseek_mapping_block():
    block = torch.nn.Module()
    block.input_layernorm = torch.nn.LayerNorm(16)
    block.self_attn = torch.nn.Module()
    block.self_attn.q_a_proj = _linear()
    block.self_attn.kv_a_proj_with_mqa = _linear()
    block.self_attn.q_b_proj = _linear()
    block.self_attn.kv_b_proj = _linear()
    block.self_attn.o_proj = _linear()
    block.post_attention_layernorm = torch.nn.LayerNorm(16)
    block.mlp = torch.nn.Module()
    block.mlp.gate_proj = _linear(16, 32)
    block.mlp.up_proj = _linear(16, 32)
    block.mlp.down_proj = _linear(32, 16)
    block.mlp.expert_weights = torch.nn.Parameter(torch.ones(2, 16, 16))
    return block


def _mapping_fixture_for_architecture(architecture):
    if architecture in {"BloomForCausalLM", "ChatGLMForConditionalGeneration"}:
        block = _bloom_mapping_block()
    elif architecture in {"Phi3ForCausalLM", "Phi3VForCausalLM"}:
        block = _phi_mapping_block()
    elif architecture == "WhisperForConditionalGeneration":
        block = _whisper_mapping_block()
    elif architecture in {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "GlmMoeDsaForCausalLM",
    }:
        block = _deepseek_mapping_block()
    elif architecture == "AfmoeForCausalLM":
        block = _standard_mapping_block(attention_gate=True)
    else:
        block = _standard_mapping_block()

    config = SimpleNamespace()
    if architecture in SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY:
        config.text_config = SimpleNamespace(
            layer_types=["full_attention"], num_hidden_layers=1
        )
        block = _standard_mapping_block(shared_expert="Moe" in architecture)
    model_type = type(architecture, (MappingFixtureModel,), {})
    return model_type(
        block,
        config=config,
        extra_linear="ConditionalGeneration" in architecture,
    )


def _resolve_mappings(model, targets=("Linear",), ignore=()):
    modifier = _OSFP4Modifier(
        scheme="NVFP4",
        targets=list(targets),
        ignore=list(ignore),
    )
    return modifier._resolve_mappings(model)


@pytest.mark.parametrize(
    "architecture",
    sorted(MAPPINGS_REGISTRY.keys() | SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY.keys()),
)
def test_every_smoothquant_architecture_resolves_complete_coverage(architecture):
    model = _mapping_fixture_for_architecture(architecture)
    mappings = _resolve_mappings(model)
    covered = [layer for mapping in mappings for layer in mapping.balance_layers]
    expected = [
        module
        for _name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    assert any(not mapping.requires_runtime_smoothing for mapping in mappings)
    assert len(covered) == len(set(covered))
    assert set(covered) == set(expected)
    assert all(
        mapping.requires_runtime_smoothing == (mapping.smooth_layer is None)
        for mapping in mappings
    )
    assert any(mapping.balance_layers == (model.lm_head,) for mapping in mappings)


def test_mapping_resolution_excludes_non_module_expert_weights():
    model = _mapping_fixture_for_architecture("DeepseekV3ForCausalLM")
    mappings = _resolve_mappings(model)

    assert isinstance(model.model.layers[0].mlp.expert_weights, torch.nn.Parameter)
    assert all(
        model.model.layers[0].mlp.expert_weights is not target
        for mapping in mappings
        for target in mapping.balance_layers
    )


def test_mapping_resolution_applies_targets_and_ignore():
    model = _mapping_fixture_for_architecture("LlamaForCausalLM")
    mappings = _resolve_mappings(model, ignore=("lm_head",))
    assert all(mapping.balance_layers != (model.lm_head,) for mapping in mappings)

    target_name = "model.layers.0.self_attn.q_proj"
    target = model.model.layers[0].self_attn.q_proj
    mappings = _resolve_mappings(model, targets=(target_name,))
    assert mappings == [
        OSFP4Mapping(
            mapping_name=target_name,
            smooth_layer=None,
            balance_layers=(target,),
        )
    ]


def test_real_representative_models_resolve_complete_coverage():
    causal_kwargs = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    cases = [
        (TransformersLlamaForCausalLM, LlamaConfig(**causal_kwargs)),
        (MistralForCausalLM, MistralConfig(**causal_kwargs)),
        (Qwen2ForCausalLM, Qwen2Config(**causal_kwargs)),
        (Qwen3ForCausalLM, Qwen3Config(**causal_kwargs, head_dim=8)),
        (Gemma2ForCausalLM, Gemma2Config(**causal_kwargs, head_dim=8)),
        (Phi3ForCausalLM, Phi3Config(**causal_kwargs)),
        (
            BloomForCausalLM,
            BloomConfig(
                hidden_size=16,
                n_layer=1,
                n_head=2,
                vocab_size=32,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            ),
        ),
        (CohereForCausalLM, CohereConfig(**causal_kwargs)),
        (
            WhisperForConditionalGeneration,
            WhisperConfig(
                d_model=16,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                encoder_ffn_dim=32,
                decoder_ffn_dim=32,
                vocab_size=32,
                num_mel_bins=16,
                max_source_positions=32,
                max_target_positions=32,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                decoder_start_token_id=1,
            ),
        ),
    ]

    for model_class, config in cases:
        model = model_class(config)
        mappings = _resolve_mappings(model)
        covered = {layer for mapping in mappings for layer in mapping.balance_layers}
        expected = {
            module
            for _name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        assert covered == expected, model_class.__name__


def test_qwen3_decoder_layer_resolves_four_expected_mapping_groups():
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3ForCausalLM(config)
    mappings = _resolve_mappings(model, ignore=("lm_head",))

    assert len(mappings) == 12
    by_name = {mapping.mapping_name: mapping for mapping in mappings}
    for layer in range(3):
        prefix = f"model.layers.{layer}"
        expected = {
            f"{prefix}.input_layernorm": (3, False),
            f"{prefix}.post_attention_layernorm": (2, False),
            f"{prefix}.self_attn.o_proj": (1, True),
            f"{prefix}.mlp.down_proj": (1, True),
        }
        for name, (target_count, runtime_scale) in expected.items():
            assert len(by_name[name].balance_layers) == target_count
            assert by_name[name].requires_runtime_smoothing is runtime_scale


def test_unknown_architecture_uses_smoothquant_default_fallback():
    UnknownArchitecture = type("UnknownArchitecture", (LlamaForCausalLM,), {})
    model = UnknownArchitecture()
    mappings = _resolve_mappings(model, ignore=("lm_head",))

    assert mappings
    assert any(not mapping.requires_runtime_smoothing for mapping in mappings)


def test_unmapped_linear_is_derived_as_runtime_smooth_quant_scale_target():
    model = LlamaForCausalLM()
    model.model.layers[0].extra = torch.nn.Linear(16, 16)
    modifier = make_osfp4_modifier()
    modifier.on_initialize(State(model=model))
    mapping = next(
        mapping
        for mapping in modifier._resolved_mappings
        if mapping.mapping_name == "model.layers.0.extra"
    )
    assert mapping.requires_runtime_smoothing
    assert mapping.balance_layers == (model.model.layers[0].extra,)


def test_resolver_preserves_smooth_layer_then_runtime_mapping_order():
    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier()
    mappings = modifier._resolve_mappings(model)
    block = model.model.layers[0]
    assert [mapping.mapping_name for mapping in mappings] == [
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
    ]
    assert mappings[0].balance_layers == (
        block.self_attn.q_proj,
        block.self_attn.k_proj,
        block.self_attn.v_proj,
    )
    assert mappings[1].balance_layers == (block.mlp.gate_proj, block.mlp.up_proj)
    assert [mapping.requires_runtime_smoothing for mapping in mappings] == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("matches", [([], []), ([], [torch.nn.LayerNorm(16)])])
def test_resolver_skips_incomplete_smooth_layer_matches(monkeypatch, matches):
    from llmcompressor_osfp4.modifiers import base

    model = LlamaForCausalLM()
    modifier = make_osfp4_modifier(targets=["model.layers.0.self_attn.q_proj"])
    if matches[1]:
        matches = ([], [model.model.layers[0].input_layernorm])
    monkeypatch.setattr(base, "match_modules_set", lambda *args: [matches])
    mappings = modifier._resolve_mappings(model)
    assert len(mappings) == 1
    assert mappings[0].requires_runtime_smoothing
    assert mappings[0].balance_layers == (model.model.layers[0].self_attn.q_proj,)


def test_resolver_rejects_ambiguous_smooth_layer_matches(monkeypatch):
    from llmcompressor_osfp4.modifiers import base

    model = LlamaForCausalLM()
    block = model.model.layers[0]
    monkeypatch.setattr(
        base,
        "match_modules_set",
        lambda *args: [
            (
                [block.self_attn.q_proj],
                [block.input_layernorm, block.post_attention_layernorm],
            )
        ],
    )
    with pytest.raises(ValueError, match="single smooth layer"):
        make_osfp4_modifier().on_initialize(State(model=model))
