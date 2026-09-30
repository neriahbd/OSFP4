import pytest
import torch
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization import fake_quantize, preset_name_to_scheme
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

from llmcompressor_osfp4.observers.fp4 import (
    quantize_e2m1,
)
from llmcompressor_osfp4.observers.scale_selection import (  # noqa: E501
    _get_e4m3_scale_grid,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_compressed_tensors_at_all_boundaries(dtype):
    boundaries = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=dtype,
    )
    lower = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    upper = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
    values = torch.cat((lower, boundaries, upper, -lower, -boundaries, -upper))
    values = torch.cat((values, torch.tensor([0.0, -0.0], dtype=dtype)))
    original = values.clone()

    actual = quantize_e2m1(values)
    expected = FP4_E2M1_DATA.cast_to_fp4(values.clone())
    assert torch.equal(values, original)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_random_finite_inputs(dtype):
    generator = torch.Generator().manual_seed(23)
    values = (torch.randn(4096, generator=generator) * 10).to(dtype)
    actual = quantize_e2m1(values)
    expected = FP4_E2M1_DATA.cast_to_fp4(values.clone())
    assert torch.equal(actual, expected)


def test_local_e2m1_matches_canonical_nvfp4_fake_quantize():
    values = torch.tensor(
        [
            -8.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            8.0,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    args = preset_name_to_scheme("NVFP4", ["Linear"]).weights
    scale = torch.ones((1, 1), dtype=torch.float32)
    zero_point = torch.zeros((1, 1), dtype=args.zp_dtype)

    actual = fake_quantize(
        values,
        scale,
        zero_point,
        args,
        global_scale=torch.ones(1),
    )
    expected = quantize_e2m1(values)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_for_every_e4m3_scale(dtype):
    scales = _get_e4m3_scale_grid("cpu", torch.float32).to(dtype).unsqueeze(1)
    codes = torch.tensor(
        [
            -6.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
        ],
        dtype=dtype,
    ).unsqueeze(0)
    normalized = (codes * scales) / scales
    actual = quantize_e2m1(normalized)
    expected = FP4_E2M1_DATA.cast_to_fp4(normalized.clone())
    assert torch.equal(actual, expected)


def test_nvfp4_pack_preserves_codes_and_signed_zero_bits():
    codes = torch.tensor(
        [[0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 6.0, -6.0]],
        dtype=torch.float32,
    )
    packed = pack_fp4_to_uint8(codes)
    unpacked = unpack_fp4_from_uint8(packed, 1, codes.shape[1], dtype=codes.dtype)

    assert torch.equal(unpacked, codes)
    assert torch.equal(torch.signbit(unpacked), torch.signbit(codes))
    assert int(packed[0, 0]) == 0x80
