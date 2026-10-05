import pytest
import torch
import torch.nn.functional as F

import my_flash_attn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("implementation", [my_flash_attn.run_v4, my_flash_attn.run_v5])
@pytest.mark.parametrize("sequence_length", [64, 256, 1024])
def test_forward_matches_sdpa(implementation, sequence_length):
    torch.manual_seed(42)
    q = torch.randn(sequence_length, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual = implementation(q, k, v)
    expected = F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None]
    )[0, 0]
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_v4_uses_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.randn(256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = my_flash_attn.run_v4(q, k, v)
        expected = F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None]
        )[0, 0]
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_v5_uses_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.randn(256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        actual = my_flash_attn.run_v5(q, k, v)
        expected = F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None]
        )[0, 0]
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_cpu_input_is_rejected():
    tensor = torch.randn(64, 64, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        my_flash_attn.run_v4(tensor, tensor, tensor)


def test_float32_input_is_rejected():
    tensor = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="float16"):
        my_flash_attn.run_v4(tensor, tensor, tensor)


def test_unaligned_sequence_is_rejected():
    tensor = torch.randn(65, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="multiple of 64"):
        my_flash_attn.run_v4(tensor, tensor, tensor)
