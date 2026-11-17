"""TAE proxy encoder contracts for the realtime H3 codec."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch



def _runtime_module():
    root = Path(__file__).resolve().parents[3] / "python/sglang/multimodal_gen/runtime"
    loader_path = root / "loader/minimax_h3_causal.py"
    spec = importlib.util.spec_from_file_location("h3_tae_runtime_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tae_path = root / "models/vaes/taeh3.py"
    tae_spec = importlib.util.spec_from_file_location("h3_tae_runtime_model", tae_path)
    tae_module = importlib.util.module_from_spec(tae_spec)
    tae_spec.loader.exec_module(tae_module)
    return module, tae_module


_RUNTIME, _TAE = _runtime_module()
H3CausalComponents = _RUNTIME.H3CausalComponents
H3CodecGroup = _RUNTIME.H3CodecGroup
MiniMaxH3TAEEncoder = _TAE.MiniMaxH3TAEEncoder


@pytest.mark.parametrize("frames", [5, 17])
def test_encoder_produces_the_realtime_latent_shape(frames):
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    encoder = MiniMaxH3TAEEncoder().eval()
    pixels = torch.rand(1, frames, 3, 32, 64)
    with torch.no_grad():
        actual = encoder.encode_video(pixels)
    assert actual.shape == (1, 2 if frames == 5 else 5, 24, 2, 4)


def test_first_tae_latent_is_causal_within_a_packet():
    encoder = MiniMaxH3TAEEncoder().eval()
    pixels = torch.rand(1, 17, 3, 32, 64)
    changed = pixels.clone()
    changed[:, 1:] = 1 - changed[:, 1:]
    with torch.no_grad():
        original = encoder.encode_video(pixels)
        updated = encoder.encode_video(changed)
    torch.testing.assert_close(original[:, :1], updated[:, :1], rtol=0, atol=0)
    assert not torch.equal(original[:, 1:], updated[:, 1:])


def test_encoder_rejects_invalid_packet_shapes():
    with pytest.raises(ValueError, match="TAE proxy"):
        MiniMaxH3TAEEncoder().encode_video(torch.zeros(1, 16, 3, 32, 64))


@pytest.mark.parametrize("eos", [False, True])
def test_proxy_path_uses_raw_rgb_and_normalized_tae_output(eos):
    captured = []

    def encode(pixels):
        captured.append(pixels)
        return torch.full((1, 2 if eos else 5, 24, 6, 12), 7.0)

    components = H3CausalComponents(
        torch.nn.Linear(1, 1), None, None, None,
        proxy_encoder="tae",
        proxy_tae=SimpleNamespace(encode_video=encode),
        proxy_encode_size=(96, 192),
        proxy_resize="nearest",
    )
    rgb = np.random.default_rng(2026).integers(
        0, 256, (5 if eos else 17, 32, 64, 3), dtype=np.uint8)
    actual = components.encode_proxy_local(rgb, eos=eos)
    expected = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    expected = expected.repeat_interleave(3, -2).repeat_interleave(3, -1)[None]
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert actual.shape == (24, 2 if eos else 5, 6, 12)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()


def test_only_encoder_group_leader_runs_tae(monkeypatch):
    calls = []
    proxy = torch.zeros(24, 5, 2, 4, dtype=torch.bfloat16)
    for rank in range(8):
        group = H3CodecGroup(SimpleNamespace(rank=rank), (2, 3, 6, 7))
        components = H3CausalComponents(
            torch.nn.Linear(1, 1), None, None, None,
            group,
            SimpleNamespace(ranks=list(range(8)), broadcast=lambda value, src: proxy),
            proxy_encoder="tae",
            proxy_tae=SimpleNamespace(
                encode_video=lambda pixels: calls.append(rank) or proxy.permute(1, 0, 2, 3)[None]),
        )
        components.encode_proxy_local(np.zeros((17, 32, 64, 3), np.uint8), eos=False)
    assert calls == [2]
