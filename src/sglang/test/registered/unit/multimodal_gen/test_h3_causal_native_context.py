# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import torch
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import H3CausalComponents
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_condition_context_with_real_native_local_attention_on_cpu():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [7] * len(text)}

        def convert_tokens_to_ids(self, token):
            return 8

    class AttentionConsumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = LocalAttention(
                num_heads=2,
                head_size=32,
                causal=True,
                required_attention_backend=AttentionBackendEnum.TORCH_SDPA,
                compute_dtype=torch.float32,
            )

        def encode_ids(self, ids, **kwargs):
            context = get_forward_context()
            assert context.current_timestep == 0 and context.attn_metadata is None
            rows = ids.float()[None, :, None, None].expand(1, -1, 2, 32).contiguous()
            return self.attention(rows, rows, rows)[0].flatten(1)

    def image_processor(**kwargs):
        return {
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }

    image_processor.merge_size = 1
    components = H3CausalComponents(
        None,
        AttentionConsumer(),
        None,
        SimpleNamespace(image_processor=image_processor, tokenizer=Tokenizer()),
    )
    assert get_forward_context_or_none() is None
    with set_forward_context(73, object()):
        outer = get_forward_context()
        hidden, tags = components.condition(
            "prompt", np.zeros((32, 32, 3), dtype=np.uint8)
        )
        assert get_forward_context() is outer
    assert get_forward_context_or_none() is None
    assert hidden.shape == (len(tags), 64) and torch.isfinite(hidden).all()
