"""Actual reduced H3 stack and an independent full S+D attention oracle."""
from contextlib import contextmanager, ExitStack
import os
from unittest.mock import patch

import torch
from torch.nn import functional as F

from projects.minimax_h3.tools.ref2va_acceptance import tiny_model


def full_window(static, dynamic):
    """Join captured adapter inputs; S reads S and D reads S+D."""
    prefix = static['x'].shape[1]
    total = prefix + dynamic['x'].shape[1]
    value = dict(dynamic)
    value.pop('past_key_values')
    value.pop('update_past_key_values')
    value.pop('rope_cache', None)
    value.pop('local_embedding_layout', None)
    for name in ('x', 'eps', 'audio_x', 'audio_eps', 'img_position_ids'):
        value[name] = torch.cat((static[name], dynamic[name]), dim=1)
    value['token_tags'] = torch.cat((static['token_tags'], dynamic['token_tags']))
    times = torch.cat([part['unique_timesteps'][part['inverse_indices']] for part in (static, dynamic)])
    value['unique_timesteps'], value['inverse_indices'] = torch.unique(times, return_inverse=True)
    for name in ('img_pos_info', 'text_pos_info', 'audio_pos_info'):
        value[name] = dict(position_ids=torch.cat((static[name]['position_ids'],
                                                   prefix+dynamic[name]['position_ids'])))
    value['img_pos_for_infer_output_info'] = dict(
        position_ids=prefix+dynamic['img_pos_for_infer_output_info']['position_ids'])
    value['prompt_embeds'] = torch.cat((static['prompt_embeds'], dynamic['prompt_embeds']))
    lengths = torch.cat([part['refiner_packed_seq_params']['cu_seqlens_q'].diff()
                         for part in (static, dynamic)])
    lengths = lengths[lengths > 0]
    value['refiner_packed_seq_params'] = dict(
        cu_seqlens_q=torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32))),
        max_seqlen_q=int(lengths.max()))
    value['sample_lens'] = [total]
    device = value['x'].device
    static_real = static['sample_lens'][0]
    dynamic_end = prefix+dynamic['sample_lens'][0]
    queries = [[0, static_real], [prefix, dynamic_end], [prefix, dynamic_end]]
    keys = [[0, static_real], [0, static_real], [prefix, dynamic_end]]
    for start, stop in ((static_real, prefix), (dynamic_end, total)):
        if stop > start:
            queries.append([start, stop])
            keys.append([start, stop])
    value.update(q_ranges=torch.tensor(queries, dtype=torch.int32, device=device),
                 k_ranges=torch.tensor(keys, dtype=torch.int32, device=device),
                 attn_type_map=torch.zeros(len(queries), dtype=torch.int32, device=device),
                 attn_workloads=[sum((b-a)*(d-c) for (a,b),(c,d) in zip(queries, keys))], attention_mask=None)
    if device.type == 'cuda':
        from torch.nn.attention.flex_attention import create_block_mask
        def mask(batch, head, query, key):
            s = (query < static_real) & (key < static_real)
            d = (query >= prefix) & (query < dynamic_end) & ((key < static_real) | ((key >= prefix) & (key < dynamic_end)))
            sp = (query >= static_real) & (query < prefix) & (key >= static_real) & (key < prefix)
            dp = (query >= dynamic_end) & (query < total) & (key >= dynamic_end) & (key < total)
            return s | d | sp | dp
        value['attention_mask'] = create_block_mask(mask, 1, 1, total, total, device=str(device), _compile=True)
    return value


def rectangle_sdpa(q, k, v, attention_mask, q_ranges, k_ranges, attn_type_map,
                   attn_workloads, sample_lens):
    assert not torch.any(attn_type_map)
    mask = torch.zeros((len(q), len(k)), dtype=torch.bool, device=q.device)
    for (a, b), (c, d) in zip(q_ranges.tolist(), k_ranges.tolist(), strict=True):
        mask[a:b, c:d] = True
    return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
        v.transpose(0, 1), attn_mask=mask).transpose(0, 1)


@contextmanager
def actual_model(*, fp32=False):
    from projects.minimax_h3.modeling.transformer import causal_model, model
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        device = os.environ.get('H3_BIDIR_TEST_DEVICE', 'cpu')
        with ExitStack() as stack:
            if fp32:
                stack.enter_context(patch.object(model, '_BF16_DTYPE', torch.float32))
                stack.enter_context(patch.object(causal_model, '_BF16_DTYPE', torch.float32))
            net = tiny_model(device=device).eval()
            if fp32:
                net.float()
                net.param_dtype = torch.float32
            if device == 'cpu':
                from utils import flash_attn
                stack.enter_context(patch.object(flash_attn, 'FLASH_ATTN_2_AVAILABLE', False))
                stack.enter_context(patch.object(flash_attn, 'FLASH_ATTN_3_AVAILABLE', False))
                stack.enter_context(patch.object(causal_model, '_CAUSAL_FLEX_ATTENTION', rectangle_sdpa))
            yield net
    finally:
        torch.set_num_threads(threads)
