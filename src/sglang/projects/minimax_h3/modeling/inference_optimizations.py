"""Opt-in, exact caches for H3 inference; no state-dict or training changes."""
from collections import OrderedDict
import os
from weakref import ref

import torch


def graph_runtime_identity():
    """Record the initialized worker runtime without inferring NCCL defaults."""
    from projects.minimax_h3.inference.settings import GRAPH_RUNTIME_ENVIRONMENT
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                nccl=list(torch.cuda.nccl.version()),
                environment={name: os.environ.get(name) for name in GRAPH_RUNTIME_ENVIRONMENT})


def configure_inference(model, *, metadata=True, kv=True, adaln=True, delayed_hc=False, rope=False, cuda_graph=False):
    """Configure one loaded model. Caches run only in eval AND no-grad mode.

    Call again after out-of-band .data writes or changing adapter behavior.
    Ordinary in-place updates/load_state_dict and tensor replacements invalidate
    caches automatically. Does not change train/eval, dtype, weights or masks.
    CUDA Graph callers must launch with NCCL_GRAPH_REGISTER=0 before CUDA/NCCL
    initialization; checking the current environment cannot prove startup history.
    """
    if type(delayed_hc) is not bool:
        raise ValueError('delayed_hc must be a boolean')
    if type(rope) is not bool:
        raise ValueError('rope must be a boolean')
    if type(cuda_graph) is not bool:
        raise ValueError('cuda_graph must be a boolean')
    if cuda_graph:
        from projects.minimax_h3.inference.settings import require_dit_graph_environment
        require_dit_graph_environment(True)
    flags = dict(metadata=bool(metadata), kv=bool(kv), adaln=bool(adaln), delayed_hc=delayed_hc,
                 rope=rope, cuda_graph=cuda_graph)
    for module in model.modules():
        module._h3_inference = flags
        module._h3_inference_generation = object()
        module.__dict__.pop("_h3_adaln_cache", None)
        module.__dict__.pop("_h3_adaln_weights", None)
        module.__dict__.pop("_h3_adaln_parameters", None)
    return model


def inference_enabled(module, option):
    return (not getattr(module, "training", True) and not torch.is_grad_enabled()
            and getattr(module, "_h3_inference", {}).get(option, False))


def _tensor_stamp(tensor):
    # Inference tensors have no version counter: never reuse them by identity.
    try:
        version = tensor._version
    except RuntimeError:
        return object()
    return (id(tensor), version, tensor.device, tensor.dtype,
            tuple(tensor.shape), tuple(tensor.stride()))


def _weight_stamp(module):
    return tuple(_tensor_stamp(t) for t in (*module.parameters(), *module.buffers()))


def prepare_adaln_input(model, value):
    if inference_enabled(model, "adaln") and not value.is_inference():
        # One small D2H per forward shared by all blocks. Key the ACTUAL input,
        # so changed time embedding/LoRA cannot hit an old entry.
        value._h3_adaln_source = _tensor_stamp(value)
        value._h3_adaln_key = (value.device, value.dtype, tuple(value.shape),
            value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(),
            torch.is_autocast_enabled(value.device.type),
            torch.get_autocast_dtype(value.device.type))
    return value


def cached_adaln(module, value, compute):
    key = getattr(value, "_h3_adaln_key", None)
    if key is None or not inference_enabled(module, "adaln"):
        return compute(value)
    if value.is_inference():
        return compute(value)
    if getattr(value, "_h3_adaln_source", None) != _tensor_stamp(value):
        prepare_adaln_input(module, value)
        key = value._h3_adaln_key
    parameters = tuple(module.parameters()) + tuple(module.buffers())
    stamp = tuple(_tensor_stamp(t) for t in parameters)
    previous = getattr(module, "_h3_adaln_parameters", ())
    if (stamp != getattr(module, "_h3_adaln_weights", None)
            or len(previous) != len(parameters)
            or any(old() is not new for old, new in zip(previous, parameters))):
        module._h3_adaln_weights = stamp
        # Do not keep old FSDP unsharded parameter storage alive.
        module._h3_adaln_parameters = tuple(ref(p) for p in parameters)
        module._h3_adaln_cache = OrderedDict()
    cache = module._h3_adaln_cache
    if key not in cache:
        cache[key] = compute(value)
        if len(cache) > 32:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return cache[key]


def inference_geometry(model, geometry):
    """Reuse caller-owned geometry, invalidating on input/rope mutations."""
    if not inference_enabled(model, "metadata"):
        return {}
    dit = getattr(model, "dit", model)
    if not hasattr(dit, "build_rope_cache"):
        return {}
    g = geometry
    keys = ("position_ids", "img_pos", "text_pos", "output_pos", "token_tags")
    stamp = (id(dit), dit._h3_inference_generation, dit._sequence_parallel_ctx(), _weight_stamp(dit.rope),
             tuple(_tensor_stamp(g[k]) for k in keys), tuple(g["refiner_lengths"]),
             tuple(g["sample_lens"]))
    previous = g.get("_h3_inference_geometry")
    if previous is not None and previous[0] == stamp:
        return previous[1]
    device = g["position_ids"].device
    empty = torch.empty(0, device=device, dtype=torch.long)
    position_ids = g["position_ids"].unsqueeze(0)
    cu = torch.tensor([0] + g["refiner_lengths"], device=device,
                      dtype=torch.int32).cumsum(0, dtype=torch.int32)
    result = dict(
        img_position_ids=position_ids,
        rope_cache=dit.build_rope_cache(position_ids, device=device),
        sample_lens=torch.tensor(g["sample_lens"], device=device, dtype=torch.int32),
        update_mask=torch.ones(len(g["output_pos"]), device=device, dtype=torch.bool),
        refiner_packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": max(g["refiner_lengths"])},
        audio_pos_info={"position_ids": empty},
    )
    # Trusted-layout ABI accepts one leading text document; preserve other paths.
    if dit._sequence_parallel_ctx()[0] == 1 and len(g["refiner_lengths"]) == 1:
        text, img = g["text_pos"], g["img_pos"]
        live = len(text) + len(img)
        if (torch.equal(text, torch.arange(len(text), device=device))
                and torch.equal(img, torch.arange(len(text), live, device=device))):
            result["local_embedding_layout"] = dict(
                text_source_start=0, text_source_stop=len(text),
                img_global_ids=img, img_row_ids=img,
                audio_global_ids=empty, audio_row_ids=empty)
    # Keep identity-keyed inputs alive, preventing Python id reuse on replacement.
    g["_h3_inference_geometry"] = (stamp, result, tuple(g[k] for k in keys), tuple(dit.rope.buffers()))
    return result


def merge_inference_kv(cache, layer, keys, values):
    """Persistent contiguous arena for one real stream (+ optional SP pad).

    Only the new tail is written during denoising. Clean commit and eviction
    remain owned by GroupedKVCache; after eviction the retained prefix is copied
    once. Multi-sample caches retain the original scatter implementation.
    """
    arenas = cache.__dict__.setdefault("_h3_kv_arenas", {})
    past_k, past_v = cache.key_cache[layer], cache.value_cache[layer]
    past_len = 0 if past_k is None else len(past_k)
    length = past_len + len(keys)
    arena = arenas.get(layer)
    if (arena is None or len(arena[0]) < length or arena[0].device != keys.device
            or arena[0].dtype != keys.dtype or arena[0].shape[1:] != keys.shape[1:]):
        # Bound resident memory to the largest context actually requested.
        capacity = length
        arena = (keys.new_empty((capacity, *keys.shape[1:])),
                 values.new_empty((capacity, *values.shape[1:])))
        arenas[layer] = arena
    for table, buffer, past, new in zip((cache.key_cache, cache.value_cache), arena, (past_k, past_v), (keys, values)):
        if past_len and past.data_ptr() != buffer.data_ptr():
            buffer[:past_len].copy_(past)
            table[layer] = buffer[:past_len]
        buffer[past_len:length].copy_(new)
    return arena[0][:length], arena[1][:length]
