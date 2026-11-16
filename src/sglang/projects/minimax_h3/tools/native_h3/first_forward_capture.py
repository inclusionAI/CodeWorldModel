# SPDX-License-Identifier: Apache-2.0
"""Capture one original bootstrap D0 forward, with observations through block 0."""

import argparse
import contextlib
import inspect
import json
import os
import pickle
from importlib import metadata
from pathlib import Path

import torch
from numerical_trace import (
    STEP600_MERGED_SHA256,
    TraceWriter,
    _load,
    difference,
    digest,
    load_run,
)

REQUIRED_STAGES = {
    "packed/x",
    "packed/audio_x",
    "packed/img_position_ids",
    "packed/unique_timesteps",
    "packed/inverse_indices",
    "packed/token_tags",
    "packed/prompt_embeds",
    "packed/update_mask",
    "packed/img_pos_info",
    "packed/text_pos_info",
    "packed/audio_pos_info",
    "packed/img_pos_for_infer_output_info",
    "packed/cu_seqlens",
    "packed/max_seqlen",
    "packed/refiner_cu_seqlens",
    "packed/refiner_max_seqlen",
    "embed/condition_projection",
    "embed/refiner",
    "embed/video_input",
    "embed/video_projection",
    "embed/time_proj_in",
    "embed/time",
    "block0/input",
    "block0/adaln_input",
    "block0/combined_indices",
    "block0/rope_cos_sin",
    "block0/rope_positions",
    "block0/modulated",
    "block0/raw_q",
    "block0/raw_k",
    "block0/raw_v",
    "weights/rope.inv_freq",
    "weights/attn.qkv_proj.weight/q",
    "weights/attn.qkv_proj.weight/k",
    "weights/attn.qkv_proj.weight/v",
    *(
        f"block0/adaln/{name}"
        for name in (
            "shift_msa",
            "scale_msa",
            "gate_msa",
            "shift_mlp",
            "scale_mlp",
            "gate_mlp",
        )
    ),
}


RECIPE = Path(__file__).resolve().parents[2] / "configs/recipes/omni-h3-longlive.json"


def first(value):
    return value[0] if isinstance(value, tuple) else value


def runtime_record(kind, model):
    record = dict(
        kind=kind,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        autocast={
            device: dict(
                enabled=torch.is_autocast_enabled(device),
                dtype=str(torch.get_autocast_dtype(device)),
            )
            for device in ("cpu", "cuda")
        },
        environment={
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "FLASH_ATTN_2_AVAILABLE",
                "FLASH_ATTN_3_AVAILABLE",
                "FLASH_ATTENTION_DETERMINISTIC",
            )
        },
        parameter_dtypes={name: str(p.dtype) for name, p in model.named_parameters()},
        effective_flags=getattr(model, "_h3_inference", {}),
        source_sha256={
            path: digest(path)
            for cls in type(model).__mro__
            if cls.__module__ != "builtins"
            for path in [inspect.getfile(cls)]
            if Path(path).is_file()
        },
    )
    record["packages"] = {}
    for name in (
        "sglang",
        "sglang-kernel",
        "flash-attn",
        "flash-attn-3",
        "flash-attn-4",
    ):
        try:
            record["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            record["packages"][name] = None
    if kind == "native":
        from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            minimax_h3_norm_modulate_active,
        )

        attention = model.blocks[0].attn
        record.update(
            fa_ver=flash_attn.fa_ver,
            fa_module=flash_attn.__file__,
            fa_source_sha256=digest(flash_attn.__file__),
            norm_modulate_fused=minimax_h3_norm_modulate_active(model.blocks[0]),
            qknorm_rope_fused=attention._use_fused_qknorm_rope,
            quant_method=type(attention.qkv_proj.quant_method).__name__,
        )
    else:
        from utils import flash_attn

        record.update(
            placement="ResidentInferencePlacement",
            dit_options={
                name: bool(record["effective_flags"].get(name, False))
                for name in (
                    "metadata",
                    "kv",
                    "adaln",
                    "rope",
                    "delayed_hc",
                    "cuda_graph",
                )
            }
            | {"fuse_first_denoise": False},
            fa2_available=flash_attn.FLASH_ATTN_2_AVAILABLE,
            fa3_available=flash_attn.FLASH_ATTN_3_AVAILABLE,
            fa_module=flash_attn.__file__,
            fa_source_sha256=digest(flash_attn.__file__),
            norm_modulate_fused=False,
            qknorm_rope_fused=model.blocks[0].attn._use_fused_qknorm_rope,
            selected_fa="fa3"
            if flash_attn.FLASH_ATTN_3_AVAILABLE
            else ("fa2" if flash_attn.FLASH_ATTN_2_AVAILABLE else "sdpa"),
        )
    return record


class FirstForwardCapture(contextlib.AbstractContextManager):
    def __init__(
        self, model, writer, *, kind, tp_rank=0, tp_size=1, up_rank=0, up_size=1
    ):
        self.model, self.writer, self.kind = model, writer, kind
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.up_rank, self.up_size = up_rank, up_size
        self.handles = []
        self.events = []
        self.runtime = None

    def save(self, name, value, axes=()):
        if name in self.events:
            raise ValueError(f"first-forward stage called more than once: {name}")
        axes = [
            dict(axis=axis, start=start, total=total) for axis, start, total in axes
        ]
        self.writer.save(name, dict(value=value, axes=axes))
        self.events.append(name)

    def local_rows(self, value):
        if value.shape[0] != self.local_rows_count:
            raise ValueError("activation does not cover the expected local packed rows")
        return ((0, self.row_start, self.global_rows),)

    def packed(self, module, args, kwargs):
        if self.runtime is not None:
            raise ValueError("probe permits exactly one DiT forward")
        self.runtime = runtime_record(self.kind, self.model)
        self.global_rows = kwargs["x"].shape[1]
        if self.global_rows % self.up_size:
            raise ValueError("packed rows are not divisible by the UP group")
        self.local_rows_count = self.global_rows // self.up_size
        self.row_start = self.up_rank * self.local_rows_count
        self.img_ids = kwargs["img_pos_info"]["position_ids"].reshape(-1)
        for name in (
            "x",
            "audio_x",
            "img_position_ids",
            "unique_timesteps",
            "inverse_indices",
            "token_tags",
            "prompt_embeds",
            "update_mask",
        ):
            self.save(f"packed/{name}", kwargs[name])
        for name in (
            "img_pos_info",
            "text_pos_info",
            "audio_pos_info",
            "img_pos_for_infer_output_info",
        ):
            self.save(f"packed/{name}", kwargs[name]["position_ids"])
        if "packed_seq_params" in kwargs:
            params = kwargs["packed_seq_params"]
            cu, maximum = params["cu_seqlens_q"], params["max_seqlen_q"]
        else:
            lengths = kwargs["sample_lens"]
            cu = (
                torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).to(torch.int32)
            )
            maximum = max(lengths)
        self.save("packed/cu_seqlens", cu)
        self.save("packed/max_seqlen", torch.tensor(maximum))
        refiner = kwargs["refiner_packed_seq_params"]
        self.save("packed/refiner_cu_seqlens", refiner["cu_seqlens_q"])
        self.save("packed/refiner_max_seqlen", torch.tensor(refiner["max_seqlen_q"]))

    def output(self, name, *, axes=None):
        def capture(module, args, result):
            value = first(result)
            self.save(name, value, () if axes is None else axes(value))

        return capture

    def block_input(self, module, args, kwargs):
        value = args[0]
        self.save("block0/input", value, self.local_rows(value))
        self.save("block0/adaln_input", kwargs["adaln_input"])
        if kwargs.get("adaln_params") is not None:
            self.adaln(module, args, kwargs["adaln_params"])
        indices = kwargs["combined_indices"]
        self.save("block0/combined_indices", indices, self.local_rows(indices))
        for name, value in zip(("cos_sin", "positions"), kwargs["rope_cache"]):
            # Positions index the rank-local RoPE cache, not the global sequence.
            if name == "positions":
                value = value + self.row_start
            self.save(f"block0/rope_{name}", value, self.local_rows(value))

    def modulated(self, module, args):
        self.save("block0/modulated", args[0], self.local_rows(args[0]))

    def adaln(self, module, args, result):
        if len(result) != 6:
            raise ValueError("block0 AdaLN requires exactly six parameter tables")
        for name, value in zip(
            (
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            ),
            result,
        ):
            self.save(f"block0/adaln/{name}", value)

    def raw_qkv(self, module, args, result):
        value = first(result)
        attention = self.model.blocks[0].attn
        heads, dim = attention.num_heads, attention.head_dim
        if value.shape != (self.local_rows_count, 3 * heads * dim):
            raise ValueError("unexpected raw QKV layout")
        for name, part in zip(
            "qkv", value.reshape(value.shape[0], 3, heads, dim).unbind(1)
        ):
            self.save(
                f"block0/raw_{name}",
                part,
                (
                    *self.local_rows(part),
                    (1, self.tp_rank * heads, heads * self.tp_size),
                ),
            )

    def video_projection(self, module, args, result):
        value = first(result)
        if self.kind == "reference":
            start, count = 0, self.img_ids.numel()
        else:
            start = int((self.img_ids < self.row_start).sum())
            count = int(
                (
                    (self.img_ids >= self.row_start)
                    & (self.img_ids < self.row_start + self.local_rows_count)
                ).sum()
            )
        if value.shape[0] != count:
            raise ValueError("video projection rows do not match packed image indexes")
        axes = ((0, start, self.img_ids.numel()),)
        self.save("embed/video_input", args[0], axes)
        self.save("embed/video_projection", value, axes)

    def weights(self):
        if self.up_rank:
            return
        heads = self.model.blocks[0].attn.num_heads
        dim = self.model.blocks[0].attn.head_dim
        for name, value in self.model.blocks[0].named_parameters():
            if name.startswith("attn.qkv_proj."):
                parts = value.reshape(3, heads, dim, *value.shape[1:]).unbind(0)
                for label, part in zip("qkv", parts):
                    self.save(
                        f"weights/{name}/{label}",
                        part,
                        ((0, self.tp_rank * heads, heads * self.tp_size),),
                    )
                continue
            if name == "mlp.fc1.weight":
                gate, up = value.chunk(2, dim=0)
                for label, part in (("gate", gate), ("up", up)):
                    self.save(
                        f"weights/{name}/{label}",
                        part,
                        ((0, self.tp_rank * len(part), len(part) * self.tp_size),),
                    )
                continue
            if name in ("attn.out_proj.weight", "mlp.fc2.weight"):
                axis = 1
            elif name.startswith("adaln_proj.linear."):
                axis = 0
            else:
                axis = None
            axes = (
                ()
                if axis is None
                else (
                    (
                        axis,
                        self.tp_rank * value.shape[axis],
                        value.shape[axis] * self.tp_size,
                    ),
                )
            )
            self.save(f"weights/{name}", value, axes)
        self.save("weights/rope.inv_freq", self.model.rope.inv_freq)

    def __enter__(self):
        try:
            self.weights()
            m, block = self.model, self.model.blocks[0]
            self.handles.append(
                m.register_forward_pre_hook(self.packed, with_kwargs=True)
            )
            self.handles.append(
                block.register_forward_pre_hook(self.block_input, with_kwargs=True)
            )
            for module, hook in (
                (m.condition_proj, self.output("embed/condition_projection")),
                (m.token_refiner, self.output("embed/refiner")),
                (m.time_embedder, self.output("embed/time")),
                (
                    m.time_embedder.proj_in,
                    self.output(
                        "embed/time_proj_in",
                        axes=lambda value: (
                            (
                                1,
                                self.tp_rank * value.shape[1],
                                value.shape[1] * self.tp_size,
                            ),
                        ),
                    ),
                ),
                (m.video_patch_proj, self.video_projection),
                (block.adaln_proj, self.adaln),
                (block.norm1, self.output("block0/norm1", axes=self.local_rows)),
                (block.attn.qkv_proj, self.raw_qkv),
            ):
                self.handles.append(module.register_forward_hook(hook))
            self.handles.append(block.attn.register_forward_pre_hook(self.modulated))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def receipt(self):
        required = {name for name in REQUIRED_STAGES if not name.startswith("weights/")}
        missing = sorted(required - set(self.events))
        if self.runtime is None or missing:
            raise ValueError(
                f"first forward did not reach the required block-0 boundaries; missing={missing}"
            )
        self.runtime.update(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            up_rank=self.up_rank,
            up_size=self.up_size,
            global_rows=self.global_rows,
            local_row_start=self.row_start,
            local_rows=self.local_rows_count,
            norm1_observed="block0/norm1" in self.events,
            scope="one original full D0; observations through block0 raw QKV; no sampling or KV commit",
        )
        if self.kind == "native":
            from sglang.multimodal_gen.runtime.layers.attention.backends import (
                flash_attn,
            )

            self.runtime.update(
                fa_ver=flash_attn.fa_ver,
                resolved_backend=self.model._resolved_attention_backend.name,
                attention_impl=type(self.model.blocks[0].attn._attention_impl).__name__,
            )
        if (
            not self.runtime["norm1_observed"]
            and not self.runtime["norm_modulate_fused"]
        ):
            raise ValueError("unfused norm1 was not observed")
        return self.runtime


def corpus(root):
    root = Path(root) / "rank0000"
    receipt = json.loads((root / "complete.json").read_text())
    identity = receipt["identity"]
    frozen = json.loads(RECIPE.read_text())
    if (
        receipt["runtime"]["kind"] != "independent-reference"
        or receipt["runtime"].get("effective_flags")
        or receipt["runtime"].get("delayed_hc")
        or receipt["runtime"].get("fuse_first_denoise")
        or identity["weights_sha256"] != STEP600_MERGED_SHA256
        or identity["recipe"] != frozen
        or (identity["heads"], identity["head_dim"], identity["layers"])
        != (56, 128, 50)
    ):
        raise ValueError("probe requires the independent frozen step600 corpus")

    def read(name):
        path = root / f"{name}.pt"
        if digest(path) != receipt["files"][name]["sha256"]:
            raise ValueError(f"reference payload changed: {name}")
        return _load(path)

    return identity, read


def reference_model(artifact_path, world):
    from common.config import CfgNode
    from common.distributed.init import configure_distributed
    from common.distributed.unified_parallel import init_unified_parallel
    from common.model import build_model
    from projects.minimax_h3.inference.artifact import load_bundle, runtime_backbone

    bundle, recipe = load_bundle(artifact_path)
    if bundle["weights"]["sha256"] != STEP600_MERGED_SHA256:
        raise ValueError("reference artifact is not the frozen merged step600")
    if bundle["deployment"]["active_up_size"] != world:
        raise ValueError("reference world must match artifact active_up_size")
    config = CfgNode(dict(distributed=bundle["deployment"]["distributed"], seed=2026))
    configure_distributed(config)
    init_unified_parallel(config.distributed.up_size, world)
    model_config = runtime_backbone(bundle)
    model_config.update(
        runtime=dict(training=False, requires_grad=False, param_dtype="bfloat16"),
        placement=dict(
            module="projects.minimax_h3.modeling.inference_runtime",
            class_name="ResidentInferencePlacement",
        ),
    )
    model = build_model("backbone", CfgNode(model_config))["model"]
    if any(
        any(getattr(module, "_h3_inference", {}).values()) for module in model.modules()
    ):
        raise ValueError(
            "reference probe requires the original all-off inference flags"
        )
    return model, recipe


def native_model(bundle_path, identity, rank, world):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3_causal import (
        MiniMaxH3CausalDiTConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.managers.minimax_h3_causal_host import (
        _initialize_native,
        _validate_causal_parallelism,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    manifest = json.loads((Path(bundle_path) / "native-runtime.json").read_text())
    settings = manifest["server_args"]
    if (
        manifest["schema"] != "h3_omni_native/v1"
        or manifest["recipe"] != identity["recipe"]
        or identity["artifact_sha256"] not in manifest["assets"].values()
        or manifest["assets"][settings["transformer_weights_path"]]
        != STEP600_MERGED_SHA256
        or world != settings["num_gpus"]
    ):
        raise ValueError("native bundle differs from the frozen reference corpus")
    args = ServerArgs(pipeline_config=MiniMaxH3PipelineConfig(), **settings)
    _initialize_native(pickle.dumps(args), rank, world, int(os.environ["MASTER_PORT"]))
    if rank == 0:
        for path, expected in manifest["assets"].items():
            if digest(path) != expected:
                raise ValueError(f"native asset changed: {path}")
    torch.distributed.barrier()
    _validate_causal_parallelism(args)
    args.pipeline_config.dit_config = MiniMaxH3CausalDiTConfig()
    args.pipeline_config.transformer_class_override = "MiniMaxH3CausalDiTModel"
    args.model_variant = args.model_variant or "ref2va"
    model, _ = TransformerLoader(device=get_local_torch_device()).load(
        args.component_paths["transformer"],
        args,
        "transformer",
        "diffusers",
        allow_native_fallback=False,
    )
    return model.eval().requires_grad_(False)


def first_frames(read, count, device, kind):
    if kind == "native":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
            _Frame,
        )
    else:
        from projects.minimax_h3.inference.raw_latent import SourceLatent as _Frame
    frames, index = [], 0
    while len(frames) < count:
        source = read(f"inputs/source{index:04d}")
        stop = len(frames) + source["target_noise"].shape[1]
        for i in range(min(source["target_noise"].shape[1], count - len(frames))):
            values = dict(
                position=len(frames),
                references=tuple(
                    v[:, i : i + 1].to(device) for v in source["references"]
                ),
                noise=source["target_noise"][:, i : i + 1].to(device),
            )
            ref_noise = tuple(
                v[:, i : i + 1].to(device) for v in source["reference_noise"]
            )
            clean = source["clean_noise"][:, i : i + 1].to(device)
            if kind == "native":
                values.update(reference_noise=ref_noise, clean_noise=clean)
            else:
                values.update(
                    reference_eps=ref_noise,
                    eps=clean,
                    source_clip=index,
                    source_stop=stop,
                    document=None,
                    prompt=None,
                )
            frames.append(_Frame(**values, video=None))
        index += 1
    return frames


@torch.no_grad()
def capture(options):
    from dataclasses import replace

    identity, read = corpus(options.reference)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    try:
        if options.kind == "reference":
            from common.distributed.unified_parallel import get_unified_parallel_rank

            if digest(options.artifact) != identity["artifact_sha256"]:
                raise ValueError("reference artifact differs from the original capture")
            model, recipe = reference_model(options.artifact, world)
            if recipe.to_dict() != identity["recipe"]:
                raise ValueError("reference recipe differs from the frozen corpus")
            dit = model.dit
            tp_rank, tp_size, up_rank, up_size = (
                0,
                1,
                get_unified_parallel_rank(),
                world,
            )
        else:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_tp_rank,
                get_tp_world_size,
                get_ulysses_ctx,
            )

            model = dit = native_model(options.bundle, identity, rank, world)
            tp_rank, tp_size = get_tp_rank(), get_tp_world_size()
            up_size, up_rank = get_ulysses_ctx()
        if (
            len(dit.blocks) != 50
            or dit.arch.num_attention_heads != 56
            or dit.arch.attention_head_dim != 128
        ):
            raise ValueError("probe requires the full step600 DiT architecture")
        device = next(model.parameters()).device
        static, d0 = read("inputs/static"), read("window0000/d0")
        frames = first_frames(read, d0["xt"].shape[1], device, options.kind)
        frames = [
            replace(f, video=d0["xt"][:, i : i + 1].to(device))
            for i, f in enumerate(frames)
        ]
        sigma = d0["sigma"].to(device)
        writer = TraceWriter(options.output, rank, world)
        with FirstForwardCapture(
            dit,
            writer,
            kind=options.kind,
            tp_rank=tp_rank,
            tp_size=tp_size,
            up_rank=up_rank,
            up_size=up_size,
        ) as probe:
            if options.kind == "reference":
                from projects.minimax_h3.inference.raw_latent import (
                    LatentKVCache,
                    RawLatentModel,
                )

                raw = RawLatentModel()
                raw.recipe = recipe
                inputs = raw.inputs(
                    {
                        "prompt_embeds": static["prompt_embeds"].to(device),
                        "text_token_tags": static["token_tags"].to(device),
                    },
                    static["first_frame"].to(device),
                    static["first_frame_noise"].to(device),
                    frames,
                    current_start=0,
                    include_static=True,
                )
                raw.forward(
                    model, inputs, frames, sigma, LatentKVCache(50, padding=world > 1)
                )
            else:
                from sglang.multimodal_gen.runtime.layers.kvcache.minimax_h3_causal import (
                    H3CausalKVCache,
                )
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
                    H3FixedHistorySession,
                    H3StaticCondition,
                )

                session = H3FixedHistorySession(
                    model, H3StaticCondition(**static), recipe=identity["recipe"]
                )
                session._forward(
                    frames,
                    H3CausalKVCache(50),
                    static=True,
                    conditions=True,
                    denoise=True,
                    sigma=sigma,
                    step_index=0,
                )
        torch.cuda.synchronize()
        writer.finish(
            dict(corpus=identity, probe="bootstrap-d0-block0/v1"), probe.receipt()
        )
        torch.distributed.barrier()
    finally:
        if options.kind == "native":
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        elif torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def bounds(record):
    value = record["value"]
    slices = [slice(0, n) for n in value.shape]
    shape = list(value.shape)
    seen = set()
    for item in record["axes"]:
        axis, start, total = item["axis"], item["start"], item["total"]
        if (
            axis in seen
            or not 0 <= axis < value.ndim
            or not 0 <= start <= total - value.shape[axis]
        ):
            raise ValueError("invalid or duplicate shard axis")
        seen.add(axis)
        shape[axis] = total
        slices[axis] = slice(start, start + value.shape[axis])
    return tuple(shape), tuple(slices)


def assemble(records):
    shape, _ = bounds(records[0])
    output = torch.empty(shape, dtype=records[0]["value"].dtype)
    # Coverage is checked on shard-boundary cells, not one byte per tensor value.
    edges = [set((0, n)) for n in shape]
    for record in records:
        actual, slices = bounds(record)
        if actual != shape or record["value"].dtype != output.dtype:
            raise ValueError("shard global shape/dtype mismatch")
        for axis, s in enumerate(slices):
            edges[axis].update((s.start, s.stop))
    edges = [sorted(e) for e in edges]
    if output.numel() == 0:
        return output
    coverage = torch.zeros([max(1, len(e) - 1) for e in edges], dtype=torch.bool)
    for record in records:
        _, slices = bounds(record)
        cells = tuple(
            slice(edges[a].index(s.start), edges[a].index(s.stop))
            for a, s in enumerate(slices)
        )
        if coverage[cells].any():
            # Replica consistency follows the same byte-equality gate as comparison.
            if not coverage[cells].all() or not torch.equal(
                output[slices].contiguous().reshape(-1).view(torch.uint8),
                record["value"].contiguous().reshape(-1).view(torch.uint8),
            ):
                raise ValueError("overlapping shards or disagreeing replicas")
        output[slices] = record["value"]
        coverage[cells] = True
    if not bool(coverage.all()):
        raise ValueError("incomplete global tensor coverage")
    return output


def compare(left_root, right_root, report_path):
    left_ranks, left = load_run(left_root)
    right_ranks, right = load_run(right_root)
    if (
        left_ranks[0]["identity"] != right_ranks[0]["identity"]
        or left_ranks[0]["identity"].get("probe") != "bootstrap-d0-block0/v1"
    ):
        raise ValueError("first-forward identities differ")
    for ranks, events, kind in (
        (left_ranks, left, "reference"),
        (right_ranks, right, "native"),
    ):
        if not REQUIRED_STAGES <= set(events):
            raise ValueError("incomplete first-forward capture")
        for rank in ranks:
            required = (
                REQUIRED_STAGES
                if rank["runtime"]["up_rank"] == 0
                else {
                    name for name in REQUIRED_STAGES if not name.startswith("weights/")
                }
            )
            if rank["runtime"]["kind"] != kind or not required <= set(rank["files"]):
                raise ValueError("incomplete first-forward rank")
    required = set(left) | set(right)
    optional = {"block0/norm1"}
    if (set(left) ^ set(right)) - optional:
        raise ValueError("first-forward stages differ")
    results = {}
    for name in sorted(required):
        if name not in left or name not in right:
            results[name] = dict(
                exact=False, unavailable="fused path has no standalone norm1 output"
            )
            continue
        reference = assemble([_load(path) for path, _ in left[name]])
        records = [_load(path) for path, _ in right[name]]
        native = assemble(records)
        results[name] = difference(reference, native)
    exact = all(result["exact"] for result in results.values())
    report = dict(
        exact=exact,
        results=results,
        reference_runtime=[rank["runtime"] for rank in left_ranks],
        native_runtime=[rank["runtime"] for rank in right_ranks],
    )
    with Path(report_path).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(dict(exact=exact, stages=len(results), report=str(report_path))))
    return 0 if exact else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("--kind", choices=("reference", "native"), required=True)
    run.add_argument(
        "--reference", required=True, help="existing completed independent corpus"
    )
    run.add_argument("--output", required=True)
    run.add_argument("--artifact")
    run.add_argument("--bundle")
    diff = commands.add_parser("compare")
    diff.add_argument("reference")
    diff.add_argument("native")
    diff.add_argument("--report", required=True)
    options = parser.parse_args()
    if options.command == "compare":
        return compare(options.reference, options.native, options.report)
    if (
        options.kind == "reference"
        and not options.artifact
        or options.kind == "native"
        and not options.bundle
    ):
        parser.error(
            "reference capture needs --artifact; native capture needs --bundle"
        )
    capture(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
