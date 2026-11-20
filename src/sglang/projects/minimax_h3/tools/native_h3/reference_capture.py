# SPDX-License-Identifier: Apache-2.0
"""Instrument the existing reference file CLI, in its own Python environment."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numerical_trace import (
    STEP600_MERGED_SHA256,
    CommitLedger,
    TraceWriter,
    committed_signature,
    digest,
    plan_record,
)


RECIPE = Path(__file__).resolve().parents[2] / "configs/recipes/omni-h3-longlive.json"


def reference_options(argv):
    from projects.minimax_h3.inference.settings import (
        inspection_parser,
        parse_run_arguments,
    )

    return parse_run_arguments(
        inspection_parser("infer_ref_raw_latent"),
        argv,
        entrypoint="infer_ref_raw_latent",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    if "--" not in sys.argv:
        parser.parse_args()
        parser.error("separate the existing reference CLI arguments with --")
    separator = sys.argv.index("--")
    options = parser.parse_args(sys.argv[1:separator])
    reference_args = sys.argv[separator + 1 :]
    artifact_args = reference_options(reference_args)
    from common.distributed.unified_parallel import (
        get_unified_parallel_rank,
        get_unified_parallel_world_size,
    )
    from projects.minimax_h3.inference import raw_latent
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
    from projects.minimax_h3.tools.infer_ref_raw_latent import main as reference_main

    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    writer = TraceWriter(options.output, rank, world)
    frozen = json.loads(RECIPE.read_text())
    artifact = json.loads(Path(artifact_args.artifact).read_text())
    if artifact["weights"]["sha256"] != STEP600_MERGED_SHA256:
        raise ValueError("oracle requires the frozen step600 EMA merged weights")
    identity = dict(
        recipe=frozen,
        artifact_sha256=digest(artifact_args.artifact),
        weights_sha256=STEP600_MERGED_SHA256,
        heads=56,
        head_dim=128,
        layers=50,
    )
    encoder_index = 0
    session_seen = False
    completed = False

    condition, image, chunk = (
        FrozenCWMEncoders.condition,
        FrozenCWMEncoders.image,
        FrozenCWMEncoders.encode_chunk,
    )

    def capture_condition(self, prompt, videos=(), images=()):
        if self.posterior != "mean" or len(videos) or len(images) != 1:
            raise ValueError("capture requires one static image and posterior mean")
        result = condition(self, prompt, videos=videos, images=images)
        writer.save(
            "inputs/condition",
            dict(prompt=prompt, rgb=torch.from_numpy(np.array(images[0]))),
        )
        writer.save(
            "encoder/condition",
            dict(
                prompt_embeds=result["prompt_embeds"],
                token_tags=result["text_token_tags"],
            ),
        )
        return result

    def capture_image(self, frame, **kwargs):
        result = image(self, frame, **kwargs)
        writer.save("inputs/image", torch.from_numpy(np.array(frame)))
        writer.save("encoder/image", result)
        return result

    def capture_chunk(self, frames, **kwargs):
        nonlocal encoder_index
        result = chunk(self, frames, **kwargs)
        writer.save(
            f"inputs/proxy{encoder_index:04d}",
            dict(rgb=torch.from_numpy(np.array(frames)), eos=kwargs.get("eos", False)),
        )
        writer.save(f"encoder/proxy{encoder_index:04d}", result)
        encoder_index += 1
        return result

    (
        FrozenCWMEncoders.condition,
        FrozenCWMEncoders.image,
        FrozenCWMEncoders.encode_chunk,
    ) = capture_condition, capture_image, capture_chunk
    original_init, original_push, original_generate = (
        raw_latent.RawLatentSession.__init__,
        raw_latent.RawLatentSession.push,
        raw_latent.RawLatentSession._generate,
    )

    def export_sources(session):
        for source in sorted(
            {frame.source_clip for frame in session.frames.values()} - session._exported
        ):
            frames = sorted(
                (f for f in session.frames.values() if f.source_clip == source),
                key=lambda f: f.position,
            )
            count = frames[-1].source_stop - source * 5
            if len(frames) != count or count not in (2, 5):
                raise ValueError("partial source export")
            if rank == 0:
                writer.save(
                    f"inputs/source{source:04d}",
                    dict(
                        references=[
                            torch.cat([f.references[j] for f in frames], 1)
                            for j in range(len(frames[0].references))
                        ],
                        reference_noise=[
                            torch.cat([f.reference_eps[j] for f in frames], 1)
                            for j in range(len(frames[0].reference_eps))
                        ],
                        target_noise=torch.cat([f.noise for f in frames], 1),
                        clean_noise=torch.cat([f.eps for f in frames], 1),
                        eos=count == 2,
                    ),
                )
            session._exported.add(source)

    def initialize(session, *args, **kwargs):
        nonlocal session_seen
        if session_seen:
            raise ValueError("one session per capture")
        session_seen = True
        original_init(session, *args, **kwargs)
        flags = getattr(session.backbone, "_h3_inference", {})
        if (
            session.recipe.to_dict() != frozen
            or session.delayed_hc
            or session.fuse_first_denoise
            or flags.get("cuda_graph")
            or flags.get("spectrum")
        ):
            raise ValueError(
                "oracle requires frozen independent C/4D/cleanH without graph/Spectrum"
            )
        if (
            len(session.backbone.dit.blocks) != 50
            or get_unified_parallel_world_size() != world
        ):
            raise ValueError("oracle requires the full 50-layer TP1 single UP group")
        session._exported = set()
        session._commit_ledger = CommitLedger()
        if rank == 0:
            writer.save(
                "inputs/static",
                dict(
                    prompt_embeds=session.static["prompt_embeds"],
                    token_tags=session.static["text_token_tags"],
                    first_frame=session.picture,
                    first_frame_noise=session.picture_eps,
                ),
            )

    def generate(session, plan):
        export_sources(session)
        window = f"window{plan.index:04d}"
        original_forward = session.model.forward
        steps = writes = 0

        def forward(
            backbone,
            inputs,
            frames,
            timestep,
            cache,
            *,
            store=False,
            cache_segments=None,
        ):
            nonlocal steps, writes
            if cache_segments is not None:
                raise ValueError("combined oracle commits are forbidden")

            def layers():
                return {
                    layer: (key, cache.value_cache[layer])
                    for layer, key in cache.key_cache.items()
                    if key is not None
                }

            before = None if store else committed_signature(cache.owners, layers())
            row_start = len(cache.owners)
            result = original_forward(
                backbone, inputs, frames, timestep, cache, store=store
            )
            if store:
                owners = [
                    position for position, source in cache.owners if source is None
                ]
                if len(owners) != cache.kvlens[0] or len(owners) != len(cache.owners):
                    raise ValueError("unexpected document or cache padding owners")
                session._commit_ledger.commit(owners, layers(), row_start)
                for layer in range(50):
                    k, v = (
                        cache.key_cache[layer][: len(owners)],
                        cache.value_cache[layer][: len(owners)],
                    )
                    writer.save(
                        f"{window}/write{writes}/layer{layer:02d}",
                        dict(
                            owners=owners,
                            row_start=row_start,
                            key=k[row_start:],
                            value=v[row_start:],
                        ),
                        head_start=get_unified_parallel_rank() * k.shape[1],
                    )
                writes += 1
            else:
                if before != committed_signature(cache.owners, layers()):
                    raise ValueError("denoise changed committed KV contents or owners")
                if rank == 0:
                    writer.save(
                        f"{window}/d{steps}",
                        dict(
                            xt=torch.cat(
                                [
                                    f.video if f.video is not None else f.noise
                                    for f in frames
                                ],
                                1,
                            ),
                            sigma=timestep,
                            x0=result[0][0],
                        ),
                    )
                steps += 1
            return result

        session.model.forward = forward
        try:
            result = original_generate(session, plan)
        finally:
            session.model.forward = original_forward
        if steps != 4 or (plan.bootstrap and writes):
            raise ValueError("oracle did not execute independent four-step sampling")
        if rank == 0:
            writer.save(
                f"{window}/output", dict(plan=plan_record(plan), video=result.video)
            )
        return result

    def push(session, item):
        nonlocal completed
        result = original_push(session, item)
        export_sources(session)
        if item.get("eos"):
            if len(session._exported) < 4 or (
                rank == 0 and encoder_index != len(session._exported)
            ):
                raise ValueError("capture requires three normal packets plus EOS")
            inputs = [
                {
                    name: info["sha256"]
                    for name, info in writer.files.items()
                    if name.startswith("inputs/")
                }
                if rank == 0
                else None
            ]
            if world > 1:
                torch.distributed.broadcast_object_list(inputs, src=0)
            identity["inputs"] = inputs[0]
            writer.finish(
                identity,
                dict(
                    kind="independent-reference",
                    tp_size=1,
                    up_rank=get_unified_parallel_rank(),
                    up_size=get_unified_parallel_world_size(),
                    delayed_hc=session.delayed_hc,
                    fuse_first_denoise=session.fuse_first_denoise,
                    source_sha256=digest(raw_latent.__file__),
                    torch=torch.__version__,
                    effective_flags=getattr(session.backbone, "_h3_inference", {}),
                    kv_readonly_guard="logical-row-sha256",
                    kv_history_guard="owner-block-sha256",
                ),
            )
            completed = True
        return result

    (
        raw_latent.RawLatentSession.__init__,
        raw_latent.RawLatentSession.push,
        raw_latent.RawLatentSession._generate,
    ) = initialize, push, generate
    reference_main(reference_args)
    if not completed:
        raise RuntimeError("reference exited without a completed EOS trace")


if __name__ == "__main__":
    main()
