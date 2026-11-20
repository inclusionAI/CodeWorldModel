# SPDX-License-Identifier: Apache-2.0
"""Replay one input receipt through the upstream native engine's H3 realtime session.

Launches native SGLang the way Omni's native_generation stage does
(DiffGenerator.from_server_args), opens one session, then advances 15 packets
of 17 frames, the held 5-frame tail and an EOS. Prints the RGB sha256, which is
comparable with the host path for the same receipt and seed 2026, and the
per-advance timings. ``--self-check`` builds the config and request plan on CPU.
"""

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.omni_h3_realtime import PIPELINE, pipeline_config  # noqa: E402


def plan(packets, *, first_image, seed, overlap):
    """[(block_idx, request)]: open, one advance per packet, then a frameless EOS."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.realtime import (
        H3RealtimeAdvance,
        H3RealtimeOpen,
        H3RealtimeSource,
    )

    sources = [H3RealtimeSource(index, frames, False) for index, frames in enumerate(packets)]
    sources.append(H3RealtimeSource(len(packets), None, True))
    requests = [(0, H3RealtimeOpen(first_image, seed))]
    for index, source in enumerate(sources):
        following = sources[index + 1] if overlap and not source.eos else None
        requests.append((index + 1, H3RealtimeAdvance(source, following)))
    return requests


def receipt_inputs(receipt_path):
    # Same packetization and seed as the host-path client.
    sys.path.insert(0, str(ROOT / "experiments/2026-10-27-native-vs-local"))
    from native import input_packets

    packets, request, first, _ = input_packets(json.loads(Path(receipt_path).read_text()))
    frames = [np.frombuffer(packet.rgb, dtype=np.uint8).reshape(
        len(packet.frames), packet.height, packet.width, 3) for packet in packets]
    return frames, request.prompt, np.ascontiguousarray(first), request.seed


def launch(manifest, *, overlap):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = ServerArgs(**manifest["server_args"], pipeline_class_name=PIPELINE,
        pipeline_config=pipeline_config(manifest, overlap=overlap))
    return DiffGenerator.from_server_args(args)


def send(session_id, block_idx, request, prompt):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client

    request_id = f"{session_id}-{block_idx}"
    req = Req(sampling_params=SamplingParams(request_id=request_id, prompt=prompt, save_output=False),
        realtime_session_id=session_id, block_idx=block_idx,
        condition_inputs={"h3": request})
    # Req.validate attaches metrics, which carry Scheduler.return_result.* timings back.
    start = time.perf_counter()
    result = sync_scheduler_client.forward(req)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if result.error is not None:
        raise RuntimeError(result.error)
    stages = {} if result.metrics is None else dict(result.metrics.stages)
    return result.output, elapsed_ms, stages


def replay(options):
    from inference.omni_h3_assets import read_native_bundle
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
        ReleaseRealtimeSessionReq,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client

    manifest = read_native_bundle(options.bundle)
    frames, prompt, first, seed = receipt_inputs(options.receipt)
    requests = plan(frames, first_image=first, seed=seed,
                    overlap=options.overlap)
    generator = launch(manifest, overlap=options.overlap)
    session_id = f"h3-replay-{uuid.uuid4().hex[:8]}"
    checksum, total, rows = hashlib.sha256(), 0, []
    try:
        for block_idx, request in requests:
            rgb, elapsed_ms, stages = send(session_id, block_idx, request, prompt)
            count = 0 if rgb is None else len(rgb)
            if block_idx and rgb is None:
                raise RuntimeError("output rank returned no RGB array for an advance")
            if count:
                checksum.update(np.ascontiguousarray(rgb).tobytes())
            total += count
            row = dict(block=block_idx, frames=count, ms=round(elapsed_ms, 2),
                       return_result={k: round(v, 3) for k, v in stages.items()
                                      if k.startswith("Scheduler.return_result.")},
                       stages={k: round(v, 3) for k, v in stages.items()
                               if not k.startswith("Scheduler.return_result.")})
            rows.append(row)
            print(json.dumps(row), flush=True)
        sync_scheduler_client.forward(ReleaseRealtimeSessionReq(session_id))
    finally:
        generator.shutdown()
    summary = dict(overlap=options.overlap, frames=total, rgb_sha256=checksum.hexdigest(),
                   steady_ms=[row["ms"] for row in rows[2:]])
    print(json.dumps(summary), flush=True)
    if options.output:
        Path(options.output).write_text(json.dumps(dict(summary, rows=rows), indent=2) + "\n")


def self_check():
    manifest = dict(recipe=json.loads((ROOT / "configs/recipes/omni-h3-longlive.json").read_text()),
        codec=dict(encoder="h3", decoder="causal", backend="fast", compile=False,
                   cuda_graph=False, window=16, decoder_config="d.yaml",
                   decoder_weights="d.safetensors", suffix_sdpa_backend="cudnn"),
        codec_parallel=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1]),
        dit=dict(fuse_first_denoise=True, norm_quality="lossless"))
    config = pipeline_config(manifest, overlap=True)
    assert config.decoder.cudnn_sdpa and config.lossless_norm and config.fuse_first_denoise
    assert config.transformer_class_override == "MiniMaxH3CausalDiTModel"
    rgb = np.zeros((260, 64, 64, 3), np.uint8)
    packets = [rgb[i:i + 17] for i in range(0, 260, 17)]
    for overlap in (False, True):
        requests = plan(packets, first_image=rgb[0], seed=2026, overlap=overlap)
        advances = [request for _, request in requests[1:]]
        assert [block for block, _ in requests] == list(range(18))
        assert [0 if a.source.frames is None else len(a.source.frames)
                for a in advances] == [17] * 15 + [5, 0]
        assert advances[-1].source.eos and not any(a.source.eos for a in advances[:-1])
        expected = [a.source for a in advances[1:]] + [None] if overlap else [None] * 17
        assert all(a.next_source is b for a, b in zip(advances, expected, strict=True))
    print("realtime_replay self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="prepared native bundle directory")
    parser.add_argument("--receipt", help="input receipt JSON with 260 proxy frames")
    parser.add_argument("--overlap", action="store_true", help="enable codec overlap")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("--self-check", action="store_true")
    options = parser.parse_args()
    if options.self_check:
        self_check()
    elif not options.bundle or not options.receipt:
        parser.error("--bundle and --receipt are required")
    else:
        replay(options)


if __name__ == "__main__":
    main()
