# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen reference corpus through the real native loader and session."""

import argparse
import json
import os
import pickle
from pathlib import Path

import torch
from numerical_trace import (
    STEP600_MERGED_SHA256,
    CommitLedger,
    TraceWriter,
    _load,
    committed_signature,
    digest,
    plan_record,
)


RECIPE = Path(__file__).resolve().parents[2] / "configs/recipes/omni-h3-longlive.json"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", required=True, help="prepared native bundle directory"
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_tp_rank,
        get_ulysses_parallel_rank,
    )
    from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import (
        load_h3_causal_components,
    )
    from sglang.multimodal_gen.runtime.managers.minimax_h3_causal_host import (
        _initialize_native,
        _validate_causal_parallelism,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
        H3FixedHistorySession,
        H3SourceLatents,
        H3StaticCondition,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    manifest = json.loads((Path(options.bundle) / "native-runtime.json").read_text())
    reference = Path(options.reference) / "rank0000"
    receipt = json.loads((reference / "complete.json").read_text())
    identity = receipt["identity"]
    frozen = json.loads(RECIPE.read_text())
    if (
        manifest["schema"] != "h3_omni_native/v1"
        or manifest["recipe"] != frozen
        or identity["recipe"] != frozen
        or receipt["runtime"]["kind"] != "independent-reference"
        or identity["artifact_sha256"] not in manifest["assets"].values()
        or identity["weights_sha256"] != STEP600_MERGED_SHA256
        or manifest["assets"][manifest["server_args"]["transformer_weights_path"]]
        != STEP600_MERGED_SHA256
        or identity["heads"] != 56
        or identity["head_dim"] != 128
        or identity["layers"] != 50
    ):
        raise ValueError(
            "native bundle and independent step600 corpus identities disagree"
        )
    settings = manifest["server_args"]
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if world != settings["num_gpus"]:
        parser.error("torchrun world size must equal bundle num_gpus")
    args = ServerArgs(pipeline_config=MiniMaxH3PipelineConfig(), **settings)
    _validate_causal_parallelism(args)
    _initialize_native(pickle.dumps(args), rank, world, int(os.environ["MASTER_PORT"]))
    writer = TraceWriter(options.output, rank, world)
    if rank == 0:
        for path, expected in manifest["assets"].items():
            if digest(path) != expected:
                raise ValueError(f"native asset changed: {path}")
        for name, expected in identity["inputs"].items():
            if digest(reference / f"{name}.pt") != expected:
                raise ValueError(f"reference input changed: {name}")
    torch.distributed.barrier()
    components = load_h3_causal_components(args)

    def oracle_input(name):
        return _load(reference / f"inputs/{name}.pt")

    if rank == 0:
        for name in identity["inputs"]:
            writer.save(name, _load(reference / f"{name}.pt"))
    condition = oracle_input("condition")
    hidden, tags = components.condition(condition["prompt"], condition["rgb"].numpy())
    image = components.encode_image(oracle_input("image").numpy())
    if rank == 0:
        writer.save("encoder/condition", dict(prompt_embeds=hidden, token_tags=tags))
        writer.save("encoder/image", image)
    for name in sorted(identity["inputs"]):
        if name.startswith("inputs/proxy"):
            value = _load(reference / f"{name}.pt")
            encoded = components.encode_proxy(value["rgb"].numpy(), eos=value["eos"])
            if rank == 0:
                writer.save(name.replace("inputs/", "encoder/"), encoded)

    class ReplaySession(H3FixedHistorySession):
        def _generate(self, plan):
            self.window = f"window{plan.index:04d}"
            self.steps = self.writes = 0
            result = super()._generate(plan)
            if self.steps != 4 or (plan.bootstrap and self.writes):
                raise ValueError(
                    "native session did not execute independent four-step sampling"
                )
            if rank == 0:
                writer.save(
                    f"{self.window}/output",
                    dict(plan=plan_record(plan), video=result.video),
                )
            return result

        def _forward(self, frames, cache, **kwargs):
            before = (
                None
                if kwargs.get("write")
                else committed_signature(cache.owners, cache.layers)
            )
            row_start = len(cache.owners)
            result = super()._forward(frames, cache, **kwargs)
            if kwargs.get("write"):
                if len(cache.layers) != 50:
                    raise ValueError("expected every native cache layer")
                self.commit_ledger.commit(cache.owners, cache.layers, row_start)
                for layer, (k, v) in cache.layers.items():
                    head_start = (
                        get_tp_rank() * (56 // args.tp_size)
                        + get_ulysses_parallel_rank() * k.shape[1]
                    )
                    writer.save(
                        f"{self.window}/write{self.writes}/layer{layer:02d}",
                        dict(
                            owners=list(cache.owners),
                            row_start=row_start,
                            key=k[row_start:],
                            value=v[row_start:],
                        ),
                        head_start=head_start,
                    )
                self.writes += 1
            else:
                if before != committed_signature(cache.owners, cache.layers):
                    raise ValueError("denoise changed committed KV contents or owners")
                if rank == 0:
                    xt = torch.cat(
                        [f.video if f.video is not None else f.noise for f in frames], 1
                    )
                    sigma = kwargs["sigma"]
                    x0 = (
                        xt.to(torch.bfloat16).float()
                        + (1 - (1.0 - sigma.float())) * result
                    )
                    writer.save(
                        f"{self.window}/d{self.steps}", dict(xt=xt, sigma=sigma, x0=x0)
                    )
                self.steps += 1
            return result

    session = ReplaySession(
        components.model, H3StaticCondition(**oracle_input("static")), recipe=frozen
    )
    session.commit_ledger = CommitLedger()
    sources = sorted(
        name for name in identity["inputs"] if name.startswith("inputs/source")
    )
    if len(sources) < 4:
        raise ValueError("expected at least three ordinary sources and EOS")
    for index, name in enumerate(sources):
        value = _load(reference / f"{name}.pt")
        eos = value.pop("eos")
        if type(eos) is not bool or eos != (index == len(sources) - 1):
            raise ValueError("corpus must end with exactly one EOS")
        value["references"] = tuple(value["references"])
        value["reference_noise"] = tuple(value["reference_noise"])
        (session.finish if eos else session.append)(H3SourceLatents(**value))
    components.synchronize()
    writer.finish(
        identity,
        dict(
            kind="native",
            settings=settings,
            backend=components.model._resolved_attention_backend.name,
            tp_rank=get_tp_rank(),
            up_rank=get_ulysses_parallel_rank(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            kv_readonly_guard="logical-row-sha256",
            kv_history_guard="owner-block-sha256",
        ),
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
