"""Generate independent CWM SFT/TF/TSCD/DMD trials without loading any models.

Run from the RAVEN root with ``python -m projects.minimax_h3.tools.cwm_config
--help``. Later stages consume full or merged RAVEN exports, never a DCP
resume directory or an adapter-only file.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml

from projects.minimax_h3.modeling.cwm_provenance import source_identity
from projects.minimax_h3.data.ref_latent import (DEFAULT_WINDOW, DEFAULT_ROLE_SPAN, ROLLOUT_PROTOCOL,
    LEGACY_CONDITIONING_MODE, causal_layout, conditioning_mode)

STAGES = ("sft", "tf", "tscd", "dmd")
PROJECT = "projects.minimax_h3"


def node(module, class_name, **args):
    return dict(module=module, class_name=class_name, args=args)


def grid(steps, shift, *, skip=0):
    return node("common.diffusion.timestep.sampling.trailing", "TrailingSamplingTimesteps",
                T=1.0, num_sampling_steps=steps, shift=shift, sampling_skip_max=int(skip))


def training(shift, *, paired=False, clamp_min=0.02, clamp_max=0.98):
    result = node(PROJECT + ".modeling.training_timesteps" if paired else
                  "common.diffusion.timestep.training.uniform",
                  "PairedUniformTrainingTimesteps" if paired else "UniformTrainingTimesteps",
                  T=1.0, shift=shift, clamp_min=clamp_min, clamp_max=clamp_max)
    if paired:
        result["args"]["audio_shift"] = 3.0
    return result


def model(args, architecture, path, *, causal, trainable, source="raven", ema=False):
    result = node(PROJECT + ".modeling.build",
                  "MiniMaxH3CausalX0DiTSP" if causal else "MiniMaxH3X0DiTSP",
                  **copy.deepcopy(architecture))
    result.update(meta_init=True, weight=dict(
        module=PROJECT + ".modeling.weights" if source == "released" else "common.model.weights",
        class_name="MiniMaxH3WeightLoader" if source == "released" else "DefaultWeightLoader",
        path=str(Path(path).expanduser().resolve())),
        runtime=dict(training=trainable, param_dtype="bfloat16"),
        placement=dict(weight_dtype="float32", plugins=[dict(
            module=PROJECT + ".modeling.fp32_placement", class_name="MiniMaxH3FP32ModulePlacement")],
            fsdp=dict(wrap_modules=["CausalMiniMaxH3DiTBlock" if causal else "MiniMaxH3DiTBlock",
                                    "MiniMaxH3TokenRefinerBlock"],
                      param_dtype="bfloat16", reduce_dtype="float32", shard_size=args.shard_size,
                      cpu_offload=args.cpu_offload, cast_forward_inputs=False)))
    if trainable:
        result["runtime"]["plugins"] = [dict(module="common.plugin.gradient_checkpointing",
                                            class_name="NativeGradientCheckpointing", enabled=True)]
        result["optimizer"] = [dict(module="torch.optim", class_name="AdamW", args=dict(
            lr=args.lr if args.lr is not None else (1e-4 if args.mode == "lora" else 2e-6),
            betas=[0.9, 0.999], weight_decay=0.01, foreach=False))]
        if args.mode == "lora":
            # PEFT has already selected trainable parameters when runtime runs.
            # Even `requires_grad: null` would override that selection.
            result["adapter"] = dict(r=args.lora_rank, lora_alpha=args.lora_rank,
                target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
                custom_module_mapping=[dict(
                    source=dict(module=PROJECT + ".modeling.transformer.model", class_name="_Linear"),
                    target=dict(module=PROJECT + ".modeling.lora", class_name="MiniMaxH3LoraLinear"))])
        else:
            result["runtime"]["requires_grad"] = True
    else:
        result["runtime"]["requires_grad"] = False
    if ema:
        result["ema"] = {}
    return result


def make_config(args, stage, architecture, max_seqlen, init, source, teacher=None):
    entry = ("tscd", "TrajectorySegmentedConsistencyDistillation") if stage == "tscd" else (
        ("dmd", "DistributionMatchingDistillation") if stage == "dmd" else
        ("diffusion_finetuning", "DiffusionFinetuning"))
    schedule = node("common.diffusion.schedule.lerp", "LinearInterpolationSchedule", T=1.0, pred_type="x_0")
    steps = args.tscd_grid if stage == "tscd" else (args.rollout_steps if stage == "dmd" else args.fm_steps)
    diffusion = dict(schedule=schedule, sampling_timesteps=grid(steps, 12.0, skip=stage == "tscd"),
        audio_sampling_timesteps=grid(steps, 3.0, skip=stage == "tscd"),
        # Full rollouts use Euler; TSCD's teacher advances one adjacent grid
        # interval with the same deterministic ODE operator.
        sampler=node(PROJECT + ".modeling.sampling", "EulerSampler"))
    mm = dict(module=PROJECT + ".meta_models.minimax_h3_ref2va",
              class_name="MiniMaxH3Ref" + stage.upper(), audio_loss_weight=0.0)
    engine = dict(resume=args.resume, training_steps=args.steps, ga_steps=args.ga_steps,
        up_size=dict(train=args.up_size, validation=args.up_size), seed=args.seed,
        log_interval=args.log_interval, save_interval=args.save_interval,
        val_interval=args.val_interval, val_before_train=False, save_before_train=False, clip_grad_norm=1.0)
    config = dict(entry=dict(module="engines." + entry[0], class_name=entry[1]), logging={},
        engine=engine, distributed=dict(up_size=[args.up_size]),
        persistence=dict(proj_name=args.run_name, exp_name=stage, output_dir=str(Path(args.runs_dir).resolve()),
                         checkpoint={"async": False, "keep_latest_k": 2}),
        data=dict(module=PROJECT + ".data.ref_latent", class_name="RefLatentDataset",
                  num_workers=args.num_workers, prefetch_factor=2 if args.num_workers else None,
                  pin_memory=True, args=dict(manifest=str(Path(args.manifest).resolve()),
                      max_seqlen=max_seqlen, max_samples=args.max_samples, stage=stage,
                      conditioning_mode=getattr(args,'conditioning_mode',LEGACY_CONDITIONING_MODE),
                      window_size=None if stage == "sft" else args.window_size,
                      rope_reference_latents=args.rope_reference_latents)),
        meta_model=mm, diffusion=diffusion, models=dict(backbone=model(
            args, architecture, init, causal=stage != "sft", trainable=True,
            source=source, ema=stage == "tscd")))
    if args.swanlab:
        config['logging']['plugins'] = [dict(module='common.plugin.swanlab', class_name='SwanLabPlugin',
            mode=args.swanlab_mode, project=args.swanlab_project, workspace=args.swanlab_workspace)]
    if args.resume_dir:
        engine["resume_dir"] = str(Path(args.resume_dir).resolve())
    validation = dict(seed=args.seed + 1, num_samples=args.val_samples, validate_ema=False,
        sampling_timesteps=grid(args.rollout_steps, 12.0),
        audio_sampling_timesteps=grid(args.rollout_steps, 3.0),
        sampler=node(PROJECT + ".modeling.sampling", "EulerSampler"))
    if args.val_manifest:
        validation["manifest"] = str(Path(args.val_manifest).resolve())
        validation["output_dir"] = str(Path(args.runs_dir).resolve() / args.run_name / stage / "validation")
    config["validation"] = validation
    if stage in ("sft", "tf"):
        diffusion["training_timesteps"] = (node(PROJECT + ".modeling.training_timesteps",
            "RefSFTTrainingTimesteps", T=1.0, shift=12.0) if stage == "sft" else
            training(12.0, paired=True, clamp_min=1e-4, clamp_max=None))
        if args.guidance is not None:
            mm["guidance"] = copy.deepcopy(args.guidance)
        if stage == "sft":
            mm["objective"] = args.sft_objective
            optimizer = config["models"]["backbone"]["optimizer"][0]
            optimizer["args"]["lr"] = args.lr if args.lr is not None else 1e-4
            # Engine t is zero-based optimizer steps. Warmup uses (t + 1)/N:
            # update N reaches the peak and the final update reaches min_lr.
            optimizer["lr_scheduler"] = node("common.lr_scheduler.cosine", "CosineLRScheduler",
                warmup_t=args.sft_warmup_steps, warmup_init_lr=0.0,
                t_start=max(0, args.sft_warmup_steps - 1), t_max=args.steps - 1,
                min_lr=args.sft_min_lr)
        else:
            mm["objective"] = args.tf_objective
            if args.resampling_warmup is not None:
                mm["resampling"] = dict(warmup_steps=args.resampling_warmup, shift=args.resampling_shift,
                                        steps=args.resampling_steps)
    else:
        config["models"]["tea_model"] = model(args, architecture, teacher,
            causal=stage == "tscd", trainable=False)
        if stage == "tscd":
            engine["ema_decay"] = dict(backbone=args.ema_decay)
            mm.update(num_segments=args.num_segments, loss_type="x_t")
            diffusion.update(tea_schedule=copy.deepcopy(schedule),
                tea_sampler=node(PROJECT + ".modeling.sampling", "EulerSampler"))
        else:
            mm.update(fake_loss_type="v_lerp", fake_grad_enabled=False, real_grad_enabled=False,
                      dmd_loss=dict(type="dmd", norm_clip_min=1e-5))
            engine.update(fake_update_interval=1, gen_update_interval=2,
                          fake_step_models=["fake_model"], gen_step_models=["backbone"])
            config["models"]["fake_model"] = model(args, architecture, teacher, causal=False, trainable=True)
            diffusion["fake_schedule"] = copy.deepcopy(schedule)
            for name in ("fake_training_timesteps", "score_timesteps"):
                diffusion[name] = training(12.0)
                diffusion["audio_" + name] = training(3.0)
    config["lineage"] = dict(stage=stage, init_format=source, initialization=str(Path(init).resolve()),
                             references=args.references,
                             conditioning_mode=getattr(args,'conditioning_mode',LEGACY_CONDITIONING_MODE))
    config['lineage']['sampling_protocol'] = ROLLOUT_PROTOCOL
    if args.guidance is not None:
        if stage in ("sft", "tf"):
            config['lineage']['guidance'] = copy.deepcopy(args.guidance)
        else:
            # The all-stage plan records the source training semantics, but
            # TSCD/DMD retain their existing single-pass teacher contract.
            config['lineage']['teacher_guidance'] = dict(
                training=copy.deepcopy(args.guidance), inference_scale=1.0)
    if stage != "sft":
        config['lineage']['causal_layout'] = dict(zip(
            ('name','window_size','rope_reference_latents'),
            causal_layout(args.window_size,args.rope_reference_latents)))
    if teacher:
        config["lineage"].update(teacher_stage="tf" if stage == "tscd" else "sft",
                                teacher=str(Path(teacher).resolve()), teacher_sha256=None)
        hash_cache = str(Path(args.runs_dir).resolve() / ".cwm_weight_hashes")
        config["persistence"]["plugins"] = [dict(module=PROJECT + ".modeling.cwm_provenance",
            class_name="CWMTeacherProvenance", hash_cache=hash_cache)]
        if Path(teacher).expanduser().is_file():
            config["lineage"]["teacher_sha256"] = source_identity(teacher, hash_cache)["sha256"]
    return config


def _manifest(path, *, guidance_drop=None):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries or any(e.get("schema") != "cwm_ref2va_v1" for e in entries):
        raise ValueError(f"{path}: expected a nonempty prepared CWM cache manifest")
    if len({e["fingerprint"] for e in entries}) != 1:
        raise ValueError(f"{path}: mixed cache fingerprints")
    if len({conditioning_mode(e) for e in entries}) != 1:
        raise ValueError(f"{path}: mixed conditioning modes")
    if guidance_drop is not None and any(e.get('guidance_drop') != guidance_drop for e in entries):
        raise ValueError(f"{path}: --guidance-drop {guidance_drop} requires matching encoded weak conditions "
                         "in every manifest entry; prepare a new cache with encode --guidance-drop")
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    parser.add_argument("--manifest", required=True, help="Prepared training cache manifest.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init", required=True, help="SFT: released Ref2VA; TF: B_sft or (--init-format released) Ref2VA; later stages: full/merged RAVEN weights")
    parser.add_argument("--init-format", choices=("released", "raven"), help="Default: released for SFT/all, raven otherwise")
    parser.add_argument("--sft-checkpoint", help="All: future B_sft export; DMD: frozen B_sft teacher and fake initialization")
    parser.add_argument("--tf-checkpoint", help="All: future C_tf export; TSCD: teacher override, defaults to --init")
    parser.add_argument("--tscd-checkpoint", help="All: future TSCD export used to initialize DMD generator")
    parser.add_argument("--mode", choices=("full", "lora"), default="lora")
    parser.add_argument("--lora-rank", type=int, default=384)
    parser.add_argument("--lr", type=float, help="Peak SFT lr / constant other-stage lr; SFT defaults to 1e-4")
    parser.add_argument("--sft-warmup-steps", type=int, default=500,
                        help="SFT linear warmup in optimizer steps; explicitly shorten for smoke runs")
    parser.add_argument("--sft-min-lr", type=float, default=1e-5,
                        help="SFT cosine endpoint on the final optimizer update")
    parser.add_argument("--steps", type=int, default=1000, help="Example optimizer windows per stage; tune on held-out rollouts")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--ga-steps", type=int, default=1)
    parser.add_argument("--up-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-seqlen", type=int, help="Default: maximum actual packed length in the manifest")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW,
                        help="Previous proxy/RGB time groups retained (default 6); 0 means current only")
    parser.add_argument("--rope-reference-latents", type=int, default=DEFAULT_ROLE_SPAN,
                        help="Fixed role spacing in H3 latent-frame units (default 37); independent of stream length")
    parser.add_argument("--fm-steps", type=int, default=20)
    parser.add_argument("--tscd-grid", type=int, default=20)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--sft-objective", choices=("velocity", "x0"), default="velocity")
    parser.add_argument("--tf-objective", choices=("gaussian_velocity", "x0"), default="gaussian_velocity")
    parser.add_argument("--guidance-mode", choices=("plain", "cfg", "gft"), default="plain",
                        help="SFT/TF: ordinary fitting, condition-dropout CFG, or fixed-beta GFT")
    parser.add_argument("--guidance-drop", choices=("proxy", "prompt", "prompt_proxy"), default="proxy",
                        help="Weak branch removes these conditions; first frame and RGB history remain")
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--gft-beta", type=float, default=0.5,
                        help="Fixed GFT beta in (0,1]; learned guidance strength is approximately 1/beta")
    parser.add_argument("--cfg-scale", type=float,
                        help="Validation/rollout CFG scale: default 2 for cfg, 1 for plain/gft")
    parser.add_argument("--dropout-seed", type=int, default=20261010)
    parser.add_argument("--resampling-warmup", type=int,
                        help="TF: switch to Resampling Forcing after this many plain-TF optimizer steps; omit for plain TF")
    parser.add_argument("--resampling-shift", type=float, default=0.6,
                        help="Resampling Forcing t_s shift; <1 favors light resampling (paper 0.6)")
    parser.add_argument("--resampling-steps", type=int, default=20,
                        help="TF full resampling Euler steps from t_s to 0 (default 20; paper setting: 1)")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-interval", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=2)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-name", default="cwm_ref2va")
    parser.add_argument("--swanlab", action="store_true", help="Log metrics to SwanLab using SDK login or SWANLAB_API_KEY")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--swanlab-project", default="cwm-ref2va")
    parser.add_argument("--swanlab-workspace", help="Default: the logged-in user's workspace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default="auto", help="Same-stage only: auto, never, or integer checkpoint step")
    parser.add_argument("--resume-dir", help="Same-stage DCP seed directory; never cross-stage initialization")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.guidance = None
        scale = args.cfg_scale if args.cfg_scale is not None else (2.0 if args.guidance_mode == "cfg" else 1.0)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("--cfg-scale must be finite and nonnegative")
        if args.guidance_mode != "cfg" and scale != 1:
            raise ValueError("plain/GFT use a single branch and require --cfg-scale 1")
        if args.guidance_mode != "plain":
            if args.stage not in ("sft", "tf", "all"):
                raise ValueError("--guidance-mode cfg/gft is supported only for SFT/TF (or --stage all)")
            if not 0 <= args.cond_dropout < 1 or (args.guidance_mode == "cfg" and args.cond_dropout == 0):
                raise ValueError("--cond-dropout must be in (0,1) for CFG and [0,1) for GFT")
            if not 0 < args.gft_beta <= 1:
                raise ValueError("--gft-beta must be in (0,1]")
            args.guidance = dict(mode=args.guidance_mode, drop=args.guidance_drop,
                dropout=args.cond_dropout, beta=args.gft_beta, scale=scale, seed=args.dropout_seed)
        for name in ("steps", "save_interval", "log_interval", "ga_steps", "up_size", "shard_size",
                     "max_samples", "lora_rank", "fm_steps", "tscd_grid", "num_segments", "rollout_steps", "val_samples"):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.num_workers < 0 or args.val_interval < 0 or not 0 <= args.ema_decay < 1:
            raise ValueError("invalid worker count, validation interval, or EMA decay")
        causal_layout(args.window_size,args.rope_reference_latents)
        if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
            raise ValueError("--lr must be finite and positive")
        if args.stage in ("sft", "all"):
            if args.steps < 2 or not 0 <= args.sft_warmup_steps < args.steps:
                raise ValueError("SFT requires --steps >= 2 and --sft-warmup-steps in [0, --steps); "
                                 "short smoke runs must explicitly shorten the default 500-step warmup")
            peak_lr = args.lr if args.lr is not None else 1e-4
            if not math.isfinite(args.sft_min_lr) or not 0 <= args.sft_min_lr <= peak_lr:
                raise ValueError("--sft-min-lr must be finite and in [0, the SFT peak lr]")
        if args.tscd_grid % args.num_segments or args.tscd_grid < 2:
            raise ValueError("--tscd-grid must be >= 2 and divisible by --num-segments")
        for value in (args.steps, args.save_interval, args.val_interval):
            if value * args.ga_steps % args.up_size:
                raise ValueError("steps, save-interval, val-interval times ga-steps must be divisible by up-size")
        if args.val_interval and not args.val_manifest:
            raise ValueError("--val-interval requires --val-manifest")
        if args.resampling_warmup is not None and (args.stage not in ("tf", "all")
                or not 0 <= args.resampling_warmup < args.steps
                or not args.resampling_shift > 0 or args.resampling_steps < 1):
            raise ValueError("--resampling-warmup is TF-only and must be in [0, --steps); "
                             "--resampling-shift must be > 0 and --resampling-steps >= 1")
        if args.resume not in ("auto", "never"):
            args.resume = int(args.resume)
            if args.resume < 0:
                raise ValueError("--resume step cannot be negative")
        if args.resume_dir and (args.stage == "all" or args.resume == "never"):
            raise ValueError("--resume-dir requires a single stage and resume enabled")
        stages = STAGES if args.stage == "all" else (args.stage,)
        source = args.init_format or ("released" if args.stage in ("sft", "all") else "raven")
        if args.stage in ("tscd", "dmd") and source != "raven":
            raise ValueError("TSCD/DMD initialize from RAVEN full/merged domain checkpoints")
        if args.stage == "all" and not all((args.sft_checkpoint, args.tf_checkpoint, args.tscd_checkpoint)):
            raise ValueError("--stage all requires explicit --sft-checkpoint, --tf-checkpoint, --tscd-checkpoint export paths")
        if args.stage == "dmd" and not args.sft_checkpoint:
            raise ValueError("DMD requires --sft-checkpoint for both the bidirectional teacher and fake initialization")
        guidance_drop = args.guidance_drop if args.guidance is not None and args.guidance_drop != 'proxy' else None
        entries = _manifest(args.manifest, guidance_drop=guidance_drop)
        # Record the cache's reference recipe (None for manifests written before it existed).
        args.references = entries[0].get("references")
        args.conditioning_mode = conditioning_mode(entries[0])
        if args.val_manifest:
            validation = _manifest(args.val_manifest, guidance_drop=guidance_drop)
            if {e["take_id"] for e in entries} & {e["take_id"] for e in validation}:
                raise ValueError("training and validation share takes; overlapping CWM windows must remain in one split")
            if args.conditioning_mode != conditioning_mode(validation[0]):
                raise ValueError("training and validation use different conditioning modes")
            if entries[0]["fingerprint"] != validation[0]["fingerprint"]:
                raise ValueError("training and validation use different encoder/preprocessing fingerprints")
        architecture = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/dit.yaml").read_text())
        configs = {}
        for stage in stages:
            budget_key = "sft" if stage == "sft" else ("dmd" if stage == "dmd" else "tf")
            required = max(e["lengths"][budget_key] for e in entries)
            budget = required if args.max_seqlen is None else args.max_seqlen
            if budget < required:
                raise ValueError(f"{stage} needs --max-seqlen >= {required}; refusing silent sample drops")
            init = args.init if args.stage != "all" or stage == "sft" else {
                "tf": args.sft_checkpoint, "tscd": args.tf_checkpoint, "dmd": args.tscd_checkpoint}[stage]
            teacher = (args.tf_checkpoint or init) if stage == "tscd" else args.sft_checkpoint if stage == "dmd" else None
            # --init-format describes --init only; chained --stage all inits are RAVEN exports.
            configs[stage] = make_config(args, stage, architecture, budget, init,
                source if stage == "sft" or args.stage != "all" else "raven", teacher)
        root = Path(args.out_dir).expanduser().resolve()
        for stage in configs:
            if (root / f"{stage}.yaml").exists() and not args.overwrite:
                raise ValueError(f"{root / (stage + '.yaml')} exists; use --overwrite to replace")
        root.mkdir(parents=True, exist_ok=True)
        for stage, config in configs.items():
            path = root / f"{stage}.yaml"
            if stage != 'sft':
                from projects.minimax_h3.modeling.ref_recipe import RefRecipe
                config['inference_recipe'] = RefRecipe.from_training(config).to_dict()
            path.write_text("# Generated CWM trial; schedules are starting points, not convergence claims.\n" +
                            yaml.safe_dump(config, sort_keys=False))
            print(path)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
