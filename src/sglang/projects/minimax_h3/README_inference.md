# Invoker-Render inference contract / 推理约定

## Default model contract / 默认模型契约

The default inference path uses the CWM causal recipe described below. Keep the recipe, checkpoint layout, conditioning assets, and runtime configuration compatible when preparing a bundle.

默认推理路径采用下表中的 CWM 因果生成契约。准备 bundle 时，需保持 recipe、checkpoint 布局、条件资产与运行配置一致。

| Contract | Value |
| --- | --- |
| Public model name | `invoker-render` |
| Frozen recipe | `configs/recipes/omni-h3-cwm.json` |
| Protocol / position policy | `cwm_causal/v1` / `cwm_frame_aligned_v1` |
| Sampler | Deterministic Euler, 4 steps, shift 12 |
| Conditioning VAE | BF16 encoder; posterior mean, no sampled noise |
| Qwen first-image presentation | Single image |
| Prompt | Frozen system prompt recorded in the artifact |
| Stream windows | Bootstrap 9; regular chunks 5; segment boundaries at latents 45 and 70; EOS short tail |
| History | One cache for proxy and clean video; 12 most recent latents |
| Sink | First latent's proxy and clean video (`sink_latents=1`), pinned next to the recent window |
| Long-stream positions | 47-latent reference; same-frame Proxy and target share coordinates; static prefix recomputed when the shift changes |
| Source / output | Original proxy canvas (448×256 reference), nearest resize to 1344×768 before encoding; 1344×768 output |

The shared VAE encoding path applies to both first-image and proxy latents and uses the posterior mean. The request seed only drives the initial target, conditioning and history noise.

Proxy encoding can be selected per bundle with `--proxy-encoder tae --tae-weights PATH`. TAE emits normalized H3 proxy latents through the same realtime stage. The static first image continues to use the full H3 encoder.

The public repository does not ship a model-specific `artifact.json`. Generate it with `invoker_render.py export-artifact` from the matching training configuration, generator checkpoint, encoder checkpoint, and frozen system prompt. The artifact is the source of the asset hashes, prompt, geometry, and provenance used by `prepare`, `infer`, and the browser adapter.

首图和 Proxy 共享 VAE 编码路径，均取 posterior 均值，不采样噪声。请求 seed 仅用于初始目标噪声、条件噪声与历史噪声。

## Deployment / 部署

The CLI defaults to `--recipe cwm --deployment-profile production`: TP1/Ulysses8, E4 encoder ranks 2/3/6/7, decoder TP4 ranks 0/1/4/5, fast causal decoding with FA3 spatial attention, codec overlap, and CPU RGB delivery. CWM enables online AdaLN, first-denoise fusion and the lossless DiT norm; decoder compile, CUDA Graph and fused post remain disabled.

`reference` provides a conservative codec profile. `overlap` enables additional compiler/Graph tuning. Legacy `longlive` / `continuous5` recipes require an explicit non-production profile and retain their own fixed-v2 contract.

默认 `production` 使用上述部署配置。`reference` 对应保守 codec 路径；`overlap` 启用额外的编译与 Graph 调优，需单独检查软硬件兼容性和实际效果。

Merged HF weights are loaded directly; do not apply the training LoRA again. Checkpoint class/QKV detection, FFN conversion, and FP32 RoPE frequency reconstruction are preserved. Keep the upstream architecture names in checkpoint configuration files.

部署命令、依赖、输入格式和输出检查见[部署指南](../../../../docs/INFERENCE.md)。
