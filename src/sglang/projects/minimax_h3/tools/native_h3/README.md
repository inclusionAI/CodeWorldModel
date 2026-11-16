# Native causal H3

This computation variant subclasses the native SGLang H3 DiT. It keeps the native
Qwen encoder, TP/SP groups, projections, blocks, weight loading and attention
kernels. The fixed-history session accepts prepared tensors and explicit noise;
the media bridge owns RGB decoding, RNG, output transfer and the causal decoder.

## Component loading

In an initialized SGLang worker, call:

```python
from sglang.multimodal_gen.runtime.loader.minimax_h3_causal import load_h3_causal_components

components = load_h3_causal_components(server_args)
prompt_embeds, token_tags = components.condition(prompt, first_rgb)
first_frame = components.encode_image(first_rgb)
proxy = components.encode_proxy(rgb_chunk, eos=False)
```

`server_args.pipeline_config` must be `MiniMaxH3PipelineConfig` and
`component_paths` must contain `transformer`, `text_encoder`, `video_vae` and
`processor`. Use native component configs with their registered class names.
The transformer asset config must declare `_class_name: MiniMaxH3CausalDiTModel`
and `checkpoint_qkv_layout: native`. Set `transformer_weights_path` to the original
step600 merged safetensors. `MiniMaxH3CausalDiTConfig` strips `dit.` in the loader;
the merged file's QKV already uses q_all/k_all/v_all and needs no payload rewrite.
The released grouped checkpoint requires an explicit `grouped` layout instead.

The helper requires all three models resident on the worker device. It rejects
ring parallelism and fallback component implementations. Qwen uses the native
text loader's `encoder_parallel` setting. `auto` keeps its native policy;
`fold` must resolve to the complete replica's TP group or loading fails.
`replicate` disables folding across SP, while retaining the DiT TP group when
TP exceeds one. `dp` requires native stage-level batch scheduling and is rejected
by this single-document helper. The existing VAE loader temporarily
loads the entire VAE, then the helper releases its unused offline decoder. Only
the encoder is called. The separate causal decoder stays with the media bridge.

RGB inputs are uint8 HWC or THWC with H/W divisible by 32, already cropped by
the caller. Image encoding yields normalized BF16 C1HW. A normal 17-frame proxy
yields C5HW; EOS accepts exactly 5 frames, repeats the last frame to reach 17,
encodes 5 latents and returns the first 2. Encoding uses posterior mean with
BF16 VAE weights/input, FP32 normalization and BF16 returned latents. It consumes
no RNG. The helper never rescales the canvas or invents an extra source frame.

## Tensor session

```python
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.causal_latent import (
    H3FixedHistorySession, H3SourceLatents, H3StaticCondition,
)

static = H3StaticCondition(prompt_embeds, token_tags, first_frame, first_frame_noise)
session = H3FixedHistorySession(components.model, static, recipe=recipe)
spans = session.append(H3SourceLatents(
    references=(proxy,), reference_noise=(proxy_noise,),
    target_noise=target_noise, clean_noise=history_noise,
))
# The final source packet is mandatory; finish consumes two source latents.
tail = session.finish(final_source)
```

All latents and their noise are CTHW. `append` takes 5 latent frames and may
return no spans while bootstrap waits. `finish` takes 2 and drains the tail.
Each returned span has `.video` and `.plan`, including the absolute half-open
`.plan.current.start/stop`, window index and bootstrap/EOS flags. `reset()` clears
all session state. A failed model call makes the session unusable until reset;
partial layer KV never becomes published history.

`configs/recipes/omni-h3-longlive.json` records bootstrap9, current5, sink1/recent12, boundaries
45/70, absolute native positions, role span37 and Euler4/shift12. The closed
recipe parser rejects unsupported schedules, conditioning and guidance.

The component helper scopes Qwen encoding with native
`set_forward_context(current_timestep=0, attn_metadata=None)`. Each session model
call also owns its scope: D calls use the actual zero-based loop index, restarting
at 0 for each window; C/H writes use encoding-stage index 0. The physical H3
timesteps remain in the packed tensors (`1 - sigma.float()` for D and `0.999`
for clean conditioning). AdaLN lookup uses those exact timestep plans, rather
than the context's loop index. Both helpers restore any caller context, including
when an encoder or model raises.

### Optional first-denoise fusion

`H3FixedHistorySession(..., fuse_first_denoise=True)` defers each continuation's
clean-history write until the next window. It packs previous clean H, current
proxy C and noisy D1 into one native forward, followed by the ordinary D2-D4
forwards. Fusion currently supports five pending H latents and five current
latents. Short windows and their neighboring non-five-latent history use separate
forwards; pending H commits before cache eviction in that path. The default is
`False`; the frozen recipe is unchanged.

H queries the complete previous cache and all pending H rows before eviction.
C queries retained previous cache plus retained H and current C. D1 queries that
same set plus current noisy D1. The existing post-exchange hook executes three
dense attentions for these row ranges. All native projections, norm/RoPE,
head-group exchanges and output projections remain shared. Only retained old/H
and C rows commit, after every layer and head slice completes. D and padding rows
never enter the cache.

Clean H/C timestep embeddings and block AdaLN projections retain the separate
path's single-row GEMM shape. D1 retains its original two-row projection. This
avoids clean-cache rounding drift from combining the two timestep rows; native
TP gathers, parameter splits and model kernels are unchanged.

Bootstrap and the first continuation's cache setup use the separate path. Pending
H retains its clean tensors independently of the frame window. EOS flushes the
final H, and reset discards all pending state. Failures retain the existing
reset-required session behavior. Fusion changes GEMM row counts, so real-weight
numerical and performance acceptance remains necessary before deployment.

```bash
PYTHONPATH=python pytest -q test/registered/unit/multimodal_gen/test_h3_causal_fused.py
```

## Distributed execution and validation

The component helper does not create processes. Reuse the native initialization
sequence in `runtime/managers/gpu_worker.py`: publish global diffusion ServerArgs,
set rank/device and rendezvous, call
`maybe_init_distributed_environment_and_model_parallel` with matching TP/SP/
Ulysses degrees, and publish the SRT runtime context. All ranks must load and
invoke the components/session in the same order. Local `components.synchronize()`
only waits for CUDA on that rank; stage barriers and input broadcast belong to
the host. A single-process factory cannot create a multi-rank layout.

The single-node `H3CausalHost` provides that process group for a trusted local
bridge factory. It derives its worker count from `server_args.num_gpus` and owns
one active session, ordered calls, stage barriers, cancellation and failure
teardown. An explicit `world_size` must match ServerArgs. CPU factories without
ServerArgs default to one worker:

```python
from sglang.multimodal_gen.runtime.managers.minimax_h3_causal_host import H3CausalHost

with H3CausalHost(
    worker_factory=trusted_factory_symbol,  # importable "module:build_bridge"
    factory_kwargs=bridge_asset_paths,
    server_args=server_args,
) as host:
    host.open(identity, request)
    for packet in source_packets:
        output_packets = host.append(identity, packet)
    output_packets = host.finish(identity)
    host.close(identity)
```

Execution settings belong in ServerArgs, separately from the frozen checkpoint
and numerical recipe. Pass `num_gpus`, `tp_size`, `sp_degree`, `ulysses_degree`,
`ring_degree`, `encoder_parallel` and `attention_backend`. This host requires
`nnodes=1`, `dp_size=1`, `enable_cfg_parallel=False`, `cfg_parallel_degree=1`,
`kv_gather_degree=1`, `ring_degree=1`, `sp_degree=ulysses_degree` and
`num_gpus=tp_size*sp_degree`. Set Ulysses explicitly, including SP2; the native
SP2 default can otherwise choose KV gather, which this causal path rejects.
No encoder or decoder rank placement is part of the model recipe.

For the current dense architecture, native H3 checks TP divisibility of every
sharded width, `(56/tp_size) % ulysses_degree == 0` and
`64 % sp_degree == 0`. The 56 attention heads and audio output width 32 permit
TP1/2/4/8. The following combinations pass those source-level constraints;
none is a performance recommendation or a causal GPU acceptance result:

| GPUs | TP × Ulysses |
| --- | --- |
| 1 | 1 × 1 |
| 2 | 1 × 2, 2 × 1 |
| 4 | 1 × 4, 2 × 2, 4 × 1 |
| 8 | 1 × 8, 2 × 4, 4 × 2, 8 × 1 |

For example, an unmeasured eight-GPU candidate uses `num_gpus=8`, `tp_size=2`,
`sp_degree=4`, `ulysses_degree=4`, `ring_degree=1`, `encoder_parallel="auto"`
and `attention_backend="fa"`, together with the explicit single-replica settings
above. Different assets still pass through the native model's dimension checks.
CPU ServerArgs adjusts some TP launches for NUMA; a CPU configuration check is
not validation of GPU argument resolution or collectives.

Causal attention keeps native TP projections/reductions, fused QK norm/RoPE,
destination-major packed QKV Ulysses exchange and output exchange. Dense backend
enums accepted here are `FA`, `TORCH_SDPA`, `TORCH_CUDNN_SDPA` and
`DYNAMIC_CUDNN_SDPA`; ServerArgs accepts their lower-case names. The native dynamic
cuDNN implementation selects FA where its hardware/shape policy requires it and
logs a cuDNN failure before pinning its FA fallback. An explicitly selected
unsupported algorithm fails; the host does not rewrite it. After native
resolution, an explicit component/global choice must match the resolved enum,
so an unavailable FA request cannot silently become Torch SDPA. Platform and
backend availability checks remain native. Leaving the backend unspecified uses
the native default, which must resolve to one of these dense implementations.

Ordinary H3 also implements FA ring/USP, but the causal cache currently stores
complete live rows after Ulysses rather than ring-local rows. Ring therefore
requires a cache-aware algorithm and remains rejected. KV gather is rejected
because this path implements Ulysses exchange. Sparse/hybrid architectures and
Spectrum remain rejected. Resident components are still required.
These are current execution limits, not frozen numerical semantics.

Causal attention uses the native H3 norm/RoPE, packed exchange and head-group
pipeline. A post-exchange hook receives explicit `head_start` and `head_count`
within this rank's exchanged head partition, reads that slice of historical KV,
and calls the selected dense backend with the longer key sequence. The official
reverse exchange and output projection restore the original layout. Each group
owns its KV snapshot; all layers and head slices must finish before the forward
publishes a complete cache. Ordinary H3 and the token refiner use their original
attention path without this hook.

Set `SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS=1` to pin sequential exchange.
For step600 TP2/U4, each rank holds 7 heads after exchange, so explicit groups=7
is divisible and groups=2 is not. The native default `-1` selects groups only when
the transport's backend, shape, IPC and memory checks permit it, otherwise it
keeps sequential exchange. Both ordinary pipelining and QK norm/RoPE direct-to-
staging retain the native eligibility checks. GPU correctness and performance
must be verified on the actual deployment; a requested group count does not
prove either path ran. Real-weight comparisons retain `rtol=atol=0.01` and record
exact/max-absolute/relative-RMS diagnostics without relaxing the semantic checks.

Source entry points are the native H3
[TP and SP validation, projections and output gathers](../../frameworks/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py),
[Ulysses transport](../../frameworks/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py),
[encoder folding policy and bound TP group](../../frameworks/sglang/python/sglang/multimodal_gen/runtime/models/encoders/base.py),
and [dense SDPA/cuDNN implementations](../../frameworks/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sdpa.py).

The factory runs as `factory(context, **factory_kwargs)` on each rank after
native platform/plugin/group initialization. It returns the media bridge and
uses `context.server_args` to load native components. Only the configured output
rank's CPU RGB packets return to the caller. The present media bridge requires
matching decoded frame counts on every rank to drain each source ledger; a
non-output codec must not silently return `None` for all frames. Factory symbols
and pickle payloads are trusted local configuration, never network input.
Any worker failure or transport deadline invalidates the complete host. Construct
another host after failure; no partially advanced KV or decoder state is reused.

```bash
python -m pytest -q test/registered/unit/multimodal_gen/test_h3_causal_host.py
```

These host tests use real spawned CPU processes and real pipes. They validate
control flow and teardown, including blocked large sends and incomplete receives;
they do not validate native TP/Ulysses/NCCL, actual model assets or the real codec.

The following smoke runs actual native H3 modules with small random weights,
checks registry/mapping, compares dense bootstrap against the ordinary H3,
and executes bootstrap, cache writes, continuation, short EOS and reset:

```bash
PYTHONPATH=frameworks/sglang/python torchrun --standalone --nproc-per-node=1 tools/native_h3/native_smoke.py
PYTHONPATH=frameworks/sglang/python torchrun --standalone --nproc-per-node=8 tools/native_h3/native_smoke.py --sp-size 8
PYTHONPATH=frameworks/sglang/python torchrun --standalone --nproc-per-node=4 tools/native_h3/native_smoke.py --tp-size 2 --sp-size 2 --attention-backend FA
PYTHONPATH=python pytest -q test/registered/unit/multimodal_gen/test_h3_causal_semantics.py
PYTHONPATH=python pytest -q test/registered/unit/multimodal_gen/test_h3_causal_native_context.py
PYTHONPATH=python pytest -q test/registered/unit/multimodal_gen/test_h3_causal_native_attention.py
```

The native context test is a Linux CPU integration check using the real native
`LocalAttention` and Torch SDPA through the conditioning helper. Its tokenizer,
processor and input features are small synthetic fixtures; it does not load
Qwen weights. The CUDA smoke additionally observes actual H3 model entry on
every D and C/H call, checking loop indices, sigma plans, context cleanup and
TP/Ulysses cache-head shape. Its small architecture permits TP1/2/4/8 with an
audio output width of eight. Repeat a supported GPU layout with each chosen
backend; the smoke accepts the same four backend enum names listed above.
Smoke and production workers publish the supported SRT `model_path="dummy"`
context, matching the ordinary diffusion worker. SRT skips model/Hugging Face
configuration resolution for that sentinel; the smoke constructs its small DiT
from its local architecture object. Production component loaders separately use
the native ServerArgs' actual local model/component paths.

The native attention CPU tests exercise real causal attention with longer
historical K/V through native SDPA classes and native architecture validation.
The cuDNN class uses CPU SDPA on CPU; this does not execute a cuDNN GPU kernel.
The test inputs and attention object fields are small fixtures, not a full DiT.

## Real-weight numerical capture and replay

The validation scripts use the frozen step600 EMA merged SHA256
`131703bc616164df65c1290da1497c153a56d3905c6d025225252ca86ed5c802`.
They have CPU transport/comparator tests; real-weight GPU execution is pending.
Run the reference and native stages in separate processes and environments.
The reference requires the parent's Raven `common` and `projects.minimax_h3`
imports. Native replay requires the installed native SGLang fork and a prepared
`native-runtime.json` bundle whose asset paths resolve on the execution machine.

First run the parent's existing `infer_ref_raw_latent --check-only` with the
intended artifact, input, prompt and codec options. Keep those exact arguments
for capture, remove `--check-only`, and use a new reference output directory.
`--run-config reference56.yaml` is forwarded through the parent's real settings
parser; do not repeat YAML fields as CLI arguments. A staged layout must expose
`projects/minimax_h3` from the integration root before Raven's older copy, and
`common` from that project's pinned `frameworks/raven`. Include the project's
`data/` directory. Check the imported `raw_latent.__file__` before GPU execution.
Use the h3 encoder, posterior mean and independent C/4D/clean-H configuration.
Capture rejects delayed H/C, first-D fusion, CUDA Graph and a different recipe
or merged checkpoint. The smallest input is 56 RGB frames: three ordinary17
packets and EOS5. It produces17 latent frames and three windows. Use260 RGB
frames, 15 ordinary packets plus EOS5, for45/70 boundaries and recent pruning.

The following commands are for the execution owner. Set `REF_ARGS` to the
checked reference command's argument array, `ORACLE_GPUS` to its verified UP
group size, and `NATIVE_GPUS` to the prepared bundle's `num_gpus`. The two
topologies can differ. `SCRIPTS` is this directory's absolute path.

```bash
torchrun --standalone --nproc-per-node="$ORACLE_GPUS" \
  "$SCRIPTS/reference_capture.py" --output "$ORACLE_TRACE" -- "${REF_ARGS[@]}"
PYTHONPATH="$NATIVE_ROOT/python:$PYTHONPATH" \
torchrun --standalone --nproc-per-node="$NATIVE_GPUS" \
  "$SCRIPTS/native_replay.py" --bundle "$NATIVE_BUNDLE" \
  --reference "$ORACLE_TRACE" --output "$NATIVE_TRACE"
python "$SCRIPTS/numerical_trace.py" "$ORACLE_TRACE" "$NATIVE_TRACE" \
  --report "$NEW_REPORT_JSON"
```

Every rank writes a new `rankNNNN` directory. `complete.json` registers each
tensor file's SHA256 and head range. The files load with
`torch.load(weights_only=True)` and contain tensors plus primitive metadata.

| Record | Content |
| --- | --- |
| `inputs/condition`, `inputs/image`, `inputs/proxyNNNN` | Exact preprocessed uint8 RGB, prompt and EOS flags |
| `encoder/condition`, `encoder/image`, `encoder/proxyNNNN` | All Qwen rows/tags and normalized posterior means |
| `inputs/static`, `inputs/sourceNNNN` | Actual reference features and first-frame/reference/target/clean noise |
| `windowNNNN/d0` through `d3` | Input xt, sigma and actual x0 prediction |
| `windowNNNN/output` | Canonical plan and final Euler latent |
| `windowNNNN/writeN/layerLL` | Full logical owners, row_start and appended post-RoPE K/V for every layer and rank-local head range |

The native encoder results are compared separately. Native DiT always consumes
the frozen reference features and noise. Its x0 conversion preserves
`xt.bfloat16().float() + (1 - (1.0 - sigma.float())) * velocity` exactly.
Each next D input and the final window output expose the Euler states without
implementing another reference sampler. QKV layout is bound by the merged asset
identity and native bundle config; all50 layers' clean K/V provide its numerical
check. Head ranges use actual TP and Ulysses ranks. Owners must agree in row order
before KV values are compared. Four-D read-only checks hash only committed rows,
so legitimate arena tail writes remain allowed.
Each C/H write also checks retained owner blocks against the preceding commit's
SHA256. It rejects changes to any retained historical bytes, then stores only
new rows. This proves reuse of previously compared K/V and avoids repeatedly
dumping immutable history. A failed history guard aborts capture and leaves it
incomplete; it does not claim numerical parity.

The comparator rejects incomplete ranks, changed payloads, missing encoder
results, missing D calls, missing C/H commits and incomplete head/layer coverage.
It reports byte equality, dtype/shape, finite status and maximum/mean/RMS absolute
error. Exit0 requires every comparison to be exact and finite; numerical
differences return1 with their report. Invalid records raise an error. There is
no tolerance option. The report is created exclusively and cannot overwrite an
existing report.

Reserve disk before capture. Appended clean-KV dumps cost the sum of new commit
row counts multiplied by50 layers ×56 heads ×128 dimensions ×2 K/V tensors ×2
BF16 bytes, across all ranks, per run. At768x1344 with256x448 proxies, the56-frame
sample has20,048 + L appended rows, where L is the actual Qwen token count.
Its KV payload is28.74 GB +1,433,600×L bytes per run. Full snapshots would use
109.50 GB +8,601,600×L bytes. These are source-derived BF16 payload estimates,
excluding inputs, D states and serialization metadata. Replay writes another
complete run. Hashing and
tensor dumps add CPU transfers and synchronization; these scripts provide no
performance evidence. RGB decoding, serving integration and uninstrumented
topology timing remain separate checks.

```bash
python -m pytest -q test/registered/unit/multimodal_gen/test_h3_numerical_trace.py
```
## First-forward diagnosis

`first_forward_capture.py` reuses a completed independent corpus and the original
model loaders, packing and forward methods. It loads only the DiT. It does not
load Qwen or VAE, generate another sampling grid, run another D step, or commit
KV. It runs **one complete original 50-layer bootstrap D0 forward** on every
rank and observes only the packed inputs, embedding/refiner and block 0 through
raw QKV. Completing that forward preserves the causal model's all-blocks
contract and permits normal distributed cleanup without exception-based stops.
The output of this diagnostic forward is discarded.

Use the same reference environment and `PYTHONPATH` as the accepted
`reference_capture.py` run. `REF_ARTIFACT` must be the exact original artifact
file. The reference probe selects its resident BF16/declared-FP32 placement and
rejects a corpus with active reference inference flags. The current independent
capture has no such flags. The native probe uses the existing bundle's
TransformerLoader settings and causal class override. Neither probe changes
precision, fusion, backend selection or comparison tolerances.
The reference build does not call `configure_inference`, whose default metadata,
KV and AdaLN settings are enabled. Its receipt records the effective flags and
all seven explicit false DiT options. This matches `reference-56.yaml`'s
resident placement and the original capture's empty effective-flags receipt.

The probe runs from a checkout of this repository, which provides
`first_forward_capture.py`, `numerical_trace.py` and the frozen `recipe.json`, against
the already installed production sources. Record their SHA256 values together
with the runtime source hashes. No framework reinstall is required.

The execution owner must apply GPU isolation before **each** capture and restore
the previous monitor state afterwards. With `SCRIPTS`, `ORACLE_TRACE`,
`ORACLE_GPUS`, `NATIVE_ROOT`, `NATIVE_BUNDLE` and `NATIVE_GPUS` set as above, choose
fresh output directories and a fresh report path:

```bash
torchrun --standalone --nproc-per-node="$ORACLE_GPUS" \
  "$SCRIPTS/first_forward_capture.py" capture --kind reference \
  --artifact "$REF_ARTIFACT" --reference "$ORACLE_TRACE" --output "$FIRST_REF"
PYTHONPATH="$NATIVE_ROOT/python:$PYTHONPATH" \
torchrun --standalone --nproc-per-node="$NATIVE_GPUS" \
  "$SCRIPTS/first_forward_capture.py" capture --kind native \
  --bundle "$NATIVE_BUNDLE" --reference "$ORACLE_TRACE" --output "$FIRST_NATIVE"
python "$SCRIPTS/first_forward_capture.py" compare "$FIRST_REF" "$FIRST_NATIVE" \
  --report "$FIRST_REPORT"
```

The reference hook sits inside the x0 wrapper, at the raw DiT, so `packed/*`
contains the actual H3 timesteps and already noised clean context. `xt` and
`sigma` come directly from `window0000/d0`; static features and source noises
come from their hashed corpus records. No native encoder result enters this
comparison. Each rank writes a TraceWriter completion receipt after the forward,
then synchronizes and destroys its distributed groups.

| Events | Coordinates and meaning |
| --- | --- |
| `packed/*` | Actual global tensors and normalized attention/refiner sequence metadata |
| `embed/condition_projection`, `embed/refiner`, `embed/time` | Full output dimensions, replicated across UP/TP where applicable |
| `embed/time_proj_in` | Time GEMM output with an explicit TP output-column range |
| `embed/video_input`, `embed/video_projection` | Rows indexed by the ordinal in `packed/img_pos_info`, not by packed row number |
| `block0/input`, `norm1`, `modulated`, `combined_indices`, `rope_*` | UP-local packed rows with an explicit global row interval |
| `block0/adaln/*` | Six gathered AdaLN parameter tables before indexing |
| `block0/raw_q`, `raw_k`, `raw_v` | Pre-QKNorm, pre-RoPE, pre-USP tensors with explicit row and TP-head intervals |
| `weights/*` | Actual first-block parameter values/dtypes and RoPE inverse frequencies; saved only at UP rank 0 per TP shard |

Every tensor record holds `value` and `axes`. An axis entry declares its tensor
axis, global start and global length. Unlisted axes cover their full dimension.
Before USP, native TP2/U4 raw QKV has `S/4` rows and 28 heads per Q/K/V;
reference TP1/U8 has `S/8` rows and 56 heads. QKV weights and activations split
Q/K/V before head assembly. MLP fc1 splits gate/up before assembly. Output
projection and fc2 weights split their input columns; AdaLN weights split output
rows. Replicas must agree. Missing ranges, inconsistent replicas, changed hashes
or absent required stages reject the trace.

Hooks immediately copy through TraceWriter because later operations can mutate
norm, residual and QKV storage. When actual norm/modulation fusion bypasses the
norm module, `norm1_observed` is false and comparison marks that stage unavailable.
It does not recompute an eager norm or disable fusion. The receipt records actual
native FA version/backend, reference FA availability/selection, package versions,
source hashes, all parameter dtypes, fusion state, TF32, autocast and matmul
precision. Runtime metadata is included in the report separately from tensors.
Native batched AdaLN bypasses the projection module hook. The block pre-hook
captures its actual supplied `adaln_params`, already gathered across TP; ordinary
execution uses the projection output hook. Both paths require all six tables.
Missing-event errors list the absent keys and leave the run incomplete.

Comparison requires exact finite values and returns 1 for a difference, including
an unavailable standalone norm stage. It reports shape/dtype and max/mean/RMS
absolute error without a tolerance option. It is a localization artifact, not
performance evidence or a replacement for the full numerical gate. Reserve disk
for the packed and first-layer activations plus first-layer weights; replicated
global packed/embedding events are retained per rank for consistency checks.

```bash
python -m pytest -q test/registered/unit/multimodal_gen/test_h3_first_forward_capture.py
```

CPU tests exercise the real hooks and TraceWriter with small synthetic modules,
including in-place mutation, cleanup, TP logical matrix layout, different UP
row partitions, empty tensors, missing stages and the comparator CLI. They do
not certify Linux model imports, real weights or GPU execution.
