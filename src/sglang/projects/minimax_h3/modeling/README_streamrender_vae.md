# StreamRender H3 video codec

`StreamRenderH3VideoVAE` combines the native TAE-H3 encoder with the
StreamRender causal ViT24 decoder. Existing H3 VAE configuration and CWM
cache preparation are unchanged. This is the PyTorch reference decoder;
it does not include the StreamRender FA4/CUDA deployment optimizations.

## Load and configure

Three local assets are required: the released TAE state dictionary, the
causal ViT student safetensors, and its decoder YAML/JSON architecture config.

The config argument expects the decoder architecture, not the full runtime
JSON. To use the released runtime JSON directly, extract its nested dict:

```python
import json
from pathlib import Path
assets = Path("/path/to/StreamRender-H3")
runtime = json.loads((assets / "configs/runtime.json").read_text())
decoder_config = runtime["inference"]["models"]["video_decoder"]["args"]["decoder_config"]
# Pass decoder_config above, assets / "tae/taeh3_official.pth", and
# assets / "causal_decoder/causal_vit24_step6000.safetensors" to the constructor.
```

The constructor loads both checkpoints strictly, without loading the full
H3 VAE teacher. TAE's small original decoder stays in its checkpoint module
for strict loading; decoding through this adapter uses the causal ViT.

```python
import torch
from projects.minimax_h3.modeling.streamrender_vae import StreamRenderH3VideoVAE

codec = StreamRenderH3VideoVAE(
    tae_weights="/shared/models/taeh3.safetensors",
    decoder_weights="/shared/models/causal_vit24.safetensors",
    decoder_config="/shared/models/causal_vit24.yaml",
).to("cuda").eval()

# pixels: [N,T,3,H,W], float RGB in [0,1]; H/W are divisible by 16.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    z = codec.encode_video(pixels)                    # [N,L,24,H/16,W/16]
    rgb = codec.decode_base(z.transpose(1, 2))        # [N,3,T,H,W]
```

For a new trial, replace only that trial's `models.video_vae` node with
the following fragment and set the three asset paths. Constructor loading
requires `meta_init: false`; omit the separate `weight` node.

```yaml
video_vae:
  module: projects.minimax_h3.modeling.streamrender_vae
  class_name: StreamRenderH3VideoVAE
  meta_init: false
  args:
    tae_weights: /shared/models/taeh3.safetensors
    decoder_weights: /shared/models/causal_vit24.safetensors
    decoder_config: /shared/models/causal_vit24.yaml
  runtime:
    training: false
    requires_grad: false
    param_dtype: float32
    plugins:
      - module: common.plugin.autocast
        class_name: AutocastNoGradWrapper
        functions: [decode_base, encode_images, encode_videos]
        grad_enabled: false
        autocast:
          enabled: true
          dtype: bfloat16
  placement:
    device: cuda
```

Keep the decoder parameters FP32 and use CUDA autocast for BF16/FP16
compute; do not call `codec.half()` or `codec.bfloat16()`. Full-model
FSDP and mixed storage dtypes are not configured by this fragment.
The runtime plugin enters BF16 autocast and no-grad for each listed model
method, including RAVEN's direct `decode_base` calls. It does not wrap the
objects returned by `create_stream`; use the caller autocast context shown
in the stream example below. The underlying class retains gradients through
`encode_image`, `encode_video`, `decode_base` and `forward`; omit this inference
plugin when those methods are used for training. List-based encoding and
streaming methods are inference APIs.

Save the host-specific model node and extracted architecture outside the
repository, then use the model node in a new trial as follows (the file contains
only that node):

```yaml
models:
  video_vae:
    __inherit__: /path/to/raven-streamrender-vae/raven_model.yaml
```

## Layout and normalization

- All public latents are **normalized H3 diffusion latents**. The adapter's
  `latents_mean=0` and `latents_std=1` are intentional. The causal decoder
  performs the original H3 denormalization internally.
- Public pixels are RGB in [0,1]. The public processor is identity/raw;
  RAVEN's `processor.revert_tensor(decode_base(z))` remains valid.
- List-based encoding accepts images/videos like the existing VAE and
  returns CTHW latents. Direct encoding and streaming use NTCHW.
  `decode_base` and `forward` accept NCTHW.
- Native 17-frame blocks map to five latents; a final five-frame tail maps
  to two latents. Image encoding produces one latent. Whole-video
  encoding accepts 17n or 17n+5; it does not silently trim another length.
- Decoder attention uses two-latent chunks and four cached latents.
  Each latent emits four RGB frames except indices 0,5,10,..., which emit
  one. It is causal between two-latent chunks, not between both latents
  inside a chunk. An odd final latent is duplicated for the final call;
  only the original latent's frames are retained.
- `frame_num` on `decode_base` keeps the trailing requested frames, as
  in the original H3 interface. It is not a streaming prefix selector.

## Independent streams

```python
stream = codec.create_stream()
rgb_parts = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for pixels_now in pixel_fragments:                # [N,t,3,H,W]
        z_now = stream.encode(pixels_now)              # NTCHW or None
        if z_now is not None:
            rgb_now = stream.decode(z_now)            # NTCHW or None
            if rgb_now is not None:
                rgb_parts.append(rgb_now)
    stream.flush_encoder()                            # no synthetic input frames
    last = stream.flush_decoder()                     # emits pending odd latent
    if last is not None:
        rgb_parts.append(last)
rgb = torch.cat(rgb_parts, dim=1)
```

Each stream owns separate encoder/decoder state. Keep batch, resolution,
device and dtype fixed until `reset()`; create/reset streams between
unrelated videos. Encoder state resets at each native 17-frame boundary.
An incomplete final encoder group is ignored, matching the upstream
streaming contract. Decoder flush ends its stream; reset before decoding
another sequence.

Changing from the full H3 encoder to TAE changes the conditioning latents.
Existing H3 caches remain valid for decoder-only comparisons, but do not
relabel them as TAE caches. CWM's `FrozenCWMEncoders` still loads the original
VAE and its posterior-specific internal methods; selecting this model node
does not change that data pipeline or provide a VAE training loop.
