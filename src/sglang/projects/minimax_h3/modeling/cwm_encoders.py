"""Frozen encoders shared by offline CWM processing and online chunk arrival."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def qwen_inputs(processor, prompt, videos=(), images=()):
    """Native Ref2VA presentation: <Picture N> images, then 24 -> 2 FPS videos, with live row tags.

    Matches DiffSynth's presentation_ref2va with image_token_counts and
    video_token_counts. Every call is a separate Qwen document: never split
    full-video hidden states into chunks.
    """
    tokenizer = processor.tokenizer
    ids, tags, timestamps, sampled = [], [], [], []
    merge = int(processor.image_processor.merge_size)**2
    def text(value):
        rows = tokenizer(value,add_special_tokens=False)['input_ids']
        ids.extend(rows); tags.extend([1]*len(rows))
    def vision(pad, count):
        rows = [tokenizer.convert_tokens_to_ids('<|vision_start|>')]
        rows += [tokenizer.convert_tokens_to_ids(pad)] * count
        rows += [tokenizer.convert_tokens_to_ids('<|vision_end|>')]
        ids.extend(rows); tags.extend([0]*len(rows))
    extra = {}
    if len(images):
        pictures = processor.image_processor(images=list(images),return_tensors='pt')
        extra.update({k:pictures[k] for k in ('pixel_values','image_grid_thw')})
        for i,g in enumerate(pictures['image_grid_thw']):
            text(f'<Picture {i+1}>: ')
            vision('<|image_pad|>',int(g.prod())//merge)
    for frames in videos:
        indices = list(range(0,len(frames),12))
        if not indices:
            raise ValueError('empty proxy video')
        # HF's video smart_resize rejects a single frame before its internal
        # temporal padding runs (the 5-frame EOS chunk samples just one).
        sampled_indices = indices + [indices[-1]] * (len(indices)%2)
        sampled.append(np.stack([frames[i] for i in sampled_indices]))
        ts = [i/24 for i in indices]
        ts += [ts[-1]] * (len(ts)%2)
        timestamps.append([(a+b)/2 for a,b in zip(ts[::2],ts[1::2])])
    if sampled:
        clips = processor.video_processor(videos=sampled,do_sample_frames=False,return_tensors='pt')
        extra.update({k:clips[k] for k in ('pixel_values_videos','video_grid_thw')})
        for i,(g,ts) in enumerate(zip(clips['video_grid_thw'],timestamps)):
            if int(g[0]) != len(ts):
                raise ValueError('Qwen temporal patch count differs from presentation')
            text(f'<Video {i+1}>: ')
            for t in ts:
                text(f'<{t:.1f} seconds>')
                vision('<|video_pad|>',int(g[1]*g[2])//merge)
    text(prompt)
    input_ids = torch.tensor([ids],dtype=torch.long)
    token_types = torch.zeros_like(input_ids,dtype=torch.int32)
    token_types[input_ids==tokenizer.convert_tokens_to_ids('<|image_pad|>')] = 1
    token_types[input_ids==tokenizer.convert_tokens_to_ids('<|video_pad|>')] = 2
    return dict(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=token_types,**extra),torch.tensor(tags,dtype=torch.long)


class FrozenCWMEncoders:
    def __init__(self, checkpoint, *, device='cuda', posterior='mean', load_text_encoder=True):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor
        import yaml
        from projects.minimax_h3.modeling.build import MiniMaxH3VideoVAE
        from projects.minimax_h3.modeling.text_encoder import MiniMaxH3TextEncoder
        self.device, self.posterior = torch.device(device), posterior
        root = Path(checkpoint)
        cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/video_vae.yaml').read_text())
        with init_empty_weights():
            self.vae = MiniMaxH3VideoVAE(**cfg)
        state = load_file(str(root/'video_vae/source/model.safetensors'))
        self.vae.load_state_dict(state,assign=True,strict=True)
        del state
        self.vae = self.vae.to(device=self.device,dtype=torch.bfloat16).eval().requires_grad_(False)
        self.mean = torch.tensor(cfg['latents_mean'],device=self.device).view(1,-1,1,1,1)
        self.std = torch.tensor(cfg['latents_std'],device=self.device).view(1,-1,1,1,1)
        if load_text_encoder:
            self.processor = AutoProcessor.from_pretrained(root/'processor')
            config = AutoConfig.from_pretrained(root/'text_encoder')
            config.text_config.num_hidden_layers = 50
            self.qwen = MiniMaxH3TextEncoder.from_pretrained(str(root/'text_encoder'),hidden_layer=50,
                config=config,dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def condition(self, prompt, videos=(), images=()):
        kwargs,tags = qwen_inputs(self.processor,prompt,videos,images)
        kwargs = {k:v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                  for k,v in kwargs.items()}
        hidden = self.qwen(**kwargs,use_cache=False,logits_to_keep=1)[0]
        if len(hidden) != len(tags):
            raise ValueError('Qwen live rows and presentation tags disagree')
        return {'prompt_embeds':hidden.cpu().contiguous(),'text_token_tags':tags}

    def _latent(self, frames, *, seed, pad_to=None):
        pixels = torch.as_tensor(np.array(frames),device=self.device).permute(0,3,1,2).float()/255
        pixels = self.vae.processor.transform(pixels).permute(1,0,2,3).unsqueeze(0).to(torch.bfloat16)
        if pad_to and pixels.shape[2] < pad_to:
            pixels = torch.cat([pixels,pixels[:,:,-1:].expand(-1,-1,pad_to-pixels.shape[2],-1,-1)],2)
        mean,logvar = self.vae._adaptive_encode(pixels).float().chunk(2,dim=1)
        if self.posterior == 'sample':
            generator = torch.Generator(device=self.device).manual_seed(seed)
            mean = mean + torch.exp(.5*logvar.clamp(-30,20))*torch.randn(mean.shape,device=self.device,generator=generator)
        elif self.posterior != 'mean':
            raise ValueError('posterior must be mean or sample')
        return ((mean-self.mean)/self.std)[0].cpu().to(torch.bfloat16)

    @torch.no_grad()
    def encode_chunk(self, frames, *, eos=False, seed=0):
        """17 non-EOS frames -> 5 latents; final 5 padded frames -> 2.

        The VAE's temporal encoder treats blocks independently. Calling its
        low-level block encoder avoids incorrectly applying token_drop to each
        non-EOS chunk. Frozen posterior means are the deterministic default.
        """
        if len(frames) != 17 and not eos:
            raise ValueError('non-EOS proxy chunks must contain 17 frames')
        if not 1 <= len(frames) <= 17:
            raise ValueError('chunk frame count must be in [1,17]')
        latent = self._latent(frames,seed=seed,pad_to=17)
        if latent.shape[1] != 5:
            raise ValueError('VAE block encoder must return five temporal latents')
        return latent[:,:-3] if eos else latent

    @torch.no_grad()
    def image(self, frame, *, seed=0):
        """One RGB frame -> one latent through the encoder's native single-image path."""
        latent = self._latent([frame],seed=seed)
        if latent.shape[1] != 1:
            raise ValueError('single-frame encode must return one latent')
        return latent

    def video(self, frames, *, seed=0):
        if len(frames)%17 != 5:
            raise ValueError('native CWM windows must have 17*n+5 frames; no silent trimming')
        return torch.cat([self.encode_chunk(frames[a:a+17],eos=a+17>=len(frames),seed=seed+a)
                          for a in range(0,len(frames),17)],1)

    @torch.no_grad()
    def decode_strict_chunk(self, latent, *, num_frames=17):
        """Current-only right padding, then immutable pixel emission.

        This deliberately differs from the pretrained decoder's future-latent
        overlap. It enforces causality; seam quality needs held-out evaluation.
        Decoder weights/masks are unchanged. No future proxy/latent is consumed.
        """
        if latent.shape[1] not in (2,5) or num_frames != (5 if latent.shape[1]==2 else 17):
            raise ValueError('strict decode expects 5 latents/17 frames or a 2 latent/5 frame EOS')
        z = latent.to(self.device).float().unsqueeze(0)*self.std+self.mean
        z = torch.cat([z,z[:,:,-1:].expand(-1,-1,7-z.shape[2],-1,-1)],2)
        pixels = self.vae.decode_base(z.to(torch.bfloat16))[:,:,:num_frames]
        pixels = self.vae.processor.transform_rev(pixels[0].permute(1,0,2,3).float())
        return (pixels.clamp(0,1)*255).byte().permute(0,2,3,1).cpu()

    @torch.no_grad()
    def decode_video(self, latent, *, num_frames, mode='native'):
        if mode == 'strict':
            return torch.cat([self.decode_strict_chunk(latent[:,a:a+5],num_frames=min(17,num_frames-a//5*17))
                              for a in range(0,latent.shape[1],5)])
        if mode != 'native':
            raise ValueError('decode mode must be native or strict')
        z = latent.to(self.device).float().unsqueeze(0)*self.std+self.mean
        pixels = self.vae.decode_base(z.to(torch.bfloat16),frame_num=num_frames)
        pixels = self.vae.processor.transform_rev(pixels[0].permute(1,0,2,3).float())
        return (pixels.clamp(0,1)*255).byte().permute(0,2,3,1).cpu()
