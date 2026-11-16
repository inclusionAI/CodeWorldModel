"""Serving-only Qwen placement; frozen encoder source identity stays unchanged."""
from pathlib import Path

import torch

from .cwm_encoders import qwen_inputs


@torch.no_grad()
def encode_prepared(qwen, device, prepared, *, reshard=False):
    kwargs, tags = prepared
    kwargs = {k: v.to(device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
              for k, v in kwargs.items()}
    hidden = qwen(**kwargs, use_cache=False, logits_to_keep=1)[0]
    if reshard:
        # Inference has no backward to release root weights retained by FSDP.
        qwen.reshard()
    if len(hidden) != len(tags):
        raise ValueError('Qwen live rows and presentation tags disagree')
    return {'prompt_embeds': hidden.cpu().contiguous(), 'text_token_tags': tags}


class FrozenQwenEncoder:
    def __init__(self, checkpoint, *, device='cuda', mesh=None):
        from transformers import AutoConfig, AutoProcessor
        from projects.minimax_h3.modeling.text_encoder import MiniMaxH3TextEncoder
        self.device = torch.device(device)
        self._qwen_mesh = mesh
        if mesh is not None and (mesh.device_type != 'cuda' or mesh.ndim != 1
                                 or mesh.mesh.tolist() != [0, 1]):
            raise ValueError('Qwen FSDP requires the CUDA mesh on global ranks [0, 1]')
        root = Path(checkpoint)
        self.processor = AutoProcessor.from_pretrained(root/'processor')
        config = AutoConfig.from_pretrained(root/'text_encoder')
        config.text_config.num_hidden_layers = 50
        with torch.device('cpu'):
            self.qwen = MiniMaxH3TextEncoder.from_pretrained(str(root/'text_encoder'),hidden_layer=50,
                config=config,dtype=torch.bfloat16,attn_implementation='flash_attention_2').eval().requires_grad_(False)
        if mesh is None:
            self.qwen.to(self.device)
        else:
            from torch.distributed.fsdp import fully_shard
            if any(p.device.type != 'cpu' for p in self.qwen.parameters()):
                raise RuntimeError('Qwen must load on CPU before FSDP placement')
            blocks = {'Qwen3VLTextDecoderLayer': 0, 'Qwen3VLVisionBlock': 0}
            for module in list(self.qwen.modules()):
                name = type(module).__name__
                if name in blocks:
                    fully_shard(module, mesh=mesh, reshard_after_forward=True)
                    blocks[name] += 1
            if blocks['Qwen3VLTextDecoderLayer'] != 50 or not blocks['Qwen3VLVisionBlock']:
                raise RuntimeError(f'unexpected Qwen FSDP block counts: {blocks}')
            fully_shard(self.qwen, mesh=mesh, reshard_after_forward=True)

    def prepare_condition(self, prompt, videos=(), images=()):
        return qwen_inputs(self.processor,prompt,videos,images)

    def encode_condition(self, prepared):
        return encode_prepared(self.qwen, self.device, prepared, reshard=getattr(self, '_qwen_mesh', None) is not None)

    def condition(self, prompt, videos=(), images=()):
        return self.encode_condition(self.prepare_condition(prompt,videos,images))
