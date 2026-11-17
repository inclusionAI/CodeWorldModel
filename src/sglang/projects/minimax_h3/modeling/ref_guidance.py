"""Partial-condition CFG and fixed-beta GFT for Ref2VA SFT / teacher forcing."""
from __future__ import annotations

from dataclasses import replace
import math
import random

import torch

from common.seed import combine_seed
from .ref_recipe import training_guidance_options


def guidance_options(config):
    return training_guidance_options(config.meta_model, config.data.args.get('stage'))


def weak_inputs(inputs, drop):
    """A private view of validated full inputs; never slice Qwen hidden states.

    Keep the full sequence budget for sampling/weighting, but rebuild geometry
    from the selected documents and references. Target/history tensors and the
    noise of retained references are shared exactly with the full branch.
    """
    if drop not in ('proxy', 'prompt', 'prompt_proxy'):
        raise ValueError('invalid dropped condition')
    samples, layouts, noises = [], [], []
    for sample, layout, eps in zip(inputs.samples, inputs.layouts, inputs.reference_eps, strict=True):
        if not sample['references'] or sample['references'][0]['role'] != 'first_frame':
            raise ValueError('partial guidance requires a first_frame reference')
        condition = sample.get('guidance_conditions', {}).get(drop)
        if condition is None and drop == 'proxy':
            if sample['metadata'].get('caption_mode') == 'provided_static':
                raise ValueError('proxy guidance with a different static prompt needs an encoded weak document')
            static = sample['student_condition']['static']
            condition = dict(teacher_condition=static, student_condition=dict(static=static, chunks=[]))
        if condition is None:
            raise ValueError(f'{drop} guidance needs independently encoded guidance_conditions; '
                             'prepare a separate cache with cwm_prepare encode --guidance-drop')
        teacher, student = condition['teacher_condition'], condition['student_condition']
        chunks = student['chunks']
        expected_chunks = len(sample['student_condition']['chunks']) if drop == 'prompt' else 0
        if len(chunks) != expected_chunks:
            raise ValueError('weak chunk documents do not match the selected conditions')
        width = sample['teacher_condition']['prompt_embeds'].shape[1]
        for doc in (teacher, student['static'], *chunks):
            emb, tags = doc['prompt_embeds'], doc['text_token_tags']
            if emb.ndim != 2 or emb.shape[1] != width or len(emb) == 0 or tags.shape != (len(emb),):
                raise ValueError('invalid independently encoded weak Qwen document')
        kept = list(range(len(eps))) if drop == 'prompt' else [0]
        selected = dict(sample, teacher_condition=teacher, student_condition=student,
                        references=[sample['references'][j] for j in kept])
        samples.append(selected)
        layouts.append(replace(layout, text_len=len(student['static']['prompt_embeds'])))
        noises.append([eps[j] for j in kept])
    return replace(inputs, samples=samples, layouts=layouts, reference_eps=noises, geometry={})


class RefSamplingGuidance:
    def _sampling_inputs(self, inputs):
        return weak_inputs(inputs, self.guidance['drop']) if self._guidance_scale() != 1 else None

    def _sampling_forward(self, backbone, inputs, *, weak=None, weak_cache=None, **kwargs):
        native = kwargs.pop('native', False)
        forward = self._bidirectional_forward if native else self._ref_forward
        positive = forward(backbone, inputs, **kwargs)
        if weak is None:
            return positive
        negative = forward(backbone, weak, **(kwargs if native else dict(kwargs, cache=weak_cache)))
        if positive is None:  # Both caches commit the same RGB history, separately.
            return None
        scale = self._guidance_scale()
        return ([u.float()+scale*(p.float()-u.float()) for p,u in zip(positive[0],negative[0],strict=True)],
                positive[1])


class RefGuidanceMixin(RefSamplingGuidance):
    def __init__(self, config):
        self.guidance = guidance_options(config)
        super().__init__(config)

    def _guidance_forward(self, ctx, forward, **kwargs):
        cfg = getattr(self, 'guidance', {'mode': 'plain'})
        backbone, inputs = ctx['models']['backbone'], ctx['inputs']
        if cfg['mode'] == 'plain':
            return forward(backbone, inputs, **kwargs)
        weak = weak_inputs(inputs, cfg['drop'])
        # The engine supplies a shared RNG after SP sync. Use a separate stream
        # so changing guidance never changes the target noise or timesteps.
        rng = random.Random(combine_seed(ctx['rng'].seed, 'cwm_proxy_cfg_dropout_v1', cfg['seed']))
        mask = [rng.random() < cfg['dropout'] for _ in inputs.samples]
        selected = replace(inputs,
            samples=[w if d else f for d, f, w in zip(mask, inputs.samples, weak.samples)],
            layouts=[w if d else f for d, f, w in zip(mask, inputs.layouts, weak.layouts)],
            reference_eps=[w if d else f for d, f, w in zip(mask, inputs.reference_eps, weak.reference_eps)],
            geometry={})
        auxiliary = None
        if cfg['mode'] == 'gft' and cfg['beta'] < 1:
            # Always execute on every rank, even if this rank dropped all samples:
            # FSDP collectives must have the same forward-call order everywhere.
            with torch.no_grad():
                auxiliary = forward(backbone, weak, **kwargs)[0]
        video, audio = forward(backbone, selected, **kwargs)
        if auxiliary is not None:
            # Dropped samples train the weak function with its original FM loss.
            # Only full-condition samples use GFT; no extra 1/beta^2 rescaling.
            video = [p if d else cfg['beta']*p.float()+(1-cfg['beta'])*u.detach().float()
                     for p, u, d in zip(video, auxiliary, mask, strict=True)]
        ctx.setdefault('metrics', {}).update({'guidance/dropout_fraction': sum(mask)/len(mask),
                                              'guidance/beta': cfg['beta'] if cfg['mode']=='gft' else 1.0})
        return video, audio

    def _guidance_scale(self):
        # getattr also preserves test/probe objects created with __new__.
        if not hasattr(self, 'guidance'):
            return 1.0
        cfg = self.guidance
        # Historical run-local CFG subclasses own their sampling/validation;
        # their old validation.cfg_scale must not change the base plain path.
        scale = float(self.config.validation.get('cfg_scale', cfg['scale'])
                      if 'guidance' in self.config.meta_model else 1.0)
        if not math.isfinite(scale) or scale < 0 or (cfg['mode'] != 'cfg' and scale != 1):
            raise ValueError('invalid CFG scale; plain/GFT sampling requires scale=1')
        return scale
