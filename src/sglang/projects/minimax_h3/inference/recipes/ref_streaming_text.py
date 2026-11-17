"""Raw complete-text adapters for the two inference-only history protocols."""
import hashlib

import torch

from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
from projects.minimax_h3.inference.streaming_text import AppliedPrompt, PromptedLatent, own_document, validate_prompt
from projects.minimax_h3.modeling.ref_recipe import (FIXED_STREAMING_TEXT_RECIPE,
    BIDIR_STREAMING_TEXT_RECIPE, LEGACY_CONDITIONING_MODE, STATIC_CONDITIONING_MODE)


def _stream(parent, backbone, static_condition, chunks, *, encode_text, first_frame, **kwargs):
    if not callable(encode_text):
        raise ValueError('streaming text requires encode_text(full_instruction)')
    if (not isinstance(first_frame, torch.Tensor) or first_frame.ndim != 4 or first_frame.shape[1] != 1
            or min(first_frame.shape) < 1 or not first_frame.is_floating_point() or not torch.isfinite(first_frame).all()):
        raise ValueError('streaming text requires a finite first_frame image latent')
    static = own_document(static_condition)
    first_frame = first_frame.detach().clone()
    text, document, applied = None, None, None

    def encoded_chunks():
        nonlocal text, document, applied
        ended = False
        for k, item in enumerate(chunks):
            if ended:
                raise ValueError('received a chunk after EOS')
            if not isinstance(item, dict) or set(item)-{'references', 'eos', 'prompt'} or 'references' not in item:
                raise ValueError('raw text chunks require references and only optional prompt/eos')
            if 'eos' in item and type(item['eos']) is not bool:
                raise ValueError('eos must be a bool')
            update = item.get('prompt')
            if update is not None:
                validate_prompt(update)
            if update is None and text is None:
                raise ValueError('chunk 0 requires a complete nonblank prompt')
            refs = item['references']
            if not isinstance(refs, (list, tuple)) or not refs:
                raise ValueError('at least one semantic reference is required')
            owned = []
            for ref in refs:
                if not isinstance(ref, dict) or not {'kind', 'role', 'video'} <= ref.keys():
                    raise ValueError('references require kind, role and video')
                video = ref['video']
                if (not isinstance(video, torch.Tensor) or video.ndim != 4 or min(video.shape) < 1
                        or not video.is_floating_point() or not torch.isfinite(video).all()):
                    raise ValueError('reference video must be a finite nonempty floating latent')
                owned.append(dict(ref, video=video.detach().clone()))
            if update is not None and update != text:
                document = own_document(encode_text(update), static=static)
                text = update
                applied = AppliedPrompt(text, hashlib.sha256(text.encode('utf-8')).hexdigest(), k,
                                        len(document['prompt_embeds']))
            ended = item.get('eos', False)
            yield dict(references=owned, eos=ended, condition=document)

    for video in parent(backbone, static, encoded_chunks(), first_frame=first_frame, **kwargs):
        yield PromptedLatent(video, applied)


class RefFixedHistoryStreamingText(RefFixedHistory):
    PROTOCOL = FIXED_STREAMING_TEXT_RECIPE

    def _condition_options(self):
        return dict(super()._condition_options(), conditioning_mode=LEGACY_CONDITIONING_MODE)

    def stream_latents(self, backbone, static_condition, chunks, *, encode_text, first_frame=None,
                       conditioning_mode=STATIC_CONDITIONING_MODE, **kwargs):
        return _stream(super().stream_latents, backbone, static_condition, chunks, encode_text=encode_text,
                       first_frame=first_frame, conditioning_mode=conditioning_mode, **kwargs)


class RefBidirHistoryStreamingText(RefBidirHistory):
    PROTOCOL = BIDIR_STREAMING_TEXT_RECIPE

    def _condition_options(self):
        return dict(super()._condition_options(), conditioning_mode=LEGACY_CONDITIONING_MODE)

    def stream_latents(self, backbone, static_condition, chunks, *, encode_text, first_frame=None,
                       conditioning_mode=STATIC_CONDITIONING_MODE, **kwargs):
        return _stream(super().stream_latents, backbone, static_condition, chunks, encode_text=encode_text,
                       first_frame=first_frame, conditioning_mode=conditioning_mode, **kwargs)
