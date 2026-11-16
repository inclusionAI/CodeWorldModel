"""Bounded Ref2VA acceptance checks; GPU commands must be scheduled explicitly.

contracts: CPU cache invariants, including rejection of stale conditioning.
models: real reduced-size H3 forwards, optimizers and streaming (one GPU).
encoders: re-encode one real prepared window with released Qwen/VAE (one GPU).
This reports numerical correctness, not full-model convergence or visual quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

MODE = 'static_image_qwen_proxy_vae_v1'


def synthetic(seed=1, frames=158, depth=False):
    if frames % 17 != 5:
        raise ValueError('closed clips require 17*n+5 frames')
    g = torch.Generator().manual_seed(seed)
    t = 5 * ((frames + 16) // 17) - 3
    rand = lambda *shape: torch.randn(*shape, generator=g)
    condition = dict(prompt_embeds=rand(6, 32).bfloat16(),
                     text_token_tags=torch.tensor([1, 0, 0, 1, 1, 1]))
    refs = [dict(kind='image', role='first_frame', video=rand(24, 1, 4, 4))]
    refs += [dict(kind='video', role=r, video=rand(24, t, 4, 4))
             for r in (('semantic', 'depth') if depth else ('semantic',))]
    return dict(video=rand(24, t, 4, 4), audio=torch.empty(2, 32, 0), references=refs,
                teacher_condition=condition,
                student_condition=dict(static=condition, chunks=[]),
                metadata=dict(schema='cwm_ref2va_v1', conditioning_mode=MODE,
                              fps=24, normalized=True, condition_seed=seed, num_frames=frames,
                              chunk_raw_ranges=[[a, min(a+17, frames)] for a in range(0, frames, 17)],
                              chunk_latent_ranges=[[a, min(a+5, t)] for a in range(0, t, 5)]))


def contracts(args):
    from projects.minimax_h3.data.ref_latent import (
        STATIC_CONDITIONING_MODE, load_entry, packed_length, read_manifest, validate_sample,
    )
    assert STATIC_CONDITIONING_MODE == MODE
    positive = []
    for frames in (5, 39, 158):
        for depth in (False, True):
            sample = synthetic(frames=frames, depth=depth)
            validate_sample(sample)
            t, rows = sample['video'].shape[1], 4
            references = rows + (2 if depth else 1) * t * rows
            expected_sft = 6 + references + t * rows
            expected_tf = 6 + references + (2*t-2) * rows
            assert packed_length(sample, 'sft') == expected_sft
            assert packed_length(sample, 'tf') == expected_tf
            assert packed_length(sample, 'dmd') == max(expected_sft, expected_tf)
            positive.append(dict(frames=frames, depth=depth, sft_rows=expected_sft, tf_rows=expected_tf))
    sample = synthetic()
    bad = []
    def reject(name, mutate):
        changed = copy.deepcopy(sample)
        mutate(changed)
        try:
            validate_sample(changed)
        except (ValueError, KeyError):
            bad.append(name)
        else:
            raise AssertionError(f'invalid cache accepted: {name}')
    reject('old per-chunk Qwen', lambda s: s['student_condition']['chunks'].append(copy.deepcopy(s['teacher_condition'])))
    reject('old teacher caption/proxy Qwen', lambda s: s.update(teacher_condition=dict(
        prompt_embeds=s['teacher_condition']['prompt_embeds']+1,
        text_token_tags=s['teacher_condition']['text_token_tags'].clone())))
    reject('missing first image', lambda s: s['references'].pop(0))
    reject('proxy temporal misalignment', lambda s: s['references'][1].update(video=s['references'][1]['video'][:, :-1]))
    reject('wrong FPS', lambda s: s['metadata'].update(fps=30))
    reject('wrong EOS length', lambda s: s['metadata'].update(num_frames=157))
    reject('unexpected condition mode', lambda s: s['metadata'].update(conditioning_mode='unknown'))
    reject('invalid Qwen tag', lambda s: s['student_condition']['static']['text_token_tags'].__setitem__(0, 7))
    reject('non-finite proxy', lambda s: s['references'][1]['video'].__setitem__((0, 0, 0, 0), float('nan')))
    checked = []
    if args.manifest:
        path, entries, _ = read_manifest(args.manifest)
        for entry in entries[:args.limit]:
            assert entry.get('conditioning_mode') == MODE
            item = load_entry(path, entry, verify_hash=True)
            validate_sample(item)
            assert item['metadata'].get('conditioning_mode') == MODE
            assert item['metadata']['num_frames'] == 158
            assert not item['student_condition']['chunks']
            for key in ('prompt_embeds', 'text_token_tags'):
                assert torch.equal(item['teacher_condition'][key], item['student_condition']['static'][key])
            checked.append(entry['sample_id'])
    raw = bundle_contracts(args.bundle) if args.bundle else None
    return dict(positive=positive, invalid_cases_rejected=bad, real_cache_samples=checked, raw_bundle=raw)


def tiny_model(causal=True, *, device='cuda'):
    from projects.minimax_h3.modeling.transformer.config import MiniMaxH3DiTConfig, MiniMaxH3DiTArchConfig
    from projects.minimax_h3.modeling.transformer.model import MiniMaxH3DiTModel
    from projects.minimax_h3.modeling.transformer.causal_model import CausalMiniMaxH3DiTModel
    from projects.minimax_h3.modeling.transformer.x0_model import MiniMaxH3X0Model
    cfg = MiniMaxH3DiTConfig(MiniMaxH3DiTArchConfig(
        num_layers=2, token_refiner_num_layers=1, hidden_size=128,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=256,
        text_dim=32, timestep_input_dim=64, time_embed_hidden_size=128,
        time_embed_dim=64, adaln_out_features=18*128, final_adaln_out_features=2*128))
    cls = CausalMiniMaxH3DiTModel if causal else MiniMaxH3DiTModel
    torch.manual_seed(7)
    net = MiniMaxH3X0Model(cls(config=cfg, hf_config={})).to(device)
    with torch.no_grad():
        net.dit.rope.inv_freq.copy_(10000.0 ** (-torch.arange(16, device=device)/16))
        for name, value in net.named_parameters():
            if value.ndim == 1:
                value.fill_(1 if 'norm' in name and name.endswith('weight') else 0)
            else:
                value.normal_(std=.05)
    net.param_dtype = torch.bfloat16
    return net


def cached_forward(meta, net, inputs, forward_args, audit):
    from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
    cache = GroupedKVCache(meta._num_layers(net), inputs.batch_size, windows=inputs.window_sizes)
    kwargs = dict(forward_args, cache=cache)
    meta._ref_forward(net, inputs, selection=('static', -1), store_kv=True, **kwargs)
    static_rows = [sum(n for group, n in groups if group == -1) for groups in cache.groups]
    parts = [[] for _ in inputs.samples]
    peak_rows = [0 for _ in inputs.samples]
    bounds = [static_rows[i]+(inputs.window_sizes[i]+1)*5*(sum(
        r['video'].shape[-2]*r['video'].shape[-1]//4 for r in sample['references'][1:])
        +sample['video'].shape[-2]*sample['video'].shape[-1]//4)
        for i, sample in enumerate(inputs.samples)]
    for k in range(len(inputs.layouts[0].chunks)):
        cache.begin_chunk(k)
        geometry = meta._ref_geometry(inputs, ('condition', k))
        assert geometry['text_pos'].numel() == 0
        assert geometry['prompt_embeds'].shape[0] == 0
        meta._ref_forward(net, inputs, selection=('condition', k), store_kv=True, **kwargs)
        output, _ = meta._ref_forward(net, inputs, selection=('noise', k), **kwargs)
        for target, value in zip(parts, output):
            target.append(value)
        meta._ref_forward(net, inputs, selection=('history', k), store_kv=True, **kwargs)
        for i, groups in enumerate(cache.groups):
            assert groups[0] == (-1, static_rows[i])
            assert all(group >= k-inputs.window_sizes[i] for group, _ in groups[1:])
            peak_rows[i] = max(peak_rows[i], cache.kvlens[i])
            assert cache.kvlens[i] <= bounds[i], 'KV memory exceeded the configured finite window'
    audit.update(peak_rows=peak_rows, bound_rows=bounds)
    return [torch.cat(part, 1) for part in parts]


def stage_config(directory, stage):
    from common.config import CfgNode
    cfg = CfgNode.from_file(Path(directory)/f'{stage}.yaml')
    assert cfg.data.args.get('conditioning_mode') == MODE, 'use configs generated from a Ref2VA manifest'
    return cfg



def attach_adapter(net, config):
    from common.model.adapters import apply_adapters
    return apply_adapters(dict(name='acceptance', config=config, model=net))['model']

def optimizer_step(net, loss, config):
    from common.optim import build_optimizers
    from common.lr_scheduler import build_lr_schedulers
    configs = dict(backbone=config)
    optimizers = build_optimizers(dict(backbone=net), configs)
    for scheduler in build_lr_schedulers(optimizers, configs).values():
        scheduler.step(0)
    optimizer = optimizers['backbone']
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    total = sum(float(g.float().abs().sum()) for g in grads)
    assert total > 0, 'loss did not train any parameter'
    changed = next(p for p in net.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    before = changed.detach().clone()
    optimizer.step()
    delta = float((before-changed).abs().max())
    assert delta > 0
    net.zero_grad(set_to_none=True)
    return dict(loss=float(loss), gradient_l1=total, parameter_delta_max=delta,
                first_update_lr=optimizer.param_groups[0]['lr'])


def models(args):
    if not torch.cuda.is_available():
        raise RuntimeError('models is an explicitly scheduled GPU check')
    from common.seed import RandomState
    from projects.minimax_h3.meta_models.causal_minimax_h3_base import CausalMiniMaxH3Base
    from projects.minimax_h3.modeling.ref_condition import CWMRefConditionMixin
    from projects.minimax_h3.meta_models.minimax_h3_ref2va import (
        MiniMaxH3RefSFT, MiniMaxH3RefTF, MiniMaxH3RefTSCD, MiniMaxH3RefDMD,
    )
    class Harness(CWMRefConditionMixin, CausalMiniMaxH3Base):
        def __init__(self, window):
            self.config = SimpleNamespace(data=SimpleNamespace(args=dict(
                window_size=window, conditioning_mode=MODE, stage='tf')))
    result = dict(model='real H3 architecture, 2 layers / width 128; randomized weights', parity={})
    net = tiny_model().eval()
    for window in (0, 1, 6):
        meta = Harness(window)
        inputs = meta._build_inputs({'ref_samples': [synthetic(), synthetic(2, depth=True)]})
        contexts = [s['video'] for s in inputs.samples]
        kwargs = dict(video_xts=[torch.randn_like(v) for v in contexts], video_context=contexts,
                      video_eps=[torch.randn_like(v) for v in contexts],
                      video_timesteps=[torch.linspace(.2, .9, 10, device='cuda') for _ in contexts])
        with torch.no_grad():
            packed, _ = meta._ref_forward(net, inputs, **kwargs)
            cache_audit = {}
            cached = cached_forward(meta, net, inputs, kwargs, cache_audit)
            torch.testing.assert_close(packed, cached, atol=.015, rtol=.015)
            altered = copy.deepcopy(inputs.samples)
            for s in altered:
                for ref in s['references'][1:]:
                    ref['video'][:, 5:] += 20
            future_args = {**kwargs, 'video_xts': [v.clone() for v in kwargs['video_xts']],
                           'video_context': [v.clone() for v in contexts]}
            for v in future_args['video_xts'] + future_args['video_context']:
                v[:, 5:] += 20
            future, _ = meta._ref_forward(net, meta._build_inputs({'ref_samples': altered}), **future_args)
            assert all(torch.equal(a[:, :5], b[:, :5]) for a, b in zip(packed, future))
            own_context = [v.clone() for v in contexts]
            for v in own_context:
                v[:, :5] += 20
            own, _ = meta._ref_forward(net, inputs, **dict(kwargs, video_context=own_context))
            assert all(torch.equal(a[:, :5], b[:, :5]) for a, b in zip(packed, own))
            altered = copy.deepcopy(inputs.samples)
            altered[0]['references'][0]['video'] += 20
            picture, _ = meta._ref_forward(net, meta._build_inputs({'ref_samples': altered}), **kwargs)
            assert torch.equal(picture[1], packed[1]), 'cross-sample leakage'
            influence = [float((picture[0][:, a:b]-packed[0][:, a:b]).abs().max())
                         for a, b in inputs.samples[0]['metadata']['chunk_latent_ranges']]
            assert all(v > 0 for v in influence), 'first image ceased to condition a chunk'
            altered = copy.deepcopy(inputs.samples)
            altered[0]['references'][1]['video'] += 20
            proxy_changed, _ = meta._ref_forward(net, meta._build_inputs({'ref_samples': altered}), **kwargs)
            assert torch.equal(proxy_changed[1], packed[1])
            proxy_influence = [float((proxy_changed[0][:, a:b]-packed[0][:, a:b]).abs().max())
                               for a, b in inputs.samples[0]['metadata']['chunk_latent_ranges']]
            assert all(v > 0 for v in proxy_influence), 'VAE proxy was ignored'
        result['parity'][str(window)] = dict(packed_cached_max_abs=max(
            float((a-b).abs().max()) for a, b in zip(packed, cached)),
            future_leak=0, own_clean_leak=0, cross_sample_leak=0,
            proxy_text_rows=0, first_frame_influence=influence, proxy_influence=proxy_influence, cache=cache_audit)
    del net, inputs, kwargs, packed, cached, future, own, picture
    classes = dict(sft=MiniMaxH3RefSFT, tf=MiniMaxH3RefTF, tscd=MiniMaxH3RefTSCD, dmd=MiniMaxH3RefDMD)
    result['stages'] = {}
    for stage, cls in classes.items():
        cfg = stage_config(args.configs_dir, stage)
        net = attach_adapter(tiny_model(causal=stage != 'sft'), cfg.models.backbone)
        stage_models = dict(backbone=net)
        if stage == 'tscd':
            stage_models.update(backbone_ema=copy.deepcopy(net).requires_grad_(False),
                                tea_model=copy.deepcopy(net).requires_grad_(False))
            # Distinct targets ensure the backward check is not a zero-gradient identity.
            with torch.no_grad():
                next(stage_models['backbone_ema'].parameters()).add_(.01)
        if stage == 'dmd':
            stage_models.update(fake_model=attach_adapter(tiny_model(False), cfg.models.fake_model),
                                tea_model=tiny_model(False).requires_grad_(False))
        meta = cls(cfg)
        ctx = dict(models=stage_models, batch={'ref_samples': [synthetic()]}, rng=RandomState(314159))
        meta.prepare_inputs(ctx)
        if stage in ('sft', 'tf'):
            calls = ('sample_timesteps', 'add_noise', 'forward', 'compute_loss')
        elif stage == 'tscd':
            calls = ('sample_segment', 'add_noise', 'student_forward', 'solver_step', 'target_forward', 'compute_loss')
        else:
            for name in ('rollout', 'prepare_fake', 'fake_forward', 'fake_loss'):
                getattr(meta, name)(ctx)
            fake = optimizer_step(stage_models['fake_model'], ctx['fake_loss'], cfg.models.fake_model)
            for name in ('prepare_gen', 'gen_forward', 'score', 'gen_loss'):
                getattr(meta, name)(ctx)
            ctx['loss'], calls = ctx['gen_loss'], ()
        for name in calls:
            getattr(meta, name)(ctx)
        stage_result = optimizer_step(net, ctx['loss'], cfg.models.backbone)
        if stage == 'dmd':
            stage_result['fake'] = fake
        stage_result.update(lora_rank=cfg.models.backbone.get('adapter', {}).get('r'),
                            trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.eval()
        if hasattr(meta, 'validation_diffusion'):
            meta.sampling_timesteps = meta.validation_diffusion['sampling_timesteps']
            meta.audio_sampling_timesteps = meta.validation_diffusion['audio_sampling_timesteps']
            meta.sampler = meta.validation_diffusion['sampler']
        with torch.no_grad():
            video = (meta._sample_native(net, ctx['inputs'], RandomState(919)) if stage == 'sft'
                     else meta._rollout_latents(net, ctx['inputs'], RandomState(919), keep_trajectory=False)[0].video)
        assert all(torch.isfinite(v).all() for v in video)
        stage_result.update(sampler=type(meta.sampler).__name__,
                            sampling_steps=len(meta.sampling_timesteps.timesteps))
        result['stages'][stage] = stage_result
        print(json.dumps({stage: stage_result}), flush=True)
        del ctx, stage_models, meta, net
        torch.cuda.empty_cache()
    cfg = stage_config(args.configs_dir, 'tf')
    meta = MiniMaxH3RefTF(cfg)
    net, sample = attach_adapter(tiny_model(), cfg.models.backbone).eval(), synthetic(frames=192)
    def arrivals(tracker, sample=sample):
        for k, (a, b) in enumerate(sample['metadata']['chunk_latent_ranges']):
            tracker.append(k)
            yield dict(references=[dict(r, video=r['video'][:, a:b]) for r in sample['references'][1:]], eos=b-a==2)
    inputs = meta._build_inputs({'ref_samples': [sample]})
    with torch.no_grad():
        offline = meta._rollout_latents(net, inputs, RandomState(44), keep_trajectory=False)[0].video[0]
        tracker = []
        stream = meta.stream_latents(net, sample['student_condition']['static'], arrivals(tracker), seed=44,
                                    condition_seed=sample['metadata']['condition_seed'],
                                    first_frame=sample['references'][0]['video'], conditioning_mode=MODE)
        first = next(stream)
        assert tracker == [0], 'future proxy input was prefetched'
        online = torch.cat([first, *stream], 1)
        torch.testing.assert_close(offline, online, atol=.003, rtol=.003)
        single = next(arrivals([]))
        shorter = meta.stream_latents(net, sample['student_condition']['static'], iter([single]), seed=44,
                                      condition_seed=sample['metadata']['condition_seed'],
                                      first_frame=sample['references'][0]['video'], conditioning_mode=MODE)
        assert torch.equal(first, next(shorter)), 'eventual clip length changed output prefix'
        from projects.minimax_h3.modeling.ref_recipe import RefRecipe
        from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
        independent = RefFixedHistory(RefRecipe.from_training(cfg))
        independent_tracker = []
        generated = independent.stream_latents(net,sample['student_condition']['static'],arrivals(independent_tracker),
            seed=44,condition_seed=sample['metadata']['condition_seed'],
            first_frame=sample['references'][0]['video'],conditioning_mode=MODE)
        independent_first = next(generated)
        assert independent_tracker == [0], 'independent inference prefetched future proxy input'
        independent_video = torch.cat([independent_first,*generated],1)
        torch.testing.assert_close(online,independent_video,atol=.003,rtol=.003)
    result['stream'] = dict(raw_frames=192, chunks=len(tracker), no_future_prefetch=True,
                            offline_live_max_abs=float((offline-online).abs().max()), length_dependent_prefix_error=0,
                            independent_max_abs=float((online-independent_video).abs().max()))
    result['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
    return result


def bundle_contracts(bundle):
    from projects.minimax_h3.tools.ref2va_prepare import FILES, check_sources, inspect_bundle, iter_video_chunks
    import numpy as np
    bundle = Path(bundle).resolve()
    original = inspect_bundle(bundle)
    check_sources(original, checksums=True)
    raw = (bundle/'actions.jsonl').read_bytes()
    first, payload = raw.split(b'\n', 1)
    header = json.loads(first)
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    rejected = []
    def attempt(name, mutate, *, check_media=False):
        h, rs = copy.deepcopy(header), copy.deepcopy(records)
        mutate(h, rs)
        body = b''.join((json.dumps(row)+'\n').encode() for row in rs)
        h['actions_payload_sha256'] = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory(prefix='ref2va-acceptance-') as folder:
            fixture = Path(folder)
            for filename in FILES:
                if filename != 'actions.jsonl':
                    (fixture/filename).symlink_to(bundle/filename)
            (fixture/'actions.jsonl').write_bytes(json.dumps(h).encode()+b'\n'+body)
            try:
                inspected = inspect_bundle(fixture)
                if check_media:
                    check_sources(inspected, checksums=True)
            except (ValueError, KeyError, TypeError):
                rejected.append(name)
            else:
                raise AssertionError(f'invalid source accepted: {name}')
    attempt('wrong action mapping', lambda h, rs: next(r for r in rs if r['record_type']=='frame').update(rgb_frame=8))
    attempt('incomplete chunk', lambda h, rs: next(r for r in rs if r['record_type']=='chunk').update(chunk_complete=False))
    attempt('stale camera checksum', lambda h, rs: h['camera_file'].update(sha256='0'*64))
    attempt('missing trajectory identity', lambda h, rs: h['source_render'].update(source_id=''))
    attempt('media content checksum', lambda h, rs: next(m for m in h['media'] if m['path']=='proxy.mkv').update(sha256='0'*64), check_media=True)
    kwargs = dict(start=158, num_frames=158, height=128, width=224, nearest=True)
    chunks = list(iter_video_chunks(bundle/'proxy.mkv', **kwargs))
    whole = next(iter_video_chunks(bundle/'proxy.mkv', **kwargs, chunk_frames=158))
    assert [len(c) for c in chunks] == [17]*9+[5]
    assert np.array_equal(np.concatenate(chunks), whole)
    return dict(bundle=str(bundle), source_frames=original['frame_count'], source_checksums_verified=True,
                invalid_cases_rejected=rejected, chunked_vs_window_pixels_equal=True, chunk_lengths=[len(c) for c in chunks])


def encoders(args):
    if not torch.cuda.is_available():
        raise RuntimeError('encoders is an explicitly scheduled GPU check')
    from projects.minimax_h3.data.ref_latent import load_entry, read_manifest, validate_sample
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders, qwen_inputs
    from projects.minimax_h3.tools.ref2va_prepare import iter_video_chunks
    from unittest.mock import patch
    path, entries, _ = read_manifest(args.manifest)
    sample = load_entry(path, entries[args.entry], verify_hash=True)
    validate_sample(sample)
    meta = sample['metadata']
    assert meta['conditioning_mode'] == MODE and meta['num_frames'] == 158
    assert not sample['student_condition']['chunks']
    bundle, start = Path(meta['bundle_path']), meta['local_start']
    rw, rh = meta['rgb_resolution']
    pw, ph = meta['proxy_resolution']
    rgb = next(iter_video_chunks(bundle/'rgb.mkv', start=start, num_frames=158, height=rh, width=rw, chunk_frames=158))
    proxy = next(iter_video_chunks(bundle/'proxy.mkv', start=start, num_frames=158, height=ph, width=pw, chunk_frames=158, nearest=True))
    enc = FrozenCWMEncoders(args.checkpoint, posterior='mean')
    seed, text = meta['condition_seed'], meta['system_prompt']
    calls = []
    original_forward = enc.qwen.forward
    def audited_forward(*a, **kw):
        calls.append({k for k in kw if 'video' in k})
        assert 'pixel_values_videos' not in kw and 'video_grid_thw' not in kw
        return original_forward(*a, **kw)
    kwargs, tags = qwen_inputs(enc.processor, text, images=[rgb[0]])
    assert 'pixel_values_videos' not in kwargs and 'video_grid_thw' not in kwargs
    with patch.object(enc.qwen, 'forward', audited_forward):
        condition = enc.condition(text, images=[rgb[0]], videos=[])
    assert len(calls) == 1
    for key in ('prompt_embeds', 'text_token_tags'):
        torch.testing.assert_close(condition[key], sample['teacher_condition'][key], atol=0, rtol=0)
        torch.testing.assert_close(condition[key], sample['student_condition']['static'][key], atol=0, rtol=0)
    image = enc.image(rgb[0], seed=seed)
    video = enc.video(rgb, seed=seed)
    reference = enc.video(proxy, seed=seed)
    for actual, expected in ((image, sample['references'][0]['video']), (video, sample['video']),
                             (reference, sample['references'][1]['video'])):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The native temporal encoder is an independent path from the per-chunk helper.
    pixels = torch.as_tensor(rgb, device='cuda').permute(0, 3, 1, 2).float()/255
    pixels = enc.vae.processor.transform(pixels).permute(1, 0, 2, 3).unsqueeze(0).bfloat16()
    with torch.no_grad():
        moments = enc.vae.encode_temporal(pixels).float()
        native_latent = ((moments[:, :24]-enc.mean)/enc.std)[0].cpu().bfloat16()
    torch.testing.assert_close(video, native_latent, atol=0, rtol=0)
    torch.testing.assert_close(enc.encode_chunk(proxy[:17]), reference[:, :5], atol=0, rtol=0)
    strict = enc.decode_video(video, num_frames=158, mode='strict')
    changed = video.clone()
    changed[:, 5:] += 3
    altered = enc.decode_video(changed, num_frames=158, mode='strict')
    assert strict.shape == (158, rh, rw, 3) and torch.equal(strict[:17], altered[:17])
    native = enc.decode_video(video, num_frames=158, mode='native')
    assert native.shape == strict.shape
    return dict(sample_id=meta['sample_id'], bundle=str(bundle), start=start, decoded_frames=158,
                rgb_latent_shape=list(video.shape), proxy_latent_shape=list(reference.shape),
                qwen_calls=1, qwen_video_inputs=0, qwen_rows=len(tags),
                teacher_static_cached_qwen_max_abs=0, cached_vae_max_abs=0,
                native_vs_chunk_encoder_max_abs=0, strict_future_prefix_max_abs=0,
                strict_vs_native_mae_8bit=float((strict.float()-native.float()).abs().mean()),
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='JSON evidence file; reports only successful completed checks')
    sub = parser.add_subparsers(dest='command', required=True)
    cpu = sub.add_parser('contracts', help='CPU only; no encoder/model loads')
    cpu.add_argument('--manifest', type=Path)
    cpu.add_argument('--limit', type=int, default=3)
    cpu.add_argument('--bundle', type=Path, help='Optional real bundle metadata/checksum/chunk-loader checks (CPU)')
    gpu = sub.add_parser('models', help='one GPU; actual reduced H3 and four training stages')
    gpu.add_argument('--configs-dir', type=Path, required=True)
    enc = sub.add_parser('encoders', help='one GPU; actual released Qwen/VAE on one real cache entry')
    enc.add_argument('--manifest', type=Path, required=True)
    enc.add_argument('--checkpoint', type=Path, required=True)
    enc.add_argument('--entry', type=int, default=0)
    args = parser.parse_args(argv)
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    base = Path(__file__).parents[1]
    sources = [Path(__file__), base/'data/ref_latent.py', base/'modeling/ref_condition.py',
               base/'meta_models/minimax_h3_ref2va.py', base/'tools/ref2va_prepare.py', base/'modeling/cwm_encoders.py',
               base/'modeling/sampling.py', base/'modeling/training_timesteps.py',
               base/'modeling/ref_recipe.py', base/'modeling/causal_layout.py',
               base/'inference/model.py', base/'inference/recipes/ref_fixed_history.py',
               base/'meta_models/causal_minimax_h3_dmd.py']
    hashes = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = globals()[args.command](args)
    report = dict(command=args.command, conditioning_mode=MODE, started_at_utc=started_at,
                  elapsed_seconds=time.monotonic()-start, torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=hashes, result=result)
    if args.command == 'models':
        report['config_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(args.configs_dir.glob('*.yaml'))}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
