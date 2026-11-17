"""CPU regression tests through Ref2VA packing, SFT/TF losses and cached rollout."""
from __future__ import annotations

import unittest
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from common.config import CfgNode
from common.seed import RandomState
from projects.minimax_h3.data.ref_latent import load_entry, validate_sample
from projects.minimax_h3.meta_models.minimax_h3_ref2va import MiniMaxH3RefSFT, MiniMaxH3RefTF
from projects.minimax_h3.modeling.ref_guidance import guidance_options, weak_inputs
from projects.minimax_h3.tests.test_ref_static_condition import entry, sample
from projects.minimax_h3.tools.cwm_prepare import encode_guidance_condition


def document(rows, value):
    return dict(prompt_embeds=torch.full((rows,5120),float(value),dtype=torch.bfloat16),
                text_token_tags=torch.ones(rows,dtype=torch.long))


def guided_sample(index=0):
    value=sample(mode=None,frames=39)
    value['metadata']['sample_id']=f'window_{index}'
    value['metadata']['condition_seed']+=index
    value['video'].fill_(index+.25)
    value['references'][1]['video'].fill_(index+2)
    value['teacher_condition']=document(6,3)
    static=document(2,1)
    value['guidance_conditions']={
        'proxy':dict(teacher_condition=static,student_condition=dict(static=static,chunks=[])),
        'prompt':dict(teacher_condition=document(4,2),student_condition=dict(static=static,
            chunks=[document(1,k+1) for k in range(3)])),
        'prompt_proxy':dict(teacher_condition=static,student_condition=dict(static=static,chunks=[])),
    }
    return value


class PackedProbe(torch.nn.Module):
    """Replace only DiT math; retain production packing and RGB row restoration."""
    param_dtype=torch.float32

    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(2.0))
        self.calls=[]

    def forward(self, **kwargs):
        rows=kwargs['img_pos_for_infer_output_info']['position_ids']
        result=kwargs['x'][0,rows]*self.weight+len(kwargs['x'][0])/100
        if result.requires_grad:
            result.retain_grad()
        self.calls.append(dict(grad=torch.is_grad_enabled(),result=result,kwargs=kwargs))
        return result,kwargs['audio_x'][0,:0]


def meta(stage, mode='gft', **options):
    cls=MiniMaxH3RefSFT if stage=='sft' else MiniMaxH3RefTF
    result=cls.__new__(cls)
    result.config=CfgNode(dict(meta_model=dict(objective='velocity' if stage=='sft' else 'x0',
        guidance=dict(mode=mode,**options)),
        data=dict(args=dict(stage=stage,window_size=6,rope_reference_latents=37)),validation={}))
    result.guidance=guidance_options(result.config)
    result.chunk_wise_weighting=None
    result.resampling=None
    result.training_timesteps=SimpleNamespace(loss_weight=lambda t:torch.tensor(3.0))
    return result


def context(subject, count=2):
    values=[guided_sample(i) for i in range(count)]
    ctx=dict(batch={'ref_samples':values},models={'backbone':PackedProbe()},rng=RandomState(73),iter=4)
    subject.prepare_inputs(ctx)
    xts=[torch.full_like(s['video'],i+2.0) for i,s in enumerate(values)]
    audio=ctx['clean_latents'][1]
    times=torch.tensor([.25+.25*i for i in range(count)])
    if isinstance(subject,MiniMaxH3RefTF):
        times=[t.repeat(3) for t in times]
    ctx.update(noisy_latents=(xts,audio),train_timesteps=(times,times),
        context_eps=([torch.ones_like(x)*.2 for x in xts],audio))
    return ctx


class RefGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        for module in ('modeling.ref_condition','meta_models.causal_minimax_h3_base',
                       'meta_models.minimax_h3_ref2va'):
            self.stack.enter_context(patch('projects.minimax_h3.'+module+'.get_device',
                                          return_value=torch.device('cpu')))
        self.stack.enter_context(patch('utils.flex_attn.FLEX_FLASH_ATTN_AVAILABLE',True))
        self.stack.enter_context(patch('projects.minimax_h3.meta_models.minimax_h3_ref2va.all_reduce_sum'))
        self.stack.enter_context(patch('projects.minimax_h3.meta_models.minimax_h3_ref2va.get_world_size',return_value=1))

    def test_options_reject_invalid_training_and_double_guidance(self):
        for options in (dict(beta=0),dict(beta=1.1),dict(beta=float('nan')),dict(dropout=1),
                        dict(dropout=float('nan')),dict(scale=2),dict(drop='wrong')):
            with self.subTest(options=options),self.assertRaises(ValueError):
                meta('sft',**options)
        with self.assertRaises(ValueError):
            meta('sft','cfg',dropout=0)
        with self.assertRaises(ValueError):
            meta('tscd')
        subject=meta('sft'); subject.config.validation.cfg_scale=2
        with self.assertRaises(ValueError):
            subject._guidance_scale()

    def test_weak_views_preserve_targets_noise_and_rebuild_geometry(self):
        for stage in ('sft','tf'):
            subject=meta(stage); inputs=context(subject)['inputs']
            original=subject._ref_geometry(inputs,native=stage=='sft')
            for drop in ('proxy','prompt','prompt_proxy'):
                with self.subTest(stage=stage,drop=drop):
                    weak=weak_inputs(inputs,drop)
                    self.assertIsNot(weak.geometry,inputs.geometry)
                    self.assertEqual(weak.geometry,{})
                    self.assertIs(weak.seqlens,inputs.seqlens)
                    for i in range(2):
                        self.assertIs(weak.samples[i]['video'],inputs.samples[i]['video'])
                        self.assertIs(weak.reference_eps[i][0],inputs.reference_eps[i][0])
                        self.assertEqual(weak.layouts[i].chunks,inputs.layouts[i].chunks)
                        self.assertEqual(len(weak.samples[i]['references']),2 if drop=='prompt' else 1)
                        self.assertEqual(len(inputs.samples[i]['references']),2)
                    geometry=subject._ref_geometry(weak,native=stage=='sft')
                    self.assertEqual(len(geometry['output_pos']),len(original['output_pos']))
                    self.assertLess(len(geometry['position_ids']),len(original['position_ids']))
            inputs.samples[0].pop('guidance_conditions')
            self.assertIs(weak_inputs(inputs,'proxy').samples[0]['teacher_condition'],
                          inputs.samples[0]['student_condition']['static'])
            with self.assertRaisesRegex(ValueError,'independently encoded'):
                weak_inputs(inputs,'prompt')

    def test_gft_sft_packed_gradient_keeps_velocity_weight(self):
        subject=meta('sft',dropout=0,beta=.5); ctx=context(subject)
        net=ctx['models']['backbone']
        subject.forward(ctx); subject.compute_loss(ctx)
        self.assertEqual([c['grad'] for c in net.calls],[False,True])
        auxiliary,primary=[c['result'] for c in net.calls]
        self.assertFalse(auxiliary.requires_grad)
        target_rows=torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['clean_latents'][0]])
        sigma=torch.cat([t.expand(v.numel()//96) for t,v in zip(ctx['train_timesteps'][0],ctx['clean_latents'][0])])
        residual=.5*primary.detach()+.5*auxiliary-target_rows
        expected=3*residual.square()/sigma[:,None].square()
        torch.testing.assert_close(ctx['loss'],expected.double().mean(),rtol=2e-7,atol=1e-7)
        ctx['loss'].backward()
        torch.testing.assert_close(primary.grad,3*residual/sigma[:,None].square()/residual.numel())
        self.assertTrue(torch.isfinite(net.weight.grad))
        self.assertGreater(net.weight.grad.abs().item(),0)

    def test_beta_one_and_plain_match_and_drop_keeps_ordinary_loss(self):
        for stage in ('sft','tf'):
            predictions=[]
            for mode,beta in (('plain',.5),('gft',1.0)):
                subject=meta(stage,mode,dropout=0,beta=beta); ctx=context(subject)
                subject.forward(ctx); subject.compute_loss(ctx)
                self.assertEqual(len(ctx['models']['backbone'].calls),1)
                predictions.append((ctx['pred'][0],ctx['loss']))
            torch.testing.assert_close(predictions[0],predictions[1],rtol=0,atol=0)
            subject=meta(stage,dropout=.1,beta=.5); ctx=context(subject)
            # Every local sample drops, but the auxiliary FSDP call still occurs.
            with patch('projects.minimax_h3.modeling.ref_guidance.random.Random') as rng:
                rng.return_value.random.return_value=0
                subject.forward(ctx)
            calls=ctx['models']['backbone'].calls
            self.assertEqual([c['grad'] for c in calls],[False,True])
            returned=torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['pred'][0]])
            torch.testing.assert_close(returned,calls[-1]['result'],rtol=0,atol=0)
            subject.compute_loss(ctx); ctx['loss'].backward()
            self.assertGreater(calls[-1]['result'].grad.abs().sum().item(),0)

    def test_mixed_drop_pack_updates_each_sample_with_its_own_beta(self):
        subject=meta('tf',dropout=.1); ctx=context(subject)
        with patch('projects.minimax_h3.modeling.ref_guidance.random.Random') as rng:
            rng.return_value.random.side_effect=[0,1]
            subject.forward(ctx)
        auxiliary,primary=[c['result'] for c in ctx['models']['backbone'].calls]
        rows=ctx['inputs'].samples[0]['video'].numel()//96
        output=torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['pred'][0]])
        torch.testing.assert_close(output[:rows],primary[:rows])
        torch.testing.assert_close(output[rows:],.5*(primary[rows:]+auxiliary[rows:]))
        subject.compute_loss(ctx); ctx['loss'].backward()
        targets=torch.cat([subject._video_rows(v,0,v.shape[1]) for v in ctx['clean_latents'][0]])
        derivative=2*(output.detach()-targets)/output.numel()
        derivative[rows:]*=.5
        torch.testing.assert_close(primary.grad,derivative)
        self.assertEqual(ctx['metrics']['guidance/dropout_fraction'],.5)

    def test_cfg_sampling_extrapolates_restored_target_rows(self):
        subject=meta('sft','cfg',scale=3); ctx=context(subject)
        result=subject._sampling_forward(ctx['models']['backbone'],ctx['inputs'],
            weak=weak_inputs(ctx['inputs'],'proxy'),native=True,
            video_xts=ctx['noisy_latents'][0],video_timesteps=ctx['train_timesteps'][0])
        positive,negative=[c['result'] for c in ctx['models']['backbone'].calls]
        output=torch.cat([subject._video_rows(v,0,v.shape[1]) for v in result[0]])
        torch.testing.assert_close(output,negative+3*(positive-negative))

    def test_tf_resampling_once_shared_history_and_original_x0_mse(self):
        subject=meta('tf',dropout=0); ctx=context(subject)
        subject.resampling=dict(warmup_steps=1); subject._ga_steps=2
        history=[x+5 for x in ctx['clean_latents'][0]]
        calls=[]; original=subject._causal_packed_forward
        def forward(*args,**kwargs):
            calls.append(kwargs)
            return original(*args,**kwargs)
        with patch.object(subject,'_resample_history',return_value=history) as resample:
            with patch.object(subject,'_causal_packed_forward',side_effect=forward):
                subject.forward(ctx)
        resample.assert_called_once()
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertIs(call['video_context'],history)
            self.assertIs(call['video_eps'],ctx['context_eps'][0])
            self.assertIs(call['video_xts'],ctx['noisy_latents'][0])
            self.assertIs(call['video_timesteps'],ctx['train_timesteps'][0])
        subject.compute_loss(ctx)
        expected=torch.cat([(p-t).square().flatten() for p,t in zip(ctx['pred'][0],ctx['clean_latents'][0])]).double().mean()
        torch.testing.assert_close(ctx['loss'],expected)
        ctx['loss'].backward()
        self.assertGreater(ctx['models']['backbone'].weight.grad.abs().item(),0)

    def test_proxy_empty_prefill_is_safe_with_sp_padding(self):
        subject=meta('tf'); ctx=context(subject)
        weak=weak_inputs(ctx['inputs'],'proxy')
        for sp in (1,2):
            with self.subTest(sp=sp),patch.object(subject,'_sp_size',return_value=sp):
                with patch.object(subject,'_pad_rows',side_effect=lambda n:0 if sp==1 else sp-n%sp):
                    weak.geometry.clear()
                    result=subject._ref_forward(ctx['models']['backbone'],weak,selection=('condition',0),
                        cache=object(),store_kv=True,video_xts=ctx['noisy_latents'][0],
                        video_timesteps=ctx['train_timesteps'][0])
                    self.assertIsNone(result)
        self.assertEqual(ctx['models']['backbone'].calls,[])

    def test_cached_cfg_uses_two_histories_and_gft_one_branch(self):
        for mode in ('cfg','gft'):
            subject=meta('tf',mode); ctx=context(subject,count=1)
            subject._num_layers=lambda net:1
            subject.sampling_timesteps=SimpleNamespace(timesteps=torch.tensor([.5]),dynamic_shift=False,
                set_timesteps=lambda **kw:None,get_next_timesteps=lambda t:torch.zeros_like(t))
            subject.sampler=SimpleNamespace(step_to=lambda **kw:kw['pred'])
            seen=[]; original=subject._ref_forward
            def forward(net,inputs,**kwargs):
                seen.append((inputs,kwargs))
                return original(net,inputs,**kwargs)
            with patch.object(subject,'_ref_forward',side_effect=forward):
                result,_=subject._rollout_latents(ctx['models']['backbone'],ctx['inputs'],RandomState(3),keep_trajectory=False)
            caches={id(kw['cache']) for _,kw in seen}
            self.assertEqual(len(caches),2 if mode=='cfg' else 1)
            self.assertEqual(len(result.video),1)
            histories=[kw for _,kw in seen if kw['selection'][0]=='history']
            self.assertEqual(len(histories),6 if mode=='cfg' else 3)
            if mode=='cfg':
                for a,b in zip(histories[::2],histories[1::2]):
                    self.assertIsNot(a['cache'],b['cache'])
                    self.assertIs(a['video_context'],b['video_context'])
                    self.assertIs(a['video_eps'],b['video_eps'])
            self.assertTrue(all(torch.isfinite(v).all() for v in result.video))

    def test_native_gft_sampling_does_not_need_weak_cache(self):
        subject=meta('sft'); ctx=context(subject,count=1)
        ctx['inputs'].samples[0].pop('guidance_conditions')
        subject.sampling_timesteps=SimpleNamespace(timesteps=torch.tensor([.6,.2]),
            set_timesteps=lambda **kw:None,get_next_timesteps=lambda t:torch.zeros_like(t))
        subject.sampler=SimpleNamespace(step_to=lambda **kw:kw['pred'])
        result=subject._sample_native(ctx['models']['backbone'],ctx['inputs'],RandomState(5))
        self.assertEqual(len(ctx['models']['backbone'].calls),2)
        self.assertEqual(tuple(result[0].shape),tuple(ctx['inputs'].samples[0]['video'].shape))

    def test_encoder_uses_independent_correct_documents(self):
        videos=[torch.arange(39).reshape(39,1),torch.arange(39).reshape(39,1)+100]
        picture=[torch.ones(1)]; static=document(3,1); ranges=[(0,17),(17,34),(34,39)]
        for drop in ('proxy','prompt','prompt_proxy'):
            calls=[]
            def condition(prompt,vids=(),*,images=()):
                calls.append((prompt,vids,images))
                return document(len(calls),len(calls))
            result=encode_guidance_condition(SimpleNamespace(condition=condition),drop,prompt='plan',
                picture=picture,videos=videos,raw_ranges=ranges,static=static)
            self.assertEqual(calls[0][0],'plan' if drop=='proxy' else '')
            self.assertIs(calls[0][2],picture)
            self.assertEqual(len(calls[0][1]),2 if drop=='prompt' else 0)
            if drop=='prompt':
                self.assertEqual(len(calls),5)
                self.assertEqual([len(c[1][0]) for c in calls[2:]],[17,17,5])
                self.assertTrue(all(c[0]=='' and not c[2] for c in calls[2:]))
            else:
                self.assertEqual(len(calls),1)
                self.assertEqual(result['student_condition']['chunks'],[])
            if drop=='proxy':
                self.assertIs(result['student_condition']['static'],static)
        with self.assertRaisesRegex(ValueError,'first-frame'):
            encode_guidance_condition(None,'proxy',prompt='',picture=[],videos=[],raw_ranges=[],static=static)

    def test_weak_cache_rejects_invalid_values_tags_geometry_and_dtype(self):
        for field, replacement, check_values, message in (
            ('prompt_embeds', torch.full((4,5120),float('nan'),dtype=torch.bfloat16), True, 'values/tags'),
            ('text_token_tags', torch.full((4,),2,dtype=torch.long), True, 'values/tags'),
            ('prompt_embeds', torch.ones(4,5120,dtype=torch.float32), False, 'geometry/dtype'),
            ('text_token_tags', torch.ones(4,dtype=torch.int32), False, 'geometry/dtype'),
            ('prompt_embeds', torch.ones(4,32,dtype=torch.bfloat16), False, 'geometry/dtype'),
        ):
            with self.subTest(field=field,message=message):
                value=guided_sample()
                value['guidance_conditions']['prompt']['teacher_condition'][field]=replacement
                with self.assertRaisesRegex(ValueError,message):
                    validate_sample(value,check_values=check_values)
        # Lightweight runtime loading retains the existing check_values contract.
        value=guided_sample()
        value['guidance_conditions']['prompt']['teacher_condition']['prompt_embeds'].fill_(float('nan'))
        validate_sample(value,check_values=False)

    def test_weak_cache_requires_matching_chunks_and_existing_budget(self):
        for drop, count in (('prompt',0),('prompt',2),('proxy',3),('prompt_proxy',3)):
            with self.subTest(drop=drop,count=count):
                value=guided_sample()
                value['guidance_conditions'][drop]['student_condition']['chunks']=[document(1,1) for _ in range(count)]
                with self.assertRaisesRegex(ValueError,'chunk documents'):
                    validate_sample(value)
        for field in ('teacher_condition','static'):
            value=guided_sample(); branch=value['guidance_conditions']['prompt']
            if field=='teacher_condition':
                branch[field]=document(7,1)  # Retained references leave room for only six document rows.
            else:
                branch['student_condition'][field]=document(32,1)  # Only the causal budget grows.
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'sequence budget'):
                validate_sample(value)

    def test_load_entry_checks_manifest_guidance_drop_against_blob(self):
        value=guided_sample()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'sample.pt'; manifest=root/'train.jsonl'
            record=entry(value); record['guidance_drop']='prompt'
            torch.save(value,path)
            load_entry(manifest,record)
            value['guidance_conditions'].pop('prompt')
            torch.save(value,path)
            with self.assertRaisesRegex(ValueError,'guidance_drop mismatch'):
                load_entry(manifest,record)
            record['guidance_drop']='unknown'
            with self.assertRaisesRegex(ValueError,'guidance_drop mismatch'):
                load_entry(manifest,record)
            record.pop('guidance_drop')
            value.pop('guidance_conditions')
            torch.save(value,path)
            load_entry(manifest,record)  # Original caches do not require new fields.

    def test_static_image_weak_documents_match_and_require_first_frame(self):
        value=sample(frames=39); static=document(2,1)
        value['guidance_conditions']={'proxy':dict(teacher_condition=static,
            student_condition=dict(static=static,chunks=[]))}
        validate_sample(value)
        value['guidance_conditions']['proxy']['student_condition']['static']=document(2,2)
        with self.assertRaisesRegex(ValueError,'identical Qwen'):
            validate_sample(value)
        value=guided_sample(); value['references'].pop(0)
        with self.assertRaisesRegex(ValueError,'first_frame'):
            validate_sample(value)

    def test_streaming_weak_budget_uses_causal_geometry_only(self):
        value=sample(mode=None,frames=5)
        value['teacher_condition']=value['student_condition']['static']
        value['guidance_conditions']={'prompt':dict(teacher_condition=document(4,1),
            student_condition=dict(static=document(2,1),chunks=[document(1,1)]))}
        validate_sample(value,streaming=True)
        with self.assertRaisesRegex(ValueError,'sequence budget'):
            validate_sample(value)

    def test_stream_cfg_rejects_changes_to_cached_weak_static(self):
        for change in (None,'value','shape','inplace'):
            subject=meta('tf','cfg',drop='prompt_proxy')
            subject._num_layers=lambda net:1
            subject.sampling_timesteps=SimpleNamespace(timesteps=torch.tensor([.5]),dynamic_shift=False,
                set_timesteps=lambda **kw:None,get_next_timesteps=lambda t:torch.zeros_like(t))
            subject.sampler=SimpleNamespace(step_to=lambda **kw:kw['pred'])
            def chunks():
                weak=document(2,1)
                for index,(size,eos) in enumerate(((5,False),(2,True))):
                    if index:
                        if change=='value': weak=document(2,2)
                        elif change=='shape': weak=document(3,1)
                        elif change=='inplace': weak['prompt_embeds'].add_(1)
                    yield dict(references=[dict(kind='video',role='semantic',video=torch.zeros(24,size,2,4))],
                        condition=document(3,1),eos=eos,guidance_conditions={'prompt_proxy':dict(
                            teacher_condition=weak,student_condition=dict(static=weak,chunks=[]))})
            with self.subTest(change=change):
                stream=subject.stream_latents(PackedProbe().eval(),document(3,1),chunks(),
                    first_frame=torch.zeros(24,1,2,4))
                if change is None:
                    result=list(stream)
                    self.assertEqual([v.shape[1] for v in result],[5,2])
                else:
                    self.assertEqual(next(stream).shape[1],5)
                    with self.assertRaisesRegex(ValueError,'weak static Qwen condition changed'):
                        next(stream)


if __name__=='__main__':
    unittest.main()
