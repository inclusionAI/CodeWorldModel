"""CPU contract/regression coverage for static-image Qwen plus VAE-only proxies."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from projects.minimax_h3.data.ref_latent import (
    SCHEMA, STATIC_CONDITIONING_MODE, RefLatentDataset, conditioning_mode,
    load_entry, packed_length, read_manifest, validate_sample,
)
from projects.minimax_h3.modeling.ref_condition import CWMRefConditionMixin
from projects.minimax_h3.meta_models.minimax_h3_ref2va import MiniMaxH3RefTF
from projects.minimax_h3.tools.cwm_config import _manifest, main as config_main


def sample(*, mode=STATIC_CONDITIONING_MODE, frames=158):
    latents = 5*((frames+16)//17)-3
    static = dict(prompt_embeds=torch.ones(3,5120,dtype=torch.bfloat16),
                  text_token_tags=torch.tensor([0,1,1]))
    metadata = dict(schema=SCHEMA,fps=24,num_frames=frames,normalized=True,condition_seed=2026,
                    sample_id='window_0',take_id='episode_0',fingerprint='test_recipe',
                    chunk_raw_ranges=[[a,min(a+17,frames)] for a in range(0,frames,17)],
                    chunk_latent_ranges=[[a,min(a+5,latents)] for a in range(0,latents,5)])
    if mode is not None:
        metadata['conditioning_mode'] = mode
    chunks = [] if mode == STATIC_CONDITIONING_MODE else [copy.deepcopy(static) for _ in metadata['chunk_raw_ranges']]
    return dict(video=torch.zeros(24,latents,2,4),audio=torch.empty(2,32,0),
                references=[dict(kind='image',role='first_frame',video=torch.zeros(24,1,2,4)),
                            dict(kind='video',role='semantic',video=torch.zeros(24,latents,2,4))],
                teacher_condition=static,student_condition=dict(static=static,chunks=chunks),metadata=metadata)


def entry(value, path='sample.pt'):
    meta = value['metadata']
    result = dict(schema=SCHEMA,sample_id=meta['sample_id'],take_id=meta['take_id'],fingerprint=meta['fingerprint'],
                  path=str(path),latent_shape=list(value['video'].shape),references=['first_frame','semantic'],
                  lengths={stage:packed_length(value,stage) for stage in ('sft','tf','dmd')})
    if 'conditioning_mode' in meta:
        result['conditioning_mode'] = meta['conditioning_mode']
    return result


class Geometry(CWMRefConditionMixin):
    def __init__(self, mode=STATIC_CONDITIONING_MODE):
        args=dict(stage='tf',window_size=6,rope_reference_latents=37)
        if mode is not None:
            args['conditioning_mode']=mode
        self.config=SimpleNamespace(data=SimpleNamespace(args=args))

    def _to_device(self,value):
        return value

    def _pad_rows(self,rows):
        return 0

    def _sp_size(self):
        return 1


class StaticConditionTests(unittest.TestCase):
    def test_158_frames_and_budget(self):
        value=sample()
        validate_sample(value)
        self.assertEqual(tuple(value['video'].shape),(24,47,2,4))
        self.assertEqual(len(value['metadata']['chunk_raw_ranges']),10)
        # 3 static rows + (1 picture + 47 proxy + 47 RGB) * 2 spatial rows.
        self.assertEqual(packed_length(value,'sft'),193)
        # 3 static + picture/proxy + noisy RGB + 45 history latent frames.
        self.assertEqual(packed_length(value,'tf'),283)
        self.assertEqual(packed_length(value,'dmd'),283)

    def test_legacy_default_and_explicit_are_unchanged(self):
        old=sample(mode=None,frames=124)
        explicit=copy.deepcopy(old)
        explicit['metadata']['conditioning_mode']='legacy'
        for value in (old,explicit):
            validate_sample(value)
        self.assertEqual(conditioning_mode(old['metadata']),'legacy')
        for stage in ('sft','tf','dmd'):
            self.assertEqual(packed_length(old,stage),packed_length(explicit,stage))
        with patch('projects.minimax_h3.modeling.ref_condition.get_device',return_value=torch.device('cpu')):
            with patch('utils.flex_attn.FLEX_FLASH_ATTN_AVAILABLE',True):
                geometry=Geometry(mode=None)
                a=geometry._ref_geometry(geometry._build_inputs({'ref_samples':[old]}))
                b=geometry._ref_geometry(geometry._build_inputs({'ref_samples':[explicit]}))
        for key in ('position_ids','token_tags','prompt_embeds','q_ranges','k_ranges'):
            self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_invalid_new_conditions_are_rejected(self):
        value=sample(); value['student_condition']['chunks']=[value['student_condition']['static']]
        with self.assertRaisesRegex(ValueError,'empty chunk Qwen'):
            validate_sample(value)
        value=sample(); value['teacher_condition']=copy.deepcopy(value['teacher_condition'])
        value['teacher_condition']['prompt_embeds'][0,0]=2
        with self.assertRaisesRegex(ValueError,'identical Qwen'):
            validate_sample(value)
        value=sample(); value['teacher_condition']=copy.deepcopy(value['teacher_condition'])
        value['teacher_condition']['prompt_embeds']=value['teacher_condition']['prompt_embeds'].float()
        with self.assertRaisesRegex(ValueError,'identical Qwen'):
            validate_sample(value,check_values=False)
        value=sample(); value['references'].pop(0)
        with self.assertRaisesRegex(ValueError,'first_frame'):
            validate_sample(value)
        with self.assertRaisesRegex(ValueError,'unsupported conditioning'):
            validate_sample(sample(mode='typo'))

    def test_mixed_manifest_and_blob_modes_rejected(self):
        value=sample(); first=entry(value); second=copy.deepcopy(first)
        second.update(sample_id='window_1',conditioning_mode='legacy')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in (first,second)))
            for reader in (read_manifest,_manifest):
                with self.assertRaisesRegex(ValueError,'mixed conditioning modes'):
                    reader(path)
            torch.save(value,Path(directory)/'sample.pt')
            first.pop('conditioning_mode')
            with self.assertRaisesRegex(ValueError,'conditioning mode mismatch'):
                load_entry(path,first)
            path.write_text(json.dumps(entry(value))+'\n')
            with self.assertRaisesRegex(ValueError,'dataset/config conditioning mode mismatch'):
                RefLatentDataset(0,None,manifest=str(path),max_seqlen=1000,conditioning_mode='legacy')

    def test_packed_and_proxy_prefill_have_only_static_text(self):
        value=sample(); geometry=Geometry()
        with patch('projects.minimax_h3.modeling.ref_condition.get_device',return_value=torch.device('cpu')):
            inputs=geometry._build_inputs({'ref_samples':[value]})
            with patch('utils.flex_attn.FLEX_FLASH_ATTN_AVAILABLE',True):
                packed=geometry._ref_geometry(inputs)
            condition=geometry._ref_geometry(inputs,selection=('condition',0))
            native=geometry._ref_geometry(inputs,native=True)
        self.assertEqual(packed['refiner_lengths'],[3])
        self.assertEqual(len(packed['position_ids']),packed_length(value,'tf'))
        self.assertEqual(native['refiner_lengths'],[3])
        self.assertEqual(len(native['position_ids']),packed_length(value,'sft'))
        self.assertEqual(condition['refiner_lengths'],[0])
        self.assertEqual(tuple(condition['prompt_embeds'].shape),(0,5120))
        self.assertEqual(len(condition['text_pos']),0)
        self.assertEqual(len(condition['img_pos']),10)
        self.assertEqual(len(condition['blocks']),1)
        with self.assertRaisesRegex(ValueError,'one conditioning mode'):
            geometry._build_inputs({'ref_samples':[value,sample(mode=None)]})

    def test_all_stage_configs_record_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=root/'train.jsonl'
            manifest.write_text(json.dumps(entry(sample()))+'\n')
            config_main(['--stage','all','--mode','lora','--manifest',str(manifest),
                         '--init',str(root/'released.safetensors'),
                         '--sft-checkpoint',str(root/'sft.safetensors'),
                         '--tf-checkpoint',str(root/'tf.safetensors'),
                         '--tscd-checkpoint',str(root/'tscd.safetensors'),
                         '--out-dir',str(root/'configs'),'--runs-dir',str(root/'runs')])
            import yaml
            for stage in ('sft','tf','tscd','dmd'):
                config=yaml.safe_load((root/'configs'/f'{stage}.yaml').read_text())
                self.assertEqual(config['data']['args']['conditioning_mode'],STATIC_CONDITIONING_MODE)
                self.assertEqual(config['lineage']['conditioning_mode'],STATIC_CONDITIONING_MODE)

    def test_stream_without_chunk_qwen_prefills_static_once(self):
        value=sample(); calls=[]; built=[]
        meta=MiniMaxH3RefTF.__new__(MiniMaxH3RefTF)
        def build(batch,streaming=False):
            current=batch['ref_samples'][0]; validate_sample(current,streaming=streaming)
            built.append(current)
            return SimpleNamespace(samples=[current],window_sizes=[6])
        meta._build_inputs=build
        meta._num_layers=lambda backbone:1
        meta._sp_size=lambda:1
        meta._ref_forward=lambda *args,**kwargs:calls.append(kwargs['selection'])
        meta._rollout_grid=lambda *args:None
        meta._rollout_chunk=lambda backbone,inputs,k,args,*rest,**kwargs:args['video_xts'][0].fill_(1)
        cache=SimpleNamespace(begin_chunk=lambda k:None)
        def chunks():
            for size,eos in ((5,False),(2,True)):
                yield dict(references=[dict(kind='video',role='semantic',video=torch.zeros(24,size,2,4))],eos=eos)
        with patch('projects.minimax_h3.meta_models.minimax_h3_ref2va.GroupedKVCache',return_value=cache):
            result=list(meta.stream_latents(SimpleNamespace(training=False),value['teacher_condition'],chunks(),
                    first_frame=value['references'][0]['video'],conditioning_mode=STATIC_CONDITIONING_MODE))
        self.assertEqual([x.shape[1] for x in result],[5,2])
        self.assertEqual(calls,[('static',-1)])
        self.assertTrue(all(item['student_condition']['chunks']==[] for item in built))
        bad=next(chunks()); bad['condition']=None
        with self.assertRaisesRegex(ValueError,'must omit Qwen'):
            list(meta.stream_latents(SimpleNamespace(training=False),value['teacher_condition'],[bad],
                 first_frame=value['references'][0]['video'],conditioning_mode=STATIC_CONDITIONING_MODE))


if __name__=='__main__':
    unittest.main()
