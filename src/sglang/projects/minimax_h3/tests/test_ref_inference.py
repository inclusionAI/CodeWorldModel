"""Old/new causal stream differential with real packing and cache tensors."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from projects.minimax_h3.tests.ref_inference_support import legacy_class

from projects.minimax_h3.modeling.ref_condition import GroupedKVCache
from projects.minimax_h3.modeling.ref_recipe import PROJECT, RefRecipe
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.inference.artifact import export_bundle, load_bundle, sha256, validate_bundle


class Node(dict):
    def __setattr__(self, name, value):
        self[name] = nodes(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def nodes(value):
    return Node({k: nodes(v) for k, v in value.items()}) if isinstance(value, dict) else value


class Grid:
    dynamic_shift = False

    def __init__(self):
        self.timesteps = torch.tensor([1., .712345678901, .183456789012], dtype=torch.float64)

    def set_timesteps(self, **kwargs):
        pass

    def get_next_timesteps(self, ts):
        i = (self.timesteps == ts[0]).nonzero().item()
        return torch.full_like(ts, self.timesteps[i+1] if i+1 < len(self.timesteps) else 0)


class Euler:
    def step_to(self, pred, x_t, t, s, **kwargs):
        return [x + (s[i]-t[i]).float()/t[i].float()*(x-p) for i, (x,p) in enumerate(zip(x_t,pred))]


def configuration(stage='dmd', guidance='plain'):
    return nodes(dict(data=dict(args=dict(stage=stage, conditioning_mode='static_image_qwen_proxy_vae_v1',
        window_size=1, rope_reference_latents=17)), meta_model=dict(guidance=dict(mode=guidance)), validation={}))


def recipe_config(stage='dmd', guidance='plain'):
    value = configuration(stage, guidance)
    def node(module, name, **args):
        return dict(module=module,class_name=name,args=args)
    value.diffusion = dict(schedule=node('common.diffusion.schedule.lerp','LinearInterpolationSchedule',T=1.,pred_type='x_0'),
        sampling_timesteps=node('common.diffusion.timestep.sampling.trailing','TrailingSamplingTimesteps',T=1.,num_sampling_steps=3,shift=12.),
        sampler=node(PROJECT+'.modeling.sampling','EulerSampler'))
    value.validation = dict(sampling_timesteps=node('common.diffusion.timestep.sampling.trailing','TrailingSamplingTimesteps',T=1.,num_sampling_steps=2,shift=3.),
                            sampler=node(PROJECT+'.modeling.sampling','EulerSampler'))
    return value


class CacheProbe(torch.nn.Module):
    param_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.dit = SimpleNamespace(blocks=[None])
        self.calls = []
        self.eval()

    def forward(self, **kwargs):
        cache = kwargs['past_key_values']
        prior = cache.key_cache[0]
        x = kwargs['x'][0]
        out_rows = kwargs['img_pos_for_infer_output_info']['position_ids']
        past = x.new_tensor(0) if prior is None else prior.mean()
        value = x[out_rows]*.21 + past*.03 + kwargs['prompt_embeds'].float().sum()*.00001
        value = value + kwargs['img_position_ids'][0,out_rows,0,None].float()*.0001
        value = value + kwargs['unique_timesteps'][kwargs['inverse_indices'][out_rows],None]*.01
        if kwargs['update_past_key_values']:
            keys = x.clone() if prior is None else torch.cat((prior, x))
            cache.update_kvlens(kwargs['sample_lens'])
            cache.update_kvcache(0, keys, keys*.7)
        record = {key: kwargs[key].detach().clone() for key in
                  ('x','eps','img_position_ids','unique_timesteps','inverse_indices')}
        record.update(store=kwargs['update_past_key_values'], groups=copy.deepcopy(cache.groups),
                      kv=None if cache.key_cache[0] is None else cache.key_cache[0].clone())
        self.calls.append(record)
        return value, x[:0]


def inputs():
    static = dict(prompt_embeds=torch.ones(3, 32), text_token_tags=torch.tensor([1,0,1]))
    picture = torch.arange(24*2*4, dtype=torch.float32).reshape(24,1,2,4)/100
    chunks = [dict(references=[dict(kind='video',role='semantic',video=torch.full((24,t,2,4),k+.25))],eos=t==2)
              for k,t in enumerate((5,5,5,2))]
    return static, picture, chunks


def run_stream(driver):
    static, picture, chunks = inputs()
    net = CacheProbe()
    consumed = []
    def source():
        for k, item in enumerate(chunks):
            consumed.append(k)
            yield item
    stream = driver.stream_latents(net, static, source(), first_frame=picture,
        conditioning_mode='static_image_qwen_proxy_vae_v1',seed=73,condition_seed=94)
    results = []
    for k, result in enumerate(stream):
        if consumed != list(range(k+1)):
            raise AssertionError('future chunk was consumed before current yield')
        results.append(result.clone())
    return results, net.calls


class RefInferenceTests(unittest.TestCase):
    def legacy(self, mode='plain'):
        cls = legacy_class()
        result = cls.__new__(cls)
        result.config = configuration(guidance=mode)
        result.guidance = dict(mode=mode, drop='proxy', scale=2. if mode=='cfg' else 1.)
        result.sampling_timesteps, result.sampler = Grid(), Euler()
        return result

    def independent(self, mode='plain'):
        recipe = RefRecipe.from_training(recipe_config('tf' if mode!='plain' else 'dmd',mode))
        from types import ModuleType
        boundary = ModuleType('common.diffusion')
        def build(config):
            self.assertEqual(config, recipe.to_dict()['diffusion'])
            return dict(sampling_timesteps=Grid(), sampler=Euler())
        boundary.build_diffusion = build
        with patch.dict('sys.modules', {'common.diffusion':boundary}):
            return RefFixedHistory(recipe)

    def test_legacy_characterization(self):
        values, calls = run_stream(self.legacy())
        self.assertEqual([v.shape[1] for v in values], [5,5,5,2])
        self.assertTrue(all(torch.isfinite(v).all() for v in values))
        self.assertEqual([c['store'] for c in calls], [True]+[True,False,False,False,True]*4)
        self.assertEqual([g for g,_ in calls[-1]['groups'][0]], [-1,2,3])
        self.assertTrue(torch.equal(calls[2]['unique_timesteps'], torch.tensor([1.])))
        self.assertTrue(all(not v.requires_grad for v in values))

    def test_old_new_latents_and_kv_match(self):
        for mode in ('plain','cfg','gft'):
            for sp in (1,2):
                with self.subTest(mode=mode,sp=sp):
                    old, new = self.legacy(mode), self.independent(mode)
                    old._sp_size = new._sp_size = lambda: sp
                    old._pad_rows = new._pad_rows = lambda n: 0 if sp==1 else sp-n%sp
                    expected, before = run_stream(old)
                    actual, after = run_stream(new)
                    for a,b in zip(actual,expected,strict=True):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    self.assertEqual(len(before),len(after))
                    for a,b in zip(after,before,strict=True):
                        self.assertEqual(a.keys(),b.keys())
                        for key in a:
                            if isinstance(a[key],torch.Tensor):
                                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                            else:
                                self.assertEqual(a[key],b[key])

    def test_recipe_resolves_grid_and_guidance_and_rejects_drift(self):
        for stage in ('tf','tscd','dmd'):
            config = recipe_config(stage)
            recipe = RefRecipe.from_training(config)
            self.assertEqual(recipe.diffusion['sampling_timesteps']['args']['num_sampling_steps'],2 if stage=='tscd' else 3)
            config['inference_recipe'] = recipe.to_dict()
            self.assertEqual(RefRecipe.from_training(config),recipe)
            config['data']['args']['window_size'] = 0
            with self.assertRaisesRegex(ValueError,'differ'):
                RefRecipe.from_training(config)
        config = recipe_config('tf','cfg'); config.validation['cfg_scale']=3.5
        self.assertEqual(RefRecipe.from_training(config).guidance['scale'],3.5)
        config = recipe_config(); config.meta_model.pop('guidance'); config.validation['cfg_scale']=9
        self.assertEqual(RefRecipe.from_training(config).guidance['scale'],1.)
        for steps in (0,-1,True):
            with self.assertRaises(ValueError):
                RefRecipe.from_training(config).with_steps(steps)
        with self.assertRaisesRegex(ValueError,'not SFT'):
            RefRecipe.from_training(recipe_config('sft'))

    def test_invalid_streams_and_no_future_prefix(self):
        for driver in (self.legacy(),self.independent()):
            static,picture,chunks = inputs()
            def stream(items):
                return driver.stream_latents(CacheProbe(),static,items,first_frame=picture,
                    conditioning_mode='static_image_qwen_proxy_vae_v1',seed=73,condition_seed=94)
            full = list(stream(chunks))
            prefix = list(stream(chunks[:1]))
            torch.testing.assert_close(full[0],prefix[0],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'after EOS'):
                list(stream(chunks+[chunks[0]]))
            changed = copy.deepcopy(chunks)
            changed[1]['references'][0]['role']='depth'
            with self.assertRaisesRegex(ValueError,'roles or resolution'):
                list(stream(changed))
            changed = copy.deepcopy(chunks)
            changed[0]['condition']=static
            with self.assertRaisesRegex(ValueError,'omit Qwen'):
                list(stream(changed))

    def test_artifact_survives_without_training_cache_and_rejects_asset_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, encoder, path = (root/name for name in ('generator.safetensors','encoder.bin','artifact.json'))
            header = json.dumps({'dit.video_patch_proj.weight':dict(dtype='F32',shape=[1],data_offsets=[0,4]),
                                 'dit.blocks.0.weight':dict(dtype='F32',shape=[1],data_offsets=[4,8])}).encode()
            weights.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*8)
            encoder.write_bytes(b'frozen test encoder')
            cfg = recipe_config()
            cfg.meta_model.update(module=PROJECT+'.meta_models.minimax_h3_ref2va', class_name='MiniMaxH3RefDMD')
            placement = dict(weight_dtype='float32',plugins=[dict(module=PROJECT+'.modeling.fp32_placement',
                class_name='MiniMaxH3FP32ModulePlacement')],fsdp=dict(param_dtype='bfloat16',reduce_dtype='float32',
                cast_forward_inputs=False,wrap_modules=['CausalMiniMaxH3DiTBlock','MiniMaxH3TokenRefinerBlock'],shard_size=8,cpu_offload=False))
            cfg.models = dict(backbone=dict(module=PROJECT+'.modeling.build',class_name='MiniMaxH3CausalX0DiTSP',args={},
                placement=placement,optimizer={'private':'training-only'},adapter={'r':8},ema={}))
            cfg.distributed = dict(up_size=[8])
            cfg.engine = dict(up_size=dict(train=8))
            cfg.data.args['manifest'] = str(root/'missing-training-manifest.jsonl')
            encoder_recipe = dict(conditioning_mode='static_image_qwen_proxy_vae_v1',posterior='mean',
                system_prompt='fixed',model_hashes={'encoder.bin':sha256(encoder)},encoder_code={})
            bundle = export_bundle(cfg,weights,root,encoder_recipe,provenance=dict(source_stage='dmd'))
            path.write_text(json.dumps(bundle))
            actual, recipe = load_bundle(path)
            self.assertEqual(actual,bundle)
            self.assertNotIn('optimizer',actual['model'])
            self.assertNotIn('data',actual)
            self.assertEqual(recipe,RefRecipe.from_training(cfg))
            import av
            import numpy as np
            from PIL import Image
            from projects.minimax_h3.tools.infer_ref_fixed_history import main
            proxy, picture, prompt = (root/name for name in ('proxy.mkv','first.png','prompt.txt'))
            with av.open(str(proxy),'w') as container:
                stream = container.add_stream('ffv1',rate=24)
                stream.width,stream.height,stream.pix_fmt = 32,32,'yuv420p'
                for _ in range(5):
                    frame = av.VideoFrame.from_ndarray(np.zeros((32,32,3),dtype=np.uint8),format='rgb24')
                    container.mux(stream.encode(frame))
                container.mux(stream.encode())
            Image.new('RGB',(32,32)).save(picture)
            prompt.write_text('fixed')
            bundle['encoders']['recipe'].update(height=32,width=32,proxy_height=32,proxy_width=32)
            path.write_text(json.dumps(bundle))
            captured = io.StringIO()
            with redirect_stdout(captured):
                main(['--artifact',str(path),'--proxy-video',str(proxy),'--first-frame',str(picture),
                    '--prompt-file',str(prompt),'--num-frames','5','--output-dir',str(root/'output'),'--check-only'])
            self.assertEqual(json.loads(captured.getvalue())['status'],'checked')
            self.assertFalse((root/'output').exists())
            graph_config = root/'graph.json'
            graph_config.write_text(json.dumps({'dit': {'cuda_graph': True}}))
            graph_args = ['--artifact', str(path), '--proxy-video', str(proxy),
                '--first-frame', str(picture), '--prompt-file', str(prompt), '--num-frames', '5',
                '--output-dir', str(root/'output'), '--inference-config', str(graph_config)]
            with patch.dict(os.environ, {'NCCL_GRAPH_REGISTER': '1'}):
                checked = io.StringIO()
                with redirect_stdout(checked):
                    main([*graph_args, '--check-only'])
                self.assertTrue(json.loads(checked.getvalue())['runtime']['dit']['cuda_graph'])
                with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_REGISTER=0'):
                    main(graph_args)
            self.assertFalse((root/'output').exists())
            for alter in (lambda b:b['deployment'].update(placement={}),
                          lambda b:b['deployment']['placement']['fsdp'].update(cast_forward_inputs=True),
                          lambda b:b['model'].update(module='projects.minimax_h3.meta_models.bad')):
                changed = copy.deepcopy(bundle); alter(changed)
                with self.assertRaises(ValueError):
                    validate_bundle(changed)
            encoder.write_bytes(b'changed encoder')
            with self.assertRaisesRegex(ValueError,'frozen encoder differs'):
                load_bundle(path)
            weights.write_bytes(b'changed generator')
            with self.assertRaisesRegex(ValueError,'generator differs'):
                load_bundle(path)

    def test_legacy_steps_override_updates_recorded_contract(self):
        from types import ModuleType
        from projects.minimax_h3.tools.ref2va_infer import inference_config
        cfg_module = ModuleType('common.config')
        cfg_module.CfgNode = nodes
        args = SimpleNamespace(inference_config=None,decode_mode='strict',weights='/tmp/weights',output_dir='/tmp/output',
            checkpoint='/tmp/encoders',seed=4,condition_seed=None,config='/tmp/train.yaml',steps=4)
        for stage in ('tf','tscd','dmd'):
            cfg = recipe_config(stage)
            cfg.diffusion['audio_sampling_timesteps'] = copy.deepcopy(cfg.diffusion['sampling_timesteps'])
            cfg.validation['audio_sampling_timesteps'] = copy.deepcopy(cfg.validation['sampling_timesteps'])
            cfg.models = dict(backbone={})
            cfg.engine = {}; cfg.lineage = {}
            cfg['inference_recipe'] = RefRecipe.from_training(cfg).to_dict()
            with patch.dict('sys.modules',{'common.config':cfg_module}):
                actual = inference_config(args,cfg,{})
            self.assertEqual(actual.models.backbone.placement.class_name,'ResidentInferencePlacement')
            self.assertEqual(RefRecipe.from_training(actual).diffusion['sampling_timesteps']['args']['num_sampling_steps'],4)

    def test_real_raven_sampling_matches_frozen_stream(self):
        for stage in ('tf','tscd','dmd'):
            config = recipe_config(stage)
            recipe = RefRecipe.from_training(config)
            new = RefFixedHistory(recipe)
            old = self.legacy()
            from common.diffusion import build_diffusion
            diffusion = build_diffusion(recipe.to_dict()['diffusion'])
            if stage == 'tscd':
                old.validation_diffusion = diffusion
            else:
                old.sampling_timesteps, old.sampler = diffusion['sampling_timesteps'],diffusion['sampler']
            before,_ = run_stream(old)
            after,_ = run_stream(new)
            self.assertEqual(new.sampling_timesteps.timesteps.dtype,torch.float64)
            for a,b in zip(after,before,strict=True):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_acceptance_keeps_recorded_window(self):
        from projects.minimax_h3.tools.ref2va_acceptance import stage_config
        with tempfile.TemporaryDirectory() as directory:
            for stage in ('tf','tscd','dmd'):
                config = recipe_config(stage)
                config['inference_recipe'] = RefRecipe.from_training(config).to_dict()
                (Path(directory)/f'{stage}.yaml').write_text(json.dumps(config))
                actual = stage_config(directory,stage)
                self.assertEqual(actual.data.args.window_size,1)
                self.assertEqual(RefRecipe.from_training(actual).window_size,1)


if __name__ == '__main__':
    unittest.main()
