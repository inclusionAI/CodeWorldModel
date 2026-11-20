"""Run settings validation and comparison need no model runtime."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from inference.settings import (EFFECTIVE_SCHEMA, RUN_SCHEMA, backbone_dit_options, diff_settings,
    effective_settings, inspect_config, inspection_parser, load_inference_options,
    options_for_args, parse_run_arguments, read_settings, require_dit_graph_environment)
from modeling.ref_recipe import RefRecipe

ROOT = Path(__file__).resolve().parents[1]


def artifact(protocol='ref_fixed_recent_history/v1'):
    recipe = RefRecipe('static_image_qwen_proxy_vae_v1', 2, 37,
        dict(schedule=dict(module='common.diffusion.schedule.lerp', class_name='LinearInterpolationSchedule',
                           args=dict(pred_type='x_0', T=1.0)),
             sampling_timesteps=dict(module='common.diffusion.timestep.sampling.trailing',
                 class_name='TrailingSamplingTimesteps', args=dict(T=1.0, num_sampling_steps=4, shift=12)),
             sampler=dict(module='projects.minimax_h3.modeling.sampling', class_name='EulerSampler', args={})),
        dict(mode='plain', drop='proxy', scale=1.), protocol=protocol)
    return dict(schema='ref_inference_artifact/v1', recipe=recipe.to_dict(),
        model=dict(module='projects.minimax_h3.modeling.build', class_name='MiniMaxH3CausalX0DiTSP', args={}),
        weights=dict(format='raven_full_or_merged', path='/assets/weights.safetensors', sha256='a'*64),
        encoders=dict(checkpoint='/assets/frozen', recipe=dict(model_hashes={'vae.bin': 'b'*64})),
        deployment=dict(distributed=dict(up_size=[8]), active_up_size=8,
                        placement=dict(fsdp=dict(shard_size=8))))


class SettingsTests(unittest.TestCase):
    def fusion_manifest(self):
        from inference.raw_recipe import RawLatentRecipe
        record = artifact()
        original = RefRecipe(**record['recipe'])
        record['recipe'] = RawLatentRecipe(original.conditioning_mode, original.rope_reference_latents,
                                           original.diffusion, original.guidance).to_dict()
        self.artifact.write_text(json.dumps(record), encoding='utf-8')
        for name in ('decoder.bin', 'decoder.json'):
            (self.root/name).write_text('fixture')
        self.manifest.update(entrypoint='serve_ref', run=dict(artifact=['artifact.json'], output_dir='serve'),
            deployment=dict(dit=dict(delayed_hc=True, fuse_first_denoise=True),
                codec=dict(decoder='causal', decoder_weights='decoder.bin', decoder_config='decoder.json')))
        return record

    def test_fuse_first_denoise_boolean_dependencies_and_projection(self):
        defaults = load_inference_options()
        self.assertFalse(defaults['dit']['fuse_first_denoise'])
        for invalid in (1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise.*boolean'):
                load_inference_options(data=dict(dit=dict(fuse_first_denoise=invalid)))
        for changes in (dict(delayed_hc=False), dict(delayed_hc=True, cuda_graph=True)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise requires'):
                load_inference_options(data=dict(dit=dict(fuse_first_denoise=True, **changes)))
        for enabled in (False, True):
            options = load_inference_options(data=dict(dit=dict(delayed_hc=True, fuse_first_denoise=enabled)))
            before = deepcopy(options)
            projected = backbone_dit_options(options)
            expected = dict(options['dit'])
            expected.pop('fuse_first_denoise')
            self.assertEqual(projected, expected)
            projected['kv'] = True
            self.assertEqual(options, before)

    def test_fuse_first_denoise_receipt_and_recipe_guards(self):
        from inference.raw_recipe import RAW_PROTOCOLS
        record = self.fusion_manifest()
        for enabled in (False, True):
            self.manifest['deployment']['dit']['fuse_first_denoise'] = enabled
            result = inspect_config(self.write())
            self.assertIs(result['deployment']['options']['dit']['fuse_first_denoise'], enabled)
        for recipe in [artifact()['recipe'],
                       *(dict(record['recipe'], protocol=p) for p in RAW_PROTOCOLS[1:]),
                       dict(record['recipe'], guidance=dict(mode='gft', drop='proxy', scale=1.)),
                       dict(record['recipe'], guidance=dict(mode='cfg', drop='proxy', scale=1.)),
                       dict(record['recipe'], guidance=dict(mode='cfg', drop='proxy', scale=2.))]:
            with self.subTest(recipe=recipe['protocol'], guidance=recipe['guidance']):
                self.artifact.write_text(json.dumps(dict(record, recipe=recipe)), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise requires'):
                    inspect_config(self.write())
        record['recipe']['diffusion']['sampling_timesteps']['args']['dynamic_shift'] = True
        self.artifact.write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'length-independent grid'):
            inspect_config(self.write())

    def test_fuse_first_denoise_checks_every_served_artifact(self):
        record = self.fusion_manifest()
        record['recipe']['guidance']['mode'] = 'gft'
        (self.root/'other.json').write_text(json.dumps(record), encoding='utf-8')
        self.manifest['run']['artifact'].append('other.json')
        with self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise requires'):
            inspect_config(self.write())

    def test_fuse_first_denoise_rejects_resolved_dynamic_grid(self):
        from inference.raw_recipe import RawLatentRecipe
        record = self.fusion_manifest()
        args = self.parse()
        recipe = RawLatentRecipe(**deepcopy(record['recipe']))
        recipe.diffusion['sampling_timesteps']['args']['dynamic_shift'] = True
        with self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise requires a static sampling grid'):
            effective_settings(args, recipe=recipe)

    def test_fuse_first_denoise_rejected_outside_serving(self):
        from argparse import Namespace
        from inference.settings import ENTRYPOINTS
        path = self.root/'deployment.yaml'
        path.write_text('dit:\n  delayed_hc: true\n  fuse_first_denoise: true\n')
        for entrypoint in ENTRYPOINTS:
            if entrypoint == 'serve_ref':
                continue
            record = deepcopy(self.manifest)
            record.update(entrypoint=entrypoint, deployment=dict(dit=dict(delayed_hc=True, fuse_first_denoise=True)))
            if entrypoint == 'ref2va_infer':
                record['run'] = dict(config='train.yaml', weights='weights', checkpoint='encoders',
                                     bundle='input', output_dir='output')
            self.rejects(record, 'dit.fuse_first_denoise is supported only by serve_ref')
            args = Namespace(inference_config=str(path), _settings_entrypoint=entrypoint)
            with self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise is supported only by serve_ref'):
                options_for_args(args)
            with self.assertRaisesRegex(ValueError, 'dit.fuse_first_denoise is supported only by serve_ref'):
                effective_settings(args, options=load_inference_options(path))

    def test_codec_overlap_default_boolean_scope_and_receipt(self):
        from inference.raw_recipe import RawLatentRecipe
        record = artifact()
        original = RefRecipe(**record['recipe'])
        record['recipe'] = RawLatentRecipe(original.conditioning_mode, original.rope_reference_latents,
                                           original.diffusion, original.guidance).to_dict()
        self.artifact.write_text(json.dumps(record), encoding='utf-8')
        for name in ('decoder.bin', 'decoder.json'):
            (self.root/name).write_text('fixture')
        self.manifest.update(entrypoint='serve_ref',
            run=dict(artifact=['artifact.json'], output_dir='serve', runtime='ray', vae_rank=4, decoder_rank=5),
            deployment=dict(codec=dict(decoder='causal', decoder_weights='decoder.bin', decoder_config='decoder.json')))
        self.assertFalse(inspect_config(self.write())['run']['codec_overlap'])
        self.manifest['run']['codec_overlap'] = True
        self.assertTrue(inspect_config(self.write())['run']['codec_overlap'])
        for changes in (dict(runtime='torchrun'), dict(decoder_rank=4)):
            invalid = deepcopy(self.manifest)
            invalid['run'].update(changes)
            with self.assertRaisesRegex(ValueError, 'codec-overlap requires'):
                inspect_config(self.write(invalid))
        self.artifact.write_text(json.dumps(artifact()), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'codec-overlap requires'):
            inspect_config(self.write())
        self.manifest['run']['codec_overlap'] = 'true'
        self.rejects(self.manifest, 'must be a boolean')

    def test_graph_launch_environment_is_explicit_and_read_only(self):
        for register in (None, '1', '', 'false'):
            values = {} if register is None else {'NCCL_GRAPH_REGISTER': register}
            with self.subTest(register=register), patch.dict(os.environ, values, clear=True):
                require_dit_graph_environment(False)
                with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_REGISTER=0'):
                    require_dit_graph_environment(True)
                self.assertEqual(dict(os.environ), values)
        for mixing in (None, '1', '0', ''):
            values = {'NCCL_GRAPH_REGISTER': '0'}
            if mixing is not None:
                values['NCCL_GRAPH_MIXING_SUPPORT'] = mixing
            with self.subTest(mixing=mixing), patch.dict(os.environ, values, clear=True):
                if mixing in (None, '1'):
                    require_dit_graph_environment(True)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_MIXING_SUPPORT'):
                        require_dit_graph_environment(True)
                self.assertEqual(dict(os.environ), values)

    def test_dit_cuda_graph_default_validation_and_receipt(self):
        self.assertFalse(load_inference_options()['dit']['cuda_graph'])
        for invalid in (1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'boolean'):
                load_inference_options(data=dict(dit=dict(cuda_graph=invalid)))
        self.manifest['deployment'] = dict(dit=dict(cuda_graph=True))
        args = self.parse()
        self.assertTrue(effective_settings(args)['deployment']['options']['dit']['cuda_graph'])

    def test_rope_default_validation_and_receipt(self):
        self.assertFalse(load_inference_options()['dit']['rope'])
        for invalid in (1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'boolean'):
                load_inference_options(data=dict(dit=dict(rope=invalid)))
        for enabled in (True, False):
            self.manifest['deployment'] = dict(dit=dict(rope=enabled))
            args = self.parse()
            self.assertIs(options_for_args(args)['dit']['rope'], enabled)
            self.assertIs(effective_settings(args)['deployment']['options']['dit']['rope'], enabled)

    def test_delayed_hc_default_validation_and_receipt(self):
        self.assertFalse(load_inference_options()['dit']['delayed_hc'])
        with self.assertRaisesRegex(ValueError, 'boolean'):
            load_inference_options(data=dict(dit=dict(delayed_hc=1)))
        self.manifest['deployment'] = dict(dit=dict(delayed_hc=True))
        self.rejects(self.manifest, 'fixed-history v2')
        from inference.raw_recipe import RawLatentRecipe, RAW_PROTOCOLS
        record = artifact()
        original = RefRecipe(**record['recipe'])
        raw = RawLatentRecipe(original.conditioning_mode, original.rope_reference_latents,
                             original.diffusion, original.guidance)
        record['recipe'] = raw.to_dict()
        self.artifact.write_text(json.dumps(record), encoding='utf-8')
        self.manifest['entrypoint'] = 'infer_ref_raw_latent'
        self.manifest['run']['decode_mode'] = 'native'
        result = inspect_config(self.write())
        self.assertTrue(result['deployment']['options']['dit']['delayed_hc'])
        for protocol in RAW_PROTOCOLS[1::2]:
            record['recipe'] = dict(raw.to_dict(), protocol=protocol)
            self.artifact.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fixed-history v2'):
                inspect_config(self.write())
        self.manifest.update(entrypoint='serve_ref', run=dict(artifact=['artifact.json'], output_dir='serve'))
        for bad in (artifact(), record):
            self.artifact.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fixed-history v2'):
                inspect_config(self.write())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / 'artifact.json'
        self.artifact.write_text(json.dumps(artifact()), encoding='utf-8')
        self.manifest = dict(schema=RUN_SCHEMA, entrypoint='infer_ref_fixed_history',
            run=dict(artifact='artifact.json', bundle='input', output_dir='output'), deployment={})
        self.path = self.root / 'run.yaml'

    def write(self, record=None):
        self.path.write_text(yaml.safe_dump(record or self.manifest), encoding='utf-8')
        return self.path

    def parse(self, extra=()):
        return parse_run_arguments(inspection_parser(self.manifest['entrypoint']),
            ['--run-config', str(self.write()), *extra], self.manifest['entrypoint'])

    def rejects(self, record, match):
        self.write(record)
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            parse_run_arguments(inspection_parser(record['entrypoint']),
                ['--run-config', str(self.path)], record['entrypoint'])
        self.assertIn(match, errors.getvalue())

    def test_resolved_request_separates_requested_defaults_and_actual_values(self):
        self.manifest['run'].update(seed=31, window_index=2)
        implicit = inspect_config(self.write())
        self.manifest['run'].update(condition_seed=31)
        explicit = inspect_config(self.write())
        self.assertIsNone(implicit['run']['condition_seed'])
        self.assertEqual(implicit['resolved_request'], explicit['resolved_request'])
        self.assertEqual(implicit['resolved_request']['first_source_frame'], 316)
        self.assertEqual(implicit['resolved_request']['condition_seed'], 31)
        self.manifest['run'].update(start_frame=3)
        with self.assertRaisesRegex(ValueError, 'choose --start-frame'):
            inspect_config(self.write())

    def test_serving_occupier_path_is_resolved_from_run_manifest(self):
        self.manifest.update(entrypoint='serve_ref', run=dict(artifact=['artifact.json'],
            output_dir='serve', runtime='ray', gpu_occupier_config='occupier.json'))
        args = self.parse()
        self.assertEqual(args.gpu_occupier_config, str(self.root / 'occupier.json'))
        self.assertEqual(args.runtime, 'ray')

    def test_native_inspection_uses_pinned_inheritance_and_resolved_execution_config(self):
        from inference.settings import _training_config
        sample = artifact()
        self.manifest = dict(schema=RUN_SCHEMA, entrypoint='ref2va_infer', deployment={},
            run=dict(config='child.yaml', weights='weights.safetensors', checkpoint='encoders', bundle='input'))
        recipe = sample['recipe']
        parent = dict(data=dict(args=dict(stage='dmd', conditioning_mode=recipe['conditioning_mode'],
            window_size=recipe['window_size'], rope_reference_latents=recipe['rope_reference_latents'])),
            meta_model={}, diffusion=recipe['diffusion'], models=dict(backbone=dict(placement={})),
            distributed=dict(up_size=[8]), engine=dict(up_size=dict(train=8)))
        (self.root/'parent.yaml').write_text(yaml.safe_dump(parent), encoding='utf-8')
        child = self.root/'child.yaml'
        child.write_text(yaml.safe_dump({'__inherit__': 'parent.yaml',
            'override_list': ['data.args.window_size', '1']}), encoding='utf-8')
        resolved = inspect_config(self.write())
        self.assertEqual(resolved['artifacts'][0]['recipe']['window_size'], 1)
        args = self.parse(['--output-dir', str(self.root/'out')])
        training = _training_config(child)
        child.unlink()
        snapshot = effective_settings(args, training=training)
        self.assertEqual(snapshot['artifacts'][0]['recipe'], resolved['artifacts'][0]['recipe'])

    def test_deployment_omitted_defaults_and_explicit_parity(self):
        defaults = load_inference_options()
        self.assertTrue(defaults['resident_weights'])
        self.assertFalse(defaults['qwen_fsdp'])
        self.assertEqual(defaults['dit'], dict(metadata=False, kv=False, adaln=False, delayed_hc=False,
                                             fuse_first_denoise=False, rope=False, cuda_graph=False))
        self.assertEqual(defaults['codec']['window'], 16)
        self.assertFalse(defaults['output']['async_write'])
        path = self.root / 'deployment.yaml'
        path.write_text(yaml.safe_dump(defaults))
        self.assertEqual(load_inference_options(path), defaults)
        self.assertEqual(load_inference_options(data=defaults), defaults)
        self.assertEqual(load_inference_options(ROOT / 'configs/inference.yaml'),
                         load_inference_options(data=read_settings(ROOT / 'configs/inference.yaml')))

    def test_old_cli_and_manifest_have_same_effective_values(self):
        args = self.parse()
        parser = inspection_parser('infer_ref_fixed_history')
        old = parse_run_arguments(parser, ['--artifact', str(self.artifact), '--bundle', str(self.root/'input'),
            '--output-dir', str(self.root/'output')], 'infer_ref_fixed_history')
        effective = effective_settings(args)
        previous = effective_settings(old)
        self.assertEqual(effective['run'], previous['run'])
        self.assertEqual(effective['deployment'], previous['deployment'])
        self.assertEqual(options_for_args(args), options_for_args(old))

    def test_relative_paths_and_codec_assets(self):
        (self.root/'decoder.bin').write_bytes(b'not loaded by settings')
        (self.root/'decoder.json').write_text('{}')
        self.manifest['deployment'] = dict(codec=dict(decoder='causal',
            decoder_weights='decoder.bin', decoder_config='decoder.json'))
        args = self.parse()
        self.assertEqual(args.artifact, str(self.artifact))
        self.assertEqual(args.bundle, str(self.root/'input'))
        self.assertEqual(options_for_args(args)['codec']['decoder_weights'], str(self.root/'decoder.bin'))
        self.assertEqual(effective_settings(args)['units'],
                         dict(history_window='past_time_groups', decoder_window='spatial'))

    def test_types_unknown_and_choices(self):
        for key, value, error in [('steps', True, 'invalid value type'),
                                  ('seed', None, 'invalid value type'),
                                  ('decode_mode', 'bad', 'invalid choice'),
                                  ('recipe', {}, 'unknown run option'),
                                  ('artifact', ['a'], 'invalid value type')]:
            record = deepcopy(self.manifest)
            record['run'][key] = value
            self.rejects(record, error)
        for options in [dict(qwen_fsdp=2), dict(dit=dict(kv=1)), dict(codec=dict(window=True)),
                        dict(output=dict(queue_size=0)), dict(unknown=True),
                        dict(codec=dict(compile=True)), dict(codec=dict(decoder='causal'))]:
            with self.assertRaises(ValueError):
                load_inference_options(data=options)

    def test_qwen_fsdp_rejected_outside_serving_in_manifest_legacy_and_inspection(self):
        from inference.settings import ENTRYPOINTS
        path = self.root/'deployment.yaml'
        path.write_text('qwen_fsdp: true\n')
        for entrypoint in ENTRYPOINTS:
            if entrypoint == 'serve_ref':
                continue
            record = deepcopy(self.manifest)
            record.update(entrypoint=entrypoint, deployment=dict(qwen_fsdp=True))
            if entrypoint == 'ref2va_infer':
                record['run'] = dict(config='train.yaml', weights='weights', checkpoint='encoders',
                                     bundle='input', output_dir='output')
            self.rejects(record, 'qwen_fsdp is supported only by serve_ref')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                inspect_config(self.write(record))
            from argparse import Namespace
            args = Namespace(inference_config=str(path), _settings_entrypoint=entrypoint)
            with self.assertRaisesRegex(ValueError, 'qwen_fsdp is supported only by serve_ref'):
                options_for_args(args)
            with self.assertRaisesRegex(ValueError, 'qwen_fsdp is supported only by serve_ref'):
                effective_settings(args, options=load_inference_options(path))

    def test_duplicate_keys_and_cli_ambiguity(self):
        self.path.write_text('a: 1\na: 2\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_settings(self.path)
        for extra in [('--output-dir', 'other'), ('--inference-config', 'other'),
                      ('--steps', '5'), ('--run-config', str(self.path))]:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(extra)
        self.manifest['run'].pop('output_dir')
        self.assertEqual(self.parse(['--output-dir', 'external', '--check-only']).output_dir, 'external')

    def test_required_and_exclusive_groups_remain_enforced(self):
        record = deepcopy(self.manifest)
        record['run'].pop('bundle')
        self.rejects(record, 'required')
        record['run'].update(bundle='a', proxy_video='b')
        self.rejects(record, 'not allowed')
        self.manifest['entrypoint'] += '_streaming_text'
        self.rejects(self.manifest, 'required')
        self.manifest['run']['prompts_jsonl'] = 'prompts.jsonl'
        args = self.parse()
        self.assertEqual(args.prompts_jsonl, str(self.root/'prompts.jsonl'))

    def test_entrypoint_and_protocol_mismatch(self):
        self.write()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_run_arguments(inspection_parser('infer_ref_bidir_history'),
                                ['--run-config', str(self.path)], 'infer_ref_bidir_history')
        record = artifact('ref_bidir_recent_history/v1')
        self.artifact.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'artifact selects'):
            inspect_config(self.path)

    def test_full_artifact_expansion_and_override_diff(self):
        baseline = inspect_config(self.write())
        self.manifest['run']['steps'] = 8
        self.manifest['deployment'] = dict(resident_weights=False, dit=dict(kv=True))
        changed = inspect_config(self.write())
        self.assertEqual(changed['artifacts'][0]['weights']['format'], 'raven_full_or_merged')
        self.assertEqual(changed['artifacts'][0]['encoders']['recipe']['model_hashes']['vae.bin'], 'b'*64)
        self.assertEqual(changed['artifacts'][0]['source_recipe']['diffusion']['sampling_timesteps']['args']['num_sampling_steps'], 4)
        paths = {row['path'] for row in diff_settings(baseline, changed)}
        self.assertIn('deployment.options.dit.kv', paths)
        self.assertIn('deployment.selected_placement', paths)
        self.assertIn('artifacts[0].recipe.diffusion.sampling_timesteps.args.num_sampling_steps', paths)
        self.assertEqual(diff_settings({}, {'x': None}), [dict(path='x', change='added', after=None)])
        self.assertEqual(diff_settings({'x': None}, {}), [dict(path='x', change='removed', before=None)])
        self.assertEqual(diff_settings({'a': []}, {'a': [None]}), [dict(path='a[0]', change='added', after=None)])
        self.assertEqual(diff_settings({'a': False}, {}), [dict(path='a', change='removed', before=False)])

    def test_inspection_without_output_and_known_bundle_without_rereading(self):
        self.manifest['run'].pop('output_dir')
        args = self.parse(['--output-dir', 'explicit-output'])
        known = read_settings(self.artifact)
        self.artifact.unlink()
        snapshot = effective_settings(args, bundle=known)
        self.assertEqual(snapshot['artifacts'][0]['recipe'], known['recipe'])
        self.artifact.write_text(json.dumps(known))
        snapshot = inspect_config(self.write())
        self.assertIsNone(snapshot['run']['output_dir'])

    def test_serving_preserves_each_recipe_and_rejects_incompatible_options(self):
        other = self.root/'bidir.json'
        other.write_text(json.dumps(artifact('ref_bidir_recent_history/v1')))
        (self.root/'decoder.bin').write_bytes(b'x')
        (self.root/'decoder.json').write_text('{}')
        self.manifest.update(entrypoint='serve_ref', run=dict(artifact=['artifact.json', 'bidir.json'], output_dir='serve'),
            deployment=dict(codec=dict(decoder='causal', decoder_weights='decoder.bin', decoder_config='decoder.json')))
        snapshot = inspect_config(self.write())
        self.assertEqual(len(snapshot['artifacts']), 2)
        self.assertNotEqual(snapshot['artifacts'][0]['recipe'], snapshot['artifacts'][1]['recipe'])
        self.manifest['deployment']['qwen_fsdp'] = True
        snapshot = inspect_config(self.write())
        self.assertTrue(snapshot['deployment']['options']['qwen_fsdp'])
        self.assertEqual(snapshot['deployment']['qwen_ranks'], [0, 1])
        for setting in [dict(resident_weights=False), dict(output=dict(async_write=True))]:
            self.manifest['deployment'].update(setting)
            with self.assertRaisesRegex(ValueError, 'serving requires'):
                inspect_config(self.write())
            self.manifest['deployment'].pop(next(iter(setting)))

    def test_serving_accepts_all_four_recipes_and_source_provenance_drift(self):
        from inference.settings import PROTOCOLS
        paths = []
        for index, protocol in enumerate(PROTOCOLS.values()):
            record = artifact(protocol)
            record['encoders']['recipe']['encoder_code'] = {'encoder.py': str(index)}
            path = self.root/f'recipe{index}.json'
            path.write_text(json.dumps(record))
            paths.append(path.name)
        (self.root/'decoder.bin').write_bytes(b'x')
        (self.root/'decoder.json').write_text('{}')
        self.manifest.update(entrypoint='serve_ref', run=dict(artifact=paths, output_dir='serve'),
            deployment=dict(qwen_fsdp=True, codec=dict(decoder='causal',
                decoder_weights='decoder.bin', decoder_config='decoder.json')))
        snapshot = inspect_config(self.write())
        self.assertEqual([a['recipe']['protocol'] for a in snapshot['artifacts']], list(PROTOCOLS.values()))
        self.assertEqual([a['encoders']['recipe']['encoder_code']['encoder.py'] for a in snapshot['artifacts']],
                         ['0', '1', '2', '3'])
        record['weights']['sha256'] = 'c'*64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'identical resident assets'):
            inspect_config(self.write())

    def test_resolve_diff_cli_without_runtime_imports(self):
        self.write()
        output = self.root/'resolved.json'
        command = [sys.executable, str(ROOT/'tools/inference_settings.py')]
        result = subprocess.run([*command, 'resolve', str(self.path), '-o', str(output)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())['schema'], EFFECTIVE_SCHEMA)
        same = subprocess.run([*command, 'diff', str(output), str(output), '--fail-on-difference'],
                              text=True, capture_output=True)
        self.assertEqual(same.returncode, 0, same.stderr)
        self.assertEqual(json.loads(same.stdout), [])
        changed = self.root/'changed.json'
        record = json.loads(output.read_text())
        record['deployment']['options']['dit']['metadata'] = True
        changed.write_text(json.dumps(record))
        result = subprocess.run([*command, 'diff', str(output), str(changed), '--fail-on-difference'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]['path'], 'deployment.options.dit.metadata')
        probe = subprocess.run([sys.executable, '-c',
            "import sys; import inference.settings; assert 'torch' not in sys.modules"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == '__main__':
    unittest.main()
