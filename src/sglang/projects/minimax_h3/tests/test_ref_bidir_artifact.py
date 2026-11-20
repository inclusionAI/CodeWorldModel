"""Explicit recipe export provenance and crossed entry-point refusal."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from projects.minimax_h3.inference.artifact import export_bundle, load_bundle, sha256, validate_bundle
from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE, FIXED_RECIPE, PROJECT, RefRecipe
from projects.minimax_h3.tests.test_ref_inference import recipe_config


def assets(root):
    weights = root/'weights.safetensors'
    save_file({'dit.video_patch_proj.weight': torch.ones(1), 'dit.blocks.0.weight': torch.ones(1)}, weights)
    encoder = root/'encoder.bin'
    encoder.write_bytes(b'frozen encoder fixture')
    cfg = recipe_config()
    cfg.meta_model.update(module=PROJECT+'.meta_models.minimax_h3_ref2va', class_name='MiniMaxH3RefDMD')
    cfg.models = dict(backbone=dict(module=PROJECT+'.modeling.build', class_name='MiniMaxH3CausalX0DiTSP', args={},
        placement=dict(weight_dtype='float32', plugins=[dict(module=PROJECT+'.modeling.fp32_placement',
            class_name='MiniMaxH3FP32ModulePlacement')], fsdp=dict(param_dtype='bfloat16', reduce_dtype='float32',
                cast_forward_inputs=False, wrap_modules=['CausalMiniMaxH3DiTBlock','MiniMaxH3TokenRefinerBlock']))))
    cfg.distributed = dict(up_size=[1])
    cfg.engine = dict(up_size=dict(train=1))
    cfg['inference_recipe'] = RefRecipe.from_training(cfg).to_dict()
    encoder_recipe = dict(conditioning_mode=cfg.data.args.conditioning_mode, posterior='mean', system_prompt='fixed',
        model_hashes={'encoder.bin': sha256(encoder)}, encoder_code={}, height=32, width=32, proxy_height=32, proxy_width=32)
    return cfg, weights, encoder_recipe


def override_bundle(root, cfg, weights, encoders, override):
    config_path, override_path = root/'training.json', root/'recipe.json'
    config_path.write_text(json.dumps(cfg))
    override_path.write_text(json.dumps(override))
    return export_bundle(cfg, weights, root, encoders, provenance=dict(training_config_sha256=sha256(config_path)),
                         recipe_override=override, recipe_override_sha256=sha256(override_path))


class BidirArtifactTests(unittest.TestCase):
    def test_protocols_and_local_training_inference_remain_strict(self):
        source = recipe_config()
        fixed = RefRecipe.from_training(source)
        self.assertEqual(FIXED_RECIPE, 'ref_fixed_recent_history/v1')
        self.assertEqual(fixed.protocol, 'ref_fixed_recent_history/v1')
        self.assertEqual(RefFixedHistory.PROTOCOL, 'ref_fixed_recent_history/v1')
        self.assertEqual(BIDIR_RECIPE, 'ref_bidir_recent_history/v1')
        old_recipe = dict(fixed.to_dict(), protocol='ref_causal_clean_history/v1')
        with self.assertRaisesRegex(ValueError, 'unsupported Ref'):
            RefRecipe(**old_recipe)
        source['inference_recipe'] = copy.deepcopy(old_recipe)
        with self.assertRaisesRegex(ValueError, 'differ'):
            RefRecipe.from_training(source)
        self.assertEqual(source['inference_recipe'], old_recipe)
        bidir = replace(fixed, protocol=BIDIR_RECIPE)
        self.assertEqual(RefRecipe(**bidir.to_dict()), bidir)
        with self.assertRaisesRegex(ValueError, 'requires protocol'):
            RefFixedHistory(bidir)
        with self.assertRaisesRegex(ValueError, 'requires protocol'):
            RefBidirHistory(fixed)
        with self.assertRaisesRegex(ValueError, 'unsupported Ref'):
            replace(bidir, protocol='not-a-recipe')
        with self.assertRaisesRegex(ValueError, 'static-image'):
            replace(bidir, conditioning_mode='legacy')
        source['inference_recipe'] = bidir.to_dict()
        with self.assertRaisesRegex(ValueError, 'differ'):
            RefRecipe.from_training(source)

    def test_explicit_override_preserves_source_disagreement_and_never_bypasses_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, encoders = assets(root)
            override = replace(RefRecipe.from_training(cfg), protocol=BIDIR_RECIPE).to_dict()
            cfg['inference_recipe']['protocol'] = 'ref_causal_clean_history/v1'
            source = copy.deepcopy(cfg)
            bundle = override_bundle(root, cfg, weights, encoders, override)
            self.assertEqual(cfg, source)
            provenance = bundle['provenance']
            self.assertEqual(provenance['source_recorded_recipe'], source['inference_recipe'])
            self.assertEqual(provenance['source_meta_model'], source['meta_model'])
            self.assertEqual(provenance['effective_recipe'], override)
            self.assertEqual(provenance['recipe_override_sha256'], sha256(root/'recipe.json'))
            self.assertEqual(provenance['training_config_sha256'], sha256(root/'training.json'))
            self.assertEqual(provenance['training_protocol_match'], 'unverified')
            path = root/'artifact.json'
            path.write_text(json.dumps(bundle))
            self.assertEqual(load_bundle(path)[0], bundle)
            cfg.meta_model.update(module='experimental.training', class_name='CustomBidirectionalStudent')
            cfg['inference_recipe']['window_size'] = 0
            with self.assertRaisesRegex(ValueError, 'explicit inference recipe'):
                export_bundle(cfg, weights, root, encoders, provenance={})
            custom = override_bundle(root, cfg, weights, encoders, override)
            self.assertEqual(custom['provenance']['source_recorded_recipe']['window_size'], 0)
            for change in (lambda c: c.models.backbone.update(module='unsupported.model'),
                           lambda c: c.models.backbone.placement.fsdp.update(cast_forward_inputs=True)):
                bad = copy.deepcopy(cfg)
                change(bad)
                with self.assertRaises(ValueError):
                    override_bundle(root, bad, weights, encoders, override)
            for key in ('protocol', 'diffusion', 'guidance'):
                incomplete = copy.deepcopy(override)
                incomplete.pop(key)
                with self.assertRaisesRegex(ValueError, 'complete RefRecipe'):
                    override_bundle(root, cfg, weights, encoders, incomplete)
            for key in ('recipe_selection', 'training_config_sha256', 'recipe_override_sha256',
                        'source_recorded_recipe', 'source_meta_model', 'effective_recipe', 'training_protocol_match'):
                bad = copy.deepcopy(bundle)
                del bad['provenance'][key]
                with self.assertRaises(ValueError):
                    validate_bundle(bad)
            (root/'encoder.bin').write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError, 'frozen encoder differs'):
                override_bundle(root, cfg, weights, encoders, override)
            save_file({'lora_A': torch.ones(1)}, weights)
            with self.assertRaisesRegex(ValueError, 'full/merged'):
                override_bundle(root, cfg, weights, encoders, override)

    def test_export_cli_complete_override_and_each_check_only_entry(self):
        from projects.minimax_h3.tools.export_ref_inference import main as export
        from projects.minimax_h3.tools.infer_ref_bidir_history import main as bidir
        from projects.minimax_h3.tools.infer_ref_fixed_history import main as fixed
        import av
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, encoders = assets(root)
            cfg.meta_model.update(module='experimental.training', class_name='CustomBidirectionalStudent')
            cfg.data.args.manifest = str(root/'manifest.jsonl')
            manifest = dict(schema='cwm_ref2va_v1', sample_id='test', fingerprint='fixed',
                conditioning_mode=cfg.data.args.conditioning_mode, latent_shape=[24,2,2,2], path='cache.pt')
            (root/'manifest.jsonl').write_text(json.dumps(manifest)+'\n')
            (root/'config.json').write_text(json.dumps(encoders))
            (root/'training.json').write_text(json.dumps(cfg))
            (root/'prompt.txt').write_text('fixed')
            override = dict(cfg['inference_recipe'], protocol=BIDIR_RECIPE)
            (root/'recipe.json').write_text(json.dumps(override))
            path = root/'artifact.json'
            with redirect_stdout(io.StringIO()):
                export(['--config', str(root/'training.json'), '--weights', str(weights), '--checkpoint', str(root),
                    '--prompt-file', str(root/'prompt.txt'), '--output', str(path), '--recipe-override', str(root/'recipe.json')])
            exported, recipe = load_bundle(path)
            self.assertEqual(recipe.to_dict(), override)
            proxy = root/'proxy.mkv'
            with av.open(str(proxy), 'w') as container:
                stream = container.add_stream('ffv1', rate=24)
                stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
                for _ in range(5):
                    container.mux(stream.encode(av.VideoFrame.from_ndarray(np.zeros((32,32,3), dtype=np.uint8), format='rgb24')))
                container.mux(stream.encode())
            picture = root/'first.png'
            Image.new('RGB', (32,32)).save(picture)
            argv = ['--artifact', str(path), '--proxy-video', str(proxy), '--first-frame', str(picture),
                    '--prompt-file', str(root/'prompt.txt'), '--num-frames', '5', '--output-dir', str(root/'output'),
                    '--steps', '2', '--check-only']
            out = io.StringIO()
            with redirect_stdout(out):
                bidir(argv)
            report = json.loads(out.getvalue())
            self.assertEqual(report['recipe']['protocol'], BIDIR_RECIPE)
            self.assertEqual(report['recipe']['diffusion']['sampling_timesteps']['args']['num_sampling_steps'], 2)
            self.assertEqual(report['recipe_provenance'], exported['provenance'])
            self.assertTrue(report['sampling_steps_overridden'])
            self.assertFalse((root/'output').exists())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                fixed(argv)
            exported['recipe'] = cfg['inference_recipe']
            exported['provenance'] = dict(recipe_selection='inferred_local_training')
            path.write_text(json.dumps(exported))
            with redirect_stdout(io.StringIO()):
                fixed(argv)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bidir(argv)


if __name__ == '__main__':
    unittest.main()
