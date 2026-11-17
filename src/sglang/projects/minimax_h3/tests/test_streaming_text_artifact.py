"""Exact new protocol export, frozen identity and live/replay CLI dispatch."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from projects.minimax_h3.inference.artifact import ROOT, load_bundle, sha256, validate_bundle
from projects.minimax_h3.modeling.ref_recipe import RefRecipe, STREAMING_TEXT_RECIPES
from projects.minimax_h3.tests.test_ref_bidir_artifact import assets, override_bundle
from projects.minimax_h3.tests.test_streaming_text import DRIVERS


class StreamingArtifactTests(unittest.TestCase):
    def test_explicit_only_encoder_identity_source_preservation_and_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, encoder = assets(root)
            encoder['encoder_code'] = {path: sha256(ROOT/path) for path in
                ('modeling/cwm_encoders.py', 'modeling/text_encoder.py', 'tools/ref2va_prepare.py', 'configs/video_vae.yaml')}
            original = copy.deepcopy(cfg)
            for protocol, cls in zip(STREAMING_TEXT_RECIPES, DRIVERS):
                recipe = replace(RefRecipe.from_training(cfg), protocol=protocol)
                self.assertEqual(len(recipe.to_dict()), 6)
                self.assertEqual(recipe.input_options['conditioning_mode'], encoder['conditioning_mode'])
                self.assertEqual(cls(recipe)._condition_options()['conditioning_mode'], 'legacy')
                bundle = override_bundle(root, cfg, weights, encoder, recipe.to_dict())
                self.assertEqual(bundle['encoders']['recipe'], encoder)
                self.assertEqual(cfg, original)
                self.assertEqual(bundle['provenance']['source_recorded_recipe'], cfg['inference_recipe'])
                self.assertEqual(bundle['provenance']['training_protocol_match'], 'unverified')
                path = root/'artifact.json'
                path.write_text(json.dumps(bundle))
                self.assertEqual(load_bundle(path)[1], recipe)
                for key in ('recipe_selection', 'training_protocol_match', 'effective_recipe', 'source_stage',
                            'source_meta_model', 'source_recorded_recipe', 'training_config_sha256', 'recipe_override_sha256'):
                    bad = copy.deepcopy(bundle)
                    del bad['provenance'][key]
                    with self.assertRaises(ValueError):
                        validate_bundle(bad)
                bad = copy.deepcopy(bundle)
                bad['encoders']['recipe']['conditioning_mode'] = 'legacy'
                with self.assertRaisesRegex(ValueError, 'contracts differ'):
                    validate_bundle(bad)
                bad = copy.deepcopy(bundle)
                bad['encoders']['recipe']['encoder_code']['modeling/cwm_encoders.py'] = '0'*64
                path.write_text(json.dumps(bad))
                loaded, _ = load_bundle(path)
                self.assertEqual(loaded['encoders']['recipe']['encoder_code'], bad['encoders']['recipe']['encoder_code'])
                stamped = copy.deepcopy(cfg)
                stamped['inference_recipe'] = recipe.to_dict()
                with self.assertRaisesRegex(ValueError, 'differ'):
                    RefRecipe.from_training(stamped)
                for scale in (0., 1., 2.):
                    with self.assertRaisesRegex(ValueError, 'not CFG'):
                        replace(recipe, guidance=dict(mode='cfg', drop='proxy', scale=scale))
                self.assertEqual(replace(recipe, guidance=dict(mode='gft', drop='proxy', scale=1.)).protocol, protocol)
                with self.assertRaisesRegex(ValueError, 'requires protocol'):
                    cls(RefRecipe.from_training(cfg))

    def test_actual_export_cli_and_both_new_cli_sources_cross_entry_rejection(self):
        import av
        import numpy as np
        from PIL import Image
        from projects.minimax_h3.tools.export_ref_inference import main as export
        from projects.minimax_h3.tools.infer_ref_fixed_history_streaming_text import main as new_fixed
        from projects.minimax_h3.tools.infer_ref_bidir_history_streaming_text import main as new_bidir
        from projects.minimax_h3.tools.infer_ref_fixed_history import main as old_fixed
        from projects.minimax_h3.tools.infer_ref_bidir_history import main as old_bidir
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, weights, encoders = assets(root)
            cfg.data.args.manifest = str(root/'manifest.jsonl')
            (root/'manifest.jsonl').write_text(json.dumps(dict(schema='cwm_ref2va_v1', sample_id='test', fingerprint='fixed',
                conditioning_mode=cfg.data.args.conditioning_mode, latent_shape=[24, 2, 2, 2], path='cache.pt'))+'\n')
            (root/'config.json').write_text(json.dumps(encoders))
            (root/'training.json').write_text(json.dumps(cfg))
            (root/'prompt.txt').write_text('fixed')
            (root/'live.jsonl').write_text('{"prompt":"  turn 左  "}\n')
            (root/'replay.jsonl').write_text('{"chunk":0,"prompt":"  turn 左  "}\n')
            proxy = root/'proxy.mkv'
            with av.open(str(proxy), 'w') as container:
                stream = container.add_stream('ffv1', rate=24)
                stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
                for _ in range(5):
                    container.mux(stream.encode(av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')))
                container.mux(stream.encode())
            Image.new('RGB', (32, 32)).save(root/'first.png')
            for protocol, entry in zip(STREAMING_TEXT_RECIPES, (new_fixed, new_bidir)):
                artifact = root/(protocol.split('/')[0]+'.json')
                override = dict(cfg['inference_recipe'], protocol=protocol)
                (root/'recipe.json').write_text(json.dumps(override))
                with redirect_stdout(io.StringIO()):
                    export(['--config', str(root/'training.json'), '--weights', str(weights), '--checkpoint', str(root),
                        '--prompt-file', str(root/'prompt.txt'), '--output', str(artifact), '--recipe-override', str(root/'recipe.json')])
                argv = ['--artifact', str(artifact), '--proxy-video', str(proxy), '--first-frame', str(root/'first.png'),
                    '--prompt-file', str(root/'prompt.txt'), '--num-frames', '5', '--output-dir', str(root/'output'), '--check-only']
                for flag, filename in (('--prompt-updates', 'live.jsonl'), ('--prompts-jsonl', 'replay.jsonl')):
                    out = io.StringIO()
                    with redirect_stdout(out):
                        entry(argv+[flag, str(root/filename)])
                    report = json.loads(out.getvalue())
                    self.assertEqual(report['recipe']['protocol'], protocol)
                    self.assertEqual(report['effective_protocol'], protocol)
                    self.assertFalse(report['conditioning_matches_training'])
                    self.assertEqual(report['base_conditioning_mode'], encoders['conditioning_mode'])
                    self.assertEqual(report['inference_packing'], 'legacy')
                    self.assertIn(report['prompt_source']['kind'], ('live_append_jsonl', 'replay_jsonl'))
                    for crossed in (old_fixed, old_bidir, new_bidir if entry is new_fixed else new_fixed):
                        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            crossed(argv+[flag, str(root/filename)])
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    entry(argv)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    entry(argv+['--prompt-updates', str(root/'live.jsonl'), '--prompts-jsonl', str(root/'replay.jsonl')])
            self.assertFalse((root/'output').exists())
            (root/'output').mkdir()
            for name in ('prompt-events.jsonl', 'prompt-ledger.jsonl'):
                existing = root/'output'/name
                existing.write_text('OLD')
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    entry(argv+['--prompt-updates', str(root/'live.jsonl')])
                self.assertEqual(existing.read_text(), 'OLD')
                existing.unlink()


if __name__ == '__main__':
    unittest.main()
