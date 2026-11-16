"""Launch defaults follow actual codec placement, not the recipe name alone."""
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class ServeH3CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep extension modules loaded when patch.dict restores sys.modules.
        __import__('sglang_omni.models.minimax_h3.config')
        cls.script = Path(__file__).resolve().parents[1] / 'tools/serve_omni_h3.py'

    def manifest(self, *, protocol='cwm_causal/v1', placement=None, codec=None):
        result = dict(recipe=dict(protocol=protocol), canvas=dict(width=1344, height=768),
            server_args=dict(num_gpus=8), state_reservation_bytes=1024,
            codec=codec or dict(backend='reference'))
        if placement is not None:
            result['codec_parallel'] = placement
        return result

    def launch(self, manifest, flags=()):
        launcher = ModuleType('sglang_omni.serve.launcher')
        launcher.launch_server = Mock()
        assets = ModuleType('projects.minimax_h3.inference.omni_h3_assets')
        assets.read_native_bundle = lambda path: manifest
        assets.verify_native_bundle = Mock()
        with patch.dict(sys.modules, {assets.__name__: assets, launcher.__name__: launcher}), \
                patch.object(sys, 'argv', [str(self.script), '--model-path', 'fixture', *flags]):
            runpy.run_path(str(self.script), run_name='__main__')
        return launcher.launch_server.call_args.args[0], assets.verify_native_bundle

    def test_explicit_conflict_fails_before_reading_assets_or_runtime_setup(self):
        result = subprocess.run([sys.executable, '-I', str(self.script), '--model-path', 'unused',
            '--cpu-delivery', '--no-codec-overlap'], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--cpu-delivery requires --codec-overlap', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_reference_bundle_needs_no_negative_flags(self):
        for protocol, realtime in [('cwm_causal/v1', True), ('ref_fixed_history/v2', False)]:
            with self.subTest(protocol=protocol):
                config, verify = self.launch(self.manifest(protocol=protocol))
                self.assertFalse(config.codec_overlap or config.cpu_delivery)
                self.assertEqual('omni_h3_realtime' in config.runtime_factory, realtime)
                self.assertEqual([stage.name for stage in config.stages], ['h3'])
                verify.assert_not_called()

    def test_supported_disjoint_manifest_selects_overlap_and_delivery(self):
        for encoder, decoder in [([2, 3], [0, 1]), ([2, 3], [0, 1, 4, 5]),
                                  ([2, 3, 6, 7], [0, 1, 4, 5])]:
            with self.subTest(encoder=encoder, decoder=decoder):
                manifest = self.manifest(placement=dict(encoder_ranks=encoder, decoder_ranks=decoder),
                    codec=dict(backend='fast'))
                config, _ = self.launch(manifest)
                self.assertTrue(config.codec_overlap and config.cpu_delivery)
                self.assertIn('omni_h3_realtime', config.runtime_factory)
                self.assertEqual([stage.name for stage in config.stages], ['h3', 'h3_rgb_delivery'])

    def test_shared_codec_groups_do_not_default_to_overlap(self):
        manifest = self.manifest(placement=dict(encoder_ranks=[0, 1], decoder_ranks=[0, 1]),
            codec=dict(backend='fast'))
        config, _ = self.launch(manifest)
        self.assertFalse(config.codec_overlap or config.cpu_delivery)

    def test_overrides_disable_dependent_defaults_without_extra_flags(self):
        manifest = self.manifest(placement=dict(encoder_ranks=[2, 3, 6, 7], decoder_ranks=[0, 1, 4, 5]),
            codec=dict(backend='fast'))
        for flags, overlap, delivery in [(['--no-codec-overlap'], False, False),
                                         (['--no-cpu-delivery'], True, False)]:
            with self.subTest(flags=flags):
                config, _ = self.launch(manifest, flags)
                self.assertEqual((config.codec_overlap, config.cpu_delivery), (overlap, delivery))
                self.assertIn('omni_h3_realtime', config.runtime_factory)

    def test_legacy_host_remains_available_for_e2(self):
        manifest = self.manifest(protocol='ref_fixed_history/v2',
            placement=dict(encoder_ranks=[2, 3], decoder_ranks=[0, 1]), codec=dict(backend='fast'))
        config, _ = self.launch(manifest)
        self.assertIn('omni_h3_runtime', config.runtime_factory)
        self.assertTrue(config.cpu_delivery and config.codec_overlap)

    def test_realtime_only_bundles_reject_host(self):
        manifests = [self.manifest(),
            self.manifest(protocol='ref_fixed_history/v2',
                placement=dict(encoder_ranks=[2, 3, 6, 7], decoder_ranks=[0, 1, 4, 5]), codec=dict(backend='fast')),
            self.manifest(protocol='ref_fixed_history/v2', codec=dict(backend='fast', suffix_sdpa_backend='fa3'))]
        for manifest in manifests:
            with self.subTest(manifest=manifest), self.assertRaises(SystemExit) as error:
                self.launch(manifest, ['--no-realtime-sessions'])
            self.assertEqual(error.exception.code, 2)

    def test_explicit_invalid_overlap_or_delivery_is_rejected(self):
        for flags in [['--codec-overlap'], ['--cpu-delivery']]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit) as error:
                self.launch(self.manifest(), flags)
            self.assertEqual(error.exception.code, 2)

    def test_asset_verification_is_explicit(self):
        manifest = self.manifest()
        _, verify = self.launch(manifest, ['--verify-assets'])
        verify.assert_called_once_with(manifest)


if __name__ == '__main__':
    unittest.main()
