from copy import deepcopy
from dataclasses import replace
import argparse
from types import SimpleNamespace
import unittest

from inference.raw_recipe import (RAW_PROTOCOLS, RawLatentRecipe, parse_recipe, window_arguments,
                                   window_overrides, validate_online_guidance)
from modeling.ref_recipe import DEFAULT_WINDOW, RefRecipe


def raw_recipe(**kwargs):
    diffusion = dict(
        schedule=dict(module='common.diffusion.schedule.lerp', class_name='LinearInterpolationSchedule',
                      args=dict(pred_type='x_0', T=1.0)),
        sampling_timesteps=dict(module='common.diffusion.timestep.sampling.trailing',
            class_name='TrailingSamplingTimesteps', args=dict(T=1.0, num_sampling_steps=2)),
        sampler=dict(module='projects.minimax_h3.modeling.sampling', class_name='EulerSampler', args={}))
    return RawLatentRecipe('static_image_qwen_proxy_vae_v1', 37, diffusion,
        dict(mode='plain', drop='proxy', scale=1.), **kwargs)


class RecipeTests(unittest.TestCase):
    def test_cwm_requires_one_generated_sink_latent(self):
        import json
        from pathlib import Path
        record = json.loads((Path(__file__).resolve().parents[1] / 'configs/recipes/omni-h3-cwm.json').read_text())
        self.assertEqual(parse_recipe(record).sink_latents, 1)
        for sink in (0, 2, True, 1.0):
            with self.subTest(sink=sink), self.assertRaises(ValueError):
                parse_recipe(dict(record, sink_latents=sink))

    def test_closed_tagged_contract_and_all_axes(self):
        for protocol in RAW_PROTOCOLS:
            recipe = raw_recipe(protocol=protocol)
            self.assertEqual(parse_recipe(recipe.to_dict()), recipe)
            self.assertNotIn('window_size', recipe.to_dict())
            self.assertEqual(recipe.current_latent_schedule, (3,2))
        for key, values in dict(sink_latents=[-1, True, 1.5, 10],
                                recent_latents=[-1, False, 3.5],
                                bootstrap_latents=[0, -1, True],
                                current_latent_schedule=[[], [0], [True], '3,2'],
                                segment_boundaries=[[70,45], [45,45], [True], [1]]).items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    raw_recipe(**{key:value})
        with self.assertRaises(ValueError):
            parse_recipe(dict(raw_recipe().to_dict(), window_size=2))

    def test_old_recipe_unchanged_and_no_silent_default_migration(self):
        raw = raw_recipe()
        old = RefRecipe(raw.conditioning_mode, 6, 37, deepcopy(raw.diffusion), deepcopy(raw.guidance))
        self.assertEqual(DEFAULT_WINDOW, 6)
        self.assertIsInstance(parse_recipe(old.to_dict()), RefRecipe)
        self.assertEqual(parse_recipe(old.to_dict()).to_dict(), old.to_dict())
        with self.assertRaises(ValueError):
            replace(old, window_size=7)
        with self.assertRaises(ValueError):
            parse_recipe(dict(old.to_dict(), sink_latents=2))
        self.assertEqual(raw.with_steps(4).diffusion['sampling_timesteps']['args']['num_sampling_steps'], 4)
        self.assertEqual(raw.diffusion['sampling_timesteps']['args']['num_sampling_steps'], 2)

    def test_client_overrides_are_explicit_and_yaml_schedule_uses_integer_list(self):
        from inference.settings import _value_tokens
        parser = argparse.ArgumentParser()
        window_arguments(parser)
        raw = raw_recipe()
        self.assertEqual(window_overrides(parser.parse_args([]), raw.to_dict()), {})
        args = parser.parse_args(['--sink-latents','4','--current-latent-schedule','3','2'])
        self.assertEqual(window_overrides(args, raw.to_dict()),
                         dict(sink_latents=4, current_latent_schedule=[3,2]))
        with self.assertRaises(ValueError):
            window_overrides(parser.parse_args(['--window-size','2']), raw.to_dict())
        action = parser._option_string_actions['--current-latent-schedule']
        self.assertEqual(_value_tokens(action, [3,2], '.'),
                         ['--current-latent-schedule','3','2'])
        for invalid in ([True], [], '3,2'):
            with self.assertRaises(ValueError):
                _value_tokens(action, invalid, '.')

    def test_settings_report_raw_and_mixed_units_per_artifact(self):
        from inference.settings import effective_settings, load_inference_options
        raw = raw_recipe()
        old = RefRecipe(raw.conditioning_mode, 6, 37, raw.diffusion, raw.guidance)
        def bundle(recipe):
            return dict(recipe=recipe.to_dict(), weights={}, encoders={'recipe':{}}, model={}, deployment={})
        args = SimpleNamespace(artifact=['raw'], steps=None, codec_overlap=False, _settings_entrypoint='serve_ref')
        options = load_inference_options()
        options['codec']['decoder'] = 'causal'
        single = effective_settings(args, bundle=bundle(raw), options=options)
        self.assertEqual(single['units']['history_window'], 'native_latents')
        mixed = effective_settings(args, bundle={'old':(bundle(old),old), 'raw':(bundle(raw),raw)}, options=options)
        self.assertEqual(mixed['units']['history_window'], 'mixed_by_artifact')
        self.assertEqual([a['units']['history_window'] for a in mixed['artifacts']],
                         ['past_time_groups', 'native_latents'])

    def test_online_cfg_rejects_missing_weak_document_at_setup(self):
        raw = raw_recipe()
        for drop in ('proxy', 'prompt', 'prompt_proxy'):
            value = replace(raw, guidance=dict(mode='cfg', drop=drop, scale=2.))
            self.assertEqual(parse_recipe(value.to_dict()), value)
            if drop == 'proxy':
                validate_online_guidance(value)
            else:
                with self.assertRaisesRegex(ValueError, 'explicit weak document'):
                    validate_online_guidance(value)


if __name__ == '__main__':
    unittest.main()
