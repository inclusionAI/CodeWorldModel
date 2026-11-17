"""Real Ref2VA orchestration and FrozenCWMEncoders.condition with fake model assets."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from projects.minimax_h3.inference import ref2va
from projects.minimax_h3.inference.artifact import sha256
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.inference.streaming_text import encode_instruction
from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
from projects.minimax_h3.tests.test_ref_inference import CacheProbe, nodes, RefRecipe, recipe_config, Grid, Euler
from projects.minimax_h3.tests.test_streaming_text import driver, DRIVERS


class FakeProcessor:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return dict(input_ids=[ord(c) for c in text])

        def convert_tokens_to_ids(self, token):
            return {'<|vision_start|>': 900001, '<|vision_end|>': 900002,
                    '<|image_pad|>': 900003, '<|video_pad|>': 900004}[token]

    class Images:
        merge_size = 1

        def __call__(self, **kwargs):
            return dict(pixel_values=torch.zeros(1, 3), image_grid_thw=torch.ones(1, 3, dtype=torch.long))

    def __init__(self):
        self.tokenizer = self.Tokenizer()
        self.image_processor = self.Images()


class EngineTests(unittest.TestCase):
    def run_engine(self, root, cls=DRIVERS[1], *, failure=None, old=False, programmatic=False):
        from common.engine import BaseEngine
        from projects.minimax_h3.modeling import inference_runtime
        from projects.minimax_h3.tools import ref2va_prepare
        events, encoders, engines = [], [], []
        path = root/'updates.jsonl'
        path.write_text('{"prompt":"A"}\n')
        first = root/'first.png'
        Image.new('RGB', (32, 32)).save(first)
        runner = driver(cls)
        if old:
            runner = RefFixedHistory(replace(runner.recipe, protocol=RefFixedHistory.PROTOCOL))
            runner.sampling_timesteps, runner.sampler = Grid(), Euler()
        cfg = nodes(dict(inference=dict(spec=dict(first_frame=str(first), proxy='fake', start=0,
            num_frames=39, height=32, width=32, proxy_height=32, proxy_width=32, prompt='system'),
            checkpoint='fake-assets', output_dir=str(root/'out'), runtime=dict(profile=False, dit={}, codec={},
            output=dict(async_write=True, queue_size=2)), recipe=runner.recipe.to_dict(), seed=73,
            condition_seed=94, decode_mode='strict', conditioning_matches_training=True,
            prompt_source=dict(kind='live_append_jsonl', path=str(path)))))

        class Qwen:
            model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096)))

            def __call__(self, **kwargs):
                events.append(('qwen_forward', len(kwargs['input_ids'][0])))
                self_test.assertFalse(kwargs['use_cache'])
                ids = kwargs['input_ids'][0]
                return (torch.ones(len(ids), 32, dtype=torch.bfloat16)*ids.float().mean()/100,)

        self_test = self
        def frozen_init(value, *args, **kwargs):
            value.device = torch.device('cpu')
            value.qwen, value.processor = Qwen(), FakeProcessor()
            encoders.append(value)
        original_condition = FrozenCWMEncoders.condition
        def condition(value, text, videos=(), images=()):
            events.append(('condition', text, len(images), len(videos)))
            if text != 'system':
                self.assertIn(('generator_loaded',), events)
                self.assertEqual((len(images), len(videos)), (0, 0))
            if failure == 'prepare' or failure == 'encode' and text == 'B':
                raise ValueError(f'{failure} failure')
            return original_condition(value, text, videos=videos, images=images)
        class Codec:
            def __init__(self, encoders, config):
                pass
            def encode(self, frames, *, eos):
                events.append(('proxy_encode',))
                return torch.ones(24, 2 if eos else 5, 2, 2)
            def decode(self, latent, *, eos):
                return torch.zeros(5 if eos else 17, 32, 32, 3, dtype=torch.uint8)
        class Writer:
            def __init__(self, *args):
                pass
            def write(self, pixels):
                events.append(('write', len(pixels)))
            def close(self, success):
                pass
        class Net(CacheProbe):
            def forward(self, **kwargs):
                if not kwargs['update_past_key_values'] and ('updated_during_denoise',) not in events:
                    events.append(('updated_during_denoise',))
                    with path.open('a') as handle:
                        handle.write('{"prompt":"B"}\n')
                return super().forward(**kwargs)
        def base_init(value, config, meta):
            engines.append(value)
            self.assertEqual(hasattr(encoders[0], 'qwen'), not old)
            if failure == 'generator':
                raise ValueError('generator failure')
            value.config, value.meta, value.models = config, meta, dict(backbone=Net())
            events.append(('generator_loaded',))
        def raw(*args, **kwargs):
            for size in (17, 17, 5):
                events.append(('proxy_read', size))
                yield np.zeros((size, 32, 32, 3), dtype=np.uint8)
        polls = []
        def source(k):
            polls.append((k, len([event for event in events if event[0] == 'proxy_read'])))
            if failure == 'source' and k == 1:
                raise ValueError('source failure')
            return ('A', 'B', None)[k]
        with ExitStack() as stack:
            for obj, name, value in ((FrozenCWMEncoders, '__init__', frozen_init),
                    (FrozenCWMEncoders, 'condition', condition),
                    (FrozenCWMEncoders, 'image', lambda *_: torch.zeros(24, 1, 2, 2)),
                    (BaseEngine, '__init__', base_init), (inference_runtime, 'InferenceCodec', Codec),
                    (ref2va_prepare, 'iter_video_chunks', raw), (ref2va, 'VideoWriter', Writer)):
                stack.enter_context(patch.object(obj, name, value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failure:
                with self.assertRaisesRegex((RuntimeError, ValueError), 'FileExistsError' if failure == 'output' else f'{failure} failure'):
                    engine = ref2va.Ref2VAStreamEngine(cfg, runner)
                    engine.run(prompt_source=source if programmatic or failure == 'source' else None)
                report = None
            else:
                engine = ref2va.Ref2VAStreamEngine(cfg, runner)
                report = engine.run(prompt_source=source if programmatic else None)
        self.assertEqual(len(encoders), 0 if failure == 'output' else 1)
        if encoders:
            self.assertFalse(hasattr(encoders[0], 'qwen'))
            self.assertFalse(hasattr(encoders[0], 'processor'))
        return report, events, polls, engines

    def test_actual_condition_boundary_residency_async_metadata_and_counters(self):
        for cls in DRIVERS:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report, events, _, _ = self.run_engine(root, cls)
                calls = [event for event in events if event[0] == 'condition']
                self.assertEqual(calls, [('condition', 'system', 1, 0), ('condition', 'A', 0, 0), ('condition', 'B', 0, 0)])
                self.assertEqual((report['qwen_calls'], report['static_qwen_calls'], report['dynamic_qwen_calls'],
                                  report['proxy_qwen_calls']), (3, 1, 2, 0))
                self.assertEqual(report['effective_protocol'], cls.PROTOCOL)
                self.assertFalse(report['conditioning_matches_training'])
                self.assertEqual(report['qwen_residency'], 'rank0_resident_on_generator_device')
                self.assertEqual(report['qwen_device'], 'cpu')
                chunks = [torch.load(root/'out/chunks'/f'{k:06d}.pt', weights_only=True) for k in range(3)]
                self.assertEqual([c['prompt']['text'] for c in chunks], ['A', 'B', 'B'])
                self.assertEqual([c['prompt']['first_chunk'] for c in chunks], [0, 1, 1])
                ledger = [json.loads(v) for v in (root/'out/prompt-ledger.jsonl').read_text().splitlines()]
                accepted = [json.loads(v) for v in (root/'out/prompt-events.jsonl').read_text().splitlines()]
                self.assertEqual([v['chunk'] for v in accepted], [0, 1])
                self.assertEqual([v['text'] for v in accepted], ['A', 'B'])
                self.assertEqual(len(ledger), 3)
                self.assertEqual(report['prompt_events_sha256'], sha256(root/'out/prompt-events.jsonl'))
                self.assertEqual(report['prompt_ledger_sha256'], sha256(root/'out/prompt-ledger.jsonl'))

    def test_programmatic_source_precedes_proxy_and_old_path_deletes_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, polls, _ = self.run_engine(Path(directory), programmatic=True)
            self.assertEqual(polls, [(0, 0), (1, 1), (2, 2)])
            self.assertEqual(report['prompt_source']['kind'], 'programmatic_boundary_callable')
        with tempfile.TemporaryDirectory() as directory:
            report, events, _, _ = self.run_engine(Path(directory), old=True)
            self.assertEqual(report['qwen_calls'], 1)
            self.assertEqual([v[1] for v in events if v[0] == 'condition'], ['system'])

    def test_prepare_generator_source_and_encode_failures_release_qwen(self):
        for failure in ('prepare', 'generator', 'source', 'encode'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.run_engine(Path(directory), failure=failure)

    def test_engine_refuses_each_existing_prompt_provenance_file(self):
        for name in ('prompt-events.jsonl', 'prompt-ledger.jsonl'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'out').mkdir()
                existing = root/'out'/name
                original = b'{"chunk":999,"text":"OLD"}\n'
                existing.write_bytes(original)
                self.run_engine(root, failure='output')
                self.assertEqual(existing.read_bytes(), original)

    def test_actual_tokenizer_context_preflight_prevents_forward(self):
        encoder = SimpleNamespace(processor=FakeProcessor(),
            qwen=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=3)))))
        calls = []
        encoder.condition = lambda text: calls.append(text)
        with self.assertRaisesRegex(ValueError, 'tokenizer rows'):
            encode_instruction(encoder, 'long')
        self.assertEqual(calls, [])
        encode_instruction(encoder, 'ok')
        self.assertEqual(calls, ['ok'])


if __name__ == '__main__':
    unittest.main()
