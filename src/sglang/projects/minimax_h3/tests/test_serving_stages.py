from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from projects.minimax_h3.inference import serving
from projects.minimax_h3.inference.recipes.ref_fixed_history import RefFixedHistory
from projects.minimax_h3.inference.recipes.ref_bidir_history import RefBidirHistory
from projects.minimax_h3.modeling import cwm_encoders, inference_runtime, qwen_inference
from projects.minimax_h3.modeling.ref_recipe import RefRecipe, STREAMING_TEXT_RECIPES
from projects.minimax_h3.tests.test_qwen_fsdp import module
from projects.minimax_h3.tests.test_ref_inference import CacheProbe, Euler, Grid, inputs, recipe_config
from projects.minimax_h3.tests.test_streaming_text import DRIVERS, document


class LocalGroup:
    rank = 0

    def rank0(self, callback):
        return self.call_on(0, callback)

    def call_on(self, owner, callback):
        return callback() if self.rank == owner else None

    def gather(self, value):
        return [value]


class StageTests(unittest.TestCase):
    def backend(self):
        static, picture, chunks = inputs()
        backend = serving.ResidentBackend.__new__(serving.ResidentBackend)
        backend.group = LocalGroup()
        backend.vae_rank = backend.decoder_rank = 0
        backend.qwen_fsdp = False
        backend.backbone = CacheProbe()
        backend.profiler = inference_runtime.StageProfiler(False)
        backend.outputs = backend.inputs = None
        backend.prefetched_proxy = None
        backend.memory = lambda: dict(rank=0)
        backend.artifacts = {'test': (dict(encoders=dict(recipe=dict(
            height=32, width=64, proxy_height=1, proxy_width=1))), None)}
        backend.qwen_encoder = SimpleNamespace(condition=Mock(return_value=static))
        backend.encoders = SimpleNamespace(image=Mock(return_value=picture))
        latent_chunks = iter(item['references'][0]['video'] for item in chunks)
        decoded = []
        def decode(latent, *, eos):
            decoded.append(latent.clone())
            return (latent[0, :, :1, :1] * 10).byte().unsqueeze(-1).expand(-1, -1, -1, 3)
        backend.codec = SimpleNamespace(begin_session=Mock(),
            encode=Mock(side_effect=lambda *a, **kw: next(latent_chunks)), decode=Mock(side_effect=decode))
        backend._prepare_text = Mock(side_effect=lambda text: text)
        def encode_text(text):
            backend.dynamic_text_forwards += 1
            return document(text)
        backend._encode_text = Mock(side_effect=encode_text)
        return backend, decoded

    def run_recipe(self, cls, *, stages):
        backend, decoded = self.backend()
        recipe = replace(RefRecipe.from_training(recipe_config()), protocol=cls.PROTOCOL)
        metadata = dict(recipe=recipe.to_dict(), artifact_id='test', prompt='static', seed=73, condition_seed=94)
        boundary = module('common.diffusion',
            build_diffusion=lambda _: dict(sampling_timesteps=Grid(), sampler=Euler()))
        receipts, pixels = [], []
        with patch.dict('sys.modules', {'common.diffusion': boundary}):
            if stages:
                backend.begin('start', metadata)
                backend.static_condition(bytes(32 * 64 * 3))
                self.assertEqual(backend.encoders.image.call_count, 0)
                backend.first_image(bytes(32 * 64 * 3))
                backend.open_recipe()
                backend.collect()
            else:
                backend.execute('start', metadata, bytes(32 * 64 * 3))
            for index, text in enumerate(('A', 'A', 'BBB', 'A')):
                metadata = dict(prompt=dict(text=text, revision=index), eos=index == 3)
                if stages:
                    backend.begin('chunk', metadata)
                    backend.text_condition()
                    count = backend._encode_text.call_count
                    backend.proxy_encode(bytes(3))
                    backend.render()
                    self.assertEqual(backend._encode_text.call_count, count)
                    backend.decode()
                    rgb, receipt = backend.collect()
                else:
                    rgb, receipt = backend.execute('chunk', metadata, bytes(3))
                pixels.append(rgb)
                receipts.append(receipt)
                self.assertIsNone(backend.result)
                self.assertIsNone(backend.prepared_text)
                self.assertIsNone(backend.cpu_output)
                self.assertIn('rank_timings', receipt)
            if recipe.protocol in STREAMING_TEXT_RECIPES:
                self.assertEqual([r['dynamic_text_forwards'] for r in receipts], [1, 0, 1, 1])
                self.assertEqual([r['applied_prompt']['first_chunk'] for r in receipts], [0, 0, 2, 3])
                self.assertEqual(backend._prepare_text.call_count, 3)
            backend.close_session()
            self.assertIsNone(backend.outputs)
            self.assertIsNone(backend.text_document)
            self.assertIsNone(backend.static_document)
            self.assertIsNone(backend.metadata)
            self.assertEqual(backend.codec.begin_session.call_count, 2)
        return decoded, pixels, [r.get('applied_prompt') for r in receipts]

    def test_serial_and_named_stages_match_all_four_recipes(self):
        for cls in (RefFixedHistory, RefBidirHistory, *DRIVERS):
            with self.subTest(recipe=cls.PROTOCOL):
                before, before_rgb, before_prompts = self.run_recipe(cls, stages=False)
                after, after_rgb, after_prompts = self.run_recipe(cls, stages=True)
                for a, b in zip(before, after, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(before_rgb, after_rgb)
                self.assertEqual(before_prompts, after_prompts)

    def test_recipe_callback_cannot_encode_missing_text(self):
        backend, _ = self.backend()
        backend.encoded_text, backend.text_document = 'A', document('A')
        self.assertIs(backend._lookup_text('A'), backend.text_document)
        with self.assertRaisesRegex(RuntimeError, 'before text_condition'):
            backend._lookup_text('B')
        backend._encode_text.assert_not_called()

    def test_decode_uses_owner_local_latent_and_keeps_pixels_on_owner(self):
        for rank in (0, 2):
            backend, decoded = self.backend()
            backend.group.rank = rank
            backend.decoder_rank = 2
            backend.begin('chunk', dict(eos=True))
            backend.result = torch.full((24, 2, 2, 4), 7.)
            backend.decode()
            pixels, timings = backend.collect()
            self.assertEqual(bool(pixels), rank == 2)
            self.assertEqual(len(decoded), int(rank == 2))
            self.assertEqual('decode_start_ns' in timings, rank == 2)
            self.assertIsNone(backend.result)

    def test_moved_vae_and_decoder_load_only_owned_components(self):
        class Options(dict):
            __getattr__ = dict.__getitem__
        config = SimpleNamespace(inference=SimpleNamespace(checkpoint='unused',
            runtime=Options(qwen_fsdp=False, profile=False, dit={}, codec={})))
        residency = [dict(rank=rank, qwen=dict(mode='single' if rank == 0 else 'absent',
            local_parameter_bytes=100 if rank == 0 else 0)) for rank in range(3)]
        with patch.dict('sys.modules', {
            'common.engine': module('common.engine', BaseEngine=lambda *a:
                SimpleNamespace(models=dict(backbone=SimpleNamespace(eval=lambda: None)))),
            'common.distributed.ops': module('common.distributed.ops', get_device=lambda: torch.device('cpu')),
        }), patch.object(qwen_inference, 'FrozenQwenEncoder') as qwen, \
                patch.object(cwm_encoders, 'FrozenCWMEncoders') as vae, \
                patch.object(inference_runtime, 'InferenceCodec') as codec, \
                patch.object(serving.ResidentBackend, 'memory', return_value={}):
            backends = []
            for rank in range(3):
                group = LocalGroup()
                group.rank, group.gather = rank, lambda _: residency
                backends.append(serving.ResidentBackend(config, {}, group, vae_rank=1, decoder_rank=2))
            self.assertEqual(qwen.call_count, 1)
            self.assertEqual(vae.call_count, 1)
            self.assertFalse(vae.call_args.kwargs['load_text_encoder'])
            self.assertEqual([hasattr(b, 'encoders') for b in backends], [False, True, False])
            self.assertEqual([hasattr(b, 'codec') for b in backends], [False, True, True])
            self.assertEqual(codec.call_count, 2)
            self.assertFalse(codec.call_args_list[0].kwargs['load_decoder'])
            self.assertIsNone(codec.call_args_list[1].args[0])
            self.assertFalse(codec.call_args_list[1].kwargs['load_encoder'])
            self.assertEqual(codec.call_args_list[1].kwargs['device'], torch.device('cpu'))


class PublicationTests(unittest.TestCase):
    def group(self, rank, publication):
        group = serving.Distributed.__new__(serving.Distributed)
        group.rank = rank
        def broadcast(value, *, src):
            self.assertEqual(src, 2)
            if rank == src:
                publication.append(value)
            return publication[0]
        group.broadcast = broadcast
        return group

    def test_nonzero_owner_publishes_once_to_all_ranks(self):
        publication = []
        callback = Mock(return_value={'latent': 'owned'})
        for rank in (2, 0, 1):
            result = self.group(rank, publication).call_on(2, callback)
            self.assertEqual(result, {'latent': 'owned'})
        callback.assert_called_once()

    def test_nonzero_owner_error_is_recoverable_on_all_ranks(self):
        publication = []
        callback = Mock(side_effect=ValueError('bad input'))
        for rank in (2, 0, 1):
            with self.assertRaisesRegex(serving.SessionError, 'bad input'):
                self.group(rank, publication).call_on(2, callback)
        callback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
