"""Qwen ownership and collective admission without checkpoint or CUDA allocation."""
from contextlib import ExitStack
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from projects.minimax_h3.inference import serving
from projects.minimax_h3.modeling import cwm_encoders, inference_runtime, qwen_inference, ref_recipe


def module(name, **attributes):
    value = ModuleType(name)
    value.__spec__ = ModuleSpec(name, loader=None)
    value.__dict__.update(attributes)
    return value


class Qwen3VLTextDecoderLayer(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)


class Qwen3VLVisionBlock(Qwen3VLTextDecoderLayer):
    pass


class Qwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Qwen3VLTextDecoderLayer() for _ in range(50)]
                                         + [Qwen3VLVisionBlock()])
        self.moves = []
        self.reshard = Mock()
        self.failure = None

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def forward(self, **kwargs):
        if self.failure:
            raise self.failure
        assert kwargs['use_cache'] is False and kwargs['logits_to_keep'] == 1
        return torch.ones(1, kwargs['input_ids'].shape[1], 2, dtype=torch.bfloat16)


class EncoderTests(unittest.TestCase):
    def construct(self, mesh=None):
        model = Qwen()
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64))
        load = Mock(return_value=model)
        processor = object()
        wrapped = []
        def shard(value, **kwargs):
            self.assertTrue(all(p.device.type == 'cpu' for p in value.parameters()))
            self.assertEqual(kwargs, dict(mesh=mesh, reshard_after_forward=True))
            self.assertFalse(model.moves)
            wrapped.append(value)
        with patch.dict('sys.modules', {
            'transformers': module('transformers',
                AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
                AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor))),
            'projects.minimax_h3.modeling.text_encoder': module('text_encoder',
                MiniMaxH3TextEncoder=SimpleNamespace(from_pretrained=load)),
        }), patch('torch.distributed.fsdp.fully_shard', side_effect=shard):
            encoder = qwen_inference.FrozenQwenEncoder('unused', device='cpu', mesh=mesh)
        self.assertEqual(config.text_config.num_hidden_layers, 50)
        self.assertEqual(load.call_args.kwargs['dtype'], torch.bfloat16)
        self.assertEqual(load.call_args.kwargs['attn_implementation'], 'flash_attention_2')
        self.assertIs(encoder.processor, processor)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        return encoder, model, wrapped

    def test_shard_blocks_before_root_without_moving_full_model(self):
        mesh = SimpleNamespace(device_type='cuda', ndim=1, mesh=torch.tensor([0, 1]))
        encoder, model, wrapped = self.construct(mesh)
        self.assertEqual(wrapped, [*model.blocks, model])
        self.assertFalse(model.moves)
        prepared = (dict(input_ids=torch.tensor([[1, 2, 3]])), torch.tensor([0, 1, 1]))
        result = encoder.encode_condition(prepared)
        self.assertEqual(result['prompt_embeds'].shape, (3, 2))
        self.assertEqual(result['prompt_embeds'].dtype, torch.bfloat16)
        self.assertTrue(result['prompt_embeds'].is_contiguous())
        self.assertIs(result['text_token_tags'], prepared[1])
        model.reshard.assert_called_once()
        model.failure = ValueError('forward failed')
        with self.assertRaisesRegex(ValueError, 'forward failed'):
            encoder.encode_condition(prepared)
        model.reshard.assert_called_once()

    def test_unsharded_condition_keeps_fake_constructor_and_direct_attribute_contract(self):
        encoder = cwm_encoders.FrozenCWMEncoders.__new__(cwm_encoders.FrozenCWMEncoders)
        encoder.device, encoder.processor, encoder.qwen = torch.device('cpu'), object(), Qwen()
        prepared = (dict(input_ids=torch.tensor([[1, 2]])), torch.tensor([0, 1]))
        with patch.object(cwm_encoders, 'qwen_inputs', return_value=prepared) as prepare:
            result = encoder.condition('prompt', images=['image'])
        prepare.assert_called_once_with(encoder.processor, 'prompt', (), ['image'])
        self.assertEqual(result['prompt_embeds'].shape, (2, 2))
        encoder.qwen.reshard.assert_not_called()
        del encoder.qwen, encoder.processor
        self.assertFalse(hasattr(encoder, 'qwen'))
        self.assertFalse(hasattr(encoder, 'processor'))

    def test_default_loader_keeps_single_device_path(self):
        _, model, wrapped = self.construct()
        self.assertEqual(model.moves, [((torch.device('cpu'),), {})])
        self.assertFalse(wrapped)

    def test_vae_only_skips_qwen_processor_and_transformers_loading(self):
        class VAE(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
        with patch.dict('sys.modules', {
            'accelerate': module('accelerate', init_empty_weights=ExitStack),
            'safetensors.torch': module('safetensors.torch', load_file=lambda _: {}),
            'projects.minimax_h3.modeling.build': module('build', MiniMaxH3VideoVAE=VAE),
            'projects.minimax_h3.modeling.text_encoder': module('text_encoder',
                MiniMaxH3TextEncoder=SimpleNamespace(from_pretrained=Mock(side_effect=AssertionError('Qwen loaded')))),
            'transformers': module('transformers',
                AutoConfig=SimpleNamespace(from_pretrained=Mock(side_effect=AssertionError('Qwen config loaded'))),
                AutoProcessor=SimpleNamespace(from_pretrained=Mock(side_effect=AssertionError('Qwen processor loaded')))),
        }):
            encoder = cwm_encoders.FrozenCWMEncoders('unused', device='cpu', load_text_encoder=False)
        self.assertFalse(hasattr(encoder, 'qwen'))
        self.assertFalse(hasattr(encoder, 'processor'))
        self.assertTrue(hasattr(encoder, 'vae'))


class StartTests(unittest.TestCase):
    def backend(self, *, enabled=True, rank=0, failure=None, preparation_failure=None):
        events = []
        prepared = (dict(input_ids=torch.tensor([[1]])), torch.tensor([1]))
        static = dict(prompt_embeds=torch.ones(1, 2), text_token_tags=prepared[1])
        class Group:
            def __init__(self):
                self.rank = rank
                self.broadcasts = 0
            def rank0(self, callback):
                self.broadcasts += 1
                events.append('rank0')
                if rank:
                    return prepared if self.broadcasts == 1 else static
                try:
                    return callback()
                except (ValueError, serving.SessionError) as exc:
                    raise serving.SessionError(str(exc), 500) from exc
            def call_on(self, owner, callback):
                events.append('image_owner')
                return callback() if rank == owner else 'latent'
            def gather(self, value):
                return [value]
        def prepare(*args, **kwargs):
            events.append('prepare')
            if preparation_failure:
                raise preparation_failure
            return prepared
        def encode(value):
            events.append('forward')
            self.assertIs(value, prepared)
            if failure:
                raise failure
            return static
        def condition(*args, **kwargs):
            events.append('condition')
            if failure:
                raise failure
            return static
        value = serving.ResidentBackend.__new__(serving.ResidentBackend)
        value.qwen_fsdp = enabled
        value.vae_rank = value.decoder_rank = 0
        value.backbone = object()
        value.group = Group()
        value.profiler = inference_runtime.StageProfiler(False)
        value.artifacts = {'fixed': (dict(encoders=dict(recipe=dict(height=1, width=1))), None)}
        value.memory = lambda: {}
        if rank == 0:
            value.encoders = SimpleNamespace(condition=condition,
                image=lambda _: events.append('image') or 'latent')
            value.codec = SimpleNamespace(begin_session=lambda: events.append('reset'))
        value.qwen_encoder = SimpleNamespace(prepare_condition=prepare, encode_condition=encode,
                                             condition=condition) if rank < 2 else None
        return value, events

    def execute(self, backend):
        class Driver:
            def __init__(self, recipe):
                pass
            def stream_latents(self, *args, **kwargs):
                return iter(())
        with patch.dict('sys.modules', {
            'projects.minimax_h3.inference.recipes.ref_fixed_history': module('fixed', RefFixedHistory=Driver),
            'projects.minimax_h3.inference.recipes.ref_bidir_history': module('bidir', RefBidirHistory=Driver),
        }), patch.object(serving, 'parse_recipe', return_value=SimpleNamespace(protocol=ref_recipe.FIXED_RECIPE,
                                                                                 conditioning_mode='static')):
            return backend.execute('start', dict(recipe={}, artifact_id='fixed', prompt='p', seed=1,
                                                 condition_seed=1), bytes(3) if backend.group.rank == 0 else b'')

    def test_enabled_start_prepares_before_forward_then_encodes_image(self):
        backend, events = self.backend()
        self.execute(backend)
        self.assertEqual(events, ['rank0', 'prepare', 'forward', 'rank0', 'reset', 'image_owner', 'image'])

    def test_preparation_failure_rejects_before_any_forward(self):
        backend, events = self.backend(preparation_failure=ValueError('bad input'))
        with self.assertRaisesRegex(serving.SessionError, 'bad input'):
            self.execute(backend)
        self.assertNotIn('forward', events)

    def test_collective_session_error_becomes_fatal_runtime_error(self):
        backend, events = self.backend(failure=serving.SessionError('FSDP failed'))
        with self.assertRaisesRegex(RuntimeError, 'worker restart required'):
            self.execute(backend)
        self.assertNotIn('image', events)
        self.assertEqual(events.count('rank0'), 1)

    def test_disabled_condition_value_error_remains_recoverable(self):
        backend, events = self.backend(enabled=False, failure=ValueError('old failure'))
        with self.assertRaisesRegex(serving.SessionError, 'old failure'):
            self.execute(backend)
        self.assertEqual(events, ['rank0', 'condition'])

    def test_rank_one_only_forwards_and_nonmember_never_encodes(self):
        for rank in (1, 2, 7):
            backend, events = self.backend(rank=rank)
            self.execute(backend)
            self.assertEqual(events, ['rank0', 'forward', 'rank0', 'image_owner'] if rank == 1 else ['rank0', 'rank0', 'image_owner'])
            self.assertFalse(hasattr(backend, 'encoders'))


class PlacementTests(unittest.TestCase):
    def test_all_ranks_create_group_but_only_zero_and_one_load_qwen(self):
        class Options(dict):
            __getattr__ = dict.__getitem__
        config = SimpleNamespace(inference=SimpleNamespace(checkpoint='unused',
            runtime=Options(qwen_fsdp=True, profile=False, dit={}, codec={})))
        residency = [dict(rank=rank, qwen=dict(mode='fsdp2' if rank < 2 else 'absent',
                                              local_parameter_bytes=100 if rank < 2 else 0))
                     for rank in range(8)]
        backbone = SimpleNamespace(eval=Mock())
        engine = Mock(return_value=SimpleNamespace(models=dict(backbone=backbone)))
        with patch.dict('sys.modules', {
            'common.engine': module('common.engine', BaseEngine=engine),
            'common.distributed.ops': module('common.distributed.ops', get_device=lambda: torch.device('cpu')),
        }), patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=8), \
                patch('torch.distributed.new_group') as new_group, \
                patch('torch.distributed.device_mesh.DeviceMesh.from_group') as mesh, \
                patch.object(qwen_inference, 'FrozenQwenEncoder') as qwen, \
                patch.object(cwm_encoders, 'FrozenCWMEncoders') as vae, \
                patch.object(inference_runtime, 'InferenceCodec') as codec, \
                patch.object(serving.ResidentBackend, 'memory', return_value={}):
            for rank in range(8):
                def rank0(callback):
                    return callback() if rank == 0 else None
                group = SimpleNamespace(rank=rank, rank0=rank0,
                    call_on=lambda owner, callback: callback() if rank == owner else None, gather=lambda _: residency)
                backend = serving.ResidentBackend(config, {}, group)
                self.assertEqual(backend.qwen_placement['ranks'], [0, 1])
                self.assertEqual(backend.qwen_encoder is not None, rank < 2)
                self.assertEqual(hasattr(backend, 'encoders'), rank == 0)
            self.assertEqual(new_group.call_count, 8)
            new_group.assert_called_with(ranks=[0, 1], backend='nccl')
            self.assertEqual(mesh.call_count, 2)
            self.assertEqual(qwen.call_count, 2)
            self.assertEqual(vae.call_count, 1)
            self.assertFalse(vae.call_args.kwargs['load_text_encoder'])
            self.assertEqual(codec.call_count, 1)
            engine.reset_mock()
            with patch('torch.distributed.get_world_size', return_value=1):
                with self.assertRaisesRegex(ValueError, 'at least two initialized'):
                    serving.ResidentBackend(config, {}, SimpleNamespace(rank=0))
            engine.assert_not_called()

    def test_memory_records_local_parameters_and_zero_for_nonparticipant(self):
        backend = serving.ResidentBackend.__new__(serving.ResidentBackend)
        backend.group = SimpleNamespace(rank=0)
        backend.weight_load_count = 1
        with patch('torch.cuda.memory_allocated', return_value=0), \
                patch('torch.cuda.memory_reserved', return_value=0), \
                patch('torch.cuda.max_memory_allocated', return_value=0):
            backend.qwen_encoder = SimpleNamespace(qwen=torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16))
            memory = backend.memory()
            self.assertEqual(memory['qwen']['mode'], 'single')
            self.assertEqual(memory['qwen_local_parameter_bytes'], 16)
            self.assertEqual(memory['qwen']['local_parameter_numel'], 8)
            backend.qwen_encoder = None
            memory = backend.memory()
            self.assertEqual(memory['qwen']['mode'], 'absent')
            self.assertEqual(memory['qwen_local_parameter_bytes'], 0)
            self.assertEqual(memory['qwen']['parameter_count'], 0)


class DynamicTextTests(unittest.TestCase):
    def backend(self, *, rank=0, rows=2, context=4096, failure=None):
        backend = serving.ResidentBackend.__new__(serving.ResidentBackend)
        backend.profiler = inference_runtime.StageProfiler(False)
        backend.qwen_fsdp = True
        backend.dynamic_text_forwards = 0
        backend.vae_rank = backend.decoder_rank = 0
        backend.encoded_text = backend.text_document = None
        backend.submitted_text = 'AB'
        backend.prepared_text = (dict(input_ids=torch.ones((1, rows), dtype=torch.long),
            mm_token_type_ids=torch.zeros((1, rows), dtype=torch.int32)), torch.ones(rows, dtype=torch.long))
        backend.static_document = dict(prompt_embeds=torch.ones(1, 2), text_token_tags=torch.ones(1, dtype=torch.long))
        document = dict(prompt_embeds=torch.ones(rows, 2), text_token_tags=backend.prepared_text[1])
        encoder = SimpleNamespace(
            prepare_condition=Mock(return_value=backend.prepared_text),
            encode_condition=Mock(side_effect=failure, return_value=document),
            qwen=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=context)))))
        backend.qwen_encoder = encoder if rank < 2 else None
        def rank0(callback):
            if rank:
                return document
            try:
                return callback()
            except (ValueError, serving.SessionError) as exc:
                raise serving.SessionError(str(exc), 500) from exc
        backend.group = SimpleNamespace(rank=rank, rank0=Mock(side_effect=rank0))
        return backend, encoder

    def test_preflight_uses_prepared_rows_and_rejects_before_forward(self):
        for rows, context in ((0, 4096), (4097, 10000), (4, 3)):
            backend, encoder = self.backend(rows=rows, context=context)
            with self.subTest(rows=rows, context=context), self.assertRaisesRegex(serving.SessionError, 'tokenizer rows'):
                backend._prepare_text('AB')
            encoder.encode_condition.assert_not_called()
        backend, encoder = self.backend()
        self.assertIs(backend._prepare_text('AB'), backend.prepared_text)
        encoder.prepare_condition.assert_called_once_with('AB')
        backend.prepared_text[1][0] = 0
        with self.assertRaisesRegex(serving.SessionError, 'all-text'):
            backend._prepare_text('AB')
        backend.prepared_text[1][0] = 1
        backend.prepared_text[0]['mm_token_type_ids'][0, 0] = 1
        with self.assertRaisesRegex(serving.SessionError, 'all-text'):
            backend._prepare_text('AB')

    def test_arbitrary_tokenizer_failure_becomes_synchronized_rejection(self):
        backend, encoder = self.backend()
        encoder.prepare_condition.side_effect = RuntimeError('tokenizer crashed')
        with self.assertRaisesRegex(serving.SessionError, 'tokenizer crashed'):
            backend.group.rank0(lambda: backend._prepare_text('AB'))
        encoder.encode_condition.assert_not_called()

    def test_all_ranks_callback_only_participants_forward(self):
        for rank in (0, 1, 2, 7):
            backend, encoder = self.backend(rank=rank)
            document = backend._encode_text('AB')
            self.assertEqual(document['prompt_embeds'].shape, (2, 2))
            self.assertEqual(encoder.encode_condition.call_count, int(rank < 2))
            self.assertEqual(backend.dynamic_text_forwards, 1)
            backend.group.rank0.assert_called_once()

    def test_participant_errors_are_fatal_and_skip_document_broadcast(self):
        for rank in (0, 1):
            for error in (serving.SessionError('participant failed'), ValueError('shape mismatch'), RuntimeError('FSDP failed')):
                backend, _ = self.backend(rank=rank, failure=error)
                with self.assertRaisesRegex(RuntimeError, 'worker restart required'):
                    backend._encode_text('AB')
                backend.group.rank0.assert_not_called()
                self.assertEqual(backend.dynamic_text_forwards, 0)

    def test_rank_zero_document_validation_is_synchronized(self):
        backend, encoder = self.backend()
        encoder.encode_condition.return_value['prompt_embeds'][0, 0] = float('nan')
        with self.assertRaisesRegex(serving.SessionError, 'finite'):
            backend._encode_text('AB')
        backend.group.rank0.assert_called_once()

    def test_unsharded_dynamic_forward_consumes_prepared_input_once(self):
        backend, _ = self.backend()
        backend.qwen_fsdp = False
        qwen = Qwen()
        backend.encoders = SimpleNamespace(qwen=qwen, device=torch.device('cpu'), processor=object())
        backend.qwen_encoder = backend.encoders
        qwen.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096)))
        backend.static_document['prompt_embeds'] = backend.static_document['prompt_embeds'].to(torch.bfloat16)
        with patch.object(cwm_encoders, 'qwen_inputs', return_value=backend.prepared_text) as prepare:
            backend.prepared_text = backend.group.rank0(lambda: backend._prepare_text('AB'))
            result = backend._encode_text('AB')
        prepare.assert_called_once_with(backend.encoders.processor, 'AB')
        self.assertEqual(result['prompt_embeds'].shape, (2, 2))
        qwen.reshard.assert_not_called()

    def test_resident_chunks_keep_driver_dedup_aba_counts_and_applied_identity(self):
        from projects.minimax_h3.tests.test_streaming_text import DRIVERS, driver, document
        from projects.minimax_h3.tests.test_ref_inference import CacheProbe, inputs
        class Sink:
            def __init__(self, receive, **kwargs):
                self.receive = receive
            def submit(self, key, rgb):
                self.receive(key, rgb)
            def close(self):
                pass
        for cls in DRIVERS:
            backend, encoder = self.backend()
            static, picture, chunks = inputs()
            backend.streaming_text = True
            backend.static_document = static
            backend.backbone = CacheProbe()
            backend.spec = dict(proxy_height=1, proxy_width=1)
            backend.memory = lambda: {}
            backend.group.gather = lambda value: [value]
            backend.group.call_on = lambda owner, callback: callback()
            backend.inputs = serving._ChunkInput()
            runner = driver(cls)
            backend.recipe = runner.recipe
            backend.outputs = runner.stream_latents(backend.backbone, static, backend.inputs,
                first_frame=picture, encode_text=backend._lookup_text)
            references = iter(item['references'][0]['video'] for item in chunks)
            backend.codec = SimpleNamespace(encode=lambda frames, **kwargs: next(references),
                decode=lambda result, **kwargs: torch.zeros((5 if kwargs['eos'] else 17, 1, 1, 3), dtype=torch.uint8))
            def prepare(text):
                return (dict(input_ids=torch.ones((1, len(text)), dtype=torch.long),
                    mm_token_type_ids=torch.zeros((1, len(text)), dtype=torch.int32)),
                    torch.ones(len(text), dtype=torch.long))
            encoder.prepare_condition.side_effect = prepare
            encoder.encode_condition.side_effect = lambda value: document(backend.submitted_text)
            receipts = []
            with patch('torch.cuda.Event'), patch.object(inference_runtime, 'OutputSink', Sink):
                for index, text in enumerate(('A', 'A', 'BBB', 'A')):
                    eos = index == 3
                    _, timings = backend.execute('chunk', dict(prompt=dict(text=text), eos=eos),
                                                 bytes((5 if eos else 17) * 3))
                    receipts.append(timings)
            self.assertEqual([r['dynamic_text_forwards'] for r in receipts], [1, 0, 1, 1])
            self.assertEqual([r['session_dynamic_text_forwards'] for r in receipts], [1, 1, 2, 3])
            self.assertEqual([r['applied_prompt']['first_chunk'] for r in receipts], [0, 0, 2, 3])
            self.assertEqual([r['applied_prompt']['rows'] for r in receipts], [1, 1, 3, 1])
            self.assertEqual(encoder.prepare_condition.call_count, 3)
            self.assertEqual(encoder.encode_condition.call_count, 3)


if __name__ == '__main__':
    unittest.main()
