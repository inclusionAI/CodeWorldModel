from dataclasses import replace
from copy import deepcopy
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from projects.minimax_h3.inference import serving
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS, RawLatentRecipe
from projects.minimax_h3.tests.test_qwen_fsdp import module
from projects.minimax_h3.tests.test_ref_inference import RefRecipe, recipe_config, Grid, Euler
from projects.minimax_h3.tests.test_serving_stages import StageTests


def recipe(protocol):
    old = RefRecipe.from_training(recipe_config())
    return RawLatentRecipe(old.conditioning_mode, 37, old.diffusion, old.guidance, protocol=protocol)


class RawServingTests(unittest.TestCase):
    def test_prefetch_matches_serial_windows_eos_and_encodes_once(self):
        from projects.minimax_h3.inference.serving_stages import WorkKey
        results = []
        for overlap in (False, True):
            backend, decoded = StageTests().backend()
            backend.config = SimpleNamespace(inference=SimpleNamespace(
                runtime=SimpleNamespace(codec=SimpleNamespace(decoder='causal'))))
            boundary = module('common.diffusion',
                build_diffusion=lambda _: dict(sampling_timesteps=Grid(), sampler=Euler()))
            requests = [dict(session_id='raw', chunk_id=k, eos=k == 3) for k in range(4)]
            with patch.dict('sys.modules', {'common.diffusion': boundary}):
                backend.execute('start', dict(recipe=recipe(RAW_PROTOCOLS[0]).to_dict(), artifact_id='test',
                    prompt='static', seed=73, condition_seed=94), bytes(32*64*3))
                pixels, windows = [], []
                for k, metadata in enumerate(requests):
                    backend.begin('chunk', metadata)
                    backend.text_condition()
                    backend.proxy_encode(bytes(3))
                    self.assertIsNone(backend.prefetched_proxy)
                    backend.render()
                    if overlap and k < 3:
                        backend.decode_with_prefetch(WorkKey(1, 'raw', 'chunk', k+1), requests[k+1], bytes(3))
                    else:
                        backend.decode()
                    rgb, receipt = backend.collect()
                    self.assertEqual(backend.prefetched_proxy is not None, overlap and k < 3)
                    pixels.append(rgb)
                    windows.append([(row['plan'], row['sources']) for row in receipt['windows']])
                self.assertEqual(backend.codec.encode.call_count, 4)
                self.assertEqual([call.kwargs['eos'] for call in backend.codec.encode.call_args_list],
                                 [False, False, False, True])
                backend.close_session()
                self.assertIsNone(backend.prefetched_proxy)
            results.append((pixels, windows, decoded))
        self.assertEqual(results[0][:2], results[1][:2])
        self.assertEqual([len(rows) for rows in results[1][1]], [0, 1, 2, 1])
        for before, after in zip(results[0][2], results[1][2], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_collect_keeps_each_rank_graph_execution_counts_without_window_payload(self):
        backend, _ = StageTests().backend()
        backend.recipe, backend.op = recipe(RAW_PROTOCOLS[0]), 'chunk'
        backend.cpu_output = None
        backend.timings = dict(windows=[dict(plan=dict(index=4), sources=['large payload'],
            dit_cuda_graph=dict(positive=dict(captures=4, replays=12, entries=4)))])
        def gather(value):
            rows = [deepcopy(value) for _ in range(8)]
            if 'windows' in value:
                for i, row in enumerate(rows):
                    row['rank'] = i
                    row['windows'][0]['dit_cuda_graph']['positive']['replays'] += i
            return rows
        backend.group.gather = gather
        _, receipt = backend.collect()
        for i, row in enumerate(receipt['rank_timings']):
            self.assertNotIn('windows', row)
            self.assertEqual(row['dit_cuda_graph_windows'], [dict(window_index=4,
                branches=dict(positive=dict(captures=4, replays=12+i, entries=4)))])
        self.assertIn('sources', receipt['windows'][0])

    def test_max_prompt_and_eight_rank_receipt_fit_wire_budget(self):
        from projects.minimax_h3.inference.serving_http import pack, unpack
        from projects.minimax_h3.inference.streaming_text import MAX_PROMPT_BYTES
        backend, _ = StageTests().backend()
        backend.config = SimpleNamespace(inference=SimpleNamespace(
            runtime=SimpleNamespace(codec=SimpleNamespace(decoder='causal'))))
        backend.group.gather = lambda value:[dict(deepcopy(value), rank=i) for i in range(8)]
        backend._encode_text.side_effect = lambda text:dict(prompt_embeds=torch.ones(2,32),
                                                          text_token_tags=torch.ones(2,dtype=torch.long))
        boundary = module('common.diffusion',
            build_diffusion=lambda _:dict(sampling_timesteps=Grid(), sampler=Euler()))
        with patch.dict('sys.modules', {'common.diffusion':boundary}):
            backend.execute('start', dict(recipe=recipe(RAW_PROTOCOLS[2]).to_dict(), artifact_id='test',
                prompt='static', seed=73, condition_seed=94), bytes(32*64*3))
            for k in range(2):
                text = chr(ord('A')+k)*MAX_PROMPT_BYTES
                _, receipt = backend.execute('chunk', dict(prompt=dict(text=text, revision=k+1),
                                                           eos=False), bytes(3))
                metadata = dict(receipt, submitted_prompt=dict(text=text, revision=k+1, emitted_ns=k),
                    source_frames=[dict(source_frame=i, chunk_id=i//17, capture_ns=i,
                        prompt_revision=1, prompt_sha256='0'*64, prompt_first_chunk=0, prompt_rows=2)
                        for i in range(27 if k else 0)])
                decoded, _ = unpack(pack(metadata))
                self.assertEqual(decoded['submitted_prompt_encoding']['text'], text)
                self.assertTrue(all('submitted_prompt_encoding' not in row and 'windows' not in row
                                    for row in receipt['rank_timings']))
                for window in receipt['windows']:
                    self.assertTrue(all('text' not in source['prompt'] for source in window['sources']))
            backend.close_session()

    def test_real_stage_dispatch_all_recipes_preserves_zero_and_multiple_outputs(self):
        for protocol in RAW_PROTOCOLS:
            backend, decoded = StageTests().backend()
            backend.config = SimpleNamespace(inference=SimpleNamespace(
                runtime=SimpleNamespace(codec=SimpleNamespace(decoder='causal'))))
            boundary = module('common.diffusion',
                build_diffusion=lambda _: dict(sampling_timesteps=Grid(), sampler=Euler()))
            selected = recipe(protocol)
            with patch.dict('sys.modules', {'common.diffusion':boundary}):
                backend.execute('start', dict(recipe=selected.to_dict(), artifact_id='test',
                    prompt='static', seed=73, condition_seed=94), bytes(32*64*3))
                counts, sources = [], []
                for k in range(4):
                    pixels, receipt = backend.execute('chunk',
                        dict(prompt=dict(text=f'prompt {k}', revision=k+1), eos=k==3), bytes(3))
                    counts.append(len(receipt['windows']))
                    sources.extend(source for window in receipt['windows'] for source in window['sources'])
                    self.assertEqual(bool(pixels), k>0)
                    if selected.streaming_text:
                        self.assertEqual(receipt['submitted_prompt_encoding']['first_chunk'], k)
                self.assertEqual(counts, [0,1,2,1])
                self.assertEqual([x.shape[1] for x in decoded], [9,3,2,3])
                self.assertEqual([s['source_clip'] for s in sources], [0,1,1,2,2,2,3])
                backend.close_session()

    def test_controller_raw_defaults_overrides_and_delayed_prompt_fifo(self):
        selected = recipe(RAW_PROTOCOLS[2])
        bundle = dict(encoders=dict(recipe=dict(height=1, width=1, proxy_height=1,
                                                proxy_width=1, system_prompt='fixed')))
        controller = serving.SessionController({'test': (bundle, selected)})
        start = serving.Request('start', dict(session_id='raw', artifact_id='test',
            seed=73, condition_seed=94, prompt='fixed'), bytes(3))
        checked = controller.prepare(start)
        self.assertEqual(checked['recipe'], selected.to_dict())
        controller.finish(start, b'', {})
        emitted = []
        for k, returned in enumerate((0,27,16,13)):
            count = 5 if k==3 else 17
            prompt = dict(revision=k+1, text=f'prompt {k}', emitted_ns=k)
            request = serving.Request('chunk', dict(session_id='raw', chunk_id=k,
                first_source_frame=k*17, capture_ns=list(range(k*17,k*17+count)),
                assembled_ns=k*17+count, eos=k==3, prompt=prompt), bytes(count*3))
            controller.prepare(request)
            encoding = dict(text=prompt['text'], sha256=hashlib.sha256(prompt['text'].encode()).hexdigest(),
                            first_chunk=k, rows=2)
            reply = controller.finish(request, bytes(returned*3), dict(submitted_prompt_encoding=encoding))
            emitted.extend(reply.metadata['source_frames'])
        self.assertEqual([r['source_frame'] for r in emitted], list(range(56)))
        self.assertEqual([r['prompt_revision'] for r in emitted], [1]*17+[2]*17+[3]*17+[4]*5)
        self.assertEqual(reply.metadata['status'], 'complete')
        for changes in (dict(window_size=2), dict(sink_latents=True), dict(current_latent_schedule=[0])):
            other = serving.SessionController({'test': (bundle, selected)})
            with self.assertRaises(serving.SessionError):
                other.prepare(serving.Request('start', dict(start.metadata, **changes), bytes(3)))
        other = serving.SessionController({'test': (bundle, selected)})
        effective = other.prepare(serving.Request('start',
            dict(start.metadata, sink_latents=4, recent_latents=0, current_latent_schedule=[2]), bytes(3)))
        self.assertEqual(effective['recipe']['sink_latents'], 4)
        self.assertEqual(effective['recipe']['recent_latents'], 0)


if __name__ == '__main__':
    unittest.main()
