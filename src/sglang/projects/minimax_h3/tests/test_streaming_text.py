"""Raw instructions, owned documents and append/replay boundary behavior."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from projects.minimax_h3.inference.recipes.ref_streaming_text import (
    RefFixedHistoryStreamingText, RefBidirHistoryStreamingText)
from projects.minimax_h3.inference.streaming_text import (
    MAX_PROMPT_BYTES, MAX_TEXT_ROWS, LivePromptSource, ReplayPromptSource, encode_instruction)
from projects.minimax_h3.modeling.ref_recipe import RefRecipe
from projects.minimax_h3.tests.test_ref_inference import CacheProbe, Euler, Grid, inputs, recipe_config


DRIVERS = (RefFixedHistoryStreamingText, RefBidirHistoryStreamingText)


def driver(cls=RefBidirHistoryStreamingText, window=1):
    recipe = replace(RefRecipe.from_training(recipe_config()), protocol=cls.PROTOCOL, window_size=window)
    runner = cls(recipe)
    runner.sampling_timesteps, runner.sampler = Grid(), Euler()
    return runner


def document(text, dtype=torch.float32):
    return dict(prompt_embeds=torch.full((len(text), 32), sum(text.encode())/100, dtype=dtype),
                text_token_tags=torch.ones(len(text), dtype=torch.long))


class RawTextTests(unittest.TestCase):
    def test_exact_carry_dedup_aba_and_no_prefetch(self):
        for cls in DRIVERS:
            static, picture, chunks = inputs()
            text = '  turn 左\nnow  '
            values = [text, text, 'B', text]
            consumed, encoded = [], []
            def source():
                for k, item in enumerate(chunks):
                    consumed.append(k)
                    yield dict(item, prompt=values[k])
            def encode(value):
                encoded.append(value)
                return document(value)
            stream = driver(cls).stream_latents(CacheProbe(), static, source(), first_frame=picture, encode_text=encode)
            self.assertEqual(consumed, [])
            outputs = []
            for k, output in enumerate(stream):
                outputs.append(output)
                self.assertEqual(consumed, list(range(k+1)))
                self.assertEqual(output.prompt.text, values[k])
                self.assertEqual(output.prompt.rows, len(values[k]))
                self.assertEqual(output.prompt.sha256, hashlib.sha256(values[k].encode()).hexdigest())
                with self.assertRaises(FrozenInstanceError):
                    output.prompt.text = 'changed'
            self.assertEqual(encoded, [text, 'B', text])
            self.assertEqual([o.prompt.first_chunk for o in outputs], [0, 0, 2, 3])
            for k, item in enumerate(chunks):
                item['prompt'] = text if k == 0 else None
            encoded.clear()
            list(driver(cls).stream_latents(CacheProbe(), static, chunks, first_frame=picture, encode_text=encode))
            self.assertEqual(encoded, [text])

    def test_owned_static_refs_docs_history_and_outputs(self):
        for cls in DRIVERS:
            for window in (0, 1):
                def run(mutate):
                    static, picture, chunks = inputs()
                    returned = document('A')
                    records = []
                    runner = driver(cls, window)
                    build = runner._build_inputs
                    def observe(value, **kwargs):
                        result = build(value, **kwargs)
                        records.append(copy.deepcopy(result.samples[0]['student_condition']))
                        return result
                    runner._build_inputs = observe
                    def source():
                        for item in chunks:
                            yield dict(item, prompt='A')
                    stream = runner.stream_latents(CacheProbe(), static, source(), first_frame=picture,
                                                   encode_text=lambda _: returned)
                    outputs = []
                    for output in stream:
                        outputs.append(output.video.clone())
                        if mutate:
                            output.video.fill_(999)
                            static['prompt_embeds'].fill_(999)
                            picture.fill_(999)
                            returned['prompt_embeds'].fill_(999)
                    return outputs, records
                expected, before = run(False)
                actual, after = run(True)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(after, before):
                    torch.testing.assert_close(a['static']['prompt_embeds'], b['static']['prompt_embeds'], rtol=0, atol=0)
                    self.assertLessEqual(len(a['chunks']), window+1 if cls is DRIVERS[1] else 1)
                    for da, db in zip(a['chunks'], b['chunks']):
                        torch.testing.assert_close(da['prompt_embeds'], db['prompt_embeds'], rtol=0, atol=0)

    def test_invalid_raw_contract_and_documents_fail_before_forward(self):
        static, picture, chunks = inputs()
        cases = [dict(chunks[0]), dict(chunks[0], prompt=None), dict(chunks[0], prompt=''),
                 dict(chunks[0], prompt='  '), dict(chunks[0], prompt=12), dict(chunks[0], prompt='\ud800'),
                 dict(chunks[0], prompt='a'*(MAX_PROMPT_BYTES+1)),
                 dict(chunks[0], prompt='é'*(MAX_PROMPT_BYTES//2+1)), dict(chunks[0], prompt='A', eos=1),
                 dict(chunks[0], prompt='A', condition=static), dict(chunks[0], prompt='A', guidance_conditions={}),
                 dict(chunks[0], prompt='A', extra=True),
                 dict(chunks[0], prompt='A', references=[dict(kind='video', role='semantic',
                    video=torch.full((24, 5, 2, 4), float('nan')))])]
        for cls in DRIVERS:
            for item in cases:
                net = CacheProbe()
                with self.assertRaises(ValueError):
                    list(driver(cls).stream_latents(net, static, [item], first_frame=picture, encode_text=document))
                self.assertEqual(net.calls, [])
            bad_docs = [{}, dict(document('A'), extra=1), document('A', torch.float64),
                dict(document('A'), prompt_embeds=torch.ones(1, 31)),
                dict(document('A'), prompt_embeds=torch.full((1, 32), float('nan'))),
                dict(document('A'), text_token_tags=torch.zeros(1, dtype=torch.long)),
                dict(document('A'), text_token_tags=torch.ones(1)),
                dict(document('A'), text_token_tags=torch.ones(1, dtype=torch.int32)),
                document('a'*(MAX_TEXT_ROWS+1))]
            for doc in bad_docs:
                net = CacheProbe()
                with self.assertRaises(ValueError):
                    list(driver(cls).stream_latents(net, static, [dict(chunks[0], prompt='A')],
                        first_frame=picture, encode_text=lambda _: doc))
                self.assertEqual(net.calls, [])
            with self.assertRaisesRegex(ValueError, 'first_frame'):
                list(driver(cls).stream_latents(CacheProbe(), static, chunks, encode_text=document))
            chunks[0]['prompt'] = 'A'
            with self.assertRaisesRegex(ValueError, 'after EOS'):
                list(driver(cls).stream_latents(CacheProbe(), static, chunks+[chunks[0]], first_frame=picture,
                                               encode_text=document))

    def test_history_docs_keep_original_order_and_eviction(self):
        static, picture, chunks = inputs()
        runner = driver(window=1)
        windows = []
        forward = runner._ref_forward
        def observe(net, value, **kwargs):
            if kwargs['selection'][0] == 'window':
                windows.append((value.samples[0]['metadata']['stream_chunk_start'],
                    [len(d['prompt_embeds']) for d in value.samples[0]['student_condition']['chunks']]))
            return forward(net, value, **kwargs)
        runner._ref_forward = observe
        list(runner.stream_latents(CacheProbe(), static, [dict(c, prompt='a'*(i+1)) for i, c in enumerate(chunks)],
                                   first_frame=picture, encode_text=document))
        self.assertEqual(windows[::3], [(0, [1]), (0, [1, 2]), (1, [2, 3]), (2, [3, 4])])


class PromptSourceTests(unittest.TestCase):
    def test_live_partial_none_multiple_and_exact_utf8_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'live.jsonl'
            path.write_bytes(b'{"prompt":"first"}\n{"prompt":"sec')
            source = LivePromptSource(path)
            self.assertEqual(source(0), 'first')
            self.assertIsNone(source(1))
            with path.open('ab') as handle:
                handle.write(b'ond"}\n{"prompt":"latest"}\n')
            self.assertEqual(source(2), 'latest')
            self.assertIsNone(source(3))
            text = '\x01'*(MAX_PROMPT_BYTES-1)+'x'
            with path.open('ab') as handle:
                handle.write((json.dumps(dict(prompt=text))+'\n').encode())
            self.assertEqual(source(4), text)
            self.assertGreater(path.stat().st_size, MAX_PROMPT_BYTES)
            self.assertEqual(source.provenance['initial_prefix_sha256'],
                             hashlib.sha256(b'{"prompt":"first"}\n{"prompt":"sec').hexdigest())

    def test_freezes_endpoint_at_boundary_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'live'
            path.write_text('{"prompt":"initial"}\n')
            source = LivePromptSource(path)
            original = Path.open
            def append_before_open(target, *args, **kwargs):
                if target == path:
                    with original(path, 'ab') as handle:
                        handle.write(b'{"prompt":"during read"}\n')
                return original(target, *args, **kwargs)
            with patch.object(Path, 'open', append_before_open):
                self.assertEqual(source(0), 'initial')
            self.assertEqual(source(1), 'during read')

    def test_live_rejects_invalid_completed_records_replacement_and_truncation(self):
        for content in (b'', b'{"prompt":"partial"}', b'bad\n', b'{"prompt":null}\n',
                        b'{"prompt":"valid","unknown":1}\n', b'{"prompt":"\\ud800"}\n', b'x'*(128*1024+1)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'live'
                path.write_bytes(content)
                source = LivePromptSource(path)
                with self.assertRaises(ValueError):
                    source(0)
        for operation in ('replace', 'truncate'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'live'
                path.write_text('{"prompt":"valid"}\n')
                source = LivePromptSource(path)
                source(0)
                if operation == 'replace':
                    path.rename(path.with_suffix('.old'))
                path.write_bytes(b'')
                with self.assertRaisesRegex(ValueError, 'replaced|truncated'):
                    source(1)

    def test_live_rejects_same_inode_rewrites_of_consumed_or_initial_prefix(self):
        for suffix in ('', '{"prompt":"extra"}\n'):
            for polled in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'live'
                    path.write_text('{"prompt":"AAAA"}\n')
                    source = LivePromptSource(path)
                    if polled:
                        self.assertEqual(source(0), 'AAAA')
                    identity = path.stat().st_ino
                    path.write_text('{"prompt":"BBBB"}\n'+suffix)
                    self.assertEqual(path.stat().st_ino, identity)
                    with self.assertRaisesRegex(ValueError, 'modified or rewritten'):
                        source(1 if polled else 0)

    def test_replay_sorted_unique_range_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'replay'
            records = [dict(chunk=0, prompt='A'), dict(chunk=2, prompt='B')]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            source = ReplayPromptSource(path, 4)
            self.assertEqual([source(i) for i in range(4)], ['A', None, 'B', None])
            self.assertEqual(source.provenance['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            for indices in ([], [1], [0, 0], [0, -1], [0, 4], [0, True], [0, 2, 1], [0, 1.5]):
                path.write_text(''.join(json.dumps(dict(chunk=i, prompt='A'))+'\n' for i in indices))
                with self.assertRaises(ValueError):
                    ReplayPromptSource(path, 4)


if __name__ == '__main__':
    unittest.main()
