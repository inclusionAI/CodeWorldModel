from contextlib import ExitStack, redirect_stdout
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

from common.engine import BaseEngine
from projects.minimax_h3.inference import ref2va
from projects.minimax_h3.inference.raw_latent import RawLatentDriver
from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
from projects.minimax_h3.modeling import cwm_encoders, inference_runtime
from projects.minimax_h3.modeling.streaming_video_decoder import CausalViTDecodeStream
from projects.minimax_h3.tests.test_qwen_fsdp import module
from projects.minimax_h3.tests.test_raw_serving import recipe
from projects.minimax_h3.tests.test_ref_inference import CacheProbe, Grid, Euler, nodes
from projects.minimax_h3.tools import ref2va_prepare


class PairDecoder:
    latent_chunk = 2
    _fast_generation = 0
    _fast_signature = None

    def __init__(self):
        self.vit = SimpleNamespace(init_state=lambda: dict(t=0), frames_to_trim=3,
                                    block_latents=5, decode_chunk=self.decode_chunk)
        self.calls = []

    def _validate(self, value):
        assert value.ndim == 5 and value.shape[2] == 24

    def _use_fast(self):
        return False

    def _rgb(self, value):
        return value.transpose(1, 2)

    def decode_chunk(self, value, state):
        self.calls.append(value.clone())
        frames = sum(1 if t % 5 == 0 else 4 for t in range(state['t'], state['t']+len(value[0])))
        state['t'] += len(value[0])
        return torch.zeros(1,3,frames,32,32)


class FileEngineTests(unittest.TestCase):
    def test_shared_file_engine_and_real_pair_buffer_support_all_v2_recipes(self):
        for protocol in RAW_PROTOCOLS:
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first = root/'first.png'
                Image.new('RGB', (32,32)).save(first)
                selected = recipe(protocol)
                config = nodes(dict(inference=dict(spec=dict(first_frame=str(first), proxy='fixture',
                    start=0, num_frames=56, height=32, width=32, proxy_height=32, proxy_width=32, prompt='fixed'),
                    checkpoint='fixture', output_dir=str(root/'out'), runtime=dict(profile=False, dit={}, codec={},
                    output=dict(async_write=False, queue_size=2)), recipe=selected.to_dict(),
                    seed=73, condition_seed=94, decode_mode='strict')))
                decoder = PairDecoder()
                decode_latents = inference_runtime.InferenceCodec.decode

                def frozen_init(value, *args, **kwargs):
                    value.device = torch.device('cpu')
                def condition(value, text, **kwargs):
                    return dict(prompt_embeds=torch.ones(3,32), text_token_tags=torch.ones(3,dtype=torch.long))
                class Codec:
                    def __init__(self, *args):
                        self.device = torch.device('cpu')
                        self.stream = CausalViTDecodeStream(decoder)
                    def encode(self, frames, *, eos):
                        return torch.ones(24,2 if eos else 5,2,2)
                    def decode(self, value, *, eos):
                        return decode_latents(self, value, eos=eos)
                frames_written = []
                class Writer:
                    def __init__(self, *args):
                        pass
                    def write(self, pixels):
                        frames_written.append(len(pixels))
                    def close(self, success):
                        pass
                def base_init(value, cfg, meta):
                    value.config, value.meta = cfg, meta
                    value.models = dict(backbone=CacheProbe())
                def raw(*args, **kwargs):
                    return iter(np.zeros((size,32,32,3),dtype=np.uint8) for size in (17,17,17,5))
                boundary = module('common.diffusion',
                    build_diffusion=lambda _:dict(sampling_timesteps=Grid(),sampler=Euler()))
                with ExitStack() as stack:
                    for obj, name, value in (
                        (cwm_encoders.FrozenCWMEncoders,'__init__',frozen_init),
                        (cwm_encoders.FrozenCWMEncoders,'condition',condition),
                        (cwm_encoders.FrozenCWMEncoders,'image',lambda *_:torch.zeros(24,1,2,2)),
                        (BaseEngine,'__init__',base_init), (inference_runtime,'InferenceCodec',Codec),
                        (ref2va_prepare,'iter_video_chunks',raw), (ref2va,'VideoWriter',Writer),
                        (ref2va,'rank0_call',lambda fn:fn()),
                    ):
                        stack.enter_context(patch.object(obj,name,value))
                    stack.enter_context(patch.dict('sys.modules', {'common.diffusion':boundary}))
                    stack.enter_context(patch('projects.minimax_h3.inference.streaming_text.encode_instruction',
                                             side_effect=lambda enc,text:condition(enc,text)))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    engine = ref2va.Ref2VAStreamEngine(config, RawLatentDriver(selected))
                    if selected.streaming_text:
                        report = engine.run(prompt_source=lambda k:('A', None, 'B', None)[k])
                    else:
                        report = engine.run()
                self.assertEqual(sum(frames_written), 56)
                self.assertEqual(report['generation_windows'], 4)
                chunks = [torch.load(path, weights_only=True) for path in sorted((root/'out/chunks').glob('*.pt'))]
                self.assertEqual([row['video'].shape[1] for row in chunks], [9,3,2,3])
                self.assertEqual([call.shape[1] for call in decoder.calls], [2]*9)
                self.assertTrue(chunks[-1]['eos'])
                if selected.streaming_text:
                    ledger = [json.loads(line) for line in (root/'out/prompt-ledger.jsonl').read_text().splitlines()]
                    self.assertEqual([row['source_clip'] for row in ledger], [0,1,1,2,2,2,3])
                    self.assertEqual([row['prompt']['text'] for row in ledger], ['A','A','A','B','B','B','B'])


if __name__ == '__main__':
    unittest.main()
