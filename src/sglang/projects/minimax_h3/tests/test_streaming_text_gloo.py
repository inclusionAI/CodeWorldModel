"""Two-rank CPU/Gloo orchestration with fake encoders and collective model probes."""
from contextlib import ExitStack, redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, directory):
    import torch
    import torch.distributed as dist
    import numpy as np
    from common.engine import BaseEngine
    from projects.minimax_h3.inference import ref2va
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
    from projects.minimax_h3.modeling import inference_runtime
    from projects.minimax_h3.tools import ref2va_prepare
    from projects.minimax_h3.tests.test_streaming_text import driver, DRIVERS, document
    from projects.minimax_h3.tests.test_streaming_text_engine import FakeProcessor
    from projects.minimax_h3.tests.test_ref_inference import CacheProbe, nodes
    from types import SimpleNamespace
    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root/'pg').as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    summaries = []
    try:
        for cls in DRIVERS:
            for failure in (None, 'source', 'encode'):
                runner = driver(cls)
                output = root/(cls.__name__+str(failure))
                cfg = nodes(dict(inference=dict(spec=dict(first_frame=str(root/'first.png'), proxy='fake',
                    start=0, num_frames=39, height=32, width=32, proxy_height=32, proxy_width=32, prompt='system'),
                    checkpoint='fake', output_dir=str(output), runtime=dict(profile=False, dit={}, codec={},
                    output=dict(async_write=False, queue_size=2)), recipe=runner.recipe.to_dict(), seed=73,
                    condition_seed=94, decode_mode='strict')))
                forwards, polls, encodes = [], [], []
                def frozen_init(value, *args, **kwargs):
                    value.device = torch.device('cpu')
                    value.processor = FakeProcessor()
                    value.qwen = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
                        text_config=SimpleNamespace(max_position_embeddings=4096))))
                def condition(value, text, videos=(), images=()):
                    if text != 'system':
                        encodes.append(text)
                        if failure == 'encode' and text == 'BBB':
                            raise ValueError('collective encode failure')
                    return document(text)
                class Codec:
                    def __init__(self, *args):
                        pass
                    def encode(self, frames, *, eos):
                        return torch.ones(24, 2 if eos else 5, 2, 2)
                    def decode(self, latent, *, eos):
                        return torch.zeros(5 if eos else 17, 32, 32, 3, dtype=torch.uint8)
                class Writer:
                    def __init__(self, *args):
                        pass
                    def write(self, *args):
                        pass
                    def close(self, **kwargs):
                        pass
                class Net(CacheProbe):
                    def forward(self, **kwargs):
                        signature = (kwargs['update_past_key_values'], kwargs['x'].shape[1])
                        values = [None, None]
                        dist.all_gather_object(values, signature)
                        assert values == [signature, signature], values
                        forwards.append(signature)
                        return super().forward(**kwargs)
                def base_init(value, config, meta):
                    value.config, value.meta, value.models = config, meta, dict(backbone=Net())
                    if rank == 0:
                        assert hasattr(value.encoders, 'qwen')
                def raw(*args, **kwargs):
                    for size in (17, 17, 5):
                        yield np.zeros((size, 32, 32, 3), dtype=np.uint8)
                def source(k):
                    assert rank == 0
                    polls.append(k)
                    if failure == 'source' and k == 1:
                        raise ValueError('collective source failure')
                    return ('A', 'BBB', None)[k]
                error = None
                with ExitStack() as stack:
                    for obj, name, value in ((FrozenCWMEncoders, '__init__', frozen_init),
                        (FrozenCWMEncoders, 'condition', condition),
                        (FrozenCWMEncoders, 'image', lambda *_: torch.zeros(24, 1, 2, 2)),
                        (BaseEngine, '__init__', base_init), (inference_runtime, 'InferenceCodec', Codec),
                        (ref2va_prepare, 'iter_video_chunks', raw), (ref2va, 'VideoWriter', Writer)):
                        stack.enter_context(patch.object(obj, name, value))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    engine = ref2va.Ref2VAStreamEngine(cfg, runner)
                    try:
                        engine.run(prompt_source=source)
                    except RuntimeError as exc:
                        error = str(exc)
                if rank == 0:
                    assert not hasattr(engine.encoders, 'qwen')
                gathered = [None, None]
                dist.all_gather_object(gathered, dict(error=error, forwards=forwards, polls=polls, encodes=encodes))
                assert gathered[0]['error'] == gathered[1]['error'], gathered
                assert gathered[0]['forwards'] == gathered[1]['forwards'], gathered
                assert gathered[1]['polls'] == gathered[1]['encodes'] == [], gathered
                expected = (16 if cls is DRIVERS[0] else 10) if failure is None else (6 if cls is DRIVERS[0] else 4)
                assert len(forwards) == expected, gathered
                if failure:
                    assert f'collective {failure} failure' in error, gathered
                else:
                    assert error is None, gathered
                    assert gathered[0]['encodes'] == ['A', 'BBB'], gathered
                    assert gathered[0]['polls'] == [0, 1, 2], gathered
                summaries.append(dict(protocol=cls.PROTOCOL, failure=failure, forwards=len(forwards)))
        if rank == 0:
            (root/'results.json').write_text(json.dumps(summaries))
    finally:
        dist.destroy_process_group()


def probe():
    from PIL import Image
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new('RGB', (32, 32)).save(root/'first.png')
        processes = [subprocess.Popen([sys.executable, str(ROOT/'tools/launch.py'),
            'projects.minimax_h3.tests.test_streaming_text_gloo', '--worker', str(rank), directory],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in range(2)]
        try:
            for process in processes:
                out, err = process.communicate(timeout=60)
                if process.returncode:
                    raise RuntimeError(out+err)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        print((root/'results.json').read_text())


class GlooTextTests(unittest.TestCase):
    def test_two_rank_update_forward_order_and_source_encode_error_fanout(self):
        result = subprocess.run([sys.executable, str(ROOT/'tools/launch.py'),
            'projects.minimax_h3.tests.test_streaming_text_gloo', '--probe'], text=True,
            capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('"forwards": 16', result.stdout)
        self.assertIn('"forwards": 10', result.stdout)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        worker(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1:] == ['--probe']:
        probe()
    else:
        unittest.main()
