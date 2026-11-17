"""Two real Gloo ranks must agree on every v2 forward and window boundary."""
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, directory):
    import torch
    import torch.distributed as dist
    from projects.minimax_h3.inference.raw_latent import RawLatentSession
    from projects.minimax_h3.inference.raw_recipe import RAW_PROTOCOLS
    from projects.minimax_h3.inference.serving import Distributed
    from projects.minimax_h3.tests.test_raw_serving import recipe
    from projects.minimax_h3.tests.test_ref_inference import CacheProbe, Grid, Euler, inputs
    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root/'pg').as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    group = Distributed()
    summaries = []
    try:
        for protocol in RAW_PROTOCOLS:
            calls = []
            class Net(CacheProbe):
                def forward(self, **kwargs):
                    signature = (kwargs['update_past_key_values'], tuple(kwargs['x'].shape),
                                 kwargs['sample_lens'], kwargs['unique_timesteps'].tolist())
                    values = [None, None]
                    dist.all_gather_object(values, signature)
                    assert values == [signature, signature], values
                    calls.append(signature)
                    return super().forward(**kwargs)
            static, picture, chunks = inputs()
            def text(value):
                return group.rank0(lambda:dict(prompt_embeds=torch.ones(2,32),
                    text_token_tags=torch.ones(2,dtype=torch.long)))
            stream = RawLatentSession(Net(), recipe(protocol), static, first_frame=picture, encode_text=text)
            stream.timesteps, stream.sampler = Grid(), Euler()
            counts, plans = [], []
            for k in range(4):
                item = group.rank0(lambda:dict(chunks[k], **(dict(prompt=f'prompt {k}', prompt_revision=k+1)
                    if stream.recipe.streaming_text else {})))
                result = stream.push(item)
                counts.append(len(result))
                plans.extend(output.plan.record() for output in result)
            assert counts == [0,1,2,1], counts
            gathered = [None, None]
            dist.all_gather_object(gathered, plans)
            assert gathered[0] == gathered[1], gathered
            summaries.append(dict(protocol=protocol, windows=len(plans), forwards=len(calls)))
        if rank == 0:
            (root/'results.json').write_text(json.dumps(summaries))
    finally:
        dist.destroy_process_group()


class RawGlooTests(unittest.TestCase):
    def test_all_protocols_have_identical_collective_order_and_window_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            processes = [subprocess.Popen([sys.executable, str(ROOT/'tools/launch.py'),
                'projects.minimax_h3.tests.test_raw_gloo', '--worker', str(rank), directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in range(2)]
            try:
                for process in processes:
                    out, err = process.communicate(timeout=60)
                    self.assertEqual(process.returncode, 0, out+err)
                result = json.loads((Path(directory)/'results.json').read_text())
                self.assertEqual(len(result), 4)
                self.assertTrue(all(row['windows'] == 4 for row in result))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        unittest.main()
