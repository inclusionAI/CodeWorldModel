"""Resident text preflight and participant failure across real CPU collectives."""
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, directory, mode):
    import torch.distributed as dist
    from projects.minimax_h3.inference.serving import Distributed, SessionError
    from projects.minimax_h3.tests.test_qwen_fsdp import DynamicTextTests
    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root/'pg').as_uri(), rank=rank, world_size=3,
                            timeout=timedelta(seconds=8))
    group = Distributed()
    participants = dist.new_group([0, 1], backend='gloo', timeout=timedelta(seconds=8))
    backend, encoder = DynamicTextTests().backend(rank=rank)
    backend.group = group
    calls = []
    document = encoder.encode_condition.return_value
    def encode(prepared):
        calls.append('forward')
        if mode == 'fatal' and rank == 1:
            raise SessionError('injected participant SessionError')
        values = [None, None]
        dist.all_gather_object(values, prepared[0]['input_ids'].tolist(), group=participants)
        assert values[0] == values[1]
        return document
    encoder.encode_condition.side_effect = encode
    if mode == 'prepare' and rank == 0:
        encoder.prepare_condition.side_effect = RuntimeError('injected tokenizer error')
    try:
        error = None
        try:
            backend.prepared_text = group.rank0(lambda: backend._prepare_text('AB'))
            result = backend._encode_text('AB')
            assert result['prompt_embeds'].shape == (2, 2)
        except SessionError as exc:
            error = str(exc)
        if mode == 'fatal':
            raise AssertionError('participant failure returned as a recoverable session')
        rows = group.gather(dict(rank=rank, calls=calls, error=error,
                                 forwards=backend.dynamic_text_forwards))
        if mode == 'prepare':
            assert all(row['error'] and 'injected tokenizer error' in row['error'] for row in rows), rows
            assert all(row['calls'] == [] and row['forwards'] == 0 for row in rows), rows
        else:
            assert [row['calls'] for row in rows] == [['forward'], ['forward'], []], rows
            assert all(row['error'] is None and row['forwards'] == 1 for row in rows), rows
        if rank == 0:
            (root/'results.json').write_text(json.dumps(rows))
    finally:
        dist.destroy_process_group()


class ServingTextGlooTests(unittest.TestCase):
    def probe(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            processes = [subprocess.Popen([sys.executable, str(ROOT/'tools/launch.py'),
                'projects.minimax_h3.tests.test_serving_text_gloo', '--worker', str(rank), directory, mode],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in range(3)]
            try:
                results = []
                for process in processes:
                    out, err = process.communicate(timeout=45)
                    results.append((process.returncode, out+err))
                if mode == 'fatal':
                    self.assertTrue(all(code != 0 for code, _ in results), results)
                    self.assertTrue(any('worker restart required' in log for _, log in results), results)
                    self.assertFalse((Path(directory)/'results.json').exists())
                else:
                    self.assertTrue(all(code == 0 for code, _ in results), results)
                    self.assertEqual(len(json.loads((Path(directory)/'results.json').read_text())), 3)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_participants_forward_and_nonmember_only_receives_document(self):
        self.probe('normal')

    def test_arbitrary_preparation_failure_reaches_all_ranks_before_forward(self):
        self.probe('prepare')

    def test_participant_session_error_terminates_every_process(self):
        self.probe('fatal')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        unittest.main()
