from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


def worker(rank, directory, fatal=False):
    import torch.distributed as dist
    from projects.minimax_h3.inference.serving import Distributed, Request, SessionController, Worker
    from projects.minimax_h3.inference.serving_http import Ledger, SessionMailbox
    from projects.minimax_h3.tests.test_serving import FakeBackend, artifacts, chunk, start
    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root/'pg').as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=8))
    group = Distributed()
    trace = []

    class Backend(FakeBackend):
        def execute(self, op, metadata, pixels):
            assert threading.current_thread() is threading.main_thread()
            if op == 'chunk':
                def encode():
                    if self.seed == 13:
                        raise ValueError('rank zero encode failed')
                group.rank0(encode)
                if fatal and rank == 1:
                    raise RuntimeError('injected rank one collective failure')
                signatures = [None, None]
                dist.all_gather_object(signatures, (op, metadata['chunk_id']))
                assert signatures[0] == signatures[1]
            trace.append((op, metadata.get('session_id'), metadata.get('chunk_id')))
            return super().execute(op, metadata, pixels)

        def close_session(self):
            trace.append(('close', None, None))
            super().close_session()

    backend = Backend(group)
    ledger = Ledger(root/'server.jsonl') if rank == 0 else None
    box = SessionMailbox(ledger, timeout=3) if rank == 0 else None
    controller = SessionController(artifacts()) if rank == 0 else None
    driver_error = []

    def drive():
        def exchange(request, ok=True):
            job = box.submit(request)
            assert job.ready.wait(15)
            box.acknowledge(job, ok, None if ok else 'injected socket failure')
            assert job.released.wait(15)
            return job.reply
        try:
            exchange(start())
            if fatal:
                exchange(chunk())
                raise AssertionError('fatal collective unexpectedly returned')
            for index in range(4):
                reply = exchange(chunk(index, eos=index==3))
                assert reply.status == 200
            assert reply.metadata['total_frames'] == 56
            exchange(start('disconnect', 'bidir'))
            exchange(chunk(sid='disconnect', eos=True), ok=False)
            exchange(start('encode-error', seed=13))
            assert exchange(chunk(sid='encode-error')).status == 500
            exchange(start('cancel'))
            exchange(chunk(sid='cancel'))
            exchange(Request('cancel', dict(session_id='cancel')))
            exchange(start('reset'))
            assert exchange(chunk(sid='reset', eos=True)).pixels == bytes([7])*15
            exchange(Request('shutdown', {}))
        except BaseException as exc:
            driver_error.append(str(exc))
            box.stop()

    driver = None
    if rank == 0:
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
    try:
        Worker(backend, group, controller, box).serve()
        if driver is not None:
            driver.join(10)
            assert not driver.is_alive()
            assert not driver_error, driver_error
        traces = [None, None]
        dist.all_gather_object(traces, trace)
        assert traces[0] == traces[1], traces
        assert backend.weight_load_count == 1 and backend.sessions == 5
        if rank == 0:
            (root/'results.json').write_text(json.dumps(dict(sessions=backend.sessions,
                closes=backend.closed, same_rank_trace=True, operations=trace)))
    finally:
        if box is not None:
            box.fail('test worker exited')
        if ledger is not None:
            ledger.close()
        dist.destroy_process_group()


def probe(fatal=False):
    with tempfile.TemporaryDirectory() as directory:
        processes = [subprocess.Popen([sys.executable, str(ROOT/'tools/launch.py'),
            'projects.minimax_h3.tests.test_serving_gloo', '--worker', str(rank), directory,
            'fatal' if fatal else 'normal'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for rank in range(2)]
        results = []
        try:
            for process in processes:
                out, err = process.communicate(timeout=45)
                results.append((process.returncode, out+err))
            if fatal:
                assert all(code != 0 for code, _ in results), results
                assert any('injected rank one collective failure' in log for _, log in results), results
                print('fatal collective terminated both ranks')
            else:
                assert all(code == 0 for code, _ in results), results
                print((Path(directory)/'results.json').read_text())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


class ServingGlooTests(unittest.TestCase):
    def test_terminal_delivery_cancel_and_encode_error_reach_both_ranks(self):
        probe()

    def test_collective_error_cannot_recover_as_a_session(self):
        probe(fatal=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        worker(int(sys.argv[2]), sys.argv[3], sys.argv[4]=='fatal')
    else:
        unittest.main()
