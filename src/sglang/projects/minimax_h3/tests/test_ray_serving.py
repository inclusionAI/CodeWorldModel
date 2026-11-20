from datetime import timedelta
import hashlib
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from projects.minimax_h3.inference.serving_stages import StageCursor, STAGES, WorkKey
from tools.ray_worker import RankActor


class CursorTests(unittest.TestCase):
    def test_prefetch_requires_immediate_chunk_in_same_epoch_and_session(self):
        current = WorkKey(3, 'one', 'chunk', 4)
        current.check_next(WorkKey(3, 'one', 'chunk', 5))
        for other in (WorkKey(2, 'one', 'chunk', 5), WorkKey(3, 'two', 'chunk', 5),
                      WorkKey(3, 'one', 'chunk', 6), WorkKey(3, 'one', 'start', 5)):
            with self.assertRaises(RuntimeError):
                current.check_next(other)

    def test_rejects_duplicate_chunk_stale_epoch_and_stage_reordering(self):
        cursor = StageCursor()
        start = WorkKey(1, 'session', 'start')
        for stage in STAGES['start']:
            cursor.check(start, stage)
            cursor.completed(start)
        chunk = WorkKey(1, 'session', 'chunk', 0)
        with self.assertRaises(RuntimeError):
            cursor.check(chunk, 'render')
        for stage in STAGES['chunk']:
            cursor.check(chunk, stage)
            cursor.completed(chunk)
        with self.assertRaises(RuntimeError):
            cursor.check(chunk, 'text_condition')
        cursor.reset()
        with self.assertRaises(RuntimeError):
            cursor.check(start, 'static_condition')
        cursor.check(WorkKey(2, 'session', 'start'), 'static_condition')


class GraphLaunchTests(unittest.TestCase):
    def test_reset_completion_waits_for_gpu_cleanup(self):
        actor = RankActor.__new__(RankActor)
        actor.cursor = StageCursor()
        actor.cursor.epoch = 1
        reset_started, cleanup_release, finished = (threading.Event() for _ in range(3))
        actor.backend = Mock()
        actor.backend.close_session.side_effect = reset_started.set
        def synchronize():
            if reset_started.is_set():
                self.assertTrue(cleanup_release.wait(3))
        def close():
            actor.close_session(1)
            finished.set()
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.synchronize', side_effect=synchronize):
            thread = threading.Thread(target=close)
            thread.start()
            try:
                self.assertTrue(reset_started.wait(2))
                self.assertFalse(finished.wait(.05))
            finally:
                cleanup_release.set()
                thread.join(3)
            self.assertTrue(finished.is_set())

    def test_preflight_rejects_before_asset_verification(self):
        from projects.minimax_h3.inference.settings import load_inference_options
        actor = RankActor.__new__(RankActor)
        options = load_inference_options(data={'dit': {'cuda_graph': True}})
        with patch.dict(os.environ, {'NCCL_GRAPH_REGISTER': '1'}), \
                patch('projects.minimax_h3.inference.artifact.verify_bundle_assets') as verify:
            with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_REGISTER=0'):
                actor.preflight({}, options, True)
            verify.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_REGISTER=0'):
                actor.initialize('', 0, {}, options, 1)

    def test_existing_cluster_actor_receives_explicit_graph_environment(self):
        from projects.minimax_h3.inference.serving_ray import RayResidentGroup
        ray = Mock()
        ray.is_initialized.return_value = True
        actor = Mock()
        ray.remote.return_value.return_value = actor
        actor.options.return_value.remote.side_effect = RuntimeError('stop after actor options')
        options = {'dit': {'cuda_graph': True}}
        with patch.dict(os.environ, {'NCCL_GRAPH_REGISTER': '0', 'NCCL_CUMEM_HOST_ENABLE': '0'}, clear=True), \
                patch.dict(sys.modules, {'ray': ray, 'ray.util': Mock(),
                    'ray.util.placement_group': Mock(), 'ray.util.scheduling_strategies': Mock()}), \
                patch.object(RayResidentGroup, 'close'):
            with self.assertRaisesRegex(RuntimeError, 'stop after actor options'):
                RayResidentGroup({'fixture': ({'deployment': {'active_up_size': 8}}, None)}, options)
        environment = actor.options.call_args.kwargs['runtime_env']['env_vars']
        self.assertEqual(environment['NCCL_GRAPH_REGISTER'], '0')
        self.assertEqual(environment['NCCL_GRAPH_MIXING_SUPPORT'], '1')
        self.assertEqual(environment['NCCL_CUMEM_HOST_ENABLE'], '0')
        self.assertNotIn('PYTORCH_ALLOC_CONF', environment)


class GatedRGB:
    def __init__(self, tensor, gate, *, fail=False):
        self.tensor, self.gate, self.fail = tensor, Path(gate), fail
        self.device = tensor.device

    def numel(self):
        return self.tensor.numel()

    def element_size(self):
        return self.tensor.element_size()

    def numpy(self):
        self.gate.with_suffix('.entered').touch()
        deadline = time.monotonic() + 15
        while not self.gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('test serializer gate was not released')
            time.sleep(.01)
        if self.fail:
            raise RuntimeError('injected CPU serialization failure')
        return self.tensor.numpy()


class CPUBackend:
    def __init__(self, rank, vae_rank, decoder_rank):
        from projects.minimax_h3.tests.test_serving import FakeBackend
        from projects.minimax_h3.inference.serving import Distributed
        self.fake = FakeBackend()
        self.rank, self.vae_rank, self.decoder_rank = rank, vae_rank, decoder_rank
        self.group = Distributed()
        self.pixels = b''
        self.prefetched_proxy = None
        from types import SimpleNamespace
        from projects.minimax_h3.modeling.inference_runtime import StageProfiler
        self.inputs = SimpleNamespace(item=None)
        self.profiler = StageProfiler(False)
        self.streaming_text = False
        self.encodes = []

    def begin(self, op, metadata):
        self.op, self.metadata = op, metadata
        self.trace = []
        self.pixels = b''
        self.timings = {}

    def static_condition(self, payload):
        self.trace.append('static_condition')
        self.fake.execute('start', self.metadata, b'')

    def first_image(self, payload):
        self.trace.append('first_image')

    def open_recipe(self):
        self.trace.append('open_recipe')

    def text_condition(self):
        from projects.minimax_h3.inference.serving import SessionError
        self.trace.append('text_condition')
        def prepare():
            if self.metadata.get('reject_text'):
                raise SessionError('injected invalid text')
        self.group.rank0(prepare)
        if self.metadata.get('fail_rank') == self.rank:
            raise RuntimeError('injected rank failure')
        if self.metadata.get('stall'):
            time.sleep(10)

    def rendezvous(self, metadata, owner):
        if not metadata.get('decode_rendezvous' if owner == 'decode' else 'rendezvous'):
            return
        import torch.distributed as dist
        store = dist.distributed_c10d._get_default_store()
        prefix = metadata['session_id']
        store.set(f'{prefix}/{owner}', b'entered')
        other = 'decode' if owner == 'encode' else 'encode'
        store.wait([f'{prefix}/{other}'], timedelta(seconds=5))

    def _encode_proxy(self, payload, metadata, timings, profiler):
        self.rendezvous(metadata, 'encode')
        if metadata.get('reject_encode'):
            raise ValueError('injected invalid proxy')
        self.encodes.append(metadata['chunk_id'])
        timings['encode_start_ns'] = time.perf_counter_ns()
        return dict(sha=hashlib.sha256(payload).hexdigest())

    def proxy_encode(self, payload):
        from projects.minimax_h3.inference.serving import ResidentBackend
        ResidentBackend.proxy_encode(self, payload)
        self.input_sha = self.inputs.item['sha']
        self.trace.append('proxy_encode')

    def decode_with_prefetch(self, key, metadata, payload):
        from projects.minimax_h3.inference.serving import ResidentBackend
        return ResidentBackend.decode_with_prefetch(self, key, metadata, payload)

    def render(self):
        import torch
        import torch.distributed as dist
        value = torch.tensor(self.rank + 1)
        dist.all_reduce(value)
        assert value.item() == 3
        self.trace.append('render')
        self.inputs.item = None

    def decode(self):
        def decode():
            self.rendezvous(self.metadata, 'decode')
            self.pixels, _ = self.fake.execute('chunk', self.metadata, b'')
        self.group.call_on(self.decoder_rank, decode)
        self.trace.append('decode')

    def collect(self, *, defer_output=False):
        import torch
        trace = self.group.gather(dict(rank=self.rank, pid=os.getpid(), stages=self.trace,
                                      encodes=list(self.encodes)))
        pixels = torch.tensor(list(self.pixels), dtype=torch.uint8) if defer_output else self.pixels
        if defer_output and self.rank == self.decoder_rank and self.metadata.get('output_gate'):
            pixels = GatedRGB(pixels, self.metadata['output_gate'], fail=self.metadata.get('output_fail', False))
        return pixels, dict(rank_trace=trace, input_sha=getattr(self, 'input_sha', None))

    def close_session(self):
        self.fake.close_session()
        self.prefetched_proxy = None
        self.encodes.clear()


class CPURankActor(RankActor):
    def preflight(self, bundle, options, verify_assets):
        if options.get('fail_preflight') and self.rank == 1:
            raise ValueError('injected missing asset')
        return self.describe()

    def initialize(self, host, port, artifacts, options, timeout):
        import torch.distributed as dist
        dist.init_process_group('gloo', init_method=f'tcp://{host}:{port}', rank=self.rank,
                                world_size=self.world_size, timeout=timedelta(seconds=timeout))
        self.backend = CPUBackend(self.rank, self.vae_rank, self.decoder_rank)
        return dict(self.describe(), residency=[dict(rank=i) for i in range(self.world_size)],
                    qwen_placement=dict(mode='fixture', ranks=[]))


def unused_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


class RayProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
        cls.ray = ray
        ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)

    @classmethod
    def tearDownClass(cls):
        cls.ray.shutdown()

    def group(self, *, options=None, timeout=10, overlap=False):
        from projects.minimax_h3.inference.serving_ray import RayResidentGroup
        return RayResidentGroup({'fixture': (dict(deployment=dict(active_up_size=2)), None)},
            options or {}, cpus_per_worker=1, master_port=unused_port(), stage_timeout=timeout,
            startup_timeout=90, vae_rank=1, decoder_rank=0,
            codec_overlap=overlap,
            _actor_class=CPURankActor, _gpus_per_worker=0)

    def assert_released(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.ray.available_resources().get('CPU', 0) == 4:
                return
            time.sleep(.1)
        self.fail(str(self.ray.available_resources()))

    def test_codec_owners_overlap_once_and_reset_after_encode_error(self):
        from projects.minimax_h3.inference.serving import SessionError
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group(overlap=True)
        try:
            for sid in ('overlap', 'rejected', 'reset-pending', 'recovered'):
                request = start(sid)
                group.execute(request.op, request.metadata, request.pixels)
                current, following = chunk(sid=sid), chunk(1, sid=sid, eos=True)
                following.metadata['rendezvous'] = True
                current.metadata['decode_rendezvous'] = True
                def peek():
                    return following.metadata, following.pixels
                if sid == 'rejected':
                    following.metadata['reject_encode'] = True
                if sid == 'rejected':
                    with self.assertRaisesRegex(SessionError, 'injected invalid proxy'):
                        group.execute('chunk', current.metadata, current.pixels, lookahead=peek)
                    self.assertFalse(group.failed)
                else:
                    first, _ = group.execute('chunk', current.metadata, current.pixels, lookahead=peek)
                    if sid != 'reset-pending':
                        following.metadata['rendezvous'] = False
                        last, receipt = group.execute('chunk', following.metadata, following.pixels)
                        self.assertEqual(first + last, bytes([7]) * 22 * 3)
                        self.assertEqual(receipt['rank_trace'][1]['encodes'], [0, 1])
                        self.assertEqual(receipt['input_sha'], hashlib.sha256(following.pixels).hexdigest())
                        self.assertTrue(receipt['codec_prefetch']['reused'])
                        proxy = next(row for row in receipt['orchestration']['stages'] if row['stage'] == 'proxy_encode')
                        self.assertTrue(proxy['reused'])
                group.close_session()
                self.assertIsNone(group.prefetch_bounds)
        finally:
            group.close()
        self.assert_released()

    def test_real_process_payload_eos_cancel_and_reset(self):
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group()
        try:
            self.assertEqual(len({w['pid'] for w in group.health['workers']}), 2)
            self.assertNotIn(os.getpid(), {w['pid'] for w in group.health['workers']})
            for sid in ('first', 'after-cancel'):
                request = start(sid)
                group.execute(request.op, request.metadata, request.pixels)
                request = chunk(sid=sid, eos=sid == 'after-cancel')
                pixels, timings = group.execute(request.op, request.metadata, request.pixels)
                self.assertEqual(len(pixels), (13 if sid == 'first' else 5)*3)
                self.assertEqual(timings['input_sha'], hashlib.sha256(request.pixels).hexdigest())
                self.assertTrue(all(row['stages'] == list(STAGES['chunk'][:-1]) for row in timings['rank_trace']))
                group.close_session()
            self.assertEqual(group.epoch, 2)
        finally:
            group.close()
        self.assert_released()

    def test_next_gpu_work_completes_during_cpu_serialization_and_reset_drains_output(self):
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group()
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'release'
            try:
                request = start()
                group.execute(request.op, request.metadata, request.pixels)
                request = chunk()
                request.metadata['output_gate'] = str(gate)
                first, first_timing = group.execute('chunk', request.metadata, request.pixels, defer_output=True)
                deadline = time.monotonic() + 5
                while not gate.with_suffix('.entered').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(gate.with_suffix('.entered').exists(), 'CPU serializer did not start')
                request = chunk(1, eos=True)
                last, last_timing = group.execute('chunk', request.metadata, request.pixels, defer_output=True)
                self.assertFalse(gate.exists())
                self.assertGreater(last_timing['gpu_done_upper_bound_ns'], first_timing['gpu_done_upper_bound_ns'])
                self.assertEqual((len(first), len(last)), (13*3, 9*3))
                reset_done, reset_errors = threading.Event(), []
                def reset():
                    try:
                        group.close_session()
                    except Exception as exc:
                        reset_errors.append(exc)
                    finally:
                        reset_done.set()
                reset_thread = threading.Thread(target=reset)
                reset_thread.start()
                self.assertFalse(reset_done.wait(.1), 'reset returned before pending CPU output drained')
                gate.touch()
                self.assertTrue(reset_done.wait(5))
                reset_thread.join()
                self.assertEqual(reset_errors, [])
                first_rgb, receipt = first.read()
                last_rgb, _ = last.read()
                self.assertIsInstance(first_rgb, memoryview)
                self.assertTrue(first_rgb.readonly)
                self.assertEqual(bytes(first_rgb) + bytes(last_rgb), bytes([7]) * 22*3)
                self.assertEqual(receipt['output_delivery']['status'], 'complete')
                self.assertEqual(first_timing['input_sha'], hashlib.sha256(chunk().pixels).hexdigest())
                request = start('after-drain')
                group.execute(request.op, request.metadata, request.pixels)
                self.assertEqual(group.epoch, 2)
                group.close_session()
            finally:
                gate.touch()
                group.close()
        self.assert_released()

    def test_async_serialization_failure_poison_group_until_restart(self):
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group()
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / 'release'
            try:
                request = start()
                group.execute(request.op, request.metadata, request.pixels)
                request = chunk()
                request.metadata.update(output_gate=str(gate), output_fail=True)
                output, _ = group.execute('chunk', request.metadata, request.pixels, defer_output=True)
                gate.touch()
                wait = group._wait
                def receive_after_reset(refs, timeout=None):
                    replies = wait(refs, timeout)
                    group._wait = wait
                    with self.assertRaisesRegex(Exception, 'injected CPU serialization failure'):
                        output.read()
                    return replies
                group._wait = receive_after_reset
                group.close_session()
                self.assertTrue(group.failed)
                self.assertEqual(group.health['state'], 'unavailable; restart required')
                self.assertEqual(group.actors, [])
                with self.assertRaisesRegex(RuntimeError, 'unhealthy'):
                    group.execute('start', start('retry').metadata, b'')
            finally:
                gate.touch()
                group.close()
        self.assert_released()

    def test_failed_rank_invalidates_group_and_releases_placement(self):
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group()
        try:
            request = start()
            group.execute(request.op, request.metadata, request.pixels)
            self.ray.kill(group.actors[1], no_restart=True)
            request = chunk()
            with self.assertRaises(Exception):
                group.execute(request.op, request.metadata, request.pixels)
            self.assertTrue(group.failed)
            with self.assertRaisesRegex(RuntimeError, 'unhealthy'):
                group.execute(request.op, request.metadata, request.pixels)
        finally:
            group.close()
        self.assert_released()

    def test_collective_preparation_rejection_allows_new_session(self):
        from projects.minimax_h3.inference.serving import SessionError
        from projects.minimax_h3.tests.test_serving import start, chunk
        group = self.group()
        try:
            request = start('rejected')
            group.execute(request.op, request.metadata, request.pixels)
            request = chunk(sid='rejected')
            request.metadata['reject_text'] = True
            with self.assertRaisesRegex(SessionError, 'injected invalid text'):
                group.execute(request.op, request.metadata, request.pixels)
            self.assertFalse(group.failed)
            group.close_session()
            request = start('recovered')
            group.execute(request.op, request.metadata, request.pixels)
            request = chunk(sid='recovered', eos=True)
            pixels, _ = group.execute(request.op, request.metadata, request.pixels)
            self.assertEqual(len(pixels), 5*3)
            self.assertEqual(group.epoch, 2)
            group.close_session()
        finally:
            group.close()
        self.assert_released()

    def test_timeout_and_partial_startup_release_placement(self):
        from projects.minimax_h3.tests.test_serving import start, chunk
        with self.assertRaises(Exception):
            self.group(options=dict(fail_preflight=True))
        self.assert_released()
        group = self.group()
        try:
            request = start()
            group.execute(request.op, request.metadata, request.pixels)
            group.timeout = .2
            request = chunk()
            request.metadata['stall'] = True
            with self.assertRaises(TimeoutError):
                group.execute(request.op, request.metadata, request.pixels)
            self.assertTrue(group.failed)
        finally:
            group.close()
        self.assert_released()


if __name__ == '__main__':
    unittest.main()
