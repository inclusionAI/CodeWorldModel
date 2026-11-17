import tempfile
import threading
import unittest
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.serving import Request
from projects.minimax_h3.tests.test_serving import Harness, start, chunk


class Occupier:
    def __init__(self):
        self.paused = False
        self.events = []

    def pause(self):
        self.events.append('pause')
        self.paused = True

    def resume(self):
        self.events.append('resume')
        self.paused = False

    def restore_after_shutdown(self):
        self.events.append('confirm_gpu_exit')
        self.resume()


class OccupierSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.occupier = Occupier()
        self.h = Harness(self.directory.name, occupier=self.occupier)
        close = self.h.backend.close_session
        def closed():
            self.occupier.events.append('cleanup')
            close()
        self.h.backend.close_session = closed

    def tearDown(self):
        self.h.close()
        self.directory.cleanup()

    def test_terminal_reply_waits_for_cleanup_and_resume_then_model_reused(self):
        for index in range(2):
            sid = str(index)
            self.assertEqual(self.h.exchange(start(sid)).metadata['gpu_occupier'], 'paused')
            gate = threading.Event()
            original = self.occupier.resume
            def resume():
                gate.wait(3)
                original()
            self.occupier.resume = resume
            job = self.h.box.submit(chunk(sid=sid, eos=True))
            self.assertFalse(job.ready.wait(.05))
            gate.set()
            self.assertTrue(job.ready.wait(3))
            self.assertEqual(job.reply.metadata['gpu_occupier'], 'running')
            self.h.box.acknowledge(job, True)
            self.assertTrue(job.released.wait(3))
            self.occupier.resume = original
        self.assertEqual(self.occupier.events, ['pause', 'cleanup', 'resume'] * 2)
        self.assertEqual(self.h.backend.weight_load_count, 1)

    def test_rejected_start_and_wrong_cancel_do_not_change_occupier(self):
        self.h.exchange(start(artifact='unknown'))
        self.assertEqual(self.occupier.events, [])
        self.h.exchange(start())
        self.h.exchange(start('other'), delivered=False)
        self.h.exchange(Request('cancel', dict(session_id='wrong')))
        self.assertEqual(self.occupier.events, ['pause'])
        self.assertTrue(self.occupier.paused)
        self.h.exchange(Request('cancel', dict(session_id='one')))
        self.assertEqual(self.occupier.events, ['pause', 'cleanup', 'resume'])

    def test_delivery_failure_releases_after_cleanup(self):
        self.h.exchange(start(), delivered=False)
        self.assertEqual(self.occupier.events, ['pause', 'cleanup', 'resume'])
        self.assertIsNone(self.h.controller.active)

    def test_idle_expiry_cleans_and_resumes(self):
        self.h.worker.idle_timeout = .02
        resumed = threading.Event()
        original = self.occupier.resume
        def resume():
            original()
            resumed.set()
        self.occupier.resume = resume
        self.h.exchange(start())
        self.assertTrue(resumed.wait(2))
        self.assertEqual(self.occupier.events, ['pause', 'cleanup', 'resume'])
        self.assertIsNone(self.h.controller.active)

    def test_pause_failure_does_not_execute_model(self):
        def fail():
            raise RuntimeError('pause failed')
        self.occupier.pause = fail
        job = self.h.box.submit(start())
        self.assertTrue(job.ready.wait(3))
        self.h.thread.join(3)
        self.assertEqual(self.h.backend.calls, [])
        self.assertIn('pause failed', str(self.h.error))

    def test_cleanup_failure_does_not_resume(self):
        self.h.exchange(start())
        def fail():
            raise RuntimeError('GPU cleanup failed')
        self.h.backend.close_session = fail
        job = self.h.box.submit(chunk(eos=True))
        self.assertTrue(job.ready.wait(3))
        self.h.thread.join(3)
        self.assertEqual(self.occupier.events, ['pause'])
        self.assertTrue(self.occupier.paused)

    def test_poisoned_group_waits_for_confirmed_gpu_exit_before_resume(self):
        self.h.exchange(start())
        self.h.backend.failed = True
        checking, exited = threading.Event(), threading.Event()
        original = self.occupier.restore_after_shutdown
        def restore():
            checking.set()
            self.assertTrue(exited.wait(3))
            original()
        self.occupier.restore_after_shutdown = restore
        job = self.h.box.submit(chunk(eos=True))
        try:
            self.assertTrue(checking.wait(2))
            self.assertFalse(job.ready.is_set())
            self.assertTrue(self.occupier.paused)
        finally:
            exited.set()
        self.assertTrue(job.ready.wait(2))
        self.h.box.acknowledge(job, True)
        self.assertTrue(job.released.wait(2))
        self.assertEqual(self.occupier.events, ['pause', 'cleanup', 'confirm_gpu_exit', 'resume'])

    def test_poisoned_group_cannot_resume_when_gpu_exit_is_unconfirmed(self):
        self.h.exchange(start())
        self.h.backend.failed = True
        def restore():
            raise RuntimeError('new GPU processes remain')
        self.occupier.restore_after_shutdown = restore
        job = self.h.box.submit(chunk(eos=True))
        self.assertTrue(job.ready.wait(2))
        self.h.thread.join(2)
        self.assertIn('new GPU processes remain', str(self.h.error))
        self.assertEqual(self.occupier.events, ['pause', 'cleanup'])
        self.assertTrue(self.occupier.paused)


class ServingShutdownTests(unittest.TestCase):
    def test_guarded_sigterm_closes_model_before_restore_and_restores_handler(self):
        from projects.minimax_h3.tools import serve_ref
        from projects.minimax_h3.tests.test_serving import LocalGroup
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / 'artifact.json'
            artifact.write_text('{"deployment":{"active_up_size":1}}')
            args = SimpleNamespace(timeout=1, idle_timeout=1, max_body_mib=128, artifact=[str(artifact)],
                runtime='ray', ray_address=None, vae_rank=0, decoder_rank=0, ray_cpus_per_worker=1, codec_overlap=False,
                ray_master_port=29500, ray_startup_timeout=1, output_dir=str(Path(directory)/'output'),
                gpu_occupier_config='guard.json', ray_placement='PACK', bind='127.0.0.1:0')
            events = []
            guard = Mock()
            guard.pause.side_effect = lambda: events.append('pause')
            guard.resume.side_effect = lambda: events.append('idle_resume')
            guard.restore_after_shutdown.side_effect = lambda: events.append('shutdown_restore')
            backend = Mock(weight_load_count=1, residency={}, qwen_placement={}, health={})
            backend.close.side_effect = lambda: events.append('model_close')
            options = dict(codec=dict(decoder='causal', decoder_weights=None, decoder_config=None, tae_weights=None), resident_weights=True,
                           output=dict(async_write=False), qwen_fsdp=False, dit=dict(cuda_graph=False))
            bundle = dict(deployment=dict(active_up_size=1), encoders=dict(recipe={}), weights=dict(sha256='x'))
            recipe = Mock()
            recipe.to_dict.return_value = {}
            def terminate():
                events.append('serve')
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            worker = Mock()
            worker.serve.side_effect = terminate
            modules = {
                'common.distributed.init': SimpleNamespace(configure_distributed=lambda x: x, destroy_distributed=Mock()),
                'common.logging': SimpleNamespace(configure_logging=lambda x: x),
                'projects.minimax_h3.inference.serving_ray': SimpleNamespace(RayResidentGroup=Mock(return_value=backend))}
            previous = signal.getsignal(signal.SIGTERM)
            with patch.dict('sys.modules', modules), \
                    patch.object(serve_ref, 'parse_run_arguments', return_value=args), \
                    patch.object(serve_ref, 'validate_bundle'), \
                    patch.object(serve_ref, 'options_for_args', return_value=options), \
                    patch.object(serve_ref, 'runtime_config', return_value={}), \
                    patch.object(serve_ref, 'effective_settings', return_value={}), \
                    patch.object(serve_ref, 'settings_receipt', return_value={}), \
                    patch.object(serve_ref, 'Distributed', return_value=LocalGroup()), \
                    patch.object(serve_ref, 'load_artifacts', return_value={'id': (bundle, recipe)}), \
                    patch.object(serve_ref, 'ServingHTTPServer'), \
                    patch.object(serve_ref, 'Worker', return_value=worker), \
                    patch('projects.minimax_h3.inference.gpu_occupier.GpuOccupier', return_value=guard):
                with self.assertRaises(KeyboardInterrupt):
                    serve_ref.main([])
            self.assertEqual(events, ['pause', 'idle_resume', 'serve', 'model_close', 'shutdown_restore'])
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


if __name__ == '__main__':
    unittest.main()
