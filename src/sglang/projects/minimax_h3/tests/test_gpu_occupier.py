from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from inference.gpu_occupier import GpuOccupier


class GpuOccupierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.guard = GpuOccupier.__new__(GpuOccupier)
        self.guard.argv = ['/python', '/gpu_occupy.py']
        self.guard.root = (10, 100)
        self.guard.members = {(10, 100), (11, 110)}
        self.guard.log_path = Path(self.directory.name) / 'occupier.log'
        self.guard.paused = True
        self.guard.child = None
        self.guard.baseline_gpu_pids = {11}
        self.rows = {
            10: dict(pid=10, start_ticks=100, ppid=1, state='T', argv=self.guard.argv),
            11: dict(pid=11, start_ticks=110, ppid=10, state='T', argv=['child'])}
        self.guard.processes = lambda: self.rows.copy()
        self.guard.process = lambda pid: self.rows.get(pid)

    def test_tree_tracks_children_and_rejects_orphans_duplicates_and_replacement(self):
        self.assertEqual(self.guard.tree(), {(10, 100), (11, 110)})
        del self.rows[10]
        with self.assertRaisesRegex(RuntimeError, 'children remain'):
            self.guard.tree()
        self.rows[12] = dict(pid=12, start_ticks=120, ppid=1, state='R', argv=self.guard.argv)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.guard.tree()
        self.rows[13] = dict(pid=13, start_ticks=130, ppid=1, state='R', argv=self.guard.argv)
        with self.assertRaisesRegex(RuntimeError, 'multiple'):
            self.guard.tree()

    def test_pid_reuse_is_never_signalled(self):
        self.rows[10]['start_ticks'] = 999
        with patch('os.pidfd_open', return_value=42, create=True), patch('os.close') as close, \
                patch('signal.pidfd_send_signal', create=True) as send:
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                self.guard.signal((10, 100), signal.SIGTERM)
            send.assert_not_called()
            close.assert_called_once_with(42)

    def test_existing_tree_resumes_without_launch(self):
        def resume(identity, sig):
            self.rows[identity[0]]['state'] = 'R'
        self.guard.signal = Mock(side_effect=resume)
        with patch('subprocess.Popen') as launch, patch.object(signal, 'SIGCONT', 18, create=True):
            self.guard.resume()
            self.guard.resume()
            launch.assert_not_called()
            self.assertFalse(self.guard.paused)
            self.assertEqual(self.guard.signal.call_args_list[0].args[0], (11, 110))

    def test_missing_tree_restarts_once_with_exact_argv(self):
        self.rows.clear()
        child = Mock(pid=20)
        child.poll.return_value = None
        def launch(*args, **kwargs):
            self.rows[20] = dict(pid=20, start_ticks=200, ppid=1, state='R', argv=self.guard.argv)
            return child
        with patch('subprocess.Popen', side_effect=launch) as popen, \
                patch('time.monotonic', side_effect=[0, 2, 3]), \
                patch.object(signal, 'SIGCONT', 18, create=True):
            self.guard.resume()
            self.guard.signal = Mock()
            self.guard.resume()
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.args, (self.guard.argv,))
            self.assertNotIn('shell', popen.call_args.kwargs)
            self.assertEqual(self.guard.root, (20, 200))

    def test_restart_that_immediately_exits_is_failure(self):
        self.rows.clear()
        child = Mock(pid=20)
        child.poll.return_value = 1
        with patch('subprocess.Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'restart failed'):
                self.guard.resume()
        self.assertTrue(self.guard.paused)

    def test_shutdown_waits_until_model_gpu_processes_are_gone(self):
        self.guard.gpu_pids = Mock(side_effect=[{11, 99}, {11}])
        self.guard.resume = Mock()
        with patch('time.sleep'):
            self.guard.restore_after_shutdown()
        self.assertEqual(self.guard.gpu_pids.call_count, 2)
        self.guard.resume.assert_called_once()

    def test_shutdown_unknown_gpu_process_leaves_occupier_paused(self):
        self.guard.gpu_pids = Mock(return_value={11, 99})
        self.guard.resume = Mock()
        with patch('time.monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(RuntimeError, 'left paused'):
                self.guard.restore_after_shutdown()
        self.guard.resume.assert_not_called()


if __name__ == '__main__':
    unittest.main()
