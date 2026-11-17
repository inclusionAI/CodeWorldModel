"""Trusted, local Linux occupier control for one resident serving process."""
import json
import ctypes
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def _pidfd_syscall(number, *args):
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise ValueError('Python without pidfd wrappers requires Linux x86_64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


class GpuOccupier:
    def __init__(self, path):
        config = json.loads(Path(path).read_text())
        if set(config) != {'workflow_id', 'argv', 'pid', 'start_ticks', 'log_path'}:
            raise ValueError('occupier config requires workflow_id, argv, pid, start_ticks, log_path')
        if not isinstance(config['workflow_id'], str) or os.environ.get('WORKFLOW_ID') != config['workflow_id']:
            raise ValueError('occupier workflow_id does not match this machine')
        if sys.platform != 'linux':
            raise ValueError('occupier control requires Linux pidfd support')
        self.argv = config['argv']
        if (not isinstance(self.argv, list) or len(self.argv) < 2
                or any(not isinstance(arg, str) or not arg or '\0' in arg for arg in self.argv)
                or not all(Path(arg).is_absolute() and Path(arg).is_file() for arg in self.argv[:2])):
            raise ValueError('occupier argv requires absolute executable and script paths')
        if any(type(config[key]) is not int or config[key] <= 0 for key in ('pid', 'start_ticks')):
            raise ValueError('occupier pid and start_ticks must be positive integers')
        self.log_path = Path(config['log_path'])
        if not self.log_path.is_absolute() or not self.log_path.parent.is_dir():
            raise ValueError('occupier log_path must have an existing absolute parent directory')
        self.root = (config['pid'], config['start_ticks'])
        self.members = {self.root}
        self.paused = False
        self.child = None
        row = self.process(self.root[0])
        if row is None or row['start_ticks'] != self.root[1] or row['argv'] != self.argv:
            raise ValueError('initial occupier identity does not match trusted config')
        self.tree()
        self.baseline_gpu_pids = self.gpu_pids()

    @staticmethod
    def process(pid):
        try:
            directory = Path('/proc') / str(pid)
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = (directory / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            return dict(pid=pid, start_ticks=int(stat[19]), ppid=int(stat[1]), state=stat[0], argv=argv)
        except (FileNotFoundError, ProcessLookupError):
            return None

    @classmethod
    def processes(cls):
        return {int(path.name): row for path in Path('/proc').iterdir() if path.name.isdigit()
                if (row := cls.process(int(path.name))) is not None and row['state'] != 'Z'}

    @staticmethod
    def gpu_pids():
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=10)
        return {int(line) for line in result.stdout.splitlines() if line.strip()}

    def signal(self, identity, sig):
        pid, ticks = identity
        try:
            fd = pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            row = self.process(pid)
            if row is None or row['start_ticks'] != ticks:
                raise RuntimeError(f'occupier PID identity changed: {pid}')
            pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)

    def tree(self):
        rows = self.processes()
        roots = [row for row in rows.values() if row['argv'] == self.argv]
        if len(roots) > 1:
            raise RuntimeError('multiple matching occupiers; refusing to control them')
        live = {identity for identity in self.members if identity[0] in rows
                and rows[identity[0]]['start_ticks'] == identity[1]}
        if roots:
            root = (roots[0]['pid'], roots[0]['start_ticks'])
            if root != self.root:
                raise RuntimeError('occupier root identity changed; update trusted config')
            pids = {root[0]}
            while True:
                children = {row['pid'] for row in rows.values() if row['ppid'] in pids}
                if children <= pids:
                    break
                pids.update(children)
            live.update((pid, rows[pid]['start_ticks']) for pid in pids)
        elif live:
            raise RuntimeError('occupier root disappeared but tracked children remain')
        self.members.update(live)
        return live

    def pause(self):
        # Stop the parent first so it cannot create an untracked GPU child.
        live = self.tree()
        if self.root in live:
            self.signal(self.root, signal.SIGSTOP)
        self.paused = True
        deadline = time.monotonic() + 30
        while True:
            live = self.tree()
            for identity in live:
                self.signal(identity, signal.SIGSTOP)
            if all((row := self.process(pid)) is None or row['state'] in ('T', 't', 'Z') for pid, _ in live):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('occupier did not stop')
            time.sleep(.05)
        idle_samples = 0
        while idle_samples < 2:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True, timeout=10)
            samples = result.stdout.splitlines()
            if not samples:
                raise RuntimeError('nvidia-smi returned no GPU utilization samples')
            idle_samples = idle_samples + 1 if all(int(line) == 0 for line in samples) else 0
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU work has not drained after occupier pause')
            time.sleep(.25)

    def resume(self):
        live = self.tree()
        if live:
            for identity in sorted(live, key=lambda item: item == self.root):
                self.signal(identity, signal.SIGCONT)
            deadline = time.monotonic() + 5
            while any((row := self.process(pid)) is not None and row['state'] in ('T', 't') for pid, _ in live):
                if time.monotonic() >= deadline:
                    raise TimeoutError('occupier did not resume')
                time.sleep(.05)
        else:
            with self.log_path.open('ab') as log:
                self.child = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + 1
            while True:
                row = self.process(self.child.pid)
                if self.child.poll() is not None or row is None or row['state'] == 'Z':
                    raise RuntimeError('occupier restart failed; inspect configured log_path')
                if time.monotonic() >= deadline:
                    if row['argv'] != self.argv:
                        raise RuntimeError('restarted occupier argv does not match configured command')
                    break
                time.sleep(.05)
            self.root = (row['pid'], row['start_ticks'])
            self.members = {self.root}
            self.tree()
        self.paused = False

    def restore_after_shutdown(self):
        deadline = time.monotonic() + 30
        while True:
            live = self.tree()
            if not self.gpu_pids() - self.baseline_gpu_pids - {pid for pid, _ in live}:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('new GPU processes remain; occupier left paused after service failure')
            time.sleep(.25)
        self.resume()
