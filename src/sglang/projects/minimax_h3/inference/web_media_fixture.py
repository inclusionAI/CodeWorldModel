import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

from .serving import Worker
from .serving_http import Ledger, SessionMailbox, ServingHTTPServer
from .web_serving import FixtureBackend


class MediaFixtureBackend(FixtureBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.directory = tempfile.TemporaryDirectory(prefix='h3-media-fixture-')
        self.ledger = Ledger(Path(self.directory.name) / 'server.jsonl')
        self.mailbox = SessionMailbox(self.ledger, timeout=15)
        self.worker = Worker(self, SimpleNamespace(rank=0, broadcast=lambda value: value,
            rank0=lambda function: function(), barrier=lambda: None), self.controller, self.mailbox)

    def execute(self, op, metadata, pixels):
        if op != 'chunk':
            return b'', {}
        from PIL import Image
        spec = self.spec
        stride = spec['proxy_width'] * spec['proxy_height'] * 3
        result = b''.join(Image.frombytes('RGB', (spec['proxy_width'], spec['proxy_height']),
            pixels[i:i+stride]).resize((spec['width'], spec['height'])).tobytes()
            for i in range(0, len(pixels), stride))
        return result, {}

    def close_session(self):
        self.pending = b''

    async def start(self):
        self.server = ServingHTTPServer(('127.0.0.1', 0), self.mailbox, self.health, timeout=15)
        if not self.health.get('media'):
            self.server.server_close()
            self.ledger.close()
            self.directory.cleanup()
            raise RuntimeError('CPU media fixture requires PyAV with libx264')
        self.worker_thread = threading.Thread(target=self.worker.serve, daemon=True)
        self.http_thread = threading.Thread(target=self.server.serve_forever, kwargs=dict(poll_interval=.1), daemon=True)
        self.worker_thread.start()
        self.http_thread.start()
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    async def close(self):
        def stop():
            self.mailbox.stop()
            self.worker_thread.join(20)
            if self.worker_thread.is_alive():
                raise RuntimeError('CPU media fixture worker did not stop')
            self.server.shutdown()
            self.server.server_close()
            self.http_thread.join(3)
            self.ledger.close()
            self.directory.cleanup()
        await asyncio.to_thread(stop)
