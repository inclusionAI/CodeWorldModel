"""Bounded CPU-only HTTP rendezvous. No model or collective runs on a handler."""
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
import http.client
import json
from socketserver import ThreadingMixIn
import struct
import threading
import time
from urllib.parse import urlsplit

from .serving import Reply, Request, SessionError

MAX_HEADER = 64 << 10
MAX_BODY = 128 << 20


def pack(metadata, pixels=b''):
    header = json.dumps(metadata, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8')
    if len(header) > MAX_HEADER:
        raise SessionError('metadata exceeds 64 KiB')
    return struct.pack('!I', len(header)) + header + pixels


def unpack(body):
    if len(body) < 4:
        raise SessionError('missing metadata length')
    size, = struct.unpack('!I', body[:4])
    if not 2 <= size <= MAX_HEADER or size + 4 > len(body):
        raise SessionError('invalid metadata length')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    try:
        def invalid_constant(value):
            raise ValueError('nonfinite JSON number')
        metadata = json.loads(body[4:4+size], object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SessionError('invalid JSON metadata') from exc
    if not isinstance(metadata, dict):
        raise SessionError('metadata must be an object')
    return metadata, body[4+size:]


class Ledger:
    def __init__(self, path):
        self.file = open(path, 'x', encoding='utf-8')
        self.lock = threading.Lock()

    def record(self, event, **fields):
        with self.lock:
            self.file.write(json.dumps(dict(event=event, server_ns=time.perf_counter_ns(), **fields)) + '\n')
            self.file.flush()

    def close(self):
        self.file.close()


@dataclass
class Job:
    request: Request
    internal: bool = False
    admitted_ns: int = field(default_factory=time.perf_counter_ns)
    ready: threading.Event = field(default_factory=threading.Event)
    delivered: threading.Event = field(default_factory=threading.Event)
    released: threading.Event = field(default_factory=threading.Event)
    reply: Reply | None = None
    delivery_ok: bool = False
    media: object | None = None
    reserved: bool = False
    cleaned: bool = False
    waits: dict | None = None


class SessionMailbox:
    def __init__(self, ledger, *, timeout=300):
        self.ledger, self.timeout = ledger, timeout
        self.condition = threading.Condition()
        self.current = self.pending = self.control = None
        self.stopping = False
        self.media = None

    def submit(self, request, *, internal=False, media=None):
        if request.op == 'shutdown' and (request.metadata or request.pixels):
            raise SessionError('shutdown takes no payload')
        control = request.op in ('cancel', 'shutdown')
        with self.condition:
            if self.stopping:
                raise SessionError('worker is shutting down', 503)
            if self.media is not None and self.media.active is not None and media is None:
                if request.op == 'shutdown':
                    media = self.media.active
                    self.media.abort(media, 'worker is shutting down', schedule=False)
                else:
                    raise SessionError('a media session is active or draining', 409)
            if control:
                if self.control is not None:
                    raise SessionError('a control operation is already queued', 409)
            else:
                media_pending = (media is not None and request.op == 'chunk' and self.current is not None
                    and self.current.media is media and self.current.request.op in ('start', 'chunk'))
                if self.current is not None and self.current.delivered.is_set() and not media_pending:
                    self.condition.wait_for(lambda: self.current is None or self.stopping, self.timeout)
                if (self.current is not None and not media_pending) or self.pending is not None or self.control is not None:
                    raise SessionError('one request is already in flight', 409)
            job = Job(request, internal=internal, media=media)
            if media is not None and media.observe_waits and request.op in ('start', 'chunk'):
                job.waits = dict(request_complete_ns=request.received_ns, enqueued_ns=job.admitted_ns)
            if control:
                self.control = job
            else:
                self.pending = job
            if request.op == 'shutdown':
                self.stopping = True
            self.condition.notify_all()
            return job

    def peek_pending_chunk(self):
        with self.condition:
            current, pending = self.current, self.pending
            if (self.stopping or self.control is not None or current is None or pending is None
                    or current.media is None or current.media.cancelled or pending.media is not current.media
                    or current.request.op != 'chunk' or current.request.metadata['eos']
                    or pending.request.op != 'chunk'):
                return None
            return pending.request

    def stop(self):
        with self.condition:
            if self.stopping:
                return
            if self.control is not None:
                self.control.reply = Reply(dict(status='error', error='worker is shutting down'), status=503)
                self.control.ready.set()
            media = self.media.active if self.media is not None else None
            if media is not None:
                self.media.abort(media, 'worker is shutting down', schedule=False)
            self.control = Job(Request('shutdown', {}), internal=True, media=media)
            self.stopping = True
            self.condition.notify_all()

    def take(self, idle_timeout, session_id):
        with self.condition:
            available = self.condition.wait_for(lambda: self.control is not None or self.pending is not None,
                                                timeout=idle_timeout)
            if not available:
                media = self.media.active if self.media is not None else None
                if media is not None:
                    self.media.abort(media, 'session idle timeout', schedule=False)
                job = Job(Request('expire', dict(session_id=session_id, reason='session idle timeout')),
                    internal=True, media=media)
            elif self.control is not None:
                job, self.control = self.control, None
            else:
                job, self.pending = self.pending, None
            self.current = job
            if job.waits is not None:
                job.waits['taken_ns'] = time.perf_counter_ns()
            return job

    def before_prepare(self, job, controller):
        if job.media is not None:
            self.media.reserve(job, controller)

    def prepared(self, job):
        if job.media is not None:
            self.media.admitted(job)

    def session_closed(self):
        if self.current is not None:
            self.current.cleaned = True

    def deliver(self, job, reply):
        self.ledger.record('response_ready', operation=job.request.op,
            session_id=job.request.metadata.get('session_id'), chunk_id=job.request.metadata.get('chunk_id'),
            response=reply.metadata, http_status=reply.status)
        if job.media is not None:
            delivery = self.media.result(job, reply)
            if delivery is not None:
                job.delivery_ok = delivery
                return delivery
        job.reply = reply
        job.ready.set()
        if job.internal:
            return True
        if not job.delivered.wait(self.timeout):
            self.acknowledge(job, False, 'delivery acknowledgement timeout')
        if job.media is not None and not job.delivery_ok:
            self.media.abort(job.media, 'HTTP response delivery failed')
        return job.delivery_ok

    def acknowledge(self, job, success, error=None):
        with self.condition:
            if job.delivered.is_set():
                return
            job.delivery_ok = success
            self.ledger.record('send_end' if success else 'send_error',
                session_id=job.request.metadata.get('session_id'), chunk_id=job.request.metadata.get('chunk_id'),
                operation=job.request.op, error=error)
            job.delivered.set()
            if job.media is not None and not success:
                self.media.abort(job.media, error or 'HTTP response delivery failed')
            self.condition.notify_all()

    def release(self, job):
        with self.condition:
            self.ledger.record('ranks_released', operation=job.request.op,
                session_id=job.request.metadata.get('session_id'), delivery_ok=job.delivery_ok or job.internal)
            self.current = None
            job.released.set()
            if job.cleaned and self.media is not None:
                self.media.worker_closed(job)
            self.condition.notify_all()

    def fail(self, message):
        with self.condition:
            self.stopping = True
            if self.media is not None and self.media.active is not None:
                self.media.abort(self.media.active, message, schedule=False)
            for job in (self.current, self.pending, self.control):
                if job is not None:
                    job.reply = Reply(dict(status='error', error=message), status=503)
                    job.ready.set()
                    job.delivered.set()
            self.condition.notify_all()


class ServingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address, mailbox, health, *, timeout=300, max_body=MAX_BODY):
        self.mailbox, self.health = mailbox, health
        self.io_timeout, self.max_body = timeout, max_body
        self.slots = threading.BoundedSemaphore(4)
        from .serving_media import MediaSessions
        mailbox.media = MediaSessions(mailbox, health)
        self.media = mailbox.media
        super().__init__(address, Handler)

    def server_close(self):
        try:
            session = self.media.active
            if session is not None:
                if self.mailbox.stopping:
                    self.media.abort(session, 'HTTP server is closing', schedule=False)
                    session.encoder_done.wait(self.io_timeout)
                else:
                    self.media.cancel(dict(session_id=session.session_id, reason='HTTP server is closing'))
        finally:
            super().server_close()

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            body = pack(dict(status='error', error='HTTP handler limit reached'))
            try:
                request.settimeout(.1)
                request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: ' +
                    str(len(body)).encode('ascii') + b'\r\nConnection: close\r\n\r\n' + body)
            except OSError:
                pass
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.io_timeout)

    def log_message(self, *args):
        pass

    def respond(self, reply):
        body = pack(reply.metadata, reply.pixels)
        self.send_response(reply.status)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        parts = self.path.strip('/').split('/')
        if len(parts) == 4 and parts[:2] == ['media', 'session'] and parts[3] == 'events':
            self.media_events(parts[2])
            return
        if self.path != '/health':
            self.respond(Reply(dict(status='error', error='unknown route'), status=404))
            return
        self.respond(Reply(dict(self.server.health, status='stopping' if self.server.mailbox.stopping else 'ready')))

    def media_events(self, sid):
        session = None
        complete = False
        try:
            from .serving_media import MAX_RECORD
            session = self.server.media.attach(sid)
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            while True:
                event = self.server.media.next_event(session)
                body = pack(event.metadata, event.pixels)
                if len(body) > MAX_RECORD:
                    raise SessionError('media record exceeds limit', 500)
                self.wfile.write(struct.pack('!I', len(body)) + body)
                self.wfile.flush()
                if event.metadata['op'] in ('complete', 'error'):
                    complete = event.metadata['op'] == 'complete'
                    break
        except (SessionError, OSError, ValueError) as exc:
            if session is None:
                try:
                    self.respond(Reply(dict(status='error', error=str(exc)), status=getattr(exc, 'status', 400)))
                except OSError:
                    pass
        finally:
            if session is not None:
                self.server.media.detach(session, complete)

    def do_POST(self):
        job = None
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise SessionError('exactly one Content-Length is required', 411)
            try:
                size = int(lengths[0])
            except ValueError as exc:
                raise SessionError('invalid Content-Length') from exc
            if not 6 <= size <= self.server.max_body:
                raise SessionError('body exceeds limits', 413)
            body = self.rfile.read(size)
            if len(body) != size:
                raise SessionError('truncated request body')
            metadata, pixels = unpack(body)
            parts = self.path.strip('/').split('/')
            if self.path == '/media/session':
                job = self.server.media.start(metadata, pixels)
            elif len(parts) == 4 and parts[:2] == ['media', 'session'] and parts[3] in ('chunk', 'cancel'):
                if metadata.get('session_id') != parts[2]:
                    raise SessionError('path and metadata session IDs differ')
                if parts[3] == 'cancel':
                    self.respond(self.server.media.cancel(metadata, pixels))
                    return
                job = self.server.media.submit(metadata, pixels)
            elif self.path == '/session':
                op = 'start'
            elif self.path == '/shutdown':
                op = 'shutdown'
            elif len(parts) == 3 and parts[0] == 'session' and parts[2] in ('chunk', 'cancel'):
                op = parts[2]
                if metadata.get('session_id') != parts[1]:
                    raise SessionError('path and metadata session IDs differ')
            else:
                raise SessionError('unknown route', 404)
            if job is None:
                job = self.server.mailbox.submit(Request(op, metadata, pixels))
            if not job.ready.wait(self.server.io_timeout):
                raise SessionError('worker response timeout', 504)
            self.respond(job.reply)
            self.server.mailbox.acknowledge(job, True)
        except (SessionError, OSError, ValueError) as exc:
            if job is not None:
                self.server.mailbox.acknowledge(job, False, str(exc))
                if job.ready.is_set():
                    return
            try:
                self.respond(Reply(dict(status='error', error=str(exc)), status=getattr(exc, 'status', 400)))
            except OSError:
                pass


class RefClient:
    def __init__(self, url, *, timeout=300, max_body=MAX_BODY):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or not parsed.hostname or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError('expected an http://host:port service URL')
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.timeout, self.max_body = timeout, max_body

    def call(self, path, metadata=None, pixels=b'', *, method='POST'):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        send_ns = time.perf_counter_ns()
        try:
            body = None if method == 'GET' else pack(metadata or {}, pixels)
            connection.request(method, path, body=body, headers={'Content-Type': 'application/octet-stream'})
            response = connection.getresponse()
            headers_ns = time.perf_counter_ns()
            size = response.getheader('Content-Length')
            if size is None or not 6 <= int(size) <= self.max_body:
                raise SessionError('invalid response size', 502)
            raw = response.read(int(size))
            receive_ns = time.perf_counter_ns()
            if len(raw) != int(size):
                raise SessionError('truncated response', 502)
            data, rgb = unpack(raw)
            if response.status != 200:
                raise SessionError(data.get('error', 'request failed'), response.status)
            return Reply(data, rgb), dict(send_ns=send_ns, headers_ns=headers_ns, receive_ns=receive_ns)
        finally:
            connection.close()
