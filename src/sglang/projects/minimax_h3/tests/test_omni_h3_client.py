import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from aiohttp import WSMsgType, web
import av
import msgpack
import numpy as np
from PIL import Image
import pytest

from inference.omni_h3_assets import prepare_native_bundle
from inference import omni_h3_client
from inference.omni_h3_client import infer_files
from test_omni_h3_assets import assets


ROOT = Path(__file__).resolve().parents[1]


def video(path, count=39, *, rate=24, width=16, timestamps=None):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, 16, 'yuv420p'
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((16, width, 3), index, np.uint8), 'rgb24')
            if timestamps is not None:
                frame.pts = timestamps[index]
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def files(tmp_path, assets):
    artifact, bundle, options = assets
    bundle['encoders']['recipe'].update(height=16, width=16, proxy_height=16, proxy_width=16)
    artifact.write_text(json.dumps(bundle))
    image, proxy, prompt = (tmp_path / name for name in ('first.png', 'proxy.mp4', 'prompt.txt'))
    Image.new('RGB', (16, 16), 'red').save(image)
    video(proxy)
    prompt.write_text('Arbitrary static instruction.\n', encoding='utf-8')
    return dict(model_path=tmp_path/'bundle', artifact=artifact, first_image=image,
        proxy_video=proxy, prompt_file=prompt, output_dir=tmp_path/'result', seed=42), options


class FakeOmni:
    def __init__(self, mode='ok'):
        self.mode = mode
        self.opens, self.appends = [], []
        self.closed = self.unfinished = self.max_unfinished = 0
        self.two_appends = asyncio.Event()
        self.release = asyncio.Event()
        if mode != 'gated':
            self.release.set()

    async def handle(self, request):
        socket = web.WebSocketResponse(max_msg_size=1 << 20)
        await socket.prepare(request)
        identity = dict(id='file-session', open_index=1)
        sequence = 0
        buffered = []
        queue = asyncio.Queue()

        async def send(value):
            await socket.send_bytes(msgpack.packb(value, use_bin_type=True))

        async def output(seq, payload=None, *, kind='data', eos=False):
            nonlocal sequence
            chunk = dict(session_identity=identity, seq=sequence, input_seq=seq, kind=kind,
                modality='video', format='h3.rgb24', t_start_ms=0., duration_ms=0.,
                payload=payload, eos=eos, turn=None)
            if self.mode == 'bad_identity':
                chunk['session_identity'] = dict(id='stale', open_index=1)
            if self.mode == 'bad_sequence':
                chunk['seq'] += 1
            await send(dict(op='output', chunk=chunk))
            sequence += 1

        async def process():
            while True:
                chunk = await queue.get()
                seq = chunk['seq']
                await self.release.wait()
                buffered.extend(chunk['payload']['frames'])
                if seq:
                    while buffered:
                        stamps, buffered[:] = deepcopy(buffered[:8]), buffered[8:]
                        if self.mode == 'bad_source':
                            stamps[0]['source_frame_id'] = 'wrong'
                        rgb = b''.join(bytes([int(stamp['source_frame_id']) if
                            stamp['source_frame_id'].isdigit() else 0])*16*16*3 for stamp in stamps)
                        if self.mode == 'bad_pixels':
                            rgb = rgb[:-1]
                        await output(seq, dict(width=16, height=16, frames=stamps, rgb=rgb))
                    if self.mode == 'early_eos':
                        await output(seq, eos=True)
                if chunk['eos'] and self.mode != 'omit_eos':
                    await output(seq, eos=True)
                    if self.mode == 'after_eos':
                        await output(seq, dict(width=16, height=16, frames=[], rgb=b''))
                self.unfinished -= 1
                await output(seq, kind='input_done', eos=not chunk['eos'] if self.mode == 'bad_done' else chunk['eos'])
                if self.mode == 'done_before_accepted':
                    await send(dict(op='accepted', seq=seq))

        worker = asyncio.create_task(process())
        try:
            async for message in socket:
                if message.type != WSMsgType.BINARY:
                    break
                value = msgpack.unpackb(message.data, raw=False)
                if value['op'] == 'open':
                    self.opens.append(value['params'])
                    await send(dict(op='opened', session=identity))
                elif value['op'] == 'append':
                    chunk = value['chunk']
                    self.appends.append(chunk)
                    self.unfinished += 1
                    self.max_unfinished = max(self.max_unfinished, self.unfinished)
                    if len(self.appends) == 2:
                        self.two_appends.set()
                    if self.mode == 'reject':
                        await send(dict(op='rejected', seq=chunk['seq'], reason='test rejection'))
                    elif self.mode == 'disconnect':
                        await socket.close()
                    elif self.mode == 'server_error':
                        await send(dict(op='error', message='test failure'))
                    elif self.mode != 'timeout':
                        if self.mode != 'done_before_accepted':
                            await send(dict(op='accepted', seq=chunk['seq']))
                        if self.mode == 'duplicate_accepted':
                            await send(dict(op='accepted', seq=chunk['seq']))
                        await queue.put(chunk)
                elif value['op'] == 'close':
                    self.closed += 1
                    if self.mode == 'unconfirmed_close':
                        await socket.close()
                    elif self.mode == 'cleanup_timeout':
                        await asyncio.sleep(1)
                    else:
                        await send(dict(op='closed'))
                    break
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return socket


@asynccontextmanager
async def server(fake):
    app = web.Application()
    app.router.add_get('/v1/session', fake.handle)
    runner = web.AppRunner(app, shutdown_timeout=.1)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    try:
        yield f'ws://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1/session'
    finally:
        await runner.cleanup()


@pytest.mark.parametrize('name', ['longlive', 'continuous5', 'legacy'])
def test_file_inference_preserves_provenance_and_publishes_only_after_close(files, name):
    args, options = files
    manifest = prepare_native_bundle(args['artifact'], args['model_path'],
        recipe='continuous5' if name == 'continuous5' else 'longlive', **options)
    if name == 'legacy':
        del manifest['recipe_selection']
        (args['model_path']/'native-runtime.json').write_text(json.dumps(manifest))
    original = args['artifact'].read_bytes()

    async def run():
        fake = FakeOmni('done_before_accepted')
        async with server(fake) as uri:
            receipt = await infer_files(**args, uri=uri, timeout=2)
        assert fake.closed == 1 and fake.max_unfinished <= 2
        assert fake.opens[0]['prompt'] == args['prompt_file'].read_bytes().decode('utf-8')
        assert [len(chunk['payload']['frames']) for chunk in fake.appends] == [17, 17, 5]
        assert receipt['native_eos'] and receipt['native_close_confirmed']
        assert receipt['accepted_appends'] == receipt['completed_appends'] == 3
        assert receipt['input_frames'] == receipt['output_frames'] == 39
        assert receipt['effective_recipe'] == manifest['recipe']
        assert receipt['source_recipe']['segment_boundaries'] == [45, 70]
        assert receipt['bundle']['provenance'] == 'local_verified_remote_unverified'
        with av.open(str(args['output_dir']/'video.mp4')) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 39
        assert int(frames[0].to_ndarray(format='rgb24').mean()) <= 1
        assert abs(int(frames[-1].to_ndarray(format='rgb24').mean()) - 38) <= 2
        assert json.loads((args['output_dir']/'inference.json').read_text()) == receipt
    asyncio.run(run())
    assert args['artifact'].read_bytes() == original


def test_input_lead_stays_at_two_while_output_is_blocked(files):
    args, options = files
    video(args['proxy_video'], 90)
    prepare_native_bundle(args['artifact'], args['model_path'], **options)

    async def run():
        fake = FakeOmni('gated')
        async with server(fake) as uri:
            task = asyncio.create_task(infer_files(**args, uri=uri, timeout=2))
            await asyncio.wait_for(fake.two_appends.wait(), 2)
            await asyncio.sleep(.05)
            assert len(fake.appends) == 2 and not task.done()
            assert not (args['output_dir']/'video.mp4').exists()
            fake.release.set()
            assert (await task)['output_frames'] == 90
        assert fake.max_unfinished == 2
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['reject', 'disconnect', 'server_error', 'bad_identity',
    'bad_sequence', 'bad_source', 'duplicate_accepted', 'omit_eos', 'timeout', 'unconfirmed_close',
    'cleanup_timeout', 'bad_pixels', 'bad_done', 'early_eos', 'after_eos'])
def test_wire_failures_never_publish_success(files, mode):
    args, options = files
    prepare_native_bundle(args['artifact'], args['model_path'], **options)

    async def run():
        fake = FakeOmni(mode)
        async with server(fake) as uri:
            with pytest.raises((ExceptionGroup, RuntimeError, TimeoutError)):
                await infer_files(**args, uri=uri, timeout=.15)
        receipt = json.loads((args['output_dir']/'inference.json').read_text())
        assert receipt['status'] == 'failed' and receipt['error']
        assert not (args['output_dir']/'video.mp4').exists()
        if mode not in ('disconnect', 'unconfirmed_close', 'cleanup_timeout'):
            assert receipt['native_close_confirmed']
        else:
            assert not receipt['native_close_confirmed']
    asyncio.run(run())


def test_writer_failure_confirms_cleanup_and_keeps_output_partial(files, monkeypatch):
    args, options = files
    prepare_native_bundle(args['artifact'], args['model_path'], **options)
    def fail_write(self, rgb):
        raise OSError('test output disk failure')
    monkeypatch.setattr(omni_h3_client._VideoWriter, 'write', fail_write)

    async def run():
        fake = FakeOmni()
        async with server(fake) as uri:
            with pytest.raises(ExceptionGroup):
                await infer_files(**args, uri=uri, timeout=1)
        receipt = json.loads((args['output_dir']/'inference.json').read_text())
        assert receipt['status'] == 'failed' and 'disk failure' in receipt['error']
        assert receipt['native_close_confirmed'] and fake.closed == 1
        assert not (args['output_dir']/'video.mp4').exists()
    asyncio.run(run())


def test_cancellation_confirms_close_before_returning(files):
    args, options = files
    prepare_native_bundle(args['artifact'], args['model_path'], **options)

    async def run():
        fake = FakeOmni('gated')
        async with server(fake) as uri:
            task = asyncio.create_task(infer_files(**args, uri=uri, timeout=1))
            await asyncio.wait_for(fake.two_appends.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        receipt = json.loads((args['output_dir']/'inference.json').read_text())
        assert receipt['status'] == 'failed' and receipt['native_close_confirmed']
        assert fake.closed == 1 and not (args['output_dir']/'video.mp4').exists()
    asyncio.run(run())


@pytest.mark.parametrize('timestamps, accepted', [(list(range(39)), True),
    (list(range(20)) + list(range(21, 40)), False), (list(range(20)) + list(range(19, 38)), False)])
def test_millisecond_container_cadence_accepts_constant_rate_only(tmp_path, timestamps, accepted):
    path = tmp_path / 'proxy.mkv'
    video(path, timestamps=timestamps)
    frames = omni_h3_client.proxy_frames(path, dict(proxy_width=16, proxy_height=16))
    if accepted:
        assert sum(1 for _ in frames) == 39
    else:
        with pytest.raises(ValueError, match='cadence'):
            sum(1 for _ in frames)


@pytest.mark.parametrize('problem', ['short', 'remainder', 'fps', 'canvas', 'first_image',
    'prompt', 'source_changed', 'existing'])
def test_invalid_files_fail_before_native_open(files, problem):
    args, options = files
    prepare_native_bundle(args['artifact'], args['model_path'], **options)
    if problem == 'short':
        video(args['proxy_video'], 22)
    elif problem == 'remainder':
        video(args['proxy_video'], 40)
    elif problem == 'fps':
        video(args['proxy_video'], rate=25)
    elif problem == 'canvas':
        video(args['proxy_video'], width=32)
    elif problem == 'first_image':
        Image.new('RGBA', (16, 16)).save(args['first_image'])
    elif problem == 'prompt':
        args['prompt_file'].write_text('a' * 16385)
    elif problem == 'source_changed':
        args['artifact'].write_text('{}')
    else:
        args['output_dir'].mkdir()
        (args['output_dir']/'keep.txt').write_text('keep')

    async def run():
        fake = FakeOmni()
        async with server(fake) as uri:
            with pytest.raises((ValueError, FileExistsError)):
                await infer_files(**args, uri=uri)
        assert not fake.opens
        assert not (args['output_dir']/'video.mp4').exists()
        if problem == 'existing':
            assert (args['output_dir']/'keep.txt').read_text() == 'keep'
        else:
            assert not args['output_dir'].exists()
    asyncio.run(run())


def test_public_cli_runs_against_loopback_and_help_imports_no_gpu_runtime(files):
    args, options = files
    prepare_command = [sys.executable, 'tools/launch.py', 'projects.minimax_h3.tools.prepare_omni_h3',
        '--artifact', str(args['artifact']), '--output', str(args['model_path']),
        '--recipe', 'continuous5', '--deployment-profile', 'reference',
        '--decoder-config', str(options['decoder_config']),
        '--decoder-weights', str(options['decoder_weights']), '--state-reservation-gib', '1',
        '--execution-config', str(options['execution_config'])]
    result = subprocess.run(prepare_command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    module = 'projects.minimax_h3.tools.infer_omni_h3'
    help_script = """
import runpy, sys
sys.argv = ['tools/launch.py', 'projects.minimax_h3.tools.infer_omni_h3', '--help']
try:
    runpy.run_path('tools/launch.py', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not {'torch', 'sglang', 'sglang_omni'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, '-c', help_script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--proxy-video' in result.stdout

    async def run():
        fake = FakeOmni()
        async with server(fake) as uri:
            command = [sys.executable, 'tools/launch.py', module, '--uri', uri]
            for key, value in args.items():
                command.extend(['--'+key.replace('_', '-'), str(value)])
            process = await asyncio.create_subprocess_exec(*command, cwd=ROOT,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, stderr.decode()
            assert '39 frames, complete' in stdout.decode()
        with av.open(str(args['output_dir']/'video.mp4')) as container:
            assert sum(1 for _ in container.decode(video=0)) == 39
        receipt = json.loads((args['output_dir']/'inference.json').read_text())
        assert receipt['recipe_name'] == 'continuous5' and receipt['native_close_confirmed']
    asyncio.run(run())
