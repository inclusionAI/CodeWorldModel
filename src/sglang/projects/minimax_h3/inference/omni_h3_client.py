"""Bounded file inference over Omni's public binary-msgpack session endpoint."""
import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from urllib.parse import urlsplit

from .omni_h3_assets import digest, native_bundle_artifact, read_native_bundle
from .streaming_text import MAX_PROMPT_BYTES, validate_prompt

FPS = 24


# One session step may wait for cold decoder compilation (minutes with autotune).
STEP_TIMEOUT = 900.


def frame_stamp(index):
    start, end = index * 1000. / FPS, (index + 1) * 1000. / FPS
    return dict(source_frame_id=str(index), pts_ms=start, duration_ms=end-start)


def proxy_frames(path, canvas):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('proxy video must contain exactly one video stream')
        stream = container.streams.video[0]
        if stream.average_rate != Fraction(FPS):
            raise ValueError('proxy video must be exactly 24 FPS')
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            if (frame.width, frame.height) != (canvas['proxy_width'], canvas['proxy_height']):
                raise ValueError('proxy video geometry differs from the bundle canvas')
            if frame.pts is None or frame.time_base is None:
                raise ValueError('proxy video requires presentation timestamps')
            timestamp = frame.pts * frame.time_base
            if origin is None:
                origin = timestamp
            # Half-frame tolerance accepts millisecond container time bases (MKV) yet still catches drops and repeats.
            if abs(timestamp - origin - Fraction(index, FPS)) >= Fraction(1, 2 * FPS):
                raise ValueError('proxy video timestamps must follow a constant 24 FPS cadence')
            yield frame


class _VideoWriter:
    def __init__(self, path, canvas):
        import av
        self.av = av
        self.container = av.open(str(path), mode='w')
        self.stream = self.container.add_stream('libx264', rate=FPS)
        self.stream.width, self.stream.height = canvas['width'], canvas['height']
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = dict(crf='18', preset='fast')
        self.count = 0

    def write(self, rgb):
        import numpy as np
        stride = self.stream.width * self.stream.height * 3
        for offset in range(0, len(rgb), stride):
            array = np.frombuffer(rgb, dtype=np.uint8, count=stride, offset=offset).reshape(
                self.stream.height, self.stream.width, 3)
            frame = self.av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts, frame.time_base = self.count, Fraction(1, FPS)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.count += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


@dataclass
class _Pending:
    eos: bool
    accepted: bool = False
    done: bool = False


async def _session(uri, params, proxy, canvas, count, writer, receipt, timeout):
    import msgpack
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    pending = {}
    slots = asyncio.Semaphore(2)
    sent = received = wire_seq = 0
    native_eos = False
    close_attempted = False

    async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=timeout)) as http:
        async with asyncio.timeout(timeout):
            socket = await http.ws_connect(uri, compress=0, heartbeat=None,
                max_msg_size=64 * canvas['width'] * canvas['height'] * 3 + (64 << 10))

        async def send(value):
            async with asyncio.timeout(timeout):
                await socket.send_bytes(msgpack.packb(value, use_bin_type=True))

        async def read():
            message = await socket.receive(timeout=timeout)
            if message.type != WSMsgType.BINARY:
                raise RuntimeError('native session disconnected before confirmed close')
            answer = msgpack.unpackb(message.data, raw=False)
            if not isinstance(answer, dict):
                raise RuntimeError('native session frame must be a map')
            return answer

        async def close(*, failed=False):
            nonlocal close_attempted
            close_attempted = True
            async with asyncio.timeout(timeout):
                await send(dict(op='close'))
                while True:
                    answer = await read()
                    if answer.get('op') == 'closed':
                        receipt['native_close_confirmed'] = True
                        return
                    if not failed:
                        raise RuntimeError(f'native close failed: {answer.get("message", answer.get("op"))}')

        async def produce():
            nonlocal sent
            frames = proxy_frames(proxy, canvas)
            try:
                for seq in range((count - 5) // 17 + 1):
                    async with asyncio.timeout(timeout):
                        await slots.acquire()
                    size = min(17, count - sent)
                    stamps = [frame_stamp(i) for i in range(sent, sent + size)]
                    rgb = b''.join(next(frames).to_ndarray(format='rgb24').tobytes() for _ in stamps)
                    eos = sent + size == count
                    pending[seq] = _Pending(eos)
                    sent += size
                    await send(dict(op='append', chunk=dict(modality='video', format='h3.rgb24',
                        seq=seq, t_start_ms=stamps[0]['pts_ms'],
                        duration_ms=stamps[-1]['pts_ms']+stamps[-1]['duration_ms']-stamps[0]['pts_ms'],
                        payload=dict(width=canvas['proxy_width'], height=canvas['proxy_height'],
                            frames=stamps, rgb=rgb), eos=eos, turn=None)))
                if next(frames, None) is not None:
                    raise ValueError('proxy video changed after validation')
            finally:
                frames.close()

        async def consume(identity):
            nonlocal received, wire_seq, native_eos
            while True:
                answer = await read()
                op = answer.get('op')
                if op in ('accepted', 'rejected'):
                    seq = answer.get('seq')
                    item = pending.get(seq) if type(seq) is int else None
                    if item is None or item.accepted:
                        raise RuntimeError('native acknowledgment has unknown or duplicate input')
                    if op == 'rejected':
                        raise RuntimeError(f'native rejected input {seq}: {answer.get("reason")}')
                    item.accepted = True
                    receipt['accepted_appends'] += 1
                elif op == 'output':
                    chunk = answer.get('chunk')
                    if (not isinstance(chunk, dict) or chunk.get('session_identity') != identity
                            or type(chunk.get('seq')) is not int or chunk['seq'] != wire_seq):
                        raise RuntimeError('native output has stale identity or sequence')
                    wire_seq += 1
                    seq = chunk.get('input_seq')
                    item = pending.get(seq) if type(seq) is int else None
                    if item is None or item.done:
                        raise RuntimeError('native output has unknown or completed input')
                    if chunk.get('kind') == 'input_done':
                        if chunk.get('payload') is not None or chunk.get('eos') is not item.eos:
                            raise RuntimeError('invalid native input_done')
                        item.done = True
                        receipt['completed_appends'] += 1
                    elif chunk.get('kind') == 'data':
                        if chunk.get('modality') != 'video' or chunk.get('format') != 'h3.rgb24':
                            raise RuntimeError('native output must be H3 RGB video')
                        if chunk.get('eos') is True:
                            if native_eos or not item.eos or chunk.get('payload') is not None:
                                raise RuntimeError('invalid native output EOS')
                            native_eos = True
                        elif chunk.get('eos') is False:
                            packet = chunk.get('payload')
                            if native_eos or not isinstance(packet, dict):
                                raise RuntimeError('invalid native RGB packet or RGB after EOS')
                            stamps, rgb = packet.get('frames'), packet.get('rgb')
                            if (packet.get('width') != canvas['width'] or packet.get('height') != canvas['height']
                                    or not isinstance(stamps, list) or not 1 <= len(stamps) <= 64
                                    or not isinstance(rgb, bytes)
                                    or len(rgb) != len(stamps)*canvas['width']*canvas['height']*3
                                    or received + len(stamps) > sent):
                                raise RuntimeError('invalid native RGB geometry or frame count')
                            if stamps != [frame_stamp(i) for i in range(received, received+len(stamps))]:
                                raise RuntimeError('native frame provenance differs from submitted source')
                            writer.write(rgb)
                            received += len(stamps)
                            receipt['output_frames'] = received
                        else:
                            raise RuntimeError('native output requires boolean EOS')
                    else:
                        raise RuntimeError('unknown native output kind')
                else:
                    raise RuntimeError(f'native session failed: {answer.get("message", op)}')
                if item.accepted and item.done:
                    del pending[seq]
                    slots.release()
                if native_eos and not pending:
                    if sent != count or received != count:
                        raise RuntimeError('native EOS left source frames unaccounted')
                    receipt['native_eos'] = True
                    return

        try:
            await send(dict(op='open', params=params))
            answer = await read()
            identity = answer.get('session')
            if answer.get('op') != 'opened' or not isinstance(identity, dict):
                raise RuntimeError('native open did not return a valid session identity')
            receipt['session'] = identity
            async with asyncio.TaskGroup() as group:
                group.create_task(produce())
                await consume(identity)
            await close()
        except BaseException:
            if not close_attempted:
                try:
                    await close(failed=True)
                except Exception as exc:
                    receipt['cleanup_error'] = str(exc) or type(exc).__name__
            raise
        finally:
            async with asyncio.timeout(timeout):
                await socket.close()


async def infer_files(*, model_path, artifact, uri, first_image, proxy_video, prompt_file,
                      output_dir, seed=42, timeout=STEP_TIMEOUT):
    """Validate complete inputs, run one native stream, and publish only confirmed output.

    ``timeout`` bounds each session step, not the whole run. The first chunks of a
    freshly started service include decoder compilation, which takes minutes.
    """
    from PIL import Image

    if urlsplit(uri).scheme not in ('ws', 'wss') or not urlsplit(uri).netloc:
        raise ValueError('uri must be a ws:// or wss:// session endpoint')
    if not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if not 0 < timeout < float('inf'):
        raise ValueError('timeout must be finite and positive')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = read_native_bundle(model_path)
    artifact_id, source, effective = native_bundle_artifact(manifest, artifact)
    canvas = manifest['canvas']
    identities = {name: dict(path=str(Path(path).resolve()), sha256=digest(path)) for name, path in
        dict(first_image=first_image, proxy_video=proxy_video, prompt_file=prompt_file).items()}
    with Path(prompt_file).open('rb') as handle:
        prompt = validate_prompt(handle.read(MAX_PROMPT_BYTES+1).decode('utf-8'))
    with Image.open(first_image) as image:
        if image.mode != 'RGB' or image.size != (canvas['width'], canvas['height']):
            raise ValueError('first image must be RGB and match the bundle output canvas')
        pixels = image.tobytes()
    count = sum(1 for _ in proxy_frames(proxy_video, canvas))
    if count < 39 or (count-5) % 17:
        raise ValueError('proxy video requires 17*N+5 frames and at least 39 frames for bootstrap9')
    params = dict(seed=seed, prompt=prompt, first_image=dict(width=canvas['width'], height=canvas['height'],
        rgb=pixels, frames=[dict(source_frame_id='first_image', pts_ms=0., duration_ms=1000./FPS)]))
    receipt = dict(schema='h3_omni_file_inference/v1', status='failed', seed=seed,
        bundle=dict(path=str(Path(model_path).resolve()),
            manifest_sha256=digest(Path(model_path)/'native-runtime.json'),
            provenance='local_verified_remote_unverified'),
        source_artifact=dict(path=str(Path(artifact).resolve()), sha256=artifact_id),
        recipe_name=manifest.get('recipe_selection', {}).get('name', 'longlive'),
        source_recipe=source['recipe'], effective_recipe=effective, inputs=identities,
        canvas=canvas, fps=FPS, input_frames=count, output_frames=0,
        accepted_appends=0, completed_appends=0, native_eos=False, native_close_confirmed=False)
    output.mkdir(parents=True, exist_ok=False)
    writer = None
    partial, final = output/'video.partial.mp4', output/'video.mp4'
    try:
        writer = _VideoWriter(partial, canvas)
        await _session(uri, params, proxy_video, canvas, count, writer, receipt, timeout)
        writer.close()
        writer = None
        receipt.update(status='complete', video=dict(path='video.mp4', sha256=digest(partial)))
        partial.rename(final)
        (output/'inference.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        return receipt
    except BaseException as exc:
        while isinstance(exc, BaseExceptionGroup):
            exc = exc.exceptions[0]
        receipt.update(status='failed', error=str(exc) or type(exc).__name__)
        receipt.pop('video', None)
        if final.exists():
            final.rename(partial)
        if writer is not None:
            try:
                writer.close()
            except Exception as close_error:
                receipt['writer_error'] = str(close_error)
        (output/'inference.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        raise
