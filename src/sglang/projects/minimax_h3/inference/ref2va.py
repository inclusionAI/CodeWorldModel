"""Shared Ref2VA input/output and generic RAVEN model-construction adapter."""
from pathlib import Path
from dataclasses import asdict
import gc
import json
import subprocess
import time
import traceback

from .settings import backbone_dit_options

def rank0_call(fn):
    """Broadcast either the CPU payload or the rank-zero error to all ranks."""
    import torch
    import torch.distributed as dist
    from common.distributed.ops import get_rank
    payload = [None]
    if get_rank() == 0:
        try:
            payload[0] = dict(ok=True, value=fn())
        except BaseException:
            payload[0] = dict(ok=False, error=traceback.format_exc())
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0, device=torch.device('cpu'))
    if not payload[0]['ok']:
        raise RuntimeError(payload[0]['error'])
    return payload[0]['value']


class VideoWriter:
    def __init__(self, output, height, width):
        self.closed = False
        self.final = Path(output)/'video.mp4'
        self.partial = Path(output)/'video.partial.mp4'
        self.stderr = (Path(output)/'ffmpeg.log').open('xb')
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '24',
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.partial)],
            stdin=subprocess.PIPE, stderr=self.stderr)

    def write(self, frames):
        self.process.stdin.write(frames.numpy().tobytes())

    def close(self, success):
        if self.closed:
            return
        self.closed = True
        try:
            if not self.process.stdin.closed:
                self.process.stdin.close()
        finally:
            try:
                rc = self.process.wait()
            finally:
                self.stderr.close()
        if rc:
            raise RuntimeError(f'ffmpeg exited {rc}; see {self.stderr.name}')
        if success:
            self.partial.rename(self.final)


def make_engine_class():
    from common.engine import BaseEngine
    from common.distributed.ops import get_device, get_rank
    from projects.minimax_h3.data.ref_latent import STATIC_CONDITIONING_MODE
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
    from projects.minimax_h3.tools.ref2va_prepare import iter_video_chunks
    from projects.minimax_h3.modeling.inference_runtime import InferenceCodec, OutputSink, StageProfiler
    from projects.minimax_h3.inference.raw_recipe import ALL_STREAMING_TEXT, RAW_PROTOCOLS
    from projects.minimax_h3.inference.streaming_text import encode_instruction, prompt_source as make_prompt_source
    import numpy as np
    import torch
    from PIL import Image

    class Ref2VAStreamEngine(BaseEngine):
        def __init__(self, config, meta):
            self.encoders = None
            protocol = config.inference.get('recipe', {}).get('protocol')
            self.streaming_text = protocol in ALL_STREAMING_TEXT
            self.raw_latent = protocol in RAW_PROTOCOLS
            self.static_qwen_calls = self.dynamic_qwen_calls = 0
            spec = config.inference.spec
            self.spec = spec
            self.output = Path(config.inference.output_dir)
            self.profiler = StageProfiler(config.inference.runtime.profile)

            def prepare():
                self.output.mkdir(parents=True, exist_ok=True)
                with (self.output/'inference.json').open('x') as f:
                    json.dump(dict(status='starting', **dict(config.inference)), f, indent=2)
                if 'effective_settings' in config.inference:
                    with (self.output/'effective-settings.json').open('x') as handle:
                        json.dump(dict(config.inference.effective_settings), handle, indent=2)
                (self.output/'chunks').mkdir()
                if self.streaming_text:
                    for name in ('prompt-events.jsonl', 'prompt-ledger.jsonl'):
                        with (self.output/name).open('x'):
                            pass
                if 'first_frame' in spec:
                    with Image.open(spec.first_frame) as image:
                        first = np.asarray(image.convert('RGB').resize((spec.width,spec.height),Image.Resampling.LANCZOS))
                else:
                    first = next(iter_video_chunks(spec.rgb, start=spec.start, num_frames=1,
                                 height=spec.height, width=spec.width, chunk_frames=1))[0]
                Image.fromarray(first).save(self.output/'first_frame.png')
                with self.profiler.stage('frozen_model_load'):
                    self.encoders = FrozenCWMEncoders(config.inference.checkpoint, device=get_device(), posterior='mean')
                with self.profiler.stage('static_qwen'):
                    static = self.encoders.condition(spec.prompt, images=[first])
                    self.static_qwen_calls += 1
                with self.profiler.stage('static_image_encode'):
                    first_latent = self.encoders.image(first)
                if not self.streaming_text:
                    self._release_text_encoder()
                with self.profiler.stage('online_codec_load'):
                    self.codec = InferenceCodec(self.encoders, config.inference.runtime.codec)
                return dict(static=static, first_frame=first_latent)

            try:
                payload = rank0_call(prepare)
                self.static, self.first_frame = payload['static'], payload['first_frame']
                with self.profiler.stage('generator_load'):
                    super().__init__(config, meta)
                dit_options = backbone_dit_options(config.inference.runtime)
                if any(dit_options.values()):
                    from projects.minimax_h3.modeling.inference_optimizations import configure_inference
                    configure_inference(self.models['backbone'], **dit_options)
            except BaseException:
                self._release_text_encoder()
                raise

        def _release_text_encoder(self):
            if self.encoders is not None:
                for name in ('qwen', 'processor'):
                    if hasattr(self.encoders, name):
                        delattr(self.encoders, name)
                gc.collect()
                torch.cuda.empty_cache()

        @torch.no_grad()
        def run(self, *, prompt_source=None):
            cfg, spec = self.config.inference, self.spec
            backbone = self.models['backbone']
            backbone.eval()
            raw = (iter_video_chunks(spec.proxy, start=spec.start, num_frames=spec.num_frames,
                   height=spec.proxy_height, width=spec.proxy_width, nearest=True) if get_rank() == 0 else None)
            count = (spec.num_frames+16)//17
            emitted = 0
            writer = None
            sink = None
            profiler = self.profiler
            began = time.monotonic()
            output_count = 0
            prompt_events = set()

            def chunks():
                for k in range(count):
                    eos = k == count-1
                    def encode():
                        prompt = prompt_source(k) if self.streaming_text else None
                        with profiler.stage('proxy_read', cuda=False):
                            frames = next(raw)
                        expected = 5 if eos else 17
                        if len(frames) != expected:
                            raise ValueError(f'chunk {k}: expected {expected} raw frames, received {len(frames)}')
                        with profiler.stage('proxy_encode'):
                            latent = self.codec.encode(frames, eos=eos)
                        item = dict(references=[dict(kind='video', role='semantic', video=latent)], eos=eos)
                        if self.streaming_text and prompt is not None:
                            item['prompt'] = prompt
                        return item
                    with profiler.stage('proxy_encode_broadcast'):
                        item = rank0_call(encode)
                    yield item

            def write(key, result, pixels):
                nonlocal emitted, writer
                k, prompt = key
                span = prompt if self.raw_latent else None
                if self.raw_latent:
                    prompt = None
                if not torch.isfinite(result).all():
                    raise ValueError(f'chunk {k}: generated latent contains NaN or Inf')
                path = self.output/'chunks'/f'{k:06d}.pt'
                with profiler.stage('latent_write', cuda=False):
                    with path.open('xb') as f:
                        record = dict(video=result, chunk=k, eos=span['plan']['eos'] if span else k==count-1,
                                      conditioning_mode=STATIC_CONDITIONING_MODE)
                        if span:
                            record.update(protocol=cfg.recipe.protocol, **span)
                        if prompt is not None:
                            record.update(protocol=cfg.recipe.protocol, prompt=asdict(prompt))
                        torch.save(record, f)
                if span and self.streaming_text:
                    for source in span['sources']:
                        with (self.output/'prompt-ledger.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(window=k, **source))+'\n')
                        applied = source['prompt']
                        identity = (applied['sha256'], applied['first_chunk'])
                        if identity not in prompt_events:
                            prompt_events.add(identity)
                            with (self.output/'prompt-events.jsonl').open('a') as handle:
                                handle.write(json.dumps(applied, ensure_ascii=False)+'\n')
                if prompt is not None:
                    metadata = dict(chunk=k, sha256=prompt.sha256, first_chunk=prompt.first_chunk, rows=prompt.rows)
                    with (self.output/'prompt-ledger.jsonl').open('a') as handle:
                        handle.write(json.dumps(metadata)+'\n')
                    if prompt.first_chunk == k:
                        with (self.output/'prompt-events.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(metadata, text=prompt.text), ensure_ascii=False)+'\n')
                frames = (0 if pixels is None else len(pixels)) if cfg.decode_mode == 'strict' or self.raw_latent else (5 if k==count-1 else 17)
                if pixels is not None:
                    if writer is None:
                        writer = VideoWriter(self.output, spec.height, spec.width)
                    with profiler.stage('video_write', cuda=False):
                        writer.write(pixels)
                emitted += frames
                event = dict(chunk=k, frames=frames, total_frames=emitted,
                             elapsed_seconds=time.monotonic()-began, latent=str(path))
                if span:
                    event.update(span)
                with (self.output/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(event)+'\n'); f.flush()
                print(json.dumps(event), flush=True)

            def emit(k, output):
                nonlocal sink, output_count
                output_count += 1
                if self.raw_latent:
                    result = output.video
                    prompt = dict(plan=output.plan.record(), sources=output.sources, metrics=output.metrics)
                    eos = output.plan.eos
                else:
                    result, prompt = (output.video, output.prompt) if self.streaming_text else (output, None)
                    eos = k == count-1
                if sink is None:
                    sink = OutputSink(write, asynchronous=cfg.runtime.output.async_write,
                                      queue_size=cfg.runtime.output.queue_size, profiler=profiler)
                pixels = None
                if cfg.decode_mode == 'strict':
                    with profiler.stage('video_decode'):
                        pixels = self.codec.decode(result, eos=eos)
                sink.submit((k, prompt), result, pixels)

            def encode_text(text):
                def encode():
                    with profiler.stage('dynamic_qwen'):
                        document = encode_instruction(self.encoders, text)
                    self.dynamic_qwen_calls += 1
                    return document
                return rank0_call(encode)

            success = False
            try:
                def setup_source():
                    nonlocal prompt_source
                    if prompt_source is None:
                        prompt_source = make_prompt_source(cfg.prompt_source, count)
                    if not callable(prompt_source):
                        raise ValueError('prompt_source must be a boundary callable')
                    return getattr(prompt_source, 'provenance', dict(kind='programmatic_boundary_callable'))
                if not self.streaming_text and prompt_source is not None:
                    raise ValueError('prompt_source requires a streaming-text recipe')
                source_info = rank0_call(setup_source) if self.streaming_text else None
                outputs = self.meta.stream_latents(backbone, self.static, chunks(), seed=cfg.seed,
                    condition_seed=cfg.condition_seed, first_frame=self.first_frame,
                    video_shape=(24,spec.height//16,spec.width//16),
                    conditioning_mode=STATIC_CONDITIONING_MODE,
                    profiler=profiler if profiler.enabled else None,
                    **(dict(encode_text=encode_text) if self.streaming_text else {}))
                for k, result in enumerate(outputs):
                    with profiler.stage('output_dispatch'):
                        rank0_call(lambda: emit(k, result))
                def finish():
                    nonlocal writer, emitted
                    sink.close()
                    if emitted != spec.num_frames and not (self.raw_latent and cfg.decode_mode == 'native'):
                        raise ValueError('incomplete output stream')
                    if cfg.decode_mode == 'native':
                        latents = [torch.load(self.output/'chunks'/f'{k:06d}.pt',
                                             map_location='cpu',weights_only=True)['video'] for k in range(output_count)]
                        with profiler.stage('video_decode_native'):
                            pixels = self.encoders.decode_video(torch.cat(latents,dim=1),num_frames=spec.num_frames,mode='native')
                        writer = VideoWriter(self.output,spec.height,spec.width)
                        with profiler.stage('video_write', cuda=False):
                            writer.write(pixels)
                        emitted = len(pixels)
                    with profiler.stage('video_finalize', cuda=False):
                        writer.close(success=True); writer = None
                    report = dict(status='complete', **dict(cfg), frames=emitted, chunks=count,
                                  qwen_calls=self.static_qwen_calls+self.dynamic_qwen_calls,
                                  static_qwen_calls=self.static_qwen_calls, dynamic_qwen_calls=self.dynamic_qwen_calls,
                                  proxy_qwen_calls=0, elapsed_seconds=time.monotonic()-began,
                                  output=str(self.output/'video.mp4'))
                    if self.raw_latent:
                        report.update(generation_windows=output_count, position_policy='absolute_native_v1',
                            training_protocol_match='unverified', conditioning_matches_training=False)
                    if self.streaming_text:
                        from projects.minimax_h3.inference.artifact import sha256
                        report.update(effective_protocol=cfg.recipe.protocol,
                            base_conditioning_mode=STATIC_CONDITIONING_MODE,
                            prompt_extension='independent_complete_text_per_chunk',
                            inference_packing='raw_latent' if self.raw_latent else 'legacy',
                            conditioning_matches_training=False, prompt_source=source_info,
                            qwen_residency='rank0_resident_on_generator_device',
                            qwen_device=str(self.encoders.device),
                            prompt_events_sha256=sha256(self.output/'prompt-events.jsonl'),
                            prompt_ledger_sha256=sha256(self.output/'prompt-ledger.jsonl'))
                    (self.output/'inference.json').write_text(json.dumps(report,indent=2))
                    return report
                report = rank0_call(finish)
                if profiler.enabled:
                    (self.output/f'profile.rank{get_rank()}.json').write_text(json.dumps(profiler.report(), indent=2))
                success = True
                return report
            finally:
                if get_rank() == 0:
                    try:
                        try:
                            if raw is not None:
                                raw.close()
                        finally:
                            try:
                                if sink is not None:
                                    sink.close()
                            finally:
                                if writer is not None:
                                    writer.close(success=False)
                    except Exception:
                        if success:
                            raise
                        traceback.print_exc()
                    finally:
                        self._release_text_encoder()
                        if not success:
                            (self.output/'FAILED').write_text('Inference did not complete. Preserve logs and partial output; use a new output directory.\n')
    return Ref2VAStreamEngine


def Ref2VAStreamEngine(config, meta):
    """Dynamic common.launch entry point, with imports deferred until construction."""
    return make_engine_class()(config, meta)



def input_spec(args, recipe):
    """Resolve a window explicitly; never silently trim a bundle or infer EOS."""
    if args.num_frames < 5 or args.num_frames % 17 != 5:
        raise ValueError('--num-frames must be 17*n+5 (default 158), including the final five-frame EOS')
    if args.start_frame is not None and args.window_index is not None:
        raise ValueError('choose --start-frame or --window-index, not both')
    start = args.start_frame if args.start_frame is not None else (args.window_index or 0) * 158
    if start < 0:
        raise ValueError('start/window index must be nonnegative')
    spec = dict(start=start, num_frames=args.num_frames, height=int(recipe['height']),
                width=int(recipe['width']), proxy_height=int(recipe['proxy_height']),
                proxy_width=int(recipe['proxy_width']), prompt=recipe['system_prompt'])
    if args.bundle:
        bundle = Path(args.bundle).resolve()
        with (bundle/'actions.jsonl').open() as f:
            header = json.loads(f.readline())
        if header['fps'] != 24 or start + args.num_frames > header['frame_count']:
            raise ValueError('bundle window is out of bounds or not at 24 FPS')
        spec.update(bundle=str(bundle), proxy=str(bundle/'proxy.mkv'), rgb=str(bundle/'rgb.mkv'))
    else:
        if not args.first_frame:
            raise ValueError('--proxy-video requires --first-frame')
        spec.update(proxy=str(Path(args.proxy_video).resolve()))
    if args.first_frame:
        spec['first_frame'] = str(Path(args.first_frame).resolve())
    for name in ('proxy', 'rgb', 'first_frame'):
        if name in spec and not Path(spec[name]).is_file():
            raise ValueError(f'missing {name}: {spec[name]}')
    import av
    with av.open(spec['proxy']) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError('proxy must have one video stream and no audio')
        if abs(float(container.streams.video[0].average_rate)-24) > 1e-4:
            raise ValueError('proxy FPS must be 24')
    return spec


def request_arguments(parser):
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--bundle', help='Ref2VA bundle directory with actions.jsonl, rgb.mkv, proxy.mkv')
    inputs.add_argument('--proxy-video', help='Generic 24 FPS proxy MKV; requires --first-frame')
    parser.add_argument('--first-frame', help='RGB picture; defaults to the first RGB frame of the selected bundle window')
    parser.add_argument('--window-index', type=int, help='Zero-based 158-frame bundle window (default 0)')
    parser.add_argument('--start-frame', type=int, help='Explicit proxy/bundle start; use with an explicit longer --num-frames')
    parser.add_argument('--num-frames', type=int, default=158)
    parser.add_argument('--prompt-file', default=str(Path(__file__).resolve().parents[1]/'trials/ref2va/system_prompt.txt'))
    parser.add_argument('--decode-mode', choices=('strict','native'), default='strict')
    parser.add_argument('--steps', type=int, help='Explicit sampling-step override; otherwise keep the trained stage validation/rollout grid')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--condition-seed', type=int)
    parser.add_argument('--check-only', action='store_true', help='Verify recipe, inputs and paths on CPU, without model construction')
