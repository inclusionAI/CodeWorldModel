"""Inference deployment helpers. No training configuration or encoder recipe changes."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from pathlib import Path
import queue
import threading
import time

import torch


from ..inference.settings import load_inference_options


class StageProfiler:
    """Host spans and CUDA event intervals; only report() waits for GPU work.

    Spans may overlap (notably output copies and DiT); do not sum them into
    end-to-end latency. Disabled profiling creates no events or synchronization.
    """
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.origin = time.perf_counter()
        self.records = []
        self.lock = threading.Lock()

    def stage(self, name, *, cuda=True):
        return self._stage(name, cuda) if self.enabled else nullcontext()

    @contextmanager
    def _stage(self, name, cuda):
        begin = time.perf_counter()
        events = None
        if cuda and torch.cuda.is_available():
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        try:
            yield
        finally:
            if events is not None:
                events[1].record()
            record = dict(stage=name, start_ms=(begin-self.origin)*1000,
                          host_ms=(time.perf_counter()-begin)*1000, events=events)
            with self.lock:
                self.records.append(record)

    def report(self):
        rows, totals = [], {}
        for record in sorted(self.records, key=lambda r:r['start_ms']):
            row = {k:v for k,v in record.items() if k != 'events'}
            if record['events'] is not None:
                start, stop = record['events']
                stop.synchronize()
                row['cuda_ms'] = start.elapsed_time(stop)
            total = totals.setdefault(row['stage'], dict(calls=0, host_ms=0.0, cuda_ms=0.0))
            total['calls'] += 1
            total['host_ms'] += row['host_ms']
            total['cuda_ms'] += row.get('cuda_ms', 0.0)
            rows.append(row)
        return dict(stages=totals, spans=rows, total_wall_ms=(time.perf_counter()-self.origin)*1000,
                    note='Host spans and CUDA intervals can overlap; totals are not end-to-end latency.')


class ResidentInferencePlacement:
    """Full inference replica per rank, with the same BF16/FP32 compute boundary as FSDP."""
    def __init__(self, config):
        self.config = config

    def __call__(self, state):
        from common.distributed.ops import get_device
        from .transformer.model import MINIMAX_H3_FP32_PARAM_NAMES
        model = state['model']
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('resident inference placement requires eval and frozen parameters')
        found = set()
        for name, param in model.named_parameters():
            declared = next((n for n in MINIMAX_H3_FP32_PARAM_NAMES
                             if name == n or name.endswith('.'+n)), None)
            if declared:
                found.add(declared)
            dtype = torch.float32 if declared else torch.bfloat16
            # Convert one parameter at a time, without materializing FP32 GPU masters.
            param.data = param.data.to(device=get_device(), dtype=dtype if param.is_floating_point() else param.dtype)
        if found != set(MINIMAX_H3_FP32_PARAM_NAMES):
            raise ValueError('resident placement could not find every H3 FP32 boundary parameter')
        model.to(get_device())
        return state


class InferenceCodec:
    """Online codec selection; original H3 always supplies the static RGB reference."""
    def __init__(self, encoders, options, *, device=None, load_encoder=True, load_decoder=True,
                 decoder_tp_group=None):
        self.encoders, self.options = encoders, options
        self.device = torch.device(device) if device is not None else encoders.device
        self.tae = self.decoder = self.stream = None
        if load_encoder and options['encoder'] == 'tae':
            from safetensors.torch import load_file
            from .taeh3 import MiniMaxH3NativeStreamingTAE
            self.tae = MiniMaxH3NativeStreamingTAE()
            path = options['tae_weights']
            state = (load_file(path) if Path(path).suffix in ('.safetensors', '.safetensor')
                     else torch.load(path, map_location='cpu', weights_only=True))
            self.tae.load_state_dict(state, strict=True, assign=True)
            self.tae.float().to(self.device).eval().requires_grad_(False)
        if load_decoder and options['decoder'] == 'causal':
            from .streaming_video_decoder import CausalViTVideoDecoder
            self.decoder = CausalViTVideoDecoder(options['decoder_config'], options['decoder_weights'],
                tp_group=decoder_tp_group,
                **{k:options[k] for k in ('window','backend','compile','cuda_graph')})
            self.decoder.to(self.device).eval().requires_grad_(False)
            self.stream = self.decoder.create_stream()

    def begin_session(self):
        """Discard decoder history and pending tail while retaining loaded weights."""
        if self.stream is not None:
            self.stream.reset()

    @torch.no_grad()
    def encode(self, frames, *, eos=False):
        if self.tae is None:
            return self.encoders.encode_chunk(frames, eos=eos)
        pixels = torch.as_tensor(frames.copy(), device=self.device).permute(0,3,1,2).float()/255
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            latent = self.tae.encode_video(pixels.unsqueeze(0))[0].permute(1,0,2,3)
        return latent.cpu().to(torch.bfloat16)

    @torch.no_grad()
    def decode(self, latent, *, eos=False):
        """GPU CTHW latent -> GPU THWC uint8; causal tail is flushed at EOS."""
        if self.stream is not None:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                pixels = self.stream.decode(latent.permute(1,0,2,3).unsqueeze(0))
                tail = self.stream.flush() if eos else None
            if tail is not None:
                pixels = tail if pixels is None else torch.cat((pixels, tail), dim=1)
            if pixels is None:
                return None
            pixels = pixels[0]
        else:
            if latent.shape[1] != (2 if eos else 5):
                raise ValueError('strict H3 decode expects five latents or a two-latent EOS')
            frozen = self.encoders
            z = latent.to(frozen.device).float().unsqueeze(0)*frozen.std+frozen.mean
            z = torch.cat([z,z[:,:,-1:].expand(-1,-1,7-z.shape[2],-1,-1)],2)
            pixels = frozen.vae.decode_base(z.to(torch.bfloat16))[:,:,:5 if eos else 17]
            pixels = frozen.vae.processor.transform_rev(pixels[0].permute(1,0,2,3).float())
        return (pixels.clamp(0,1)*255).byte().permute(0,2,3,1).contiguous()


class OutputSink:
    """Bounded ordered writer with owned D2H buffers and deferred error propagation.

    At most queue_size waiting jobs, one active job and one producer copy exist.
    CUDA sources are retained until their copy event completes; callbacks only
    receive ready CPU tensors. close() drains writes even after worker failure.
    """
    def __init__(self, write, *, asynchronous=False, queue_size=2, profiler=None):
        self.write, self.asynchronous = write, asynchronous
        self.profiler = profiler or StageProfiler()
        self.error = None
        self.closed = False
        self.copy_stream = None
        if asynchronous:
            self.queue = queue.Queue(maxsize=queue_size)
            self.worker = threading.Thread(target=self._work, name='h3-output', daemon=True)
            self.worker.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('asynchronous inference output failed') from self.error

    def submit(self, key, *tensors):
        if self.closed:
            raise RuntimeError('output sink is closed')
        self._check()
        sources = tuple(t.detach() if t is not None else None for t in tensors)
        ready = None
        if self.asynchronous and any(t is not None and t.is_cuda for t in sources):
            device = next(t.device for t in sources if t is not None and t.is_cuda)
            if self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(device=device)
            if self.copy_stream.device != device:
                raise ValueError('output sink accepts only one CUDA device')
            self.copy_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(self.copy_stream), self.profiler.stage('d2h'):
                cpus = []
                for value in sources:
                    if value is None:
                        cpus.append(None)
                    elif value.is_cuda:
                        dst = torch.empty_like(value, device='cpu', pin_memory=True)
                        dst.copy_(value, non_blocking=True)
                        value.record_stream(self.copy_stream)
                        cpus.append(dst)
                    else:
                        cpus.append(value.clone())
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
        else:
            with self.profiler.stage('d2h', cuda=False):
                cpus = [None if t is None else t.cpu().clone() if self.asynchronous else t.cpu() for t in sources]
        if self.asynchronous:
            with self.profiler.stage('output_queue_wait', cuda=False):
                self.queue.put((key, cpus, ready, sources))
            self._check()
        else:
            self.write(key, *cpus)

    def _work(self):
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                key, cpus, ready, sources = job
                if ready is not None:
                    with self.profiler.stage('output_copy_wait', cuda=False):
                        ready.synchronize()
                if self.error is None:
                    self.write(key, *cpus)
            except BaseException as exc:
                if self.error is None:
                    self.error = exc
            finally:
                # The next blocking get must not retain a completed GPU packet.
                job = sources = cpus = ready = None
                self.queue.task_done()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.asynchronous:
                with self.profiler.stage('output_flush', cuda=False):
                    self.queue.put(None)
                    self.queue.join()
                    self.worker.join()
        self._check()
