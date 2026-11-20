from copy import deepcopy
import os
from pathlib import Path
import time

from .serving import SessionError
from .serving_media import MAX_RGB_RESERVATIONS
from .serving_stages import CpuOutput, RAY_GPU_BOUNDARY, RAY_STAGE_COMPLETION, STAGES, STAGE_SPECS, WorkKey
from .settings import GRAPH_RUNTIME_ENVIRONMENT, require_dit_graph_environment


class RayResidentGroup:
    deferred_output = True

    def __init__(self, artifacts, options, *, address=None, strategy='PACK', cpus_per_worker=2,
                 master_port=29500, startup_timeout=900, stage_timeout=300,
                 vae_rank=0, decoder_rank=0, codec_overlap=False, _actor_class=None, _gpus_per_worker=1):
        graph_enabled = options.get('dit', {}).get('cuda_graph', False)
        require_dit_graph_environment(graph_enabled)
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from tools.ray_worker import RankActor
        self.ray = ray
        self.actors, self.placement = [], None
        self.owns_ray = not ray.is_initialized()
        self.failed = False
        self.health = dict(state='ready')
        self.epoch = 0
        self.codec_overlap = codec_overlap
        self.prefetch_bounds = None
        self.pending_outputs = {}
        self.timeout = stage_timeout
        self.vae_rank, self.decoder_rank = vae_rank, decoder_rank
        self.world_size = next(iter(artifacts.values()))[0]['deployment']['active_up_size']
        if any(type(rank) is not int or not 0 <= rank < self.world_size for rank in (vae_rank, decoder_rank)):
            raise ValueError('codec ranks must be within the artifact world size')
        root = Path(__file__).resolve().parents[1]
        pythonpath = os.pathsep.join(filter(None, (str(root), os.environ.get('PYTHONPATH'))))
        worker_env = dict(PYTHONPATH=pythonpath, OMP_NUM_THREADS=str(max(1, int(cpus_per_worker))))
        if graph_enabled:
            worker_env.update({name: os.environ[name] for name in GRAPH_RUNTIME_ENVIRONMENT if name in os.environ})
            worker_env.update(NCCL_GRAPH_REGISTER='0', NCCL_GRAPH_MIXING_SUPPORT='1')
        try:
            if self.owns_ray:
                ray.init(address=address, include_dashboard=False, log_to_driver=False)
            self.placement = placement_group(
                [dict(CPU=cpus_per_worker, **({'GPU': _gpus_per_worker} if _gpus_per_worker else {}))
                 for _ in range(self.world_size)], strategy=strategy)
            ray.get(self.placement.ready(), timeout=startup_timeout)
            actor = ray.remote(concurrency_groups={'output': 1})(_actor_class or RankActor)
            for rank in range(self.world_size):
                self.actors.append(actor.options(num_cpus=cpus_per_worker, num_gpus=_gpus_per_worker,
                    max_restarts=0, max_task_retries=0,
                    max_concurrency=1,
                    runtime_env=dict(env_vars=worker_env),
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement, placement_group_bundle_index=rank,
                        placement_group_capture_child_tasks=False)
                    ).remote(rank, self.world_size, vae_rank, decoder_rank))
            observed = self._wait([a.describe.remote() for a in self.actors], startup_timeout)
            if any(row['versions'] != observed[0]['versions'] for row in observed):
                raise RuntimeError('worker Ray/Torch versions differ across nodes')
            seen = set()
            refs = []
            bundle = next(iter(artifacts.values()))[0]
            for actor, row in zip(self.actors, observed):
                verify = row['node_id'] not in seen
                seen.add(row['node_id'])
                refs.append(actor.preflight.remote(bundle, options, verify))
            self._wait(refs, startup_timeout)
            host = observed[0]['node_ip']
            initialized = self._wait([a.initialize.remote(host, master_port, artifacts, options, stage_timeout)
                                      for a in self.actors], startup_timeout)
            self.residency = initialized[0]['residency']
            self.qwen_placement = initialized[0]['qwen_placement']
            self.weight_load_count = 1
            self.health.update(runtime='ray', world_size=self.world_size, placement_strategy=strategy,
                workers=[{k: v for k, v in row.items() if k not in ('residency', 'qwen_placement')}
                         for row in initialized],
                codec_ranks=dict(encoder=vae_rank, decoder=decoder_rank),
                codec_overlap=dict(requested=codec_overlap, enabled=codec_overlap,
                    scope='pending media proxy encode with current decode; one prefetch slot'),
                in_flight_chunks=1, stage_completion=RAY_STAGE_COMPLETION, retries=0,
                output_delivery=dict(transport='CPU ndarray via Ray; read-only byte view for media',
                    granularity='whole work', media_reservations=MAX_RGB_RESERVATIONS,
                    gpu_boundary=RAY_GPU_BOUNDARY))
        except BaseException:
            self._poison()
            self.close()
            raise

    def _poison(self):
        self.failed = True
        self.health['state'] = 'unavailable; restart required'

    def _wait(self, refs, timeout=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        pending = list(refs)
        values = {}
        while pending:
            ready, pending = self.ray.wait(pending, num_returns=1, timeout=max(0, deadline-time.monotonic()))
            if not ready:
                raise TimeoutError('Ray worker group deadline exceeded; group restart required')
            values[ready[0]] = self.ray.get(ready[0])
        return [values[ref] for ref in refs]

    def execute(self, op, metadata, payload, *, lookahead=None, defer_output=False):
        if self.failed:
            raise RuntimeError('Ray worker group is unhealthy; restart required')
        if op not in STAGES:
            return b'', {}
        if op == 'start':
            self.epoch += 1
        key = WorkKey(self.epoch, metadata['session_id'], op, metadata.get('chunk_id', -1))
        reused = self.prefetch_bounds
        if reused is not None:
            if reused['key'] != key:
                raise RuntimeError('prefetch receipt differs from current chunk')
            self.prefetch_bounds = None
        metadata = deepcopy(metadata)
        trace, bounds = [], {}
        outputs = None
        try:
            payload_ref = self.ray.put(payload)
            for stage in STAGES[op]:
                spec = STAGE_SPECS[stage]
                if self.failed:
                    raise RuntimeError('Ray worker group is unhealthy; restart required')
                prefetch = (lookahead() if spec.prefetch_next and self.codec_overlap
                            and lookahead is not None and not metadata['eos'] else None)
                if prefetch is not None:
                    next_metadata, next_payload = prefetch
                    next_key = WorkKey(self.epoch, next_metadata['session_id'], 'chunk', next_metadata['chunk_id'])
                    key.check_next(next_key)
                    next_payload_ref = self.ray.put(next_payload)
                producer = 0 if spec.payload_owner == 'rank0' else self.vae_rank
                started = time.perf_counter_ns()
                refs = []
                for rank, actor in enumerate(self.actors):
                    args = (key, metadata)
                    if spec.payload_owner is not None:
                        args += (payload_ref if rank == producer else b'',)
                    if prefetch is not None:
                        args += (next_key, next_metadata, next_payload_ref if rank == self.vae_rank else b'')
                    refs.append(getattr(actor, stage).remote(*args))
                replies = self._wait(refs)
                ended = time.perf_counter_ns()
                errors = [row for row in replies if 'error' in row]
                if errors:
                    if len(errors) != self.world_size:
                        raise RuntimeError('inconsistent stage failure; group restart required')
                    raise SessionError(errors[0]['error'], errors[0]['status'])
                trace.append(dict(stage=stage, dispatch_ns=started, complete_ns=ended,
                    wall_ms=(ended-started)/1e6,
                    workers=[dict(rank=row['rank'], elapsed_ms=row['elapsed_ms']) for row in replies]))
                bounds[stage] = (started, ended)
                if stage == 'proxy_encode' and reused is not None:
                    trace[-1]['reused'] = True
                    bounds[stage] = (reused['dispatch_ns'], reused['complete_ns'])
                if prefetch is not None:
                    self.prefetch_bounds = dict(key=next_key, dispatch_ns=started, complete_ns=ended)
                    trace[-1]['prefetch_chunk_id'] = next_key.chunk_id
                if stage == 'collect':
                    outputs = replies
            nbytes = outputs[self.decoder_rank]['result'][0]
            timings = outputs[0]['result'][1]
            if type(nbytes) is not int or nbytes < 0:
                raise RuntimeError('decoder returned an invalid CPU output size')
            pixels = b''
            if op == 'chunk':
                timings['gpu_done_upper_bound_ns'] = bounds['collect'][1]
                ref = self.actors[self.decoder_rank].output.options(concurrency_group='output').remote(key)
                self.pending_outputs[key] = ref
                pixels = CpuOutput(key, nbytes, lambda: self._read_output(key, ref, nbytes))
            if reused is not None:
                timings['codec_prefetch'] = dict(chunk_id=key.chunk_id, reused=True,
                    dispatch_ns=reused['dispatch_ns'], complete_ns=reused['complete_ns'],
                    clock='frontend_monotonic', bounds='combined decode and next encode RPC')
            # Worker monotonic clocks are local to their node; public bounds use the frontend clock.
            for stage, names in [('proxy_encode', ('encode_start_ns', 'encode_submit_ns')),
                                 ('render', ('render_start_ns', 'render_submit_ns')),
                                 ('decode', ('decode_start_ns', 'cpu_ready_ns'))]:
                if stage in bounds:
                    timings.update(zip(names, bounds[stage]))
            if 'decode' in bounds:
                timings['rgb_ready_upper_bound_ns'] = bounds['decode'][1]
            timings['orchestration'] = dict(epoch=self.epoch, stages=trace, clock='frontend_monotonic',
                rank_clock='worker_local_monotonic; compare durations only')
            if isinstance(pixels, CpuOutput) and not defer_output:
                pixels, delivery = pixels.read()
                pixels = bytes(pixels)
                timings.update(delivery)
            return pixels, timings
        except SessionError:
            raise
        except BaseException:
            self._poison()
            self.close()
            raise

    def _read_output(self, key, ref, nbytes):
        import numpy as np
        try:
            received_key, pixels, timing = self._wait([ref])[0]
            if (received_key != key or not isinstance(pixels, np.ndarray)
                    or pixels.dtype != np.uint8 or not pixels.flags.c_contiguous
                    or pixels.nbytes != nbytes):
                raise RuntimeError('CPU output differs from the completed work')
            pixels = memoryview(pixels).cast('B').toreadonly()
            received = time.perf_counter_ns()
            return pixels, dict(response_ready_ns=received,
                output_delivery=dict(status='complete', part_count=int(nbytes > 0), byte_count=nbytes,
                    worker=timing, received_ns=received, clock='frontend_monotonic'))
        except BaseException:
            self._poison()
            raise
        finally:
            self.pending_outputs.pop(key, None)

    def close_session(self):
        self.prefetch_bounds = None
        if self.failed:
            self.close()
            return
        try:
            self._wait([a.close_session.remote(self.epoch) for a in self.actors])
            if self.pending_outputs:
                self._wait(list(self.pending_outputs.values()))
                self.pending_outputs.clear()
            if self.failed:
                self.close()
        except BaseException:
            self._poison()
            self.close()
            raise

    def close(self):
        from ray.util.placement_group import remove_placement_group
        if not self.ray.is_initialized():
            return
        try:
            for actor in self.actors:
                self.ray.kill(actor, no_restart=True)
        finally:
            self.actors.clear()
            self.pending_outputs.clear()
            if self.placement is not None:
                remove_placement_group(self.placement)
                self.placement = None
            if self.owns_ray:
                self.ray.shutdown()
