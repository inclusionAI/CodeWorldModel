import os
import tempfile
import time


class RankActor:
    def __init__(self, rank, world_size, vae_rank, decoder_rank):
        from tools.launch import bootstrap
        bootstrap()
        from projects.minimax_h3.inference.serving_stages import StageCursor
        self.rank, self.world_size = rank, world_size
        self.vae_rank, self.decoder_rank = vae_rank, decoder_rank
        self.cursor = StageCursor()
        self.backend = None
        self.output_buffers = {}

    def describe(self):
        from importlib.metadata import version
        import ray
        return dict(rank=self.rank, node_id=ray.get_runtime_context().get_node_id(),
                    node_ip=ray.util.get_node_ip_address(), pid=os.getpid(),
                    accelerator_ids=ray.get_runtime_context().get_accelerator_ids(),
                    visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    versions=dict(ray=ray.__version__, torch=version('torch')))

    def preflight(self, bundle, options, verify_assets):
        from projects.minimax_h3.inference.artifact import verify_bundle_assets
        from projects.minimax_h3.inference.settings import load_inference_options, require_dit_graph_environment
        resolved = load_inference_options(data=options)
        require_dit_graph_environment(resolved['dit']['cuda_graph'])
        if verify_assets:
            verify_bundle_assets(bundle)
        return self.describe()

    def initialize(self, host, port, artifacts, options, timeout):
        from projects.minimax_h3.inference.settings import require_dit_graph_environment
        require_dit_graph_environment(options['dit']['cuda_graph'])
        import torch
        from common.distributed.init import configure_distributed
        from projects.minimax_h3.inference.serving import Distributed, ResidentBackend, runtime_config
        os.environ.update(RANK=str(self.rank), WORLD_SIZE=str(self.world_size), LOCAL_RANK='0',
                          MASTER_ADDR=host, MASTER_PORT=str(port))
        bundle = next(iter(artifacts.values()))[0]
        config = runtime_config(bundle, options, tempfile.gettempdir())
        config.distributed.timeout_minutes = timeout / 60
        configure_distributed(config)
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError('Ray rank requires exactly one visible GPU at local ordinal zero')
        self.backend = ResidentBackend(config, artifacts, Distributed(),
                                      vae_rank=self.vae_rank, decoder_rank=self.decoder_rank)
        result = dict(self.describe(), local_device=0, residency=self.backend.residency,
                      qwen_placement=self.backend.qwen_placement)
        if options['dit']['cuda_graph']:
            from projects.minimax_h3.modeling.inference_optimizations import graph_runtime_identity
            result['graph_runtime'] = graph_runtime_identity()
        return result

    def static_condition(self, key, metadata, payload=b''):
        return self._run_stage(key, 'static_condition', metadata, payload)

    def first_image(self, key, metadata, payload=b''):
        return self._run_stage(key, 'first_image', metadata, payload)

    def open_recipe(self, key, metadata):
        return self._run_stage(key, 'open_recipe', metadata)

    def text_condition(self, key, metadata):
        return self._run_stage(key, 'text_condition', metadata)

    def proxy_encode(self, key, metadata, payload=b''):
        return self._run_stage(key, 'proxy_encode', metadata, payload)

    def render(self, key, metadata):
        return self._run_stage(key, 'render', metadata)

    def decode(self, key, metadata, prefetch_key=None, prefetch_metadata=None, payload=b''):
        if prefetch_key is not None:
            key.check_next(prefetch_key)
        return self._run_stage(key, 'decode', metadata, payload,
                               prefetch_key=prefetch_key, prefetch_metadata=prefetch_metadata)

    def collect(self, key, metadata):
        return self._run_stage(key, 'collect', metadata)

    def output(self, key):
        import numpy as np
        cpu = self.output_buffers.pop(key)
        started = time.perf_counter_ns()
        if cpu is not None and cpu.device.type != 'cpu':
            raise RuntimeError('output delivery requires completed CPU storage')
        pixels = np.empty(0, dtype=np.uint8) if cpu is None else cpu.numpy()
        return key, pixels, dict(array_view_start_ns=started, array_view_end_ns=time.perf_counter_ns(),
                                clock='worker_local_monotonic')

    def _run_stage(self, key, stage, metadata, payload=b'', *, prefetch_key=None, prefetch_metadata=None):
        import torch
        from projects.minimax_h3.inference.serving import SessionError
        from projects.minimax_h3.inference.serving_stages import STAGE_SPECS
        self.cursor.check(key, stage)
        started = time.perf_counter_ns()
        try:
            if self.cursor.current is None:
                self.backend.begin(key.op, metadata)
            if stage == 'proxy_encode' and self.backend.prefetched_proxy is not None:
                if self.backend.prefetched_proxy.key != key:
                    raise RuntimeError('prefetch key differs from current stage')
            method = getattr(self.backend, stage)
            if prefetch_key is not None:
                result = self.backend.decode_with_prefetch(prefetch_key, prefetch_metadata, payload)
            elif stage == 'collect':
                cpu, timings = self.backend.collect(defer_output=True)
                size = 0 if cpu is None else cpu.numel() * cpu.element_size()
                if self.rank == self.decoder_rank and key.op == 'chunk':
                    self.output_buffers[key] = cpu
                result = size, timings
            else:
                result = method(payload) if STAGE_SPECS[stage].payload_owner is not None else method()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except SessionError as exc:
            return dict(error=str(exc), status=exc.status)
        self.cursor.completed(key)
        return dict(result=result, elapsed_ms=(time.perf_counter_ns()-started)/1e6,
                    rank=self.rank, stage=stage)

    def close_session(self, epoch):
        import torch
        if epoch != self.cursor.epoch and self.cursor.session_id is not None:
            raise RuntimeError('reset epoch differs from active session')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.backend.close_session()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.cursor.reset()

    def shutdown(self):
        from common.distributed.init import destroy_distributed
        if self.backend is not None:
            self.backend.close_session()
        destroy_distributed()
