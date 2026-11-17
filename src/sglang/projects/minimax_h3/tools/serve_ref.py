"""Serve resident fixed/bidir Ref recipes through Ray or torchrun."""
import json
import math
import os
from pathlib import Path
import signal
import threading

from projects.minimax_h3.inference.artifact import validate_bundle, sha256
from projects.minimax_h3.inference.settings import parse_run_arguments, options_for_args, effective_settings, inspection_parser, require_dit_graph_environment
from projects.minimax_h3.inference.serving import (
    Distributed, ResidentBackend, SessionController, Worker, load_artifacts, runtime_config, settings_receipt)
from projects.minimax_h3.inference.serving_http import Ledger, SessionMailbox, ServingHTTPServer


def main(argv=None):
    parser = inspection_parser('serve_ref')
    args = parse_run_arguments(parser, argv, entrypoint='serve_ref')
    if (not math.isfinite(args.timeout) or not math.isfinite(args.idle_timeout)
            or args.timeout <= 0 or args.idle_timeout <= 0 or args.max_body_mib < 1):
        parser.error('timeouts and max-body-mib must be positive')
    from common.distributed.init import configure_distributed, destroy_distributed
    from common.logging import configure_logging
    bundle = json.loads(Path(args.artifact[0]).read_text())
    validate_bundle(bundle)
    options = options_for_args(args)
    require_dit_graph_environment(options['dit']['cuda_graph'])
    if options['codec']['decoder'] != 'causal' or not options['resident_weights']:
        parser.error('serving requires causal decoder and resident_weights: true')
    if options['output']['async_write']:
        parser.error('serving requires output.async_write: false')
    size = bundle['deployment']['active_up_size']
    if options['qwen_fsdp'] and size < 2:
        parser.error('qwen_fsdp requires at least two distributed ranks')
    if args.runtime == 'torchrun' and int(os.environ.get('WORLD_SIZE', '1')) != size:
        parser.error('torchrun WORLD_SIZE must equal artifact active_up_size')
    if args.runtime == 'ray' and 'RANK' in os.environ:
        parser.error('launch the Ray frontend once with python, not torchrun')
    if args.runtime == 'torchrun' and (args.vae_rank or args.decoder_rank):
        parser.error('nonzero codec owners require the Ray runtime')
    if args.gpu_occupier_config and (args.runtime != 'ray' or args.ray_address):
        parser.error('gpu-occupier-config requires a local Ray runtime')
    if (args.ray_cpus_per_worker < 1 or not 1 <= args.ray_master_port <= 65535
            or not math.isfinite(args.ray_startup_timeout) or args.ray_startup_timeout <= 0
            or any(not 0 <= rank < size for rank in (args.vae_rank, args.decoder_rank))):
        parser.error('invalid worker resources, rendezvous port, timeout or codec rank')
    output = Path(args.output_dir).resolve()
    config = runtime_config(bundle, options, output)
    if args.runtime == 'torchrun':
        config = configure_distributed(config)
    group = Distributed()
    ledger = mailbox = server = thread = backend = None
    occupier = None
    failed = True
    previous_sigterm = None
    if args.gpu_occupier_config:
        def terminate(signum, frame):
            raise KeyboardInterrupt('serving received SIGTERM')
        previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        if args.gpu_occupier_config:
            from projects.minimax_h3.inference.gpu_occupier import GpuOccupier
            occupier = GpuOccupier(args.gpu_occupier_config)
            occupier.pause()
        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            return load_artifacts(args.artifact)
        artifacts = group.rank0(prepare)
        verified_bundle = next(iter(artifacts.values()))[0]
        if verified_bundle['deployment'] != bundle['deployment']:
            raise RuntimeError('artifact deployment changed during initialization')
        options = group.broadcast(options if group.rank == 0 else None)
        config = runtime_config(verified_bundle, options, output)
        config = configure_logging(config)
        resolved = effective_settings(args, bundle=artifacts, options=options,
            implementation="ray_serving" if args.runtime == 'ray' else "local_serving")
        receipt = settings_receipt(resolved)
        if args.runtime == 'ray':
            from projects.minimax_h3.inference.serving_ray import RayResidentGroup
            backend = RayResidentGroup(artifacts, options, address=args.ray_address,
                strategy=args.ray_placement, cpus_per_worker=args.ray_cpus_per_worker,
                master_port=args.ray_master_port, startup_timeout=args.ray_startup_timeout,
                stage_timeout=args.timeout, vae_rank=args.vae_rank, decoder_rank=args.decoder_rank,
                codec_overlap=args.codec_overlap)
        else:
            backend = ResidentBackend(config, artifacts, group, vae_rank=args.vae_rank,
                                      decoder_rank=args.decoder_rank)
        if occupier is not None:
            occupier.resume()
        def transport():
            nonlocal ledger, mailbox, server, thread
            ledger = Ledger(output/'server.jsonl')
            mailbox = SessionMailbox(ledger, timeout=args.timeout)
            codec_hashes = {key: sha256(options['codec'][key]) for key in
                ('decoder_weights', 'decoder_config', 'tae_weights') if options['codec'][key]}
            with (output/'effective-settings.json').open('x') as handle:
                json.dump(resolved, handle, indent=2)
            health = dict(effective_settings=receipt, weight_load_count=backend.weight_load_count, world_size=size, residency=backend.residency,
                qwen_placement=backend.qwen_placement,
                codec=options['codec'], codec_sha256=codec_hashes, artifacts={key: dict(
                    recipe=recipe.to_dict(), encoder_recipe=bundle['encoders']['recipe'],
                    weights_sha256=bundle['weights']['sha256']) for key, (bundle, recipe) in artifacts.items()})
            health['orchestration'] = (backend.health if args.runtime == 'ray' else
                dict(runtime='torchrun', codec_ranks=dict(encoder=args.vae_rank, decoder=args.decoder_rank)))
            host, port = args.bind.rsplit(':', 1)
            server = ServingHTTPServer((host, int(port)), mailbox, health,
                timeout=args.timeout, max_body=args.max_body_mib << 20)
            (output/'service.json').write_text(json.dumps(dict(bind=args.bind, **health), indent=2))
            thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.1), daemon=True)
            thread.start()
            print(json.dumps(dict(status='ready', bind=args.bind, artifacts=list(artifacts))), flush=True)
        group.rank0(transport)
        controller = SessionController(artifacts, settings=receipt) if group.rank == 0 else None
        Worker(backend, group, controller, mailbox, idle_timeout=args.idle_timeout, occupier=occupier).serve()
        failed = False
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if mailbox is not None:
                mailbox.fail('worker stopped' if not failed else 'worker group failed; restart required')
            if server is not None:
                server.shutdown()
                server.server_close()
            if thread is not None:
                thread.join(timeout=args.timeout)
            if ledger is not None:
                ledger.close()
            if args.runtime == 'ray' and backend is not None:
                backend.close()
            if occupier is not None:
                occupier.restore_after_shutdown()
            if not failed:
                destroy_distributed()
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
