"""Serve invoker-render on loopback using the prepared inference configuration."""
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, help='Prepared invoker-render metadata directory')
    parser.add_argument('--port', type=int, default=18860)
    parser.add_argument('--codec-overlap', action=argparse.BooleanOptionalAction, default=None,
        help='Overlap next proxy encode with current decode on separate codec GPU groups')
    parser.add_argument('--cpu-delivery', action=argparse.BooleanOptionalAction, default=None,
        help='Pack completed CPU RGB in a separate stage; requires --codec-overlap')
    parser.add_argument('--realtime-sessions', action=argparse.BooleanOptionalAction, default=None,
        help='Drive H3 through upstream SGLang realtime sessions instead of the H3 host')
    parser.add_argument('--verify-assets', action='store_true',
        help='Re-hash all pinned weights and configs before launch (slow)')
    args = parser.parse_args(argv)
    if args.cpu_delivery and args.codec_overlap is False:
        parser.error('--cpu-delivery requires --codec-overlap')
    from projects.minimax_h3.inference.omni_h3_assets import read_native_bundle, verify_native_bundle

    manifest = read_native_bundle(args.model_path)
    placement = manifest.get('codec_parallel') or {}
    encoder = tuple(placement.get('encoder_ranks', ()))
    decoder = tuple(placement.get('decoder_ranks', ()))
    codec = manifest.get('codec', {})
    cwm = manifest['recipe'].get('protocol') == 'cwm_causal/v1'
    realtime_required = (cwm or len(encoder) == 4
        or codec.get('suffix_sdpa_backend') == 'fa3'
        or codec.get('compile_mode', 'default') != 'default'
        or codec.get('tp_fused_post', False))
    if args.realtime_sessions is None:
        args.realtime_sessions = realtime_required
    if realtime_required and not args.realtime_sessions:
        parser.error('this bundle requires --realtime-sessions (CWM, E4 or realtime decoder tuning)')
    supported = (((2, 3), (0, 1)), ((2, 3), (0, 1, 4, 5)))
    if args.realtime_sessions:
        supported += (((2, 3, 6, 7), (0, 1, 4, 5)),)
    overlap_supported = (manifest['server_args']['num_gpus'] == 8
        and codec.get('backend') == 'fast'
        and (encoder, decoder) in supported)
    if args.codec_overlap is None:
        args.codec_overlap = overlap_supported
    if args.codec_overlap and not overlap_supported:
        parser.error('--codec-overlap requires a fast eight-GPU bundle with supported disjoint codec groups')
    if args.cpu_delivery is None:
        args.cpu_delivery = args.codec_overlap
    if args.cpu_delivery and not args.codec_overlap:
        parser.error('--cpu-delivery requires --codec-overlap')
    if args.verify_assets:
        verify_native_bundle(manifest)
    return args, manifest


def main():
    args, manifest = parse_args()
    from sglang_omni.config.schema import StageConfig
    from sglang_omni.models.minimax_h3.config import (
        H3_RGB_DELIVERY_FACTORY,
        H3_RGB_DELIVERY_STAGE,
        MiniMaxH3PipelineConfig,
    )
    from sglang_omni.serve.launcher import launch_server

    canvas = manifest['canvas']
    stages = [StageConfig(name='h3', process='h3',
        factory_path='sglang_omni.models.minimax_h3.stages.create_h3_session_executor',
        gpu=0, runtime_gpu_ids=list(range(manifest['server_args']['num_gpus'])),
        allow_child_processes=True, terminal=not args.cpu_delivery,
        next=H3_RGB_DELIVERY_STAGE if args.cpu_delivery else None)]
    if args.cpu_delivery:
        stages.append(StageConfig(name=H3_RGB_DELIVERY_STAGE, process=H3_RGB_DELIVERY_STAGE,
            factory_path=H3_RGB_DELIVERY_FACTORY, terminal=True))
    config = MiniMaxH3PipelineConfig(model_path=args.model_path, name=manifest.get('model_name'),
        codec_overlap=args.codec_overlap,
        cpu_delivery=args.cpu_delivery,
        runtime_factory='projects.minimax_h3.inference.{}.create_runtime'.format(
            'omni_h3_realtime' if args.realtime_sessions else 'omni_h3_runtime'),
        max_state_bytes=manifest['state_reservation_bytes'],
        max_width=canvas['width'], max_height=canvas['height'], max_output_frames=64,
        operation_timeout_seconds=600,
        stages=stages)
    launch_server(config, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
