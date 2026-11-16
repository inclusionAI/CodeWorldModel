"""Prepare invoker-render inference assets without modifying model weights."""
import argparse
from pathlib import Path

from projects.minimax_h3.inference.omni_h3_assets import (
    RECIPES, _validate_codec_options, _validate_codec_parallel, prepare_native_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--recipe', choices=RECIPES, default='cwm',
        help='CWM inference by default; legacy recipes require an explicit non-production profile')
    parser.add_argument('--deployment-profile', choices=('production', 'reference', 'overlap'), default='production',
        help='production: CWM with separate codec groups and fast FA3 decoding; reference: conservative codec; overlap: experimental compiler/Graph tuning')
    parser.add_argument('--cwm-silence', help='Frozen stereo silence, required by --recipe cwm')
    parser.add_argument('--cwm-proxy-width', type=int, choices=(448, 1344), default=1344,
        help='Proxy VAE width: 448 encodes the upload directly; 1344 upscales first (default)')
    parser.add_argument('--cwm-proxy-resize', choices=('lanczos', 'nearest'),
        help='Full-size proxy resize (default nearest); nearest runs on the encoder device')
    parser.add_argument('--output', required=True)
    parser.add_argument('--decoder-config', required=True)
    parser.add_argument('--decoder-weights', required=True)
    parser.add_argument('--proxy-encoder', choices=('h3', 'tae'), default='h3',
        help='Proxy encoder only; the static first image always uses full H3')
    parser.add_argument('--tae-weights', help='Released TAE-H3 checkpoint; required by --proxy-encoder tae')
    parser.add_argument('--state-reservation-gib', type=int, required=True)
    parser.add_argument('--execution-config', help='Native topology/backend JSON; defaults to TP1/Ulysses8 for CWM')
    parser.add_argument('--codec-parallel', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--codec-encoder-ranks', type=int, nargs='+', metavar='RANK',
        help='An encoder pair or realtime E4 ranks 2 3 6 7')
    parser.add_argument('--decoder-ranks', type=int, nargs='+', metavar='RANK')
    parser.add_argument('--decoder-backend', choices=('reference', 'fast'))
    parser.add_argument('--decoder-compile', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--decoder-compile-mode', choices=('default', 'max-autotune-no-cudagraphs'))
    parser.add_argument('--decoder-tp-fused-post', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--decoder-cuda-graph', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--decoder-suffix-sdpa-backend', choices=('default', 'cudnn', 'fa3'))
    parser.add_argument('--decoder-spatial-attention-backend', choices=('fa2', 'fa3'))
    parser.add_argument('--decoder-compact-suffix-kv', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fuse-first-denoise', action=argparse.BooleanOptionalAction, default=None,
        help='Fuse deferred clean history, current proxy and first denoise in one DiT forward (CWM default)')
    parser.add_argument('--dit-norm-quality', choices=('exact', 'lossless'))
    args = parser.parse_args(argv)
    if (args.proxy_encoder == 'tae') != bool(args.tae_weights):
        parser.error('--proxy-encoder tae requires --tae-weights, and H3 must omit it')

    def default(name, value):
        if getattr(args, name) is None:
            setattr(args, name, value)

    overlap = args.deployment_profile == 'overlap'
    production = args.deployment_profile == 'production'
    if production and args.recipe != 'cwm':
        parser.error('invoker-render production requires --recipe cwm; legacy artifacts require --deployment-profile reference or overlap')
    default('codec_parallel', production or overlap)
    if not args.codec_parallel and (args.codec_encoder_ranks is not None or args.decoder_ranks is not None):
        parser.error('explicit codec ranks require --codec-parallel')
    if args.codec_parallel:
        default('codec_encoder_ranks', [2, 3, 6, 7] if production or overlap else [0, 1])
        default('decoder_ranks', [0, 1, 4, 5] if production or overlap else [0, 1])
    default('decoder_backend', 'fast' if args.codec_parallel else 'reference')
    accelerated = overlap and args.codec_parallel and args.decoder_backend == 'fast'
    default('decoder_compile', accelerated)
    default('decoder_cuda_graph', accelerated)
    default('decoder_compile_mode', 'max-autotune-no-cudagraphs' if accelerated and args.decoder_compile else 'default')
    default('decoder_tp_fused_post', accelerated and args.decoder_compile)
    default('decoder_compact_suffix_kv', accelerated)
    default('decoder_suffix_sdpa_backend', 'fa3' if accelerated and args.decoder_compact_suffix_kv else 'default')
    default('decoder_spatial_attention_backend', 'fa3' if accelerated or (production and args.codec_parallel and args.decoder_backend == 'fast') else 'fa2')
    cwm = args.recipe == 'cwm'
    default('fuse_first_denoise', cwm)
    default('dit_norm_quality', 'lossless' if overlap or cwm else 'exact')
    default('cwm_proxy_resize', 'nearest' if cwm and args.cwm_proxy_width == 1344 else 'lanczos')
    default('execution_config', str(Path(__file__).resolve().parents[1] / 'configs/runs' /
        ('omni-h3-cwm-execution.json' if args.recipe == 'cwm' else 'omni-h3-execution.json')))
    if args.recipe == 'cwm':
        if not args.cwm_silence:
            parser.error('CWM requires --cwm-silence and a matching CWM artifact; fixed-v2 artifacts use --recipe longlive')
    elif args.cwm_silence is not None or args.cwm_proxy_width != 1344 or args.cwm_proxy_resize != 'lanczos':
        parser.error('CWM conditioning options require --recipe cwm')
    codec = dict(encoder=args.proxy_encoder, decoder='causal', backend=args.decoder_backend,
        compile=args.decoder_compile, cuda_graph=args.decoder_cuda_graph,
        compile_mode=args.decoder_compile_mode, tp_fused_post=args.decoder_tp_fused_post,
        compact_suffix_kv=args.decoder_compact_suffix_kv,
        suffix_sdpa_backend=args.decoder_suffix_sdpa_backend,
        spatial_attention_backend=args.decoder_spatial_attention_backend)
    if args.proxy_encoder == 'tae':
        codec['tae_weights'] = str(Path(args.tae_weights).resolve())
    manifest = dict(codec=codec, server_args=dict(num_gpus=8))
    if args.codec_parallel:
        manifest['codec_parallel'] = dict(encoder_ranks=args.codec_encoder_ranks, decoder_ranks=args.decoder_ranks)
    try:
        if args.decoder_backend != 'fast' and (args.decoder_compile or args.decoder_cuda_graph):
            raise ValueError('decoder compile and CUDA Graph require the fast backend')
        _validate_codec_options(codec)
        _validate_codec_parallel(manifest)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    prepare_native_bundle(args.artifact, args.output, recipe=args.recipe, cwm_silence=args.cwm_silence,
        cwm_proxy_width=args.cwm_proxy_width, cwm_proxy_resize=args.cwm_proxy_resize,
        decoder_config=args.decoder_config,
        decoder_weights=args.decoder_weights,
        state_reservation_bytes=args.state_reservation_gib * (1 << 30),
        execution_config=args.execution_config,
        codec_parallel=dict(encoder_ranks=args.codec_encoder_ranks or [0, 1],
            decoder_ranks=args.decoder_ranks) if args.codec_parallel else None,
        decoder_backend=args.decoder_backend or ('fast' if args.codec_parallel else 'reference'),
        decoder_compile=args.decoder_compile, decoder_cuda_graph=args.decoder_cuda_graph,
        decoder_compile_mode=args.decoder_compile_mode, decoder_tp_fused_post=args.decoder_tp_fused_post,
        decoder_suffix_sdpa_backend=args.decoder_suffix_sdpa_backend,
        decoder_spatial_attention_backend=args.decoder_spatial_attention_backend,
        decoder_compact_suffix_kv=args.decoder_compact_suffix_kv,
        fuse_first_denoise=args.fuse_first_denoise, dit_norm_quality=args.dit_norm_quality,
        proxy_encoder=args.proxy_encoder, tae_weights=args.tae_weights)
    print(args.output)


if __name__ == '__main__':
    main()
