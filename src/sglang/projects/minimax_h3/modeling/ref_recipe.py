"""Versioned Ref inference contracts, resolved without training objects."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

FIXED_RECIPE = 'ref_fixed_recent_history/v1'
BIDIR_RECIPE = 'ref_bidir_recent_history/v1'
FIXED_STREAMING_TEXT_RECIPE = 'ref_fixed_recent_history_streaming_text/v1'
BIDIR_STREAMING_TEXT_RECIPE = 'ref_bidir_recent_history_streaming_text/v1'
STREAMING_TEXT_RECIPES = (FIXED_STREAMING_TEXT_RECIPE, BIDIR_STREAMING_TEXT_RECIPE)
PROJECT = 'projects.minimax_h3'


CAUSAL_LAYOUT = 'fixed_role_time_v1'
DEFAULT_WINDOW = 6
DEFAULT_ROLE_SPAN = 37
LEGACY_CONDITIONING_MODE = 'legacy'
STATIC_CONDITIONING_MODE = 'static_image_qwen_proxy_vae_v1'


def conditioning_mode(record):
    mode = record.get('conditioning_mode', LEGACY_CONDITIONING_MODE)
    if mode not in (LEGACY_CONDITIONING_MODE, STATIC_CONDITIONING_MODE):
        raise ValueError(f'unsupported conditioning mode: {mode!r}')
    return mode


def causal_layout(window_size=DEFAULT_WINDOW, rope_reference_latents=DEFAULT_ROLE_SPAN):
    """Finite time-group window; role spacing covers every visible video row."""
    if type(window_size) is not int or window_size < 0:
        raise ValueError('causal window_size must be a nonnegative integer (default 6), not null')
    if type(rope_reference_latents) is not int or rope_reference_latents < 5*(window_size+1):
        raise ValueError('rope_reference_latents must be >= 5 * (window_size + 1)')
    return CAUSAL_LAYOUT, window_size, rope_reference_latents


def training_guidance_options(meta, stage):
    cfg = meta.get('guidance', {})
    mode = cfg.get('mode', 'plain')
    options = dict(mode=mode, drop=cfg.get('drop', 'proxy'),
        dropout=float(cfg.get('dropout', .1)), beta=float(cfg.get('beta', .5)),
        scale=float(cfg.get('scale', 2. if mode == 'cfg' else 1.)),
        seed=int(cfg.get('seed', 20261010)))
    if mode not in ('plain', 'cfg', 'gft') or options['drop'] not in ('proxy', 'prompt', 'prompt_proxy'):
        raise ValueError('invalid guidance mode or dropped condition')
    if not 0 <= options['dropout'] < 1 or not 0 < options['beta'] <= 1:
        raise ValueError('guidance needs dropout in [0,1) and beta in (0,1]')
    if mode == 'cfg' and options['dropout'] == 0:
        raise ValueError('CFG training needs positive condition dropout')
    if not math.isfinite(options['scale']) or options['scale'] < 0:
        raise ValueError('guidance scale must be finite and nonnegative')
    if mode != 'cfg' and options['scale'] != 1:
        raise ValueError('plain/GFT inference uses scale=1; GFT guidance is already learned')
    if mode != 'plain' and stage not in ('sft', 'tf'):
        raise ValueError('guidance training is supported only for SFT and TF')
    return options


@dataclass(frozen=True)
class RefRecipe:
    conditioning_mode: str
    window_size: int
    rope_reference_latents: int
    diffusion: dict
    guidance: dict
    protocol: str = FIXED_RECIPE

    def __post_init__(self):
        if self.protocol not in (FIXED_RECIPE, BIDIR_RECIPE, *STREAMING_TEXT_RECIPES):
            raise ValueError('unsupported Ref inference protocol')
        conditioning_mode({'conditioning_mode': self.conditioning_mode})
        if self.protocol in (BIDIR_RECIPE, *STREAMING_TEXT_RECIPES) and self.conditioning_mode != STATIC_CONDITIONING_MODE:
            raise ValueError('bidirectional history requires static-image conditioning')
        causal_layout(self.window_size, self.rope_reference_latents)
        expected = dict(schedule=('common.diffusion.schedule.lerp', 'LinearInterpolationSchedule'),
            sampling_timesteps=('common.diffusion.timestep.sampling.trailing', 'TrailingSamplingTimesteps'),
            sampler=(PROJECT+'.modeling.sampling', 'EulerSampler'))
        if set(self.diffusion) != set(expected):
            raise ValueError('recipe diffusion needs exactly schedule, sampling_timesteps and sampler')
        for name, identity in expected.items():
            node = self.diffusion[name]
            if set(node) - {'module','class_name','args'} or (node['module'],node['class_name']) != identity:
                raise ValueError(f'unsupported inference {name} constructor')
        schedule = self.diffusion['schedule'].get('args', {})
        grid = self.diffusion['sampling_timesteps'].get('args', {})
        if schedule.get('pred_type') != 'x_0' or schedule.get('T') != 1.0 or grid.get('T') != 1.0:
            raise ValueError('Ref recipe requires normalized x0 linear flow')
        if type(grid.get('num_sampling_steps')) is not int or grid['num_sampling_steps'] < 1:
            raise ValueError('sampling steps must be positive')
        if grid.get('dynamic_shift', False):
            raise ValueError('Ref recipe requires a length-independent grid')
        if set(self.guidance) != {'mode','drop','scale'}:
            raise ValueError('recipe guidance needs mode, drop and scale')
        g = self.guidance
        if self.protocol in STREAMING_TEXT_RECIPES and g['mode'] == 'cfg':
            raise ValueError('streaming text supports plain or scale-one GFT, not CFG')
        if g['mode'] not in ('plain','gft','cfg') or g['drop'] not in ('proxy','prompt','prompt_proxy'):
            raise ValueError('unsupported inference guidance')
        if not math.isfinite(g['scale']) or g['scale'] < 0 or (g['mode'] != 'cfg' and g['scale'] != 1):
            raise ValueError('invalid effective inference guidance scale')

    @property
    def input_options(self):
        return dict(conditioning_mode=self.conditioning_mode, window_size=self.window_size,
                    rope_reference_latents=self.rope_reference_latents, causal=True)

    def to_dict(self):
        return asdict(self)

    def with_steps(self, steps):
        diffusion = deepcopy(self.diffusion)
        diffusion['sampling_timesteps']['args']['num_sampling_steps'] = steps
        return replace(self, diffusion=diffusion)

    @classmethod
    def from_training(cls, config):
        args = config['data']['args']
        stage = args['stage']
        if stage not in ('tf', 'tscd', 'dmd'):
            raise ValueError('causal recipe requires TF/TSCD/DMD, not SFT')
        source = config['validation'] if stage == 'tscd' else config['diffusion']
        diffusion = dict(schedule=deepcopy(config['diffusion']['schedule']),
                         sampling_timesteps=deepcopy(source['sampling_timesteps']),
                         sampler=deepcopy(source['sampler']))
        g = training_guidance_options(config['meta_model'], stage)
        scale = config.get('validation', {}).get('cfg_scale', g['scale']) if 'guidance' in config['meta_model'] else 1.
        value = cls(args.get('conditioning_mode','legacy'), args.get('window_size',6),
                    args.get('rope_reference_latents',37), diffusion,
                    dict(mode=g['mode'],drop=g['drop'],scale=float(scale)))
        if 'inference_recipe' in config and config['inference_recipe'] != value.to_dict():
            raise ValueError('training settings differ from the recorded inference_recipe')
        return value
