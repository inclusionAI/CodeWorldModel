"""Inference-only latent units. The training/v1 recipe remains closed."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

if __package__.startswith('projects.'):
    from ..modeling.ref_recipe import BIDIR_RECIPE, FIXED_RECIPE, STREAMING_TEXT_RECIPES, RefRecipe, STATIC_CONDITIONING_MODE
else:
    from modeling.ref_recipe import BIDIR_RECIPE, FIXED_RECIPE, STREAMING_TEXT_RECIPES, RefRecipe, STATIC_CONDITIONING_MODE

RAW_FIXED = FIXED_RECIPE.replace('/v1', '/v2')
RAW_BIDIR = BIDIR_RECIPE.replace('/v1', '/v2')
RAW_STREAMING_TEXT = tuple(p.replace('/v1', '/v2') for p in STREAMING_TEXT_RECIPES)
CWM_CAUSAL = 'cwm_causal/v1'
RAW_PROTOCOLS = (RAW_FIXED, RAW_BIDIR, *RAW_STREAMING_TEXT)
ALL_STREAMING_TEXT = (*STREAMING_TEXT_RECIPES, *RAW_STREAMING_TEXT)


@dataclass(frozen=True)
class RawLatentRecipe:
    conditioning_mode: str
    rope_reference_latents: int
    diffusion: dict
    guidance: dict
    protocol: str = RAW_FIXED
    sink_latents: int = 2
    recent_latents: int = 10
    bootstrap_latents: int = 9
    current_latent_schedule: tuple[int, ...] = (3, 2)
    segment_boundaries: tuple[int, ...] = ()
    position_policy: str = 'absolute_native_v1'

    def __post_init__(self):
        if self.protocol not in (*RAW_PROTOCOLS, CWM_CAUSAL):
            raise ValueError('unsupported raw-latent inference protocol')
        if self.conditioning_mode != STATIC_CONDITIONING_MODE:
            raise ValueError('raw-latent inference requires static-image conditioning')
        for name in ('sink_latents', 'recent_latents', 'bootstrap_latents'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == 'bootstrap_latents' else 0):
                raise ValueError(f'{name} must be a {"positive" if name == "bootstrap_latents" else "nonnegative"} integer')
        if self.sink_latents > self.bootstrap_latents:
            raise ValueError('sink_latents must not exceed bootstrap_latents')
        schedule = self.current_latent_schedule
        if not isinstance(schedule, (list, tuple)) or not schedule or any(type(n) is not int or n < 1 for n in schedule):
            raise ValueError('current_latent_schedule needs positive integers')
        boundaries = self.segment_boundaries
        if (not isinstance(boundaries, (list, tuple)) or any(type(n) is not int or n < 1 for n in boundaries)
                or list(boundaries) != sorted(set(boundaries))):
            raise ValueError('segment_boundaries must be increasing positive latent positions')
        if boundaries and boundaries[0] < self.bootstrap_latents:
            raise ValueError('segment boundaries must not split bootstrap')
        cwm = self.protocol == CWM_CAUSAL
        if cwm and (self.position_policy != 'cwm_frame_aligned_v1'
                or (self.rope_reference_latents, self.sink_latents, self.recent_latents, self.bootstrap_latents) != (47, 1, 12, 9)
                or tuple(schedule) != (5,) or tuple(boundaries) != (45, 70)
                or self.guidance != dict(mode='plain', drop='proxy', scale=1.0)):
            raise ValueError('CWM requires fixed H3 history with aligned47 RoPE')
        if not cwm and self.position_policy != 'absolute_native_v1':
            raise ValueError('unsupported raw-latent position policy')
        RefRecipe(self.conditioning_mode, 0, self.rope_reference_latents,
                  self.diffusion, self.guidance, FIXED_RECIPE if cwm else self.protocol.replace('/v2', '/v1'))
        object.__setattr__(self, 'current_latent_schedule', tuple(schedule))
        object.__setattr__(self, 'segment_boundaries', tuple(boundaries))

    @property
    def bidirectional(self):
        return self.protocol in (RAW_BIDIR, RAW_STREAMING_TEXT[1])

    @property
    def streaming_text(self):
        return self.protocol in RAW_STREAMING_TEXT

    def to_dict(self):
        value = asdict(self)
        for key in ('current_latent_schedule', 'segment_boundaries'):
            value[key] = list(value[key])
        return value

    def with_steps(self, steps):
        diffusion = deepcopy(self.diffusion)
        diffusion['sampling_timesteps']['args']['num_sampling_steps'] = steps
        return replace(self, diffusion=diffusion)


def parse_recipe(record):
    if not isinstance(record, dict):
        raise ValueError('recipe must be an object')
    cls = RawLatentRecipe if record.get('protocol') in (*RAW_PROTOCOLS, CWM_CAUSAL) else RefRecipe
    try:
        return cls(**record)
    except TypeError as exc:
        raise ValueError(f'invalid {cls.__name__} fields') from exc


def validate_online_guidance(recipe):
    if (isinstance(recipe, RawLatentRecipe) and recipe.guidance['scale'] != 1
            and recipe.guidance['drop'] != 'proxy'):
        raise ValueError('raw-latent file and serving entrypoints support proxy CFG only; '
                         'prompt CFG requires an explicit weak document in RawLatentSession')


def window_arguments(parser):
    parser.add_argument('--window-size', type=int, help='Legacy v1 history blocks only')
    for name in ('sink-latents', 'recent-latents', 'bootstrap-latents'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--current-latent-schedule', type=int, nargs='+')
    parser.add_argument('--segment-boundaries', type=int, nargs='+')


def window_overrides(args, record):
    names = ('sink_latents', 'recent_latents', 'bootstrap_latents',
             'current_latent_schedule', 'segment_boundaries', 'window_size')
    values = {name:getattr(args, name) for name in names if getattr(args, name, None) is not None}
    raw = record['protocol'] in RAW_PROTOCOLS
    if raw and 'window_size' in values or not raw and set(values)-{'window_size'}:
        raise ValueError('window-size is v1-only; raw latent options require a v2 artifact')
    if values:
        parse_recipe(dict(record, **values))
    return values
