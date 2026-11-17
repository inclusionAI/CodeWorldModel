"""Offline export and online validation of causal Ref inference bundles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

from projects.minimax_h3.modeling.ref_recipe import BIDIR_RECIPE, STREAMING_TEXT_RECIPES, PROJECT, RefRecipe
from .raw_recipe import RAW_PROTOCOLS, RawLatentRecipe, parse_recipe

SCHEMA = 'ref_inference_artifact/v1'
UNMERGED_SCHEMA = 'ref_inference_artifact/v2'
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_encoders(recipe, checkpoint):
    if recipe.get('conditioning_mode') != 'static_image_qwen_proxy_vae_v1' or recipe.get('posterior') not in ('mean', 'sample_fp16_seed42'):
        raise ValueError('file inference requires static-image conditioning and posterior mean')
    if not recipe.get('system_prompt', '').strip() or not recipe.get('model_hashes'):
        raise ValueError('encoder recipe needs a fixed prompt and frozen asset hashes')
    base = Path(checkpoint).resolve()
    for relative, digest in recipe['model_hashes'].items():
        path = (base/relative).resolve()
        if not path.is_relative_to(base) or sha256(path) != digest:
            raise ValueError(f'frozen encoder differs from training: {relative}')


def verify_export(path, *, format='raven_full_or_merged'):
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 2 <= length <= 64 << 20:
            raise ValueError('expected a full or merged runtime safetensors export')
        header = json.loads(handle.read(length))
    keys = set(header)-{'__metadata__'}
    if format == 'raven_full_unmerged_lora':
        if ('base_model.model.dit.video_patch_proj.weight' not in keys
                or not any('.base_layer.weight' in k for k in keys)
                or not any('.lora_A.default.weight' in k for k in keys)
                or not any('.lora_B.default.weight' in k for k in keys)
                or any(not k.startswith('base_model.model.dit.') for k in keys)):
            raise ValueError('expected complete frozen base and unmerged LoRA, not adapter-only weights')
        return
    if format != 'raven_full_or_merged':
        raise ValueError('unsupported generator export format')
    if (not keys or not all(k.startswith('dit.') and 'lora_' not in k for k in keys)
            or 'dit.video_patch_proj.weight' not in keys or not any(k.startswith('dit.blocks.0.') for k in keys)):
        raise ValueError('expected full/merged dit.* keys, not released or adapter-only weights')


def validate_bundle(bundle):
    if set(bundle) != {'schema','model','weights','encoders','recipe','deployment','provenance'} or bundle['schema'] not in (SCHEMA, UNMERGED_SCHEMA):
        raise ValueError('unsupported inference artifact schema')
    recipe = parse_recipe(bundle['recipe'])
    if bundle['encoders']['recipe'].get('posterior') == 'sample_fp16_seed42' and recipe.protocol != 'cwm_causal/v1':
        raise ValueError('sampled FP16-rounded conditioning requires CWM causal inference')
    provenance = bundle['provenance']
    if recipe.protocol in (BIDIR_RECIPE, *STREAMING_TEXT_RECIPES, *RAW_PROTOCOLS) or provenance.get('recipe_selection') == 'explicit_override':
        if (provenance.get('recipe_selection') != 'explicit_override'
                or provenance.get('training_protocol_match') != 'unverified'
                or provenance.get('effective_recipe') != recipe.to_dict()
                or not {'source_stage', 'source_meta_model', 'source_recorded_recipe'} <= provenance.keys()):
            raise ValueError('explicit inference recipe requires source and effective recipe provenance')
        for key in ('training_config_sha256', 'recipe_override_sha256'):
            digest = provenance.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'explicit inference recipe requires {key}')
    if recipe.conditioning_mode != bundle['encoders']['recipe']['conditioning_mode']:
        raise ValueError('model and encoder conditioning contracts differ')
    model = bundle['model']
    unmerged = bundle['schema'] == UNMERGED_SCHEMA
    if set(model) != ({'module','class_name','args','adapter'} if unmerged else {'module','class_name','args'}) or model['module'] != PROJECT+'.modeling.build' or model['class_name'] not in (
            'MiniMaxH3CausalX0DiT', 'MiniMaxH3CausalX0DiTSP'):
        raise ValueError('artifact requires a supported causal x0 generator')
    expected_format = 'raven_full_unmerged_lora' if unmerged else 'raven_full_or_merged'
    if set(bundle['weights']) != {'format','path','sha256'} or bundle['weights']['format'] != expected_format:
        raise ValueError('unsupported generator export format')
    if unmerged:
        validate_adapter(model['adapter'])
    deployment = bundle['deployment']
    if set(deployment) != {'distributed','placement','active_up_size'}:
        raise ValueError('deployment requires distributed, placement and active_up_size settings')
    placement = deployment['placement']
    if placement.get('plugins') != [dict(module=PROJECT+'.modeling.fp32_placement',
                                         class_name='MiniMaxH3FP32ModulePlacement')]:
        raise ValueError('inference FSDP requires the H3 FP32 placement plugin')
    if set(placement)-{'weight_dtype','plugins','fsdp'} or set(deployment['distributed'])-{'up_size'}:
        raise ValueError('unsupported inference deployment settings')
    fsdp = placement.get('fsdp', {})
    if (placement.get('weight_dtype') != 'float32' or fsdp.get('param_dtype') != 'bfloat16'
            or fsdp.get('reduce_dtype') != 'float32' or fsdp.get('cast_forward_inputs') is not False
            or fsdp.get('wrap_modules') != ['CausalMiniMaxH3DiTBlock','MiniMaxH3TokenRefinerBlock']
            or set(fsdp)-{'param_dtype','reduce_dtype','cast_forward_inputs','wrap_modules','shard_size','cpu_offload'}):
        raise ValueError('unsupported H3 FP32/FSDP placement contract')
    sizes = deployment['distributed'].get('up_size')
    if not isinstance(sizes,list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError('deployment up_size needs positive integer group sizes')
    if type(deployment['active_up_size']) is not int or deployment['active_up_size'] not in sizes:
        raise ValueError('active_up_size must be registered in distributed.up_size')
    if model['class_name'] == 'MiniMaxH3CausalX0DiT' and any(n>1 for n in sizes):
        raise ValueError('sequence parallelism requires the SP generator')
    return recipe


def validate_adapter(adapter):
    fields = {'r', 'lora_alpha', 'target_modules', 'custom_module_mapping'}
    if not isinstance(adapter, dict) or set(adapter) != fields:
        raise ValueError('unmerged artifact requires the closed H3 LoRA adapter contract')
    if any(type(adapter[k]) is not int or not 1 <= adapter[k] <= 4096 for k in ('r', 'lora_alpha')):
        raise ValueError('LoRA rank and alpha must be positive integers up to 4096')
    targets = adapter['target_modules']
    if (not isinstance(targets, list) or not targets or any(type(t) is not str for t in targets)
            or len(set(targets)) != len(targets) or set(targets)-{'qkv_proj','out_proj','fc1','fc2'}):
        raise ValueError('unsupported H3 LoRA target modules')
    mapping = [dict(source=dict(module=PROJECT+'.modeling.transformer.model', class_name='_Linear'),
                    target=dict(module=PROJECT+'.modeling.lora', class_name='MiniMaxH3LoraLinear'))]
    if adapter['custom_module_mapping'] != mapping:
        raise ValueError('unmerged artifact requires the H3 LoRA projection implementation')


def runtime_backbone(bundle):
    """Resolve a checked artifact to the fixed model-build pipeline."""
    model = deepcopy(bundle['model'])
    adapter = model.pop('adapter', None)
    weight = dict(module='common.model.weights', class_name='DefaultWeightLoader',
                  path=bundle['weights']['path'], strict=True)
    if adapter is not None:
        weight.update(module=PROJECT+'.inference.unmerged_weights',
                      class_name='FullUnmergedWeightLoader', adapter=adapter)
    model.update(meta_init=True, weight=weight)
    return model


def export_bundle(training, weights, checkpoint, encoder_recipe, *, provenance,
                  recipe_override=None, recipe_override_sha256=None,
                  weights_format='raven_full_or_merged'):
    stage = training['data']['args']['stage']
    meta = training['meta_model']
    provenance = deepcopy(provenance)
    if recipe_override is None:
        if meta.get('module') != PROJECT+'.meta_models.minimax_h3_ref2va' or meta.get('class_name') != 'MiniMaxH3Ref'+stage.upper():
            raise ValueError('custom training meta-model needs an explicit inference recipe override')
        recipe = RefRecipe.from_training(training)
        provenance['recipe_selection'] = 'inferred_local_training'
    else:
        cls = RawLatentRecipe if isinstance(recipe_override, dict) and recipe_override.get('protocol') in (*RAW_PROTOCOLS, 'cwm_causal/v1') else RefRecipe
        if not isinstance(recipe_override, dict) or set(recipe_override) != set(cls.__dataclass_fields__):
            raise ValueError('recipe override must be a complete RefRecipe JSON record')
        recipe = parse_recipe(deepcopy(recipe_override))
        provenance.update(recipe_selection='explicit_override', training_protocol_match='unverified',
            recipe_override_sha256=recipe_override_sha256, source_stage=stage,
            source_meta_model=deepcopy(meta), source_recorded_recipe=deepcopy(training.get('inference_recipe')),
            effective_recipe=recipe.to_dict())
    source = training['models']['backbone']
    model = {key: deepcopy(source[key]) for key in ('module','class_name','args')}
    unmerged = weights_format == 'raven_full_unmerged_lora'
    if unmerged:
        adapter = source.get('adapter', {})
        if adapter.get('module') != 'common.model.adapters' or adapter.get('class_name') != 'PeftLoraAdapter':
            raise ValueError('full unmerged export requires the H3 PEFT LoRA source configuration')
        if any(adapter.get(key) for key in ('alpha_pattern', 'rank_pattern', 'modules_to_save')):
            raise ValueError('full unmerged export does not support per-module LoRA patterns or modules_to_save')
        model['adapter'] = {key: deepcopy(adapter[key]) for key in
            ('r', 'lora_alpha', 'target_modules', 'custom_module_mapping') if key in adapter}
    bundle = dict(schema=UNMERGED_SCHEMA if unmerged else SCHEMA, model=model,
        weights=dict(format=weights_format, path=str(Path(weights).resolve()), sha256=sha256(weights)),
        encoders=dict(checkpoint=str(Path(checkpoint).resolve()), recipe=deepcopy(encoder_recipe)),
        recipe=recipe.to_dict(), deployment=dict(distributed=deepcopy(training['distributed']),
            active_up_size=training['engine']['up_size']['train'],
            placement=deepcopy(source['placement'])), provenance=provenance)
    validate_bundle(bundle)
    verify_export(weights, format=weights_format)
    verify_encoders(encoder_recipe, checkpoint)
    return bundle


def load_bundle(path):
    bundle = json.loads(Path(path).read_text())
    recipe = validate_bundle(bundle)
    verify_bundle_assets(bundle)
    return bundle, recipe


def verify_bundle_assets(bundle):
    weights = bundle['weights']
    if not Path(weights['path']).is_absolute() or not Path(bundle['encoders']['checkpoint']).is_absolute():
        raise ValueError('artifact asset paths must be absolute; re-export after relocation')
    if sha256(weights['path']) != weights['sha256']:
        raise ValueError('generator differs from the inference artifact')
    verify_export(weights['path'], format=weights['format'])
    verify_encoders(bundle['encoders']['recipe'], bundle['encoders']['checkpoint'])
