import copy
import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file
from common.config import CfgNode
from common.model.adapters import PeftLoraAdapter
from projects.minimax_h3.inference.artifact import runtime_backbone, validate_adapter, verify_export
from projects.minimax_h3.inference.unmerged_weights import FullUnmergedWeightLoader
from projects.minimax_h3.modeling.transformer.model import _Linear


def adapter():
    return dict(r=2, lora_alpha=2, target_modules=['qkv_proj'], custom_module_mapping=[dict(
        source=dict(module='projects.minimax_h3.modeling.transformer.model', class_name='_Linear'),
        target=dict(module='projects.minimax_h3.modeling.lora', class_name='MiniMaxH3LoraLinear'))])


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Module()
        self.dit.qkv_proj = _Linear(4, 4, bias=False, params_dtype=torch.bfloat16)
        self.dit.video_patch_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class UnmergedTests(unittest.TestCase):
    def test_export_records_closed_adapter_without_stage_dependency(self):
        from projects.minimax_h3.inference.artifact import UNMERGED_SCHEMA, export_bundle, load_bundle
        from projects.minimax_h3.tests.test_ref_bidir_artifact import assets
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, path, encoders = assets(root)
            training.models.backbone.adapter = dict(adapter(), module='common.model.adapters',
                class_name='PeftLoraAdapter', stage_weight='/offline/stage', lora_dropout=0)
            model = PeftLoraAdapter(CfgNode(adapter()))(dict(name='tiny', model=Tiny()))['model']
            save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()}, str(path))
            bundle = export_bundle(training, path, root, encoders, provenance={},
                weights_format='raven_full_unmerged_lora')
            self.assertEqual(bundle['schema'], UNMERGED_SCHEMA)
            self.assertEqual(bundle['model']['adapter'], adapter())
            import json
            record = root/'artifact.json'
            record.write_text(json.dumps(bundle))
            self.assertEqual(load_bundle(record)[0], bundle)
            for key, value in (('alpha_pattern', {'qkv_proj': 4}),
                               ('rank_pattern', {'qkv_proj': 1}), ('modules_to_save', ['video_patch_proj'])):
                training.models.backbone.adapter[key] = value
                with self.assertRaisesRegex(ValueError, 'per-module LoRA'):
                    export_bundle(training, path, root, encoders, provenance={},
                        weights_format='raven_full_unmerged_lora')
                del training.models.backbone.adapter[key]

    def test_full_roundtrip_keeps_unmerged_arithmetic_and_rejects_missing_base(self):
        torch.manual_seed(123)
        model = PeftLoraAdapter(CfgNode(adapter()))(dict(name='tiny', model=Tiny()))['model']
        with torch.no_grad():
            for value in model.parameters():
                value.data = torch.randn_like(value).to(torch.bfloat16)
        x = torch.randn(8, 4, dtype=torch.bfloat16)
        expected = model.base_model.model.dit.qkv_proj(x)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'full.safetensors'
            values = {k:v.detach().contiguous() for k,v in model.state_dict().items()}
            save_file(values, str(path))
            verify_export(path, format='raven_full_unmerged_lora')
            loaded = FullUnmergedWeightLoader(CfgNode(dict(path=str(path), adapter=adapter())))(dict(name='tiny', model=Tiny()))['model']
            actual = loaded.base_model.model.dit.qkv_proj(x)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            partial = {k:v for k,v in values.items() if 'lora_' in k}
            save_file(partial, str(Path(temporary)/'partial.safetensors'))
            with self.assertRaises(ValueError):
                verify_export(Path(temporary)/'partial.safetensors', format='raven_full_unmerged_lora')
            values.pop('base_model.model.dit.video_patch_proj.weight')
            save_file(values, str(Path(temporary)/'missing.safetensors'))
            with self.assertRaises(RuntimeError):
                FullUnmergedWeightLoader(CfgNode(dict(path=str(Path(temporary)/'missing.safetensors'),
                    adapter=adapter())))(dict(name='tiny', model=Tiny()))

    def test_adapter_contract_rejects_dynamic_factories_and_stage_weights(self):
        validate_adapter(adapter())
        for change in (dict(r=True), dict(lora_alpha=0), dict(target_modules=['other']),
                       dict(stage_weight='/untrusted/path'), dict(custom_module_mapping=[])):
            value = dict(adapter(), **change)
            with self.assertRaises(ValueError):
                validate_adapter(value)

    def test_backbone_applies_adapter_once_inside_the_weight_loader(self):
        bundle = dict(model=dict(module='module', class_name='Model', args={}, adapter=adapter()),
                      weights=dict(path='/full.safetensors'))
        original = copy.deepcopy(bundle)
        resolved = runtime_backbone(bundle)
        self.assertNotIn('adapter', resolved)
        self.assertEqual(resolved['weight']['class_name'], 'FullUnmergedWeightLoader')
        self.assertEqual(resolved['weight']['adapter'], adapter())
        self.assertEqual(bundle, original)
        del bundle['model']['adapter']
        self.assertEqual(runtime_backbone(bundle)['weight']['class_name'], 'DefaultWeightLoader')


if __name__ == '__main__':
    unittest.main()
