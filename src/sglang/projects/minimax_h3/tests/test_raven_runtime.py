"""Exercise the pinned framework and package overlay in an isolated process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def probe():
    import torch
    import torch.distributed as dist
    from common.config import CfgNode
    from common.engine import BaseEngine
    import projects.minimax_h3.inference.artifact as artifact
    import projects
    from safetensors.torch import save_file
    from projects.minimax_h3.tools.infer_ref_fixed_history import runtime_config
    from projects.minimax_h3.modeling.ref_recipe import RefRecipe
    from projects.minimax_h3.tests.test_ref_inference import recipe_config
    from types import SimpleNamespace
    assert Path(artifact.__file__).resolve() == ROOT/'inference/artifact.py'
    assert projects.minimax_h3.__path__ == [str(ROOT)]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root/'tiny.safetensors'
        model = torch.nn.Linear(3,2)
        save_file(model.state_dict(),weights)
        bundle = dict(model={},weights=dict(path=str(weights)),encoders=dict(checkpoint=str(root)),
            deployment=dict(distributed=dict(up_size=[1]),active_up_size=1,placement={}))
        args = SimpleNamespace(inference_config=None,decode_mode='strict',output_dir=str(root/'output'),
            seed=9,condition_seed=None,artifact=str(weights),steps=None)
        recipe = RefRecipe.from_training(recipe_config())
        config = runtime_config(args,bundle,recipe,{})
        assert config.models.backbone.placement.class_name == 'ResidentInferencePlacement'
        fallback = root/'fsdp.yaml'
        fallback.write_text('resident_weights: false\n')
        args.inference_config = str(fallback)
        fallback_config = runtime_config(args,bundle,recipe,{})
        assert fallback_config.models.backbone.placement == bundle['deployment']['placement']
        assert fallback_config.distributed == config.distributed
        assert config.engine.up_size.train == 1
        assert not {'data','meta_model','diffusion','validation'} & set(config)
        config.models.backbone = CfgNode(dict(module='torch.nn',class_name='Linear',args=dict(in_features=3,out_features=2),
            meta_init=False,weight=dict(module='common.model.weights',class_name='DefaultWeightLoader',path=str(weights),strict=True),
            runtime=dict(training=False,requires_grad=False,param_dtype='bfloat16'),placement=dict(device='cpu')))
        dist.init_process_group('gloo',init_method=(root/'pg').as_uri(),rank=0,world_size=1)
        try:
            driver = object()
            engine = BaseEngine(config,driver)
            assert engine.meta is driver
            actual = engine.models['backbone']
            assert not actual.training and not any(p.requires_grad for p in actual.parameters())
            torch.testing.assert_close(actual(torch.ones(1,3)),model(torch.ones(1,3)),rtol=0,atol=0)
            assert not hasattr(engine,'optimizers') and not hasattr(engine,'dataloader')
        finally:
            dist.destroy_process_group()
    forbidden = [name for name in sys.modules if name.startswith(('projects.minimax_h3.meta_models','engines.'))]
    assert not forbidden,forbidden
    print(json.dumps(dict(overlay='current checkout',builder='real RAVEN',training_modules=forbidden)))


class RavenRuntimeTests(unittest.TestCase):
    def test_real_builder_and_overlay(self):
        result = subprocess.run([sys.executable,str(ROOT/'tools/launch.py'),
            'projects.minimax_h3.tests.test_raven_runtime','--probe'],cwd=tempfile.gettempdir(),
            text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('"training_modules": []',result.stdout)


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe']:
        probe()
    else:
        unittest.main()
