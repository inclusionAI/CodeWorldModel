from contextlib import contextmanager
from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.raw_recipe import RawLatentRecipe
from projects.minimax_h3.inference.serving import (
    Request, ResidentBackend, SessionController, settings_receipt)
from projects.minimax_h3.tests.test_inference_settings import artifact


def raw_recipe():
    value = artifact()['recipe']
    return RawLatentRecipe(value['conditioning_mode'], value['rope_reference_latents'],
                           value['diffusion'], value['guidance'])


class ServingFusionTests(unittest.TestCase):
    def test_shared_constructor_forwards_flag_and_restores_grad_scope(self):
        for selected in (None, False, True):
            for raises in (False, True):
                with self.subTest(selected=selected, raises=raises):
                    backend = ResidentBackend.__new__(ResidentBackend)
                    dit = {} if selected is None else dict(fuse_first_denoise=selected)
                    backend.config = SimpleNamespace(inference=SimpleNamespace(runtime=SimpleNamespace(
                        codec=SimpleNamespace(decoder='causal'), dit=dit)))
                    if selected is None:
                        del backend.config.inference.runtime.dit
                    backend.backbone = SimpleNamespace(_h3_inference=dict(delayed_hc=True))
                    backend.recipe = raw_recipe()
                    backend.metadata = dict(seed=7, condition_seed=11)
                    backend.static_document, backend.first_latent, backend.profiler = object(), object(), object()
                    backend.streaming_text = False
                    first_latent = backend.first_latent
                    state = dict(grad_enabled=True)
                    @contextmanager
                    def no_grad():
                        previous = state['grad_enabled']
                        state['grad_enabled'] = False
                        try:
                            yield
                        finally:
                            state['grad_enabled'] = previous
                    output = object()
                    def construct(backbone, recipe, static, **kwargs):
                        self.assertFalse(state['grad_enabled'])
                        self.assertIs(backbone, backend.backbone)
                        self.assertIs(recipe, backend.recipe)
                        self.assertIs(static, backend.static_document)
                        self.assertEqual(kwargs, dict(seed=7, condition_seed=11,
                            first_frame=first_latent, profiler=backend.profiler, encode_text=None,
                            fuse_first_denoise=bool(selected)))
                        if raises:
                            raise RuntimeError('constructor failure')
                        return output
                    modules = {'torch': SimpleNamespace(no_grad=no_grad),
                        'projects.minimax_h3.inference.raw_latent': SimpleNamespace(RawLatentSession=construct)}
                    with patch.dict(sys.modules, modules):
                        if raises:
                            with self.assertRaisesRegex(RuntimeError, 'constructor failure'):
                                backend.open_recipe()
                            self.assertIs(backend.first_latent, first_latent)
                        else:
                            backend.open_recipe()
                            self.assertIs(backend.outputs, output)
                            self.assertIsNone(backend.first_latent)
                    self.assertTrue(state['grad_enabled'])

    def test_start_receipt_preserves_selected_flag_without_mutating_settings(self):
        recipe = raw_recipe()
        bundle = dict(encoders=dict(recipe=dict(height=1, width=1, system_prompt='frozen')))
        for enabled in (False, True):
            settings = dict(artifacts=[], deployment=dict(options=dict(dit=dict(fuse_first_denoise=enabled))))
            before = deepcopy(settings)
            controller = SessionController({'fixed': (bundle, recipe)}, settings=settings_receipt(settings))
            request = Request('start', dict(session_id='one', artifact_id='fixed', seed=7,
                                           condition_seed=11, prompt='frozen'), bytes(3))
            controller.prepare(request)
            reply = controller.finish(request, b'', {})
            receipt = reply.metadata['effective_settings']
            self.assertIs(receipt['deployment']['options']['dit']['fuse_first_denoise'], enabled)
            self.assertEqual(receipt['scope'], 'session')
            self.assertEqual(settings, before)


if __name__ == '__main__':
    unittest.main()
