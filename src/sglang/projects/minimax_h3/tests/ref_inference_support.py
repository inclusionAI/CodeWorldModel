"""AST-load the frozen pre-refactor stream with the pinned RAVEN primitives."""
import ast
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from common.distributed.ops import get_device
from common.distributed.unified_parallel import (get_unified_parallel_world_size,
                                                 is_unified_parallel_initialized)
from common.seed import RandomState
from projects.minimax_h3.data.ref_latent import (LEGACY_CONDITIONING_MODE,
    STATIC_CONDITIONING_MODE, conditioning_mode as get_conditioning_mode)
from projects.minimax_h3.modeling.causal_layout import CausalLayout as _Layout
from projects.minimax_h3.modeling.ref_condition import CWMRefConditionMixin, GroupedKVCache
from projects.minimax_h3.modeling.ref_guidance import RefGuidanceMixin
from projects.minimax_h3.modeling.packed_tokens import (minimax_h3_patchify_video_latent,
                                                     minimax_h3_unpatchify_video_tokens)

ROOT = Path(__file__).resolve().parents[1]
_PATCH_SIZE = (1,2,2)
MiniMaxH3RefSFT = type('SFT', (), {})


def legacy_class():
    path = ROOT/'tests/fixtures/ref_stream_v0.py'
    tree = ast.parse(path.read_text())
    methods = [method for cls in tree.body if isinstance(cls,ast.ClassDef)
               for method in cls.body if isinstance(method,ast.FunctionDef)]
    cls = ast.ClassDef(name='Legacy',bases=[ast.Name(id='RefGuidanceMixin',ctx=ast.Load()),
        ast.Name(id='CWMRefConditionMixin',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    namespace = dict(globals())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['Legacy']
