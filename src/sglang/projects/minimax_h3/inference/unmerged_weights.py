"""Load a complete unmerged checkpoint before runtime dtype and placement."""
from pathlib import Path

from .artifact import validate_adapter


class FullUnmergedWeightLoader:
    def __init__(self, config):
        self.config = config
        validate_adapter(dict(config.adapter))

    def __call__(self, state):
        from common.config import CfgNode
        from common.model.adapters import PeftLoraAdapter
        from common.model.weights import load_state_dict_file

        state = PeftLoraAdapter(CfgNode(dict(self.config.adapter)))(state)
        values = load_state_dict_file(Path(self.config.path))
        state['model'].load_state_dict(values, strict=True, assign=True)
        if any(value.is_meta for value in list(state['model'].parameters()) + list(state['model'].buffers())):
            raise ValueError('full unmerged checkpoint left unmaterialized tensors')
        return state
