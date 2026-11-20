# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)


@dataclass
class MiniMaxH3CausalDiTArchConfig(MiniMaxH3DiTArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^dit\.(.*)$": r"\1",
            **MiniMaxH3DiTArchConfig().param_names_mapping,
        }
    )
    # Training exports already store q_all/k_all/v_all after the released reorder.
    checkpoint_qkv_layout: str | None = "native"


@dataclass
class MiniMaxH3CausalDiTConfig(MiniMaxH3DiTConfig):
    arch_config: MiniMaxH3CausalDiTArchConfig = field(
        default_factory=MiniMaxH3CausalDiTArchConfig
    )
