# SPDX-License-Identifier: Apache-2.0
"""Single-owner causal H3 sessions on Omni's bounded session graph."""

from typing import ClassVar

from pydantic import Field, model_validator

from sglang_omni.config.schema import (
    PipelineConfig,
    SessionControlEdgeConfig,
    SessionDataEdgeConfig,
    SessionNodeConfig,
    StageConfig,
)
from sglang_omni.models.minimax_h3.session import H3_CANCEL, H3_RGB24, RGB24_CHANNELS
from sglang_omni.proto.session import ControlEdge, SessionLimits

H3_STAGE = "h3"
H3_STAGE_FACTORY = "sglang_omni.models.minimax_h3.stages.create_h3_session_executor"
H3_RGB_DELIVERY_STAGE = "h3_rgb_delivery"
H3_RGB_DELIVERY_FACTORY = (
    "sglang_omni.models.minimax_h3.delivery.create_h3_rgb_delivery_executor"
)
PACKET_METADATA_BYTES = 4096
FRAME_METADATA_BYTES = 4096


class MiniMaxH3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MiniMaxH3CausalDiTModel"

    runtime_factory: str = Field(min_length=1)
    codec_overlap: bool = False
    cpu_delivery: bool = False
    max_state_bytes: int = Field(gt=0)
    max_width: int = Field(default=1280, gt=0)
    max_height: int = Field(default=720, gt=0)
    max_input_frames: int = Field(default=17, gt=0)
    max_output_frames: int = Field(default=34, gt=0)
    max_pending_chunks: int = Field(default=2, gt=0)
    max_output_chunks: int = Field(default=4, gt=0)
    operation_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    entry_stage: str = H3_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=H3_STAGE,
            process=H3_STAGE,
            factory_path=H3_STAGE_FACTORY,
            gpu=0,
            allow_child_processes=True,
            terminal=True,
        )
    ]
    session_nodes: dict[str, SessionNodeConfig] = {
        H3_STAGE: SessionNodeConfig(
            stages=[H3_STAGE],
            inputs=[
                SessionDataEdgeConfig(source="input", overflow="block"),
                SessionControlEdgeConfig(
                    kind="control", source="input", event=H3_CANCEL, preempt=True
                ),
            ],
        )
    }
    session_publish: dict[str, list[str]] = {H3_STAGE: [H3_RGB24]}

    @model_validator(mode="before")
    @classmethod
    def configure_cpu_delivery(cls, document: object) -> object:
        if not isinstance(document, dict) or document.get("cpu_delivery") is not True:
            return document
        else:
            document = document.copy()
        document.setdefault(
            "stages",
            [
                StageConfig(
                    name=H3_STAGE,
                    process=H3_STAGE,
                    factory_path=H3_STAGE_FACTORY,
                    gpu=0,
                    allow_child_processes=True,
                    next=H3_RGB_DELIVERY_STAGE,
                ),
                StageConfig(
                    name=H3_RGB_DELIVERY_STAGE,
                    process=H3_RGB_DELIVERY_STAGE,
                    factory_path=H3_RGB_DELIVERY_FACTORY,
                    terminal=True,
                ),
            ],
        )
        document.setdefault(
            "session_nodes",
            {
                H3_STAGE: SessionNodeConfig(
                    stages=[H3_STAGE, H3_RGB_DELIVERY_STAGE],
                    inputs=[
                        SessionDataEdgeConfig(source="input", overflow="block"),
                        SessionControlEdgeConfig(
                            kind="control",
                            source="input",
                            event=H3_CANCEL,
                            preempt=True,
                        ),
                    ],
                )
            },
        )
        return document

    @model_validator(mode="after")
    def validate_causal_owner(self) -> "MiniMaxH3PipelineConfig":
        graph = self.session_graph()
        route = (H3_STAGE, H3_RGB_DELIVERY_STAGE) if self.cpu_delivery else (H3_STAGE,)
        if self.cpu_delivery and not self.codec_overlap:
            raise ValueError("H3 cpu_delivery requires codec_overlap")
        else:
            pass
        if (
            len(self.stages) != len(route)
            or self.stages[0].name != H3_STAGE
            or self.stages[0].factory_path != H3_STAGE_FACTORY
            or self.entry_stage != H3_STAGE
            or any(stage.tp_size != 1 for stage in self.stages)
            or any(process.num_replicas != 1 for process in self.processes.values())
            or graph is None
            or set(graph.nodes) != {H3_STAGE}
            or graph.nodes[H3_STAGE].stages != route
            or graph.nodes[H3_STAGE].edges() != {"input": None}
            or graph.nodes[H3_STAGE].bound("input").overflow != "block"
            or graph.nodes[H3_STAGE].bound("input").feedback
            or graph.nodes[H3_STAGE].controls
            != (ControlEdge("input", H3_CANCEL, should_preempt=True),)
            or graph.publish != {H3_STAGE: (H3_RGB24,)}
        ):
            raise ValueError(
                "H3 requires one native owner, a blocking, unfiltered input edge, "
                "preemptive cancel and RGB publication"
            )
        else:
            pass
        if self.cpu_delivery and (
            self.stages[0].next != H3_RGB_DELIVERY_STAGE
            or self.stages[1].name != H3_RGB_DELIVERY_STAGE
            or self.stages[1].factory_path != H3_RGB_DELIVERY_FACTORY
            or self.stages[1].process == self.stages[0].process
            or self.stages[1].gpu is not None
            or self.stages[1].runtime_gpu_ids is not None
            or not self.stages[1].terminal
        ):
            raise ValueError(
                "H3 cpu_delivery requires its fixed route in separate GPU and CPU processes"
            )
        else:
            return self

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str | int | bool]:
        limits: dict[str, str | int | bool] = {
            "max_width": self.max_width,
            "max_height": self.max_height,
            "max_output_frames": self.max_output_frames,
            "max_state_bytes": self.max_state_bytes,
        }
        if stage_name == H3_STAGE:
            return limits | {
                "runtime_factory": self.runtime_factory,
                "max_input_frames": self.max_input_frames,
                "codec_overlap": self.codec_overlap,
                "cpu_delivery": self.cpu_delivery,
            }
        elif self.cpu_delivery and stage_name == H3_RGB_DELIVERY_STAGE:
            return limits
        else:
            raise ValueError(f"unknown H3 stage {stage_name}")

    def session_wire_limits(self) -> SessionLimits:
        frame_bytes = self.max_width * self.max_height * RGB24_CHANNELS
        input_bytes = (
            self.max_input_frames * (frame_bytes + FRAME_METADATA_BYTES)
            + PACKET_METADATA_BYTES
        )
        output_bytes = (
            self.max_output_frames * (frame_bytes + FRAME_METADATA_BYTES)
            + PACKET_METADATA_BYTES
        )
        return SessionLimits(
            max_modalities=1,
            max_pending_chunks=self.max_pending_chunks,
            max_pending_bytes=input_bytes * self.max_pending_chunks,
            max_output_chunks=self.max_output_chunks,
            max_output_bytes=output_bytes * self.max_output_chunks,
            max_chunk_bytes=input_bytes,
            operation_timeout_s=self.operation_timeout_seconds,
            max_in_flight_appends=2 if self.codec_overlap else 1,
        )


EntryClass = MiniMaxH3PipelineConfig
Variants = {"session": MiniMaxH3PipelineConfig}
