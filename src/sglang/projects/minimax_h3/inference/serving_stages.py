from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class StageSpec:
    name: str
    payload_owner: Literal['rank0', 'encoder'] | None = None
    prefetch_next: bool = False


_COLLECT = StageSpec('collect')
_PLANS = {
    'start': (StageSpec('static_condition', 'rank0'), StageSpec('first_image', 'encoder'),
              StageSpec('open_recipe'), _COLLECT),
    'chunk': (StageSpec('text_condition'), StageSpec('proxy_encode', 'encoder'),
              StageSpec('render'), StageSpec('decode', prefetch_next=True), _COLLECT),
}
STAGES = {op: tuple(stage.name for stage in plan) for op, plan in _PLANS.items()}
STAGE_SPECS = {stage.name: stage for plan in _PLANS.values() for stage in plan}

RAY_STAGE_COMPLETION = 'GPU complete'
RAY_GPU_BOUNDARY = 'all ranks synchronized, including D2H and codec prefetch'


@dataclass(frozen=True)
class WorkKey:
    epoch: int
    session_id: str
    op: str
    chunk_id: int = -1

    def check_next(self, following):
        if (self.op != 'chunk' or following !=
                WorkKey(self.epoch, self.session_id, 'chunk', self.chunk_id + 1)):
            raise RuntimeError('prefetch must be the immediate next chunk in this session')


@dataclass
class PrefetchedProxy:
    key: WorkKey
    item: dict
    timings: dict
    profile: dict | None = None


@dataclass(frozen=True)
class CpuOutput:
    key: WorkKey
    nbytes: int
    read: Callable[[], tuple[bytes | memoryview, dict]]

    def __len__(self):
        return self.nbytes


def serialize_rgb(cpu):
    if cpu is None:
        return b''
    if cpu.device.type != 'cpu':
        raise RuntimeError('output serialization requires completed CPU storage')
    return cpu.numpy().tobytes()


class StageCursor:
    def __init__(self):
        self.epoch = 0
        self.session_id = None
        self.next_chunk = 0
        self.current = None
        self.index = 0

    def check(self, key, stage):
        plan = STAGES[key.op]
        if self.current is None:
            if stage != plan[0]:
                raise RuntimeError('stage must begin at the first operation')
            if key.op == 'start':
                if self.session_id is not None or key.epoch <= self.epoch:
                    raise RuntimeError('session is active or epoch is stale')
            elif (key.epoch != self.epoch or key.session_id != self.session_id
                    or key.chunk_id != self.next_chunk):
                raise RuntimeError('chunk identity differs from the active session')
        elif key != self.current or stage != plan[self.index]:
            raise RuntimeError('stage identity or order differs from the active operation')

    def completed(self, key):
        if self.current is None:
            self.current = key
            if key.op == 'start':
                self.epoch, self.session_id = key.epoch, key.session_id
                self.next_chunk = 0
        self.index += 1
        if self.index == len(STAGES[key.op]):
            if key.op == 'chunk':
                self.next_chunk += 1
            self.current, self.index = None, 0

    def reset(self):
        self.session_id = self.current = None
        self.next_chunk = self.index = 0
