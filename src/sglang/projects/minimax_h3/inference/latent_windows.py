"""Pure absolute-latent window planning, independent of proxy and decoder batches."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LatentRange:
    start: int
    stop: int

    @property
    def count(self):
        return self.stop - self.start

    def record(self):
        return dict(start=self.start, stop=self.stop, count=self.count)


@dataclass(frozen=True)
class WindowPlan:
    index: int
    bootstrap: bool
    sink: LatentRange
    requested_recent: LatentRange
    dynamic_recent: LatentRange
    current: LatentRange
    eos: bool

    def record(self):
        value = asdict(self)
        for key in ('sink', 'requested_recent', 'dynamic_recent', 'current'):
            value[key] = getattr(self, key).record()
        return value


class LatentPlanner:
    def __init__(self, recipe):
        self.recipe = recipe
        self.position = self.index = self.phase = 0
        self.nominal_stop = recipe.bootstrap_latents
        self.ended = False
        self.available = 0

    def ready(self, available, *, eos=False):
        if self.ended:
            raise ValueError('received source after EOS')
        if type(available) is not int or available < self.available or type(eos) is not bool:
            raise ValueError('source availability must be monotonic and EOS boolean')
        self.available = available
        result = []
        while self.position < available:
            stop = next((b for b in self.recipe.segment_boundaries
                         if self.position < b < self.nominal_stop), self.nominal_stop)
            if stop > available:
                if not eos:
                    break
                stop = available
            a = self.position
            sink = LatentRange(0, min(self.recipe.sink_latents, a))
            recent = LatentRange(max(0, a-self.recipe.recent_latents), a)
            result.append(WindowPlan(self.index, a == 0, sink, recent,
                LatentRange(max(sink.stop, recent.start), a), LatentRange(a, stop),
                eos and stop == available))
            self.position, self.index = stop, self.index+1
            if stop == self.nominal_stop:
                schedule = self.recipe.current_latent_schedule
                self.nominal_stop += schedule[self.phase % len(schedule)]
                self.phase += 1
        self.ended = eos
        return result
