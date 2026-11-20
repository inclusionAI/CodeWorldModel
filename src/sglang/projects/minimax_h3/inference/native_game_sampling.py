from dataclasses import dataclass


SECOND = 1_000_000_000
FPS = 24


@dataclass
class Publication:
    evidence: dict
    jpeg: bytes
    acquired_ns: int
    pixels: bytes | None = None
    ready_ns: int | None = None


class MediaSlots:
    def __init__(self):
        self.segment = -1
        self.origin_ns = self.wall_origin_ns = None
        self.slot = 0
        self.held = self.pending = None
        self.cursor = 0
        self.skipped = 0
        self.reason = 'resume'
        self.omitted_ns = 0
        self.restart_ns = None

    def restart(self, reason, now_ns):
        if self.wall_origin_ns is not None:
            self.omitted_ns = max(0, now_ns - self.target_ns())
        self.origin_ns = self.wall_origin_ns = None
        if self.pending is not None:
            self.skipped += 1
        self.pending = None
        self.slot = 0
        self.reason = reason
        self.restart_ns = now_ns

    def target_ns(self):
        return self.wall_origin_ns + self.slot * SECOND // FPS

    def deadline_ns(self):
        return self.wall_origin_ns + ((self.slot + 1) * SECOND + FPS - 1) // FPS

    def offer(self, publication):
        evidence = publication.evidence
        self.cursor = evidence['frame_id']
        stamp = evidence['published_monotonic_ns']
        if self.origin_ns is None:
            self.segment += 1
            self.origin_ns = stamp
            self.wall_origin_ns = publication.acquired_ns
            if self.restart_ns is not None:
                self.omitted_ns += max(0, publication.acquired_ns - self.restart_ns)
        if (stamp - self.origin_ns) * FPS < (self.slot - 1) * SECOND:
            self.skipped += 1
            return
        self.pending = publication

    def take(self, now_ns):
        if self.origin_ns is None:
            return None
        fresh = self.pending is not None
        if fresh:
            self.held = self.pending
            self.pending = None
        elif now_ns < self.deadline_ns() or self.held is None:
            return None
        metadata = dict(media_segment=self.segment, media_slot=self.slot,
                        media_target_ns=self.origin_ns + self.slot * SECOND // FPS,
                        media_segment_started_ns=self.wall_origin_ns,
                        sample_kind='fresh' if fresh else 'repeat',
                        media_reason=self.reason if self.slot == 0 else 'publication' if fresh else 'slot_deadline',
                        media_skipped=self.skipped, media_omitted_ns=self.omitted_ns if self.slot == 0 else 0)
        self.slot += 1
        self.skipped = 0
        return self.held, metadata
