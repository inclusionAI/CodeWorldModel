import pytest
import torch

from inference.omni_h3_codec import NativeH3Codec


class World:
    ranks = list(range(8))

    def __init__(self, rank, message):
        self.rank, self.message = rank, message

    def broadcast(self, tensor, *, src):
        assert src == 0
        if self.rank == 0:
            self.message[:] = [tensor.clone()]
        return self.message[0].clone()


class Codec:
    def __init__(self):
        self.resets = 0

    def begin_session(self):
        self.resets += 1

    def decode(self, count, *, eos):
        return torch.zeros(count, 2, 2, 3, dtype=torch.uint8) if count else None


def test_only_owner_returns_rgb_and_all_ranks_receive_zero_tail_and_reset_counts():
    message = []
    local = [Codec(), Codec(), *[None] * 6]
    codecs = [NativeH3Codec(local[rank], World(rank, message), output_rank=0, device='cpu')
              for rank in range(8)]
    for _ in range(2):
        for codec in codecs:
            codec.begin_session()
        for count, eos in ((0, False), (13, False), (21, False), (5, True)):
            results = [codec.decode(count, eos=eos) for codec in codecs]
            assert [result.frame_count for result in results] == [count] * 8
            assert all(result.rgb is None for result in results[1:])
            assert (results[0].rgb is None) == (count == 0)
    assert local[0].resets == local[1].resets == 2


def test_participant_frame_mismatch_fails_instead_of_leaving_a_stale_ledger():
    message = []
    owner = NativeH3Codec(Codec(), World(0, message), output_rank=0, device='cpu')
    follower = NativeH3Codec(Codec(), World(1, message), output_rank=0, device='cpu')
    owner.decode(5, eos=True)
    with pytest.raises(RuntimeError, match='disagree'):
        follower.decode(4, eos=True)


def test_output_owner_must_load_decoder():
    with pytest.raises(ValueError, match='participate'):
        NativeH3Codec(None, World(0, []), output_rank=0, device='cpu')


def test_local_decode_has_no_world_collective_and_counts_publish_once_for_all_spans():
    message = []
    codecs = [NativeH3Codec(Codec() if rank < 2 else None, World(rank, message),
        output_rank=0, device='cpu') for rank in range(8)]
    local = [[codec.decode_local(count, eos=False) for count in (0, 13, 21)]
             for codec in codecs]
    assert message == []
    assert [batch.frame_count for batch in local[0]] == [0, 13, 21]
    assert all(batch.rgb is None for rank in local[1:] for batch in rank)
    counts = [codec.publish_counts([batch.frame_count for batch in batches], span_count=3)
              for codec, batches in zip(codecs, local)]
    assert counts == [(0, 13, 21)] * 8
    assert codecs[0].publish_counts([], span_count=0) == ()
    with pytest.raises(ValueError, match='count vector'):
        codecs[0].publish_counts([], span_count=1)
