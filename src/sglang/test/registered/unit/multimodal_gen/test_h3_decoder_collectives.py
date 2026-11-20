from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def coordinator(custom, communicator):
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.srt_custom_allreduce = custom
    group.device_communicator = communicator
    group.device_group = object()
    return group


def test_out_of_place_fallback_reuses_device_communicator_without_modifying_input():
    value = torch.tensor([1.0, 2.0])
    comm = SimpleNamespace(all_reduce=Mock(side_effect=lambda x, op: x + 3))
    group = coordinator(SimpleNamespace(disabled=True), comm)
    output = group._all_reduce_out_of_place(value)
    comm.all_reduce.assert_called_once_with(value, op=torch.distributed.ReduceOp.SUM)
    assert torch.equal(value, torch.tensor([1.0, 2.0]))
    assert (
        torch.equal(output, torch.tensor([4.0, 5.0]))
        and output.data_ptr() != value.data_ptr()
    )


def test_group_without_communicator_keeps_clone_and_torch_fallback(monkeypatch):
    value = torch.tensor([1.0, 2.0])
    group = coordinator(SimpleNamespace(disabled=True), None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda x, group: x.add_(3))
    output = group._all_reduce_out_of_place(value)
    assert torch.equal(value, torch.tensor([1.0, 2.0]))
    assert torch.equal(output, torch.tensor([4.0, 5.0]))


def test_custom_dispatch_remains_first_choice():
    value = torch.tensor([1.0, 2.0])
    expected = value + 3
    comm = SimpleNamespace(
        all_reduce=Mock(side_effect=AssertionError("custom fallback was reached"))
    )
    custom = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda x: True,
        custom_all_reduce=lambda x: expected,
    )
    assert coordinator(custom, comm)._all_reduce_out_of_place(value) is expected
    comm.all_reduce.assert_not_called()
