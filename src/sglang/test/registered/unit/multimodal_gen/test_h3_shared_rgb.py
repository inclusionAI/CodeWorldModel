# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.shared_rgb import (
    H3SharedRGB,
    copy_rgb_to_host,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _produce(connection):
    for value in (7, 8):
        source = torch.full((2, 3, 4, 3), value, dtype=torch.uint8)
        output = copy_rgb_to_host(source, shared=True)
        connection.send(H3SharedRGB.from_tensors([output]))
        source.zero_()
        assert connection.recv() == "mapped"
    connection.close()


def test_next_output_and_producer_exit_do_not_invalidate_retained_rgb():
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe()
    worker = ctx.Process(target=_produce, args=(sender,))
    worker.start()
    sender.close()
    buffers = []
    try:
        for _ in range(2):
            assert receiver.poll(30)
            (rgb,) = receiver.recv().materialize()
            assert rgb.is_shared()
            buffers.append(rgb)
            # Retain the first result unconsumed while the producer runs again.
            receiver.send("mapped")
        worker.join(30)
        assert worker.exitcode == 0
        assert torch.all(buffers[0] == 7) and torch.all(buffers[1] == 8)
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
            worker.join()
