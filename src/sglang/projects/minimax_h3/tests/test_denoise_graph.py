from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from projects.minimax_h3.inference.raw_latent import LatentKVCache, RawLatentSession
from projects.minimax_h3.inference.raw_recipe import RAW_FIXED
from projects.minimax_h3.inference.latent_windows import LatentPlanner
from projects.minimax_h3.modeling.inference_optimizations import (
    cached_adaln, configure_inference, prepare_adaln_input, graph_runtime_identity, merge_inference_kv)
from projects.minimax_h3.modeling.transformer import denoise_graph as dg
from projects.minimax_h3.modeling.transformer.packed_attention import PackedAttentionMetadata
from projects.minimax_h3.tests.test_raw_serving import recipe


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    def forward(self, value):
        return cached_adaln(self, value, lambda x: tuple(x * self.weight * i for i in range(6)))


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        for i, block in enumerate(self.blocks):
            block.attn = SimpleNamespace(layer_idx=i)
            block.adaln_proj = Projection()
        self.recording = None

    def _run_blocks(self, hidden, **kwargs):
        def compute():
            result = hidden + kwargs['combined_indices'][:, None]
            result = result + kwargs['rope_cache'][0] + kwargs['rope_cache'][1][:, None]
            for block in self.blocks:
                result = result + sum(block.adaln_proj(kwargs['adaln_input']))
                result = result + kwargs['past_key_values'].key_cache[block.attn.layer_idx].sum()
            return result
        if self.recording is not None:
            output = torch.empty_like(hidden)
            self.recording.replay = lambda: output.copy_(compute())
            return output
        return compute()


@contextmanager
def cpu_cuda(model):
    inspect = dg._DenoiseGraphs._inspect
    step_guard = dg._DenoiseGraphs._step_guard
    def cpu_inspect(*args):
        with patch.object(torch.Tensor, 'device', property(lambda _: torch.device('cuda:0'))):
            return inspect(*args)
    def cpu_step_guard(*args):
        with patch.object(torch.Tensor, 'device', property(lambda _: torch.device('cuda:0'))):
            return step_guard(*args)
    @contextmanager
    def capture(graph, **kwargs):
        model.recording = graph
        try:
            yield
        finally:
            model.recording = None
    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    with ExitStack() as stack:
        stack.enter_context(patch.object(dg._DenoiseGraphs, '_inspect', cpu_inspect))
        stack.enter_context(patch.object(dg._DenoiseGraphs, '_step_guard', cpu_step_guard))
        stack.enter_context(patch.object(dg, '_agree', side_effect=lambda row, *_: dg._agreed_action([row]*8)))
        stack.enter_context(patch.object(torch.cuda, 'Stream', return_value=stream))
        stack.enter_context(patch.object(torch.cuda, 'current_stream', return_value=stream))
        stack.enter_context(patch.object(torch.cuda, 'stream', side_effect=lambda _: nullcontext()))
        stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        stack.enter_context(patch.object(torch.cuda, 'CUDAGraph', side_effect=SimpleNamespace))
        stack.enter_context(patch.object(torch.cuda, 'graph', side_effect=capture))
        yield


class GraphEnvironmentTests(unittest.TestCase):
    def test_configure_rejects_before_mutating_model(self):
        model = configure_inference(Stack().eval())
        flags = model._h3_inference
        generation = model._h3_inference_generation
        with patch.dict(os.environ, {'NCCL_GRAPH_REGISTER': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'NCCL_GRAPH_REGISTER=0'):
                configure_inference(model, cuda_graph=True)
        self.assertIs(model._h3_inference, flags)
        self.assertIs(model._h3_inference_generation, generation)

    def test_runtime_receipt_preserves_absent_environment(self):
        with patch.dict(os.environ, {'NCCL_GRAPH_REGISTER': '0'}, clear=True), \
                patch.object(torch.cuda.nccl, 'version', return_value=(2, 28, 9)):
            receipt = graph_runtime_identity()
        self.assertEqual(receipt['nccl'], [2, 28, 9])
        self.assertEqual(receipt['environment']['NCCL_GRAPH_REGISTER'], '0')
        self.assertIsNone(receipt['environment']['NCCL_GRAPH_MIXING_SUPPORT'])
        self.assertEqual(json.loads(json.dumps(receipt)), receipt)


class AgreementWorkspaceTests(unittest.TestCase):
    def test_reuse_refreshes_every_rank_and_keeps_disagreement_and_reset(self):
        workspace, group = dg._AgreementWorkspace(), object()
        rows = []
        def gather(received, control, *, group):
            self.assertIs(group, workspace.group)
            for target, row in zip(received, rows):
                target.copy_(torch.tensor(row))
            self.assertEqual(control.tolist(), rows[0])
        with patch.object(dg.dist, 'get_world_size', return_value=8), \
                patch.object(dg.dist, 'all_gather', side_effect=gather) as collective:
            addresses = None
            for ordinal, mutation in enumerate((None, 'key', 'reset', None)):
                record = (dg._REPLAY, 0, ordinal, 4, dg._WARMUPS, ordinal, 7, 8, 9)
                rows[:] = [list(record) for _ in range(8)]
                if mutation is not None:
                    rows[-1][5 if mutation == 'key' else 1] += 1
                result = dg._agree(record, torch.device('cpu'), group, workspace)
                self.assertEqual(result, dg._agreed_action(rows))
                current = tuple(t.data_ptr() for t in
                                (workspace.control, workspace.received, workspace.host_received))
                if addresses is not None:
                    self.assertEqual(current, addresses)
                addresses = current
                self.assertEqual(workspace.host_received.tolist(), rows)
            self.assertEqual(collective.call_count, 4)

    def test_scope_changes_rebuild_storage_and_close_drops_owner_workspace(self):
        owner = dg._DenoiseGraphs()
        workspace, first_group, second_group = owner._agreement, object(), object()
        def gather(received, control, **_):
            for row in received:
                row.copy_(control)
        record = (dg._EAGER, 0, 0, 0, dg._WARMUPS, 1, 2, 3, 4)
        with patch.object(dg.dist, 'get_world_size') as size, \
                patch.object(dg.dist, 'all_gather', side_effect=gather):
            previous = None
            for device, group, world_size in ((torch.device('cpu'), first_group, 2),
                    (torch.device('cpu'), second_group, 2), (torch.device('cpu:0'), second_group, 2),
                    (torch.device('cpu:0'), second_group, 3)):
                size.return_value = world_size
                self.assertEqual(dg._agree(record, device, group, workspace), (dg._EAGER, False))
                self.assertIsNot(workspace.control, previous)
                self.assertIs(workspace.group, group)
                self.assertEqual(workspace.device, device)
                self.assertEqual(tuple(workspace.received.shape), (world_size, len(record)))
                previous = workspace.control
        owner.close()
        self.assertIsNot(owner._agreement, workspace)
        self.assertIsNone(owner._agreement.control)
        self.assertIsNot(owner._agreement, dg._DenoiseGraphs()._agreement)


class DenoiseGraphTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, NCCL_GRAPH_REGISTER='0', NCCL_GRAPH_MIXING_SUPPORT='1')
        environment.start()
        self.addCleanup(environment.stop)
        self.model = configure_inference(Stack().eval(), rope=True, cuda_graph=True)
        self.cache = LatentKVCache(2, padding=True)
        self.cache.supports_inference_arena = self.cache._h3_dit_graph_eligible = True
        self.cache.kvlens = [2, 0]
        self.cache._h3_kv_arenas = {}
        for i in range(2):
            arena = tuple(torch.ones(6, 1, 1, dtype=torch.bfloat16) for _ in range(2))
            self.cache._h3_kv_arenas[i] = arena
            self.cache.key_cache[i], self.cache.value_cache[i] = (t[:2] for t in arena)
        self.graphs = self.cache._h3_dit_graphs = dg._DenoiseGraphs()
        self.group = object()
        self.hidden = torch.ones(4, 1, dtype=torch.bfloat16)
        self.kwargs = dict(past_key_values=self.cache, update_past_key_values=False,
            combined_indices=torch.arange(4), rope_cache=(self.hidden.clone(), torch.arange(4)),
            attention_metadata=PackedAttentionMetadata.prepare((3, 1), (2, 0), query_len=4, device='cpu'))

    def run_key(self, value):
        self.kwargs['adaln_input'] = prepare_adaln_input(self.model, torch.tensor([value], dtype=torch.bfloat16))
        return self.graphs.run(self.model, self.hidden, self.kwargs, group=self.group, sp_size=8, tp_size=1)

    @torch.no_grad()
    def test_window_reuses_one_full_inspection_but_selects_four_actual_keys(self):
        with cpu_cuda(self.model), self.cache.denoise_graph_window(eligible=True), \
                patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect, \
                patch.object(self.graphs, '_step_guard', wraps=self.graphs._step_guard) as guard, \
                patch.object(dg, '_agree', wraps=dg._agree) as agree:
            for value in (1, 2, 3, 4):
                self.hidden.add_(1)
                output = self.run_key(value)
                torch.testing.assert_close(output, self.model._run_blocks(self.hidden, **self.kwargs))
            self.assertEqual((inspect.call_count, guard.call_count, agree.call_count), (1, 4, 4))
            self.assertEqual(len(self.graphs.variants), 4)
            self.assertIsNotNone(self.graphs._window_admission)
        self.assertIsNone(self.graphs._window_admission)
        self.assertFalse(hasattr(self.cache, '_h3_dit_graph_window'))
        self.assertFalse(self.cache._h3_dit_graph_eligible)
        self.assertEqual(self.graphs.stats['entries'], 4)

    @torch.no_grad()
    def test_unscoped_calls_always_inspect_fully(self):
        with cpu_cuda(self.model), patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect:
            for _ in range(4):
                self.run_key(1)
            self.assertEqual(inspect.call_count, 4)
            self.assertIsNone(self.graphs._window_admission)

    @torch.no_grad()
    def test_eager_arena_growth_and_rebinding_require_fresh_inspection(self):
        for repair in ('growth', 'rebind'):
            with self.subTest(repair=repair), cpu_cuda(self.model):
                self.graphs.close()
                for layer, arena in self.cache._h3_kv_arenas.items():
                    if repair == 'growth':
                        arena = self.cache._h3_kv_arenas[layer] = tuple(t[:2].clone() for t in arena)
                    self.cache.key_cache[layer], self.cache.value_cache[layer] = (t[:2].clone() for t in arena)
                original = self.model._run_blocks
                def blocks(hidden, **kwargs):
                    for layer in range(2):
                        tail = hidden.new_ones(4, 1, 1)
                        merge_inference_kv(self.cache, layer, tail, tail)
                    return original(hidden, **kwargs)
                with self.cache.denoise_graph_window(eligible=True), \
                        patch.object(self.model, '_run_blocks', side_effect=blocks), \
                        patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect:
                    self.run_key(1)
                    self.assertIsNone(self.graphs._window_admission)
                    self.run_key(1)
                    self.assertIsNotNone(self.graphs._window_admission)
                    self.run_key(1)
                    self.assertEqual(inspect.call_count, 2)
                    for layer, arena in self.cache._h3_kv_arenas.items():
                        self.assertGreaterEqual(len(arena[0]), 6)
                        self.assertEqual(self.cache.value_cache[layer].data_ptr(), arena[1].data_ptr())

    @torch.no_grad()
    def test_scope_exit_rechecks_weight_configuration_and_arena_changes(self):
        with cpu_cuda(self.model):
            with self.cache.denoise_graph_window(eligible=True):
                self.run_key(1)
            for change in ('weight', 'configuration', 'arena'):
                with self.subTest(change=change):
                    if change == 'weight':
                        self.model.blocks[0].adaln_proj.weight.add_(1)
                    elif change == 'configuration':
                        configure_inference(self.model, rope=True, cuda_graph=True)
                    else:
                        arena = tuple(t.clone() for t in self.cache._h3_kv_arenas[0])
                        self.cache._h3_kv_arenas[0] = arena
                        self.cache.key_cache[0], self.cache.value_cache[0] = (t[:2] for t in arena)
                    with self.cache.denoise_graph_window(eligible=True), \
                            patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect:
                        self.run_key(1)
                        self.assertIsNone(self.graphs._window_admission)
                        self.assertEqual(self.graphs.stats['entries'], 0)
                        self.run_key(1)
                        self.run_key(1)
                        self.assertEqual(inspect.call_count, 2)
            self.assertEqual(self.graphs.stats['invalidations'], 3)

    @torch.no_grad()
    def test_scoped_rank_disagreement_and_reset_revoke_admission(self):
        with cpu_cuda(self.model), self.cache.denoise_graph_window(eligible=True):
            for remote in ('action', 'key', 'reset'):
                self.run_key(1)
                def agreement(row, *_):
                    other = list(row)
                    other[{'action': 0, 'key': 5, 'reset': 1}[remote]] += 1
                    return dg._agreed_action([row]*7 + [other])
                with patch.object(dg, '_agree', side_effect=agreement):
                    self.run_key(1)
                self.assertIsNone(self.graphs._window_admission)
                with patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect:
                    self.run_key(1)
                    inspect.assert_called_once()

    @torch.no_grad()
    def test_changed_step_context_rechecks_eligibility_and_agrees(self):
        for change in ('generation', 'layout', 'option', 'key', 'attention', 'training'):
            with self.subTest(change=change), cpu_cuda(self.model), self.cache.denoise_graph_window(eligible=True):
                self.run_key(1)
                with ExitStack() as patches:
                    if change == 'generation':
                        patches.enter_context(patch.object(self.model, '_h3_inference_generation', object()))
                    elif change == 'layout':
                        patches.enter_context(patch.dict(self.kwargs, combined_indices=torch.arange(8)[::2]))
                    elif change == 'option':
                        patches.enter_context(patch.dict(self.model._h3_inference, rope=False))
                    elif change == 'key':
                        patches.enter_context(patch.object(self.kwargs['adaln_input'], '_h3_adaln_key', None))
                    elif change == 'attention':
                        patches.enter_context(patch.dict(self.kwargs, attention_mask=torch.ones(1)))
                    else:
                        patches.enter_context(patch.object(self.model, 'training', True))
                    with patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect, \
                            patch.object(self.graphs, '_step_guard', wraps=self.graphs._step_guard) as guard, \
                            patch.object(dg, '_agree', wraps=dg._agree) as agree:
                        self.graphs.run(self.model, self.hidden, self.kwargs, group=self.group, sp_size=8, tp_size=1)
                        self.assertEqual(inspect.call_count, int(change in ('generation', 'layout')))
                        guard.assert_called_once()
                        agree.assert_called_once()
                        self.assertIsNone(self.graphs._window_admission)

    @torch.no_grad()
    def test_ineligible_calls_defer_binding_scan_until_eligible_reentry(self):
        with cpu_cuda(self.model):
            for change in ('weight', 'arena'):
                with self.subTest(change=change), self.cache.denoise_graph_window(eligible=True):
                    self.run_key(1)
                    before = self.graphs.stats['invalidations']
                    with patch.object(self.cache, '_h3_dit_graph_eligible', False), \
                            patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect, \
                            patch.object(dg, '_agree', wraps=dg._agree) as agree:
                        if change == 'weight':
                            self.model.blocks[0].adaln_proj.weight.add_(1)
                        else:
                            arena = tuple(t.clone() for t in self.cache._h3_kv_arenas[0])
                            self.cache._h3_kv_arenas[0] = arena
                            self.cache.key_cache[0], self.cache.value_cache[0] = (t[:2] for t in arena)
                        output = self.run_key(1)
                        torch.testing.assert_close(output, self.model._run_blocks(self.hidden, **self.kwargs))
                        inspect.assert_not_called()
                        agree.assert_called_once()
                        self.assertIsNone(self.graphs._window_admission)
                        self.assertEqual(self.graphs.stats['entries'], 1)
                        self.assertEqual(self.graphs.stats['invalidations'], before)
                    with patch.object(self.graphs, '_inspect', wraps=self.graphs._inspect) as inspect:
                        self.run_key(1)
                        inspect.assert_called_once()
                        self.assertEqual(self.graphs.stats['entries'], 0)
                        self.assertEqual(self.graphs.stats['invalidations'], before + 1)
                        self.assertIsNone(self.graphs._window_admission)
                    output = self.run_key(1)
                    torch.testing.assert_close(output, self.model._run_blocks(self.hidden, **self.kwargs))
                    self.assertIsNotNone(self.graphs._window_admission)

    @torch.no_grad()
    def test_window_rejects_mutation_and_cleans_up_execution_exceptions(self):
        with cpu_cuda(self.model):
            with self.cache.denoise_graph_window(eligible=True):
                token = self.cache._h3_dit_graph_window
                for operation in (lambda: self.cache.retain(0), self.cache.fork):
                    with self.assertRaisesRegex(RuntimeError, 'denoise graph window'):
                        operation()
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    with self.cache.denoise_graph_window(eligible=True):
                        pass
                self.assertIs(self.cache._h3_dit_graph_window, token)
                self.run_key(1)
            for failure in ('capture', 'replay'):
                with self.assertRaisesRegex(RuntimeError, failure):
                    with self.cache.denoise_graph_window(eligible=True):
                        self.run_key(1)
                        target, method = ((self.graphs, '_capture') if failure == 'capture' else
                                          (next(iter(self.graphs.variants.values())).graph, 'replay'))
                        with patch.object(target, method, side_effect=RuntimeError(failure)):
                            self.run_key(2 if failure == 'capture' else 1)
                self.assertFalse(hasattr(self.cache, '_h3_dit_graph_window'))
                self.assertFalse(self.cache._h3_dit_graph_eligible)
                self.assertIsNone(self.graphs._window_admission)

    @torch.no_grad()
    def test_first_replay_live_inputs_four_value_keys_and_borrowed_output(self):
        with cpu_cuda(self.model):
            first = self.run_key(1)
            torch.testing.assert_close(first, self.model._run_blocks(self.hidden, **self.kwargs))
            self.assertEqual(self.graphs.stats['replays'], 1)
            self.hidden = self.hidden + 2
            self.kwargs['combined_indices'] = torch.arange(4).flip(0)
            self.kwargs['rope_cache'] = (self.hidden + 9, torch.arange(4) + 17)
            second = self.run_key(1)
            self.assertIs(first, second)
            torch.testing.assert_close(second, self.model._run_blocks(self.hidden, **self.kwargs))
            for value in (2, 3, 4, 5):
                output = self.run_key(value)
                torch.testing.assert_close(output, self.model._run_blocks(self.hidden, **self.kwargs))
            stats = self.graphs.stats
            self.assertEqual((stats['captures'], stats['replays'], stats['entries']), (4, 5, 4))
            self.assertEqual(stats['warmup_calls'], 4 * dg._WARMUPS)
            self.assertEqual(stats['eager_reasons'], {'variant_limit': 1})
            json.dumps(stats)
            stats['eager_reasons'].clear()
            self.assertEqual(self.graphs.stats['eager_reasons']['variant_limit'], 1)

    @torch.no_grad()
    def test_variant_slots_and_fingerprints_reuse_only_actual_key_and_layout(self):
        with cpu_cuda(self.model):
            for value in (1, 2, 3, 4):
                self.run_key(value)
            with patch.object(dg, '_fingerprint', wraps=dg._fingerprint) as fingerprint, \
                    patch.object(dg, '_agree', wraps=dg._agree) as agree:
                for value in (4, 2, 1, 3):
                    self.run_key(value)
                    self.assertEqual(agree.call_args.args[0][2], value - 1)
                fingerprint.assert_not_called()
                known_record = agree.call_args.args[0]
                self.kwargs['combined_indices'] = torch.arange(8)[::2]
                self.run_key(3)
                fingerprint.assert_called_once()
                changed_record = agree.call_args.args[0]
                self.assertEqual(changed_record[0], dg._EAGER)
                self.assertNotEqual(known_record[5:], changed_record[5:])
                self.kwargs['combined_indices'] = torch.arange(4)
                for value in (5, 6):
                    self.run_key(value)
                self.assertEqual(fingerprint.call_count, 3)
                self.assertEqual(len(self.graphs.variants), 4)
                self.assertEqual(self.graphs.stats['eager_reasons']['variant_limit'], 2)

    @torch.no_grad()
    def test_weight_generation_arena_and_group_changes_retire_before_reuse(self):
        with cpu_cuda(self.model):
            self.run_key(1)
            for change in ('weight', 'replacement', 'generation', 'arena', 'group'):
                with self.subTest(change=change):
                    if change == 'weight':
                        self.model.blocks[0].adaln_proj.weight.add_(1)
                    elif change == 'replacement':
                        self.model.blocks[0].adaln_proj.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
                    elif change == 'generation':
                        configure_inference(self.model, rope=True, cuda_graph=True)
                    elif change == 'group':
                        self.group = object()
                    else:
                        arena = tuple(t.clone() for t in self.cache._h3_kv_arenas[0])
                        self.cache._h3_kv_arenas[0] = arena
                        self.cache.key_cache[0], self.cache.value_cache[0] = (t[:2] for t in arena)
                    result = self.run_key(1)
                    self.assertEqual(self.graphs.stats['entries'], 0)
                    torch.testing.assert_close(result, self.model._run_blocks(self.hidden, **self.kwargs))
                    self.run_key(1)
                    self.assertEqual(self.graphs.stats['entries'], 1)
            self.assertEqual(self.graphs.stats['invalidations'], 5)

    @torch.no_grad()
    def test_value_history_must_alias_and_fork_drops_parent_graphs(self):
        with cpu_cuda(self.model):
            self.run_key(1)
            self.cache.value_cache[0] = self.cache.value_cache[0].clone()
            self.run_key(1)
            self.assertEqual(self.graphs.stats['eager_reasons']['history_not_in_arena'], 1)
            self.assertEqual(self.graphs.stats['captures'], 1)
            self.cache.value_cache[0] = self.cache._h3_kv_arenas[0][1][:2]
            self.run_key(1)
            self.assertEqual(self.graphs.stats['replays'], 2)
            child = self.cache.fork()
            self.assertEqual(self.graphs.stats['entries'], 0)
            for cache in (self.cache, child):
                self.assertFalse(hasattr(cache, '_h3_dit_graphs'))
                self.assertFalse(hasattr(cache, '_h3_kv_arenas'))

    @torch.no_grad()
    def test_rank_action_key_conflicts_and_global_invalidation(self):
        with cpu_cuda(self.model):
            self.run_key(1)
            for remote in ('action', 'key', 'reset'):
                def agreement(row, *_):
                    other = list(row)
                    other[{'action': 0, 'key': 5, 'reset': 1}[remote]] += 1
                    return dg._agreed_action([row]*7 + [other])
                with patch.object(dg, '_agree', side_effect=agreement):
                    self.run_key(1)
            self.assertEqual(self.graphs.stats['replays'], 1)
            self.assertEqual(self.graphs.stats['entries'], 0)
            self.assertEqual(self.graphs.stats['eager_reasons'], {'rank_disagreement': 2, 'invalidated': 1})

    @torch.no_grad()
    def test_unsupported_call_still_agrees_and_capture_error_propagates(self):
        with cpu_cuda(self.model), patch.object(dg, '_agree', return_value=(dg._EAGER, False)) as agree:
            self.cache._h3_dit_graph_eligible = False
            self.run_key(1)
            agree.assert_called_once()
        self.cache._h3_dit_graph_eligible = True
        with cpu_cuda(self.model), patch.object(torch.cuda, 'graph', side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                self.run_key(1)
        self.assertEqual(self.graphs.stats['replays'], 0)

    def test_enabled_uncached_eval_and_training_still_run_eager(self):
        self.kwargs.pop('past_key_values')
        for training in (False, True):
            self.model.train(training)
            with torch.set_grad_enabled(training), cpu_cuda(self.model), \
                    patch.object(self.model, '_run_blocks', return_value=self.hidden) as blocks, \
                    patch.object(dg, '_agree', return_value=(dg._EAGER, False)) as agree:
                self.assertIs(self.run_key(1), self.hidden)
                blocks.assert_called_once()
                agree.assert_called_once()
        self.assertEqual(self.graphs.stats['eager_reasons'], {'window_not_steady': 1, 'training_or_grad': 1})

    @torch.no_grad()
    def test_default_off_and_inference_tensor_weights_fall_back(self):
        configure_inference(self.model)
        self.assertFalse(self.model._h3_inference['cuda_graph'])
        for invalid in (None, 1, 'true'):
            with self.assertRaisesRegex(ValueError, 'cuda_graph must be a boolean'):
                configure_inference(self.model, cuda_graph=invalid)
        configure_inference(self.model, rope=True, cuda_graph=True)
        with torch.inference_mode():
            self.model.blocks[0].adaln_proj.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        with cpu_cuda(self.model):
            self.run_key(1)
        self.assertEqual(self.graphs.stats['eager_reasons'], {'untracked_or_nonresident_weights': 1})

    def test_recipe_admission_uses_full_current_and_saturated_recent(self):
        stream = RawLatentSession.__new__(RawLatentSession)
        stream.recipe = replace(recipe(RAW_FIXED), current_latent_schedule=(5,))
        stream.delayed_hc = False
        plans = LatentPlanner(stream.recipe).ready(26, eos=True)
        self.assertEqual([stream._graph_window(p, 'positive') for p in plans], [False, False, True, True, False])
        steady = plans[2]
        for changed in (replace(stream.recipe, current_latent_schedule=(3, 2)),
                        replace(stream.recipe, guidance=dict(mode='cfg', drop='proxy', scale=2.))):
            stream.recipe = changed
            self.assertFalse(stream._graph_window(steady, 'positive'))


if __name__ == '__main__':
    unittest.main()
