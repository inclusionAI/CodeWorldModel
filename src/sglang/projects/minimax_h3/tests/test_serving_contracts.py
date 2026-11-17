import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.serve_web import bootstrap

bootstrap()

from projects.minimax_h3.inference.serving import ResidentBackend
from projects.minimax_h3.inference.serving_ray import RayResidentGroup
from projects.minimax_h3.inference.serving_stages import CpuOutput, StageCursor, WorkKey
from tools.ray_worker import RankActor


class StageContractTests(unittest.TestCase):
    def test_dispatch_order_payload_owners_and_decode_only_prefetch(self):
        for overlap, eos in ((False, False), (True, False), (True, True)):
            with self.subTest(overlap=overlap, eos=eos):
                calls = []
                group = RayResidentGroup.__new__(RayResidentGroup)
                group.failed, group.epoch = False, 0
                group.prefetch_bounds = None
                group.pending_outputs = {}
                group.codec_overlap = overlap
                group.vae_rank, group.decoder_rank, group.world_size = 1, 2, 3
                group.ray = SimpleNamespace(put=lambda value: value)
                def remote(rank, stage, *args):
                    calls.append((stage, rank, args))
                    return dict(rank=rank, elapsed_ms=0, result=(0, {}))
                def wait(refs):
                    calls.append(('wait', tuple(row['rank'] for row in refs)))
                    return refs
                group._wait = wait
                group.actors = []
                for rank in range(3):
                    actor = Mock()
                    for stage in ('static_condition', 'first_image', 'open_recipe', 'text_condition',
                                  'proxy_encode', 'render', 'decode', 'collect', 'output'):
                        method = getattr(actor, stage)
                        method.options.return_value = method
                        method.remote.side_effect = lambda *args, r=rank, s=stage: remote(r, s, *args)
                    group.actors.append(actor)
                group.execute('start', dict(session_id='one'), b'first')
                next_metadata = dict(session_id='one', chunk_id=1, eos=False)
                lookahead = Mock(return_value=(next_metadata, b'next'))
                pixels, _ = group.execute('chunk', dict(session_id='one', chunk_id=0, eos=eos),
                                          b'proxy', lookahead=lookahead, defer_output=True)
                self.assertIsInstance(pixels, CpuOutput)
                self.assertEqual(pixels.key, WorkKey(1, 'one', 'chunk', 0))
                expected = []
                for stage in ('static_condition', 'first_image', 'open_recipe', 'collect',
                              'text_condition', 'proxy_encode', 'render', 'decode', 'collect'):
                    expected.extend((stage, rank) for rank in range(3))
                    expected.append(('wait', (0, 1, 2)))
                expected.append(('output', 2))
                self.assertEqual([row[:2] for row in calls], expected)
                for stage, rank, *rest in calls:
                    if stage in ('wait', 'output'):
                        continue
                    args = rest[0]
                    if stage in ('static_condition', 'first_image', 'proxy_encode'):
                        owner = 0 if stage == 'static_condition' else 1
                        value = b'proxy' if stage == 'proxy_encode' else b'first'
                        self.assertEqual(args[2:], (value if rank == owner else b'',))
                    elif stage == 'decode' and overlap and not eos:
                        self.assertEqual(args[2:], (WorkKey(1, 'one', 'chunk', 1), next_metadata,
                                                   b'next' if rank == 1 else b''))
                    else:
                        self.assertEqual(args[2:], ())
                self.assertEqual(lookahead.call_count, int(overlap and not eos))

    def test_serial_stage_arguments_are_unchanged(self):
        for op, plan in [('start', ('static_condition', 'first_image', 'open_recipe', 'collect')),
                         ('chunk', ('text_condition', 'proxy_encode', 'render', 'decode', 'collect')),
                         ('cancel', ('collect',))]:
            backend = Mock()
            self.assertIs(ResidentBackend.execute(backend, op, {}, b'input'), backend.collect.return_value)
            self.assertEqual([call[0] for call in backend.mock_calls], ['begin', *plan])
            for stage in plan:
                expected = (b'input',) if stage in ('static_condition', 'first_image', 'proxy_encode') else ()
                self.assertEqual(getattr(backend, stage).call_args.args, expected)

    def test_actor_reply_and_reset_keep_gpu_completion_order(self):
        actor = RankActor.__new__(RankActor)
        actor.rank, actor.cursor, actor.backend = 0, StageCursor(), Mock()
        observed = []
        actor.backend.begin.side_effect = lambda *args: observed.append('begin')
        actor.backend.static_condition.side_effect = lambda value: observed.append(('payload', value))
        actor.backend.close_session.side_effect = lambda: observed.append('reset')
        def synchronize():
            observed.append(('sync', actor.cursor.index))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, synchronize=synchronize))
        with patch.dict(sys.modules, torch=torch):
            actor._run_stage(WorkKey(1, 'one', 'start'), 'static_condition', {}, b'first')
            self.assertEqual(observed, ['begin', ('payload', b'first'), ('sync', 0)])
            self.assertEqual(actor.cursor.index, 1)
            observed.clear()
            actor.close_session(1)
            self.assertEqual(observed, [('sync', 1), 'reset', ('sync', 1)])
            self.assertEqual(actor.cursor.index, 0)


if __name__ == '__main__':
    unittest.main()
