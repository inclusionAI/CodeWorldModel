import unittest

import torch

from projects.minimax_h3.inference.raw_latent import LatentKVCache
from projects.minimax_h3.modeling.inference_optimizations import merge_inference_kv


class LatentKVRetainTests(unittest.TestCase):
    def cache(self, owners, *, padding=False, dtype=torch.float32):
        cache = LatentKVCache(3, padding=padding)
        cache.owners = list(owners)
        cache.kvlens[0] = len(owners)
        cache.groups[0] = [(-1, len(owners))]
        rows = torch.arange(len(owners)*8, dtype=dtype).reshape(len(owners), 4, 2)
        for layer in (0, 2):
            cache.key_cache[layer] = (rows + layer*100).transpose(1, 2)
            cache.value_cache[layer] = (rows + layer*100 + 50).transpose(1, 2)
        return cache

    def assert_rows(self, cache, previous, rows):
        self.assertEqual(cache.owners, [previous.owners[i] for i in rows])
        self.assertEqual(cache.kvlens, [len(rows)] + ([0] if cache.batch_size > 1 else []))
        self.assertEqual(cache.groups, [[(-1, len(rows))]] + ([[]] if cache.batch_size > 1 else []))
        self.assertFalse(cache.supports_inference_arena)
        for before, after in ((previous.key_cache, cache.key_cache),
                              (previous.value_cache, cache.value_cache)):
            self.assertEqual(before.keys(), after.keys())
            for layer, value in before.items():
                if value is None:
                    self.assertIsNone(after[layer])
                else:
                    torch.testing.assert_close(after[layer], value[rows], rtol=0, atol=0)

    def test_partial_blocks_preserve_row_order_duplicates_prefix_and_documents(self):
        owners = [(-1, None), (-1, None), (-1, 10), (6, 1),
                  (5, None), (6, None), (7, None), (7, None), (8, None), (9, 2),
                  (5, None), (6, None), (7, None), (7, None), (8, None)]
        for dtype in (torch.float32, torch.bfloat16):
            for padding in (False, True):
                with self.subTest(dtype=dtype, padding=padding):
                    cache = self.cache(owners, dtype=dtype, padding=padding)
                    previous = cache.fork()
                    cache.retain(7)
                    self.assert_rows(cache, previous, [0, 1, 2, 6, 7, 8, 9, 12, 13, 14])
                    self.assertEqual(cache.documents, {10, 2})
                    cache.retain(9)
                    self.assert_rows(cache, previous, [0, 1, 2, 9])
                    self.assertEqual(cache.documents, {10, 2})

    def test_no_eviction_preserves_tensor_and_metadata_identity(self):
        for owners in ([], [(-1, None), (7, 1), (8, None)]):
            with self.subTest(owners=owners):
                cache = self.cache(owners, padding=True)
                cache.groups[0] = [(-1, 1), (4, len(owners)-1)] if owners else []
                previous = cache.fork()
                metadata = cache.owners, cache.kvlens, cache.groups, cache.groups[0]
                cache.retain(7)
                for actual, original in zip((cache.owners, cache.kvlens, cache.groups, cache.groups[0]), metadata):
                    self.assertIs(actual, original)
                self.assertEqual(cache.groups, previous.groups)
                for before, after in ((previous.key_cache, cache.key_cache),
                                      (previous.value_cache, cache.value_cache)):
                    for layer, value in before.items():
                        self.assertIs(after[layer], value)

    def test_complete_eviction_keeps_empty_tensor_shapes_and_padding_metadata(self):
        cache = self.cache([(1, 1), (2, None)], padding=True, dtype=torch.bfloat16)
        previous = cache.fork()
        cache.retain(3)
        self.assert_rows(cache, previous, [])
        self.assertEqual(cache.documents, set())
        empty_key = cache.key_cache[0]
        cache.retain(4)
        self.assertIs(cache.key_cache[0], empty_key)

    def test_retaining_one_fork_does_not_change_shared_cache_or_sibling(self):
        original = self.cache([(-1, None), (5, 1), (7, 2)], padding=True)
        branch, sibling = original.fork(), original.fork()
        frozen = [(k.clone(), original.value_cache[layer].clone())
                  for layer, k in original.key_cache.items() if k is not None]
        branch.retain(7)
        self.assert_rows(branch, original, [0, 2])
        branch.key_cache[0].fill_(-1)
        branch.value_cache[0].fill_(-2)
        for cache in (original, sibling):
            self.assertEqual(cache.owners, [(-1, None), (5, 1), (7, 2)])
            self.assertEqual(cache.kvlens, [3, 0])
            self.assertEqual(cache.groups, [[(-1, 3)], []])
            for layer, (key, value) in zip((0, 2), frozen):
                torch.testing.assert_close(cache.key_cache[layer], key, rtol=0, atol=0)
                torch.testing.assert_close(cache.value_cache[layer], value, rtol=0, atol=0)


class LatentKVArenaTests(unittest.TestCase):
    def merge(self, cache, positions, *, marker=0, padding=0, store=True):
        rows = torch.arange((len(positions)+padding)*4, dtype=self.dtype).reshape(-1, 2, 2) + marker
        if store:
            cache.update_kvlens([len(positions)] + ([padding] if cache.batch_size > 1 else []))
        for layer in cache.key_cache:
            keys, values = rows + layer*100, rows + layer*100 + 50
            past_k, past_v = cache.key_cache[layer], cache.value_cache[layer]
            expected_k = keys if past_k is None else torch.cat((past_k, keys))
            expected_v = values if past_v is None else torch.cat((past_v, values))
            merged_k, merged_v = merge_inference_kv(cache, layer, keys, values)
            torch.testing.assert_close(merged_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(merged_v, expected_v, rtol=0, atol=0)
            if store:
                cache.update_kvcache(layer, merged_k, merged_v)
        if store:
            cache.owners.extend((p, None) for p in positions)
        self.assertEqual(cache.kvlens, [len(cache.owners)] + ([0] if cache.batch_size > 1 else []))
        for value in cache.key_cache.values():
            self.assertEqual(len(value), len(cache.owners))

    def snapshot(self, cache):
        return (list(cache.owners), list(cache.kvlens), [list(g) for g in cache.groups],
                [(cache.key_cache[i].clone(), cache.value_cache[i].clone()) for i in cache.key_cache])

    def assert_snapshot(self, cache, snapshot):
        actual = self.snapshot(cache)
        self.assertEqual(actual[:3], snapshot[:3])
        for pair, expected in zip(actual[3], snapshot[3], strict=True):
            for tensor, original in zip(pair, expected, strict=True):
                torch.testing.assert_close(tensor, original, rtol=0, atol=0)

    def test_fork_isolates_both_mutation_orders_siblings_and_nested_forks(self):
        for self.dtype in (torch.float32, torch.bfloat16):
            for padding in (0, 2):
                for parent_first in (False, True):
                    for start in (0, 2, 9):
                        with self.subTest(dtype=self.dtype, padding=padding, parent_first=parent_first, start=start):
                            parent = LatentKVCache(2, padding=bool(padding))
                            parent.supports_inference_arena = True
                            self.merge(parent, [0, 1, 2], padding=padding)
                            frozen = self.snapshot(parent)
                            child, sibling = parent.fork(), parent.fork()
                            self.assertTrue(child.supports_inference_arena)
                            self.assertIs(parent.key_cache[0], child.key_cache[0])
                            first, second = (parent, child) if parent_first else (child, parent)
                            first.retain(start)
                            self.merge(first, [10], marker=100, padding=padding)
                            self.assert_snapshot(second, frozen)
                            self.assert_snapshot(sibling, frozen)
                            changed = self.snapshot(first)
                            nested = first.fork()
                            self.merge(first, [11], marker=200, padding=padding)
                            self.assert_snapshot(nested, changed)
                            second.retain(9)
                            self.merge(second, [12, 13], marker=300, padding=padding)
                            self.assert_snapshot(nested, changed)
                            first_snapshot, second_snapshot = self.snapshot(first), self.snapshot(second)
                            nested.retain(11)
                            self.merge(nested, [14], marker=400, padding=padding)
                            self.assert_snapshot(first, first_snapshot)
                            self.assert_snapshot(second, second_snapshot)
                            self.assert_snapshot(sibling, frozen)

    @torch.no_grad()
    def test_inherited_chunk_compaction_cannot_write_into_fork(self):
        self.dtype = torch.float32
        parent = LatentKVCache(2)
        parent.supports_inference_arena = True
        self.merge(parent, [0, 1, 2])
        parent.groups[0], parent.windows = [(0, 1), (1, 1), (2, 1)], [0]
        child = parent.fork()
        frozen = self.snapshot(child)
        parent.begin_chunk(2)
        self.assertEqual(parent.kvlens, [1])
        self.assert_snapshot(child, frozen)

    def test_noisy_tail_fork_clean_commit_and_capacity_growth(self):
        self.dtype = torch.float32
        cache = LatentKVCache(2, padding=True)
        cache.supports_inference_arena = True
        self.merge(cache, [-1, 0], padding=2)
        frozen = self.snapshot(cache)
        self.merge(cache, [1, 2, 3], marker=100, padding=2, store=False)
        self.assert_snapshot(cache, frozen)
        child = cache.fork()
        for current in (cache, child):
            self.merge(current, [1, 2], marker=200, padding=2, store=False)
            arena = current._h3_kv_arenas[0][0]
            self.assertEqual(len(arena), 6)
            metadata = current.owners, current.kvlens, current.groups
            keys = dict(current.key_cache)
            current.retain(0)
            for actual, original in zip((current.owners, current.kvlens, current.groups), metadata):
                self.assertIs(actual, original)
            for layer in keys:
                self.assertIs(current.key_cache[layer], keys[layer])
            self.assertIs(current._h3_kv_arenas[0][0], arena)
            self.merge(current, [1], marker=300, padding=1, store=False)
            self.assertIs(current._h3_kv_arenas[0][0], arena)
            self.assert_snapshot(current, frozen)
            self.merge(current, [1, 2], marker=400, padding=2)
            self.assertIs(current._h3_kv_arenas[0][0], arena)
            committed = self.snapshot(current)
            self.merge(current, list(range(3, 9)), marker=500, padding=2, store=False)
            self.assertEqual(len(current._h3_kv_arenas[0][0]), 12)
            self.assertIsNot(current._h3_kv_arenas[0][0], arena)
            self.assert_snapshot(current, committed)
            if current is cache:
                self.assert_snapshot(child, frozen)


if __name__ == '__main__':
    unittest.main()
