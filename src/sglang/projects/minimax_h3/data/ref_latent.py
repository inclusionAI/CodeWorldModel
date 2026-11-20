"""Versioned CWM latent cache and a resumable, token-budgeted dataset."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from common.data import WorkerResumeContext, WorkerStateEnvelope, WorkerStateLoadError

from projects.minimax_h3.modeling.ref_recipe import (CAUSAL_LAYOUT, DEFAULT_WINDOW, DEFAULT_ROLE_SPAN,
    LEGACY_CONDITIONING_MODE, STATIC_CONDITIONING_MODE, conditioning_mode, causal_layout)

SCHEMA = 'cwm_ref2va_v1'
ROLLOUT_PROTOCOL = 'scheduler_native_precision_v1'


def packed_length(sample, stage='tf'):
    rows = lambda v: v.shape[1] * v.shape[2] * v.shape[3] // 4
    refs = sum(rows(r['video']) for r in sample['references'])
    rgb = rows(sample['video'])
    if stage == 'sft':
        return len(sample['teacher_condition']['prompt_embeds']) + refs + rgb
    condition = sample['student_condition']
    # Static-image mode has no chunk Qwen documents; its chunk list is empty.
    a,b = sample['metadata']['chunk_latent_ranges'][-1]
    tail = (b-a) * sample['video'].shape[2] * sample['video'].shape[3] // 4
    causal = (len(condition['static']['prompt_embeds']) +
              sum(len(c['prompt_embeds']) for c in condition['chunks']) + refs + 2*rgb-tail)
    return max(causal, packed_length(sample, 'sft')) if stage == 'dmd' else causal


def validate_sample(sample, *, check_values=True, streaming=False):
    meta = sample['metadata']
    mode = conditioning_mode(meta)
    if meta.get('schema') != SCHEMA or meta.get('fps') != 24 or not meta.get('normalized'):
        raise ValueError('cache must be normalized CWM Ref2VA v1 at 24 FPS')
    if not isinstance(meta.get('condition_seed'), int):
        raise ValueError('condition_seed is required')
    video = sample['video']
    if video.ndim != 4 or min(video.shape) <= 0 or video.shape[-1] % 2 or video.shape[-2] % 2:
        raise ValueError('video must be nonempty [C,T,H,W] with 2x2 patch alignment')
    if tuple(sample['audio'].shape[:1]) != (2,) or sample['audio'].ndim != 3 or sample['audio'].shape[-1] != 0:
        raise ValueError('CWM has no audio supervision: use [2,Ca,0], not silence')
    ranges = [list(r) for r in meta['chunk_latent_ranges']]
    expected = [[a,min(a+5,video.shape[1])] for a in range(0,video.shape[1],5)]
    if ranges != expected:
        raise ValueError('latent chunks must be consecutive 5-latent blocks plus the EOS tail')
    raw = meta['chunk_raw_ranges']
    if len(raw) != len(ranges) or raw != [[a,min(a+17,meta['num_frames'])] for a in range(0,meta['num_frames'],17)]:
        raise ValueError('raw frame chunk boundaries disagree with latent boundaries')
    open_chunk = streaming and meta['num_frames'] % 17 == 0
    if not open_chunk and meta['num_frames'] % 17 != 5:
        raise ValueError('video length violates the native H3 EOS convention')
    if video.shape[1] != 5 * ((meta['num_frames']+16)//17) - (0 if open_chunk else 3):
        raise ValueError('video length violates the native H3 EOS convention')
    if type(meta.get('stream_chunk_start',0)) is not int or meta.get('stream_chunk_start',0) < 0:
        raise ValueError('stream_chunk_start must be a nonnegative integer')
    kinds = [(r['kind'],r['role']) for r in sample['references']]
    image = kinds[:1] == [('image','first_frame')]
    if mode == STATIC_CONDITIONING_MODE and not image:
        raise ValueError('static-image conditioning requires a first_frame image reference')
    if kinds[int(image):] not in ([('video','semantic')],[('video','semantic'),('video','depth')]):
        raise ValueError('references must be an optional first_frame image, then semantic, then optional depth')
    for ref in sample['references']:
        expected = (video.shape[0],1) if ref['kind'] == 'image' else video.shape[:2]
        if ref['video'].shape[:2] != expected:
            raise ValueError('reference latent channels/time must align with RGB')
        if ref['video'].shape[-1] % 2 or ref['video'].shape[-2] % 2:
            raise ValueError('reference spatial shape must be patch aligned')
    student = sample['student_condition']
    if mode == STATIC_CONDITIONING_MODE:
        if student['chunks'] != []:
            raise ValueError('static-image conditioning requires an empty chunk Qwen list')
    elif len(student['chunks']) != len(ranges):
        raise ValueError('one independently encoded Qwen condition is required per chunk')
    conditions = [sample['teacher_condition'], student['static'], *student['chunks']]
    width = conditions[0]['prompt_embeds'].shape[-1]
    for cond in conditions:
        embed, tags = cond['prompt_embeds'], cond['text_token_tags']
        if embed.ndim != 2 or len(embed) == 0 or embed.shape[1] != width or tags.shape != (len(embed),):
            raise ValueError('Qwen embeddings and live presentation tags must have identical rows')
        if check_values and (not torch.isfinite(embed).all() or not ((tags==0)|(tags==1)).all()):
            raise ValueError('invalid condition values/tags')
    if mode == STATIC_CONDITIONING_MODE:
        teacher, static = sample['teacher_condition'], student['static']
        for key in ('prompt_embeds', 'text_token_tags'):
            same_geometry = teacher[key].shape == static[key].shape and teacher[key].dtype == static[key].dtype
            if not same_geometry or (check_values and not torch.equal(teacher[key], static[key])):
                raise ValueError('static-image teacher and student must share identical Qwen conditioning')
    guidance = sample.get('guidance_conditions', {})
    if not isinstance(guidance, dict):
        raise ValueError('guidance_conditions must map dropped conditions to encoded documents')
    for drop, branch in guidance.items():
        if drop not in ('proxy', 'prompt', 'prompt_proxy') or not image:
            raise ValueError('guidance conditions require a supported drop and first_frame reference')
        try:
            teacher, weak_student = branch['teacher_condition'], branch['student_condition']
            chunks = weak_student['chunks']
            weak_docs = [teacher, weak_student['static'], *chunks]
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid guidance condition documents') from exc
        expected = len(student['chunks']) if drop == 'prompt' else 0
        if not isinstance(chunks, list) or len(chunks) != expected:
            raise ValueError('guidance chunk documents must match the retained proxy conditions')
        original_docs = [sample['teacher_condition'], student['static'],
                         *(student['chunks'] if drop == 'prompt' else [])]
        for doc, original in zip(weak_docs, original_docs, strict=True):
            if not isinstance(doc, dict) or not all(torch.is_tensor(doc.get(k)) for k in ('prompt_embeds', 'text_token_tags')):
                raise ValueError('guidance documents require Qwen embeddings and token tags')
            embed, tags = doc['prompt_embeds'], doc['text_token_tags']
            if (embed.ndim != 2 or not len(embed) or embed.shape[1] != width or
                    tags.shape != (len(embed),) or embed.dtype != original['prompt_embeds'].dtype or
                    tags.dtype != original['text_token_tags'].dtype):
                raise ValueError('guidance Qwen geometry/dtype must match the full condition')
            if check_values and (not torch.isfinite(embed).all() or not ((tags == 0) | (tags == 1)).all()):
                raise ValueError('invalid guidance condition values/tags')
        if mode == STATIC_CONDITIONING_MODE:
            for key in ('prompt_embeds', 'text_token_tags'):
                static = weak_student['static'][key]
                if teacher[key].shape != static.shape or (check_values and not torch.equal(teacher[key], static)):
                    raise ValueError('static-image guidance teacher/student must share identical Qwen conditioning')
        weak = dict(sample, teacher_condition=teacher, student_condition=weak_student,
                    references=sample['references'] if drop == 'prompt' else sample['references'][:1])
        # A live chunk uses only causal geometry; its unused teacher document
        # is represented by the static prompt, not a full native document.
        stages = ('tf',) if streaming else ('sft', 'tf', 'dmd')
        if any(packed_length(weak, stage) > packed_length(sample, stage) for stage in stages):
            raise ValueError('guidance packed length exceeds the full-condition sequence budget')
    if check_values:
        for value in [video, *[r['video'] for r in sample['references']]]:
            if not torch.isfinite(value).all():
                raise ValueError('non-finite cached latent')


def read_manifest(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    entries = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not entries or any(e.get('schema') != SCHEMA for e in entries):
        raise ValueError(f'{path}: missing/unsupported cache manifest')
    if len({e['sample_id'] for e in entries}) != len(entries):
        raise ValueError('duplicate sample IDs')
    if len({e['fingerprint'] for e in entries}) != 1:
        raise ValueError('mixed encoder/preprocessing fingerprints')
    if len({conditioning_mode(e) for e in entries}) != 1:
        raise ValueError('mixed conditioning modes in manifest')
    # Cached rollout batches and distributed FSDP use the same chunk count.
    if len({tuple(e['latent_shape'][:2]) for e in entries}) != 1:
        raise ValueError('one manifest must have uniform channel/time geometry')
    return path, entries, hashlib.sha256(data).hexdigest()


def load_entry(manifest_path, entry, *, verify_hash=False):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = Path(manifest_path).parent / path
    if verify_hash and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'cache hash mismatch: {path}')
    sample = torch.load(path, map_location='cpu', weights_only=True)
    meta = sample['metadata']
    if meta['sample_id'] != entry['sample_id'] or meta['fingerprint'] != entry['fingerprint']:
        raise ValueError(f'cache identity mismatch: {path}')
    if conditioning_mode(meta) != conditioning_mode(entry):
        raise ValueError(f'cache/manifest conditioning mode mismatch: {path}')
    validate_sample(sample, check_values=False)
    drop = entry.get('guidance_drop')
    if drop is not None and (drop not in ('proxy', 'prompt', 'prompt_proxy') or
                             drop not in sample.get('guidance_conditions', {})):
        raise ValueError(f'cache/manifest guidance_drop mismatch: {path}')
    return sample


class RefLatentDataset(IterableDataset):
    def __init__(self, seed: int, resume_context: WorkerResumeContext, *, manifest: str,
                 max_seqlen: int, stage='tf', max_samples=1, window_size=DEFAULT_WINDOW,
                 rope_reference_latents=DEFAULT_ROLE_SPAN, conditioning_mode=None):
        self.path, self.entries, digest = read_manifest(manifest)
        if conditioning_mode is not None and conditioning_mode != self.entries[0].get('conditioning_mode', LEGACY_CONDITIONING_MODE):
            raise ValueError('dataset/config conditioning mode mismatch')
        if stage not in ('sft','tf','tscd','dmd') or max_seqlen <= 0 or max_samples <= 0:
            raise ValueError('invalid stage or batch budget')
        self.seed, self.resume_context = seed, resume_context
        self.stage, self.max_seqlen, self.max_samples = stage, max_seqlen, max_samples
        identity = [digest,seed,stage,max_seqlen,max_samples,window_size]
        if stage != 'sft':
            identity += list(causal_layout(window_size,rope_reference_latents))
        if stage == 'dmd':
            identity.append(ROLLOUT_PROTOCOL)
        self.fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.lengths = [e['lengths']['sft' if stage=='sft' else ('dmd' if stage=='dmd' else 'tf')] for e in self.entries]
        if max(self.lengths) > max_seqlen:
            raise ValueError(f'cache requires max_seqlen >= {max(self.lengths)}; no samples are silently skipped')
        self.states = {}
        for worker, raw in resume_context.committed_states.items():
            try:
                state = json.loads(raw)
                if state['fingerprint'] != self.fingerprint or state['worker'] != worker or state['cursor'] < 0:
                    raise ValueError('state identity mismatch')
                self.states[worker] = state['cursor']
            except (ValueError, KeyError, TypeError) as exc:
                raise WorkerStateLoadError('CWM manifest/config/causal layout changed since checkpoint; start a new run') from exc

    def _state(self, worker, cursor):
        return json.dumps(dict(fingerprint=self.fingerprint,worker=worker,cursor=cursor)).encode()

    def initial_worker_snapshot(self, logical_worker_id):
        return self._state(logical_worker_id, 0)

    def __iter__(self):
        context = self.resume_context
        info = get_worker_info()
        workers = context.num_workers or 1
        if (info.num_workers if info else 1) != workers:
            raise ValueError('worker topology changed')
        worker = ((info.id if info else 0)+context.next_logical_worker_id) % workers
        cursor = self.states.get(worker, 0)
        epoch, order = -1, None
        def index_at(step):
            nonlocal epoch, order
            absolute = step * context.world_size * workers + context.rank * workers + worker
            current, local = divmod(absolute,len(self.entries))
            if current != epoch:
                epoch, order = current, list(range(len(self.entries)))
                random.Random(self.seed+epoch).shuffle(order)
            return order[local]
        while True:
            samples, length = [], 0
            while len(samples) < self.max_samples:
                index = index_at(cursor)
                if samples and length+self.lengths[index] > self.max_seqlen:
                    break
                sample = load_entry(self.path,self.entries[index])
                if packed_length(sample,self.stage) != self.lengths[index]:
                    raise ValueError('cached sequence budget disagrees with manifest')
                samples.append(sample)
                length += self.lengths[index]
                cursor += 1
            yield WorkerStateEnvelope({'ref_samples':samples},worker,self._state(worker,cursor))
