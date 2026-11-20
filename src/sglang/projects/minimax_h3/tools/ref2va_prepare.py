"""Aligned RGB/proxy bundles -> shared Ref2VA cache with static image/Qwen conditioning.

Indexing freezes a validated snapshot; encoding is resumable and shards by bundle.
Only the first RGB frame and the fixed instruction enter Qwen. Proxy pixels enter
the frozen VAE, independently in 17-frame blocks (five-frame EOS).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from projects.minimax_h3.data.ref_latent import (
    SCHEMA, STATIC_CONDITIONING_MODE, load_entry, packed_length, read_manifest, validate_sample,
)
from projects.minimax_h3.tools.cwm_prepare import atomic_json, digest_file, rows, save_sample

WINDOW = 158
FILES = ('rgb.mkv', 'proxy.mkv', 'actions.jsonl', 'camera.jsonl')
PROMPT = Path(__file__).parents[1] / 'trials/ref2va/system_prompt.txt'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stamp(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def encoder_stamp(path):
    s = Path(path).stat()
    return {k: getattr(s, 'st_' + k) for k in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


@contextmanager
def lock(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def inspect_bundle(path):
    """Validate all metadata rows and declared file sizes without decoding videos."""
    path = Path(path).resolve()
    before = {name: stamp(path / name) for name in FILES}
    raw = (path / 'actions.jsonl').read_bytes()
    first, payload = raw.split(b'\n', 1)
    header = json.loads(first)
    if (header.get('record_type'), header.get('schema'), header.get('fps')) != (
        'header', 'rgb-proxy-actions/v2', 24
    ):
        raise ValueError('unsupported Ref2VA header')
    n = header['frame_count']
    if type(n) is not int or n <= 0 or n % WINDOW:
        raise ValueError('expected complete, unpadded 158-frame windows')
    if any(header.get(k) != v for k, v in dict(window_frames=WINDOW, stride_frames=WINDOW, tail_padding_frames=0).items()):
        raise ValueError('unexpected window/tail policy')
    if hashlib.sha256(payload).hexdigest() != header['actions_payload_sha256']:
        raise ValueError('actions payload checksum mismatch')
    camera_raw = (path / 'camera.jsonl').read_bytes()
    camera_info = header['camera_file']
    if camera_info['path'] != 'camera.jsonl' or camera_info['frame_count'] != n:
        raise ValueError('camera header mismatch')
    if hashlib.sha256(camera_raw).hexdigest() != camera_info['sha256']:
        raise ValueError('camera checksum mismatch')
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if any(r.get('record_type') not in ('frame', 'event', 'chunk') for r in records):
        raise ValueError('unknown actions record type')
    frames = [r for r in records if r['record_type'] == 'frame']
    cameras = [json.loads(line) for line in camera_raw.splitlines() if line.strip()]
    if len(frames) != n or len(cameras) != n:
        raise ValueError('RGB/action/camera frame counts disagree')
    for i, (frame, camera) in enumerate(zip(frames, cameras)):
        if any(frame[k] != i for k in ('frame', 'rgb_frame', 'proxy_frame', 'camera_frame')):
            raise ValueError(f'actions frame mapping differs at {i}')
        if any(camera[k] != i for k in ('frame', 'rgb_frame', 'proxy_frame', 'action_frame')):
            raise ValueError(f'camera frame mapping differs at {i}')
        for row in (frame, camera):
            if not math.isfinite(row['video_seconds']) or abs(row['video_seconds'] - i / 24) > 1e-6:
                raise ValueError(f'frame time differs at {i}')
            if not math.isfinite(row['camera_seconds']):
                raise ValueError('nonfinite camera time')
        if abs(frame['camera_seconds'] - camera['camera_seconds']) > 1e-6:
            raise ValueError('action/camera time mismatch')
    chunks = [r for r in records if r['record_type'] == 'chunk']
    expected = [(w + a, w + min(a + 17, WINDOW)) for w in range(0, n, WINDOW) for a in range(0, WINDOW, 17)]
    if [(c['frame_start'], c['frame_end_exclusive']) for c in chunks] != expected:
        raise ValueError('chunk records disagree with frame windows')
    if any(not c.get('window_complete') or not c.get('chunk_complete') for c in chunks):
        raise ValueError('incomplete chunk')
    source_id = header['source_render']['source_id']
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('missing trajectory identity')
    media = {m['path']: m for m in header['media']}
    if set(media) != {'rgb.mkv', 'proxy.mkv'}:
        raise ValueError('unexpected media paths')
    media['camera.jsonl'] = camera_info
    source_files = {}
    for name in FILES:
        info = before[name]
        if info['size'] <= 0 or info != stamp(path / name):
            raise ValueError(f'file changed during indexing: {name}')
        sha = hashlib.sha256(raw).hexdigest() if name == 'actions.jsonl' else media[name]['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid source checksum')
        if name != 'actions.jsonl' and media[name]['bytes'] != info['size']:
            raise ValueError(f'file size mismatch: {name}')
        source_files[str(path / name)] = dict(info, sha256=sha)
    return dict(bundle_path=str(path), take_id=source_id, frame_count=n,
                rgb_resolution=header['rgb_resolution'], proxy_resolution=header['proxy_resolution'],
                proxy_style=header['proxy_video']['style_version'], source_files=source_files,
                scene=header['configured_scene']['scene'], behavior=header['configured_scene']['behavior'])


def index_dataset(args):
    root, out = Path(args.data_root).resolve(), Path(args.output).resolve()
    if not 0 < args.val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    options = dict(data_root=str(root), seed=args.seed, val_fraction=args.val_fraction, window_frames=WINDOW)
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / '.index.lock'):
        if (out / 'index.jsonl').exists():
            snapshot = json.loads((out / 'snapshot.json').read_text())
            if any(snapshot[k] != v for k, v in options.items()) or digest_file(out / 'index.jsonl') != snapshot['index_sha256']:
                raise ValueError('existing snapshot differs; choose a new output directory')
            print(json.dumps(dict(event='snapshot_reused', **snapshot)), flush=True)
            return
        bundles, rejected = [], []
        for group in sorted(root.iterdir()):
            if group.name.startswith('.') or not group.is_dir():
                continue
            for path in sorted(group.iterdir()):
                if path.name.startswith('.') or not path.is_dir():
                    continue
                try:
                    if root not in path.resolve().parents:
                        raise ValueError('bundle escapes data root')
                    bundles.append(inspect_bundle(path))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    rejected.append(dict(path=str(path), error=str(error)))
        atomic_json(out / 'rejected.json', rejected)
        if rejected and not args.skip_incomplete:
            raise ValueError(f'{len(rejected)} bundles failed validation; see rejected.json (explicit --skip-incomplete to exclude)')
        splits = {b['take_id']: ('val' if int(digest([args.seed, b['take_id']])[:16], 16) / 2**64 < args.val_fraction else 'train')
                  for b in bundles}
        if set(splits.values()) != {'train', 'val'}:
            raise ValueError('snapshot needs both train and val trajectories; change seed/fraction for a small corpus')
        index = []
        for bundle in bundles:
            for start in range(0, bundle['frame_count'], WINDOW):
                sample_id = 'ref2va_' + digest([str(Path(bundle['bundle_path']).relative_to(root)), start])[:24]
                index.append(dict(bundle, sample_id=sample_id, local_start=start, local_end=start + WINDOW,
                                  split=splits[bundle['take_id']], conditioning_mode=STATIC_CONDITIONING_MODE))
        index_hash = hashlib.sha256(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in index).encode()).hexdigest()
        snapshot = dict(options, bundles=len(bundles), trajectories=len(splits), windows=len(index), rejected=len(rejected),
                        split_windows=dict(Counter(j['split'] for j in index)), index_sha256=index_hash)
        atomic_json(out / 'split.json', splits)
        atomic_json(out / 'snapshot.json', snapshot)
        atomic_json(out / 'index.jsonl', index, lines=True)  # Publish the index last.
        print(json.dumps(dict(event='indexed', **snapshot)), flush=True)


def iter_video_chunks(path, *, start=0, num_frames, height, width, chunk_frames=17, nearest=False):
    """Decode only requested ordinals; yield before reading the following chunk."""
    import av
    if min(num_frames, height, width, chunk_frames) <= 0 or start < 0:
        raise ValueError('invalid frame range or dimensions')
    resize = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    chunk, count = [], 0
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError('expected one video stream and no audio')
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - 24) > 1e-6:
            raise ValueError('expected 24 FPS')
        stream.codec_context.thread_count = 2
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.pts is None or abs(float(frame.pts * stream.time_base) - ordinal / 24) > .002:
                raise ValueError(f'video PTS disagrees with ordinal {ordinal}: {path}')
            if ordinal < start:
                continue
            pixels = frame.to_image().convert('RGB')
            if pixels.size != (width, height):
                pixels = pixels.resize((width, height), resize)
            chunk.append(np.asarray(pixels))
            count += 1
            if len(chunk) == chunk_frames or count == num_frames:
                yield np.stack(chunk)
                chunk = []
            if count == num_frames:
                return
    raise ValueError(f'truncated video: wanted {num_frames} frames at {start}, read {count}: {path}')


def encoder_recipe(args):
    out, checkpoint = Path(args.output).resolve(), Path(args.checkpoint).resolve()
    prompt = Path(args.prompt_file).read_text().strip()
    if not prompt:
        raise ValueError('fixed prompt must not be empty')
    for n in (args.height, args.width, args.proxy_height, args.proxy_width):
        if n <= 0 or n % 32:
            raise ValueError('RGB/proxy dimensions must be positive multiples of 32')
    with lock(out / '.recipe.lock'):
        hash_path = out / 'encoder_hashes.json'
        saved = json.loads(hash_path.read_text()) if hash_path.exists() else {}
        hashes, cached = {}, {}
        for folder in ('text_encoder', 'video_vae/source', 'processor'):
            paths = sorted(p for p in (checkpoint / folder).iterdir() if p.is_file())
            if not paths:
                raise ValueError(f'missing checkpoint components: {folder}')
            for p in paths:
                key, info = str(p.relative_to(checkpoint)), encoder_stamp(p)
                previous = saved.get(str(p), {})
                sha = previous.get('sha256') if previous.get('stamp') == info else digest_file(p)
                if encoder_stamp(p) != info:
                    raise ValueError(f'encoder file changed: {p}')
                hashes[key] = sha
                cached[str(p)] = dict(stamp=info, sha256=sha)
        atomic_json(hash_path, cached)
        base = Path(__file__).parents[1]
        code = {str(p.relative_to(base)): digest_file(p) for p in (
            Path(__file__), base / 'modeling/cwm_encoders.py', base / 'modeling/text_encoder.py', base / 'configs/video_vae.yaml')}
        recipe = dict(conditioning_mode=STATIC_CONDITIONING_MODE, checkpoint=str(checkpoint), model_hashes=hashes,
                      encoder_code=code, index_sha256=digest_file(args.index), system_prompt=prompt,
                      prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), height=args.height, width=args.width,
                      proxy_height=args.proxy_height, proxy_width=args.proxy_width, posterior='mean', seed=args.seed,
                      qwen_layer=50, qwen_proxy=False, first_frame=True, fps=24, num_frames=WINDOW)
        recipe['fingerprint'] = digest(recipe)
        current = out / 'recipe.json'
        if current.exists() and json.loads(current.read_text()) != recipe:
            raise ValueError('cache recipe changed; choose a new output directory')
        atomic_json(current, recipe)
        cache = out / 'samples' / recipe['fingerprint'][:16]
        cache.mkdir(parents=True, exist_ok=True)
        atomic_json(cache / 'config.json', recipe)
    return recipe, cache


def check_sources(job, *, checksums=False):
    for path, expected in job['source_files'].items():
        if stamp(path) != {k: expected[k] for k in ('size', 'mtime_ns')}:
            raise ValueError(f'source changed since indexing: {path}')
        if checksums and digest_file(path) != expected['sha256']:
            raise ValueError(f'source checksum mismatch: {path}')


def encode_dataset(args):
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
    if not 0 <= args.rank < args.shards:
        raise ValueError('invalid shard')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    recipe, cache = encoder_recipe(args)
    jobs = rows(args.index)
    wanted = set(Path(args.sample_ids).read_text().split()) if args.sample_ids else None
    if wanted is not None and not wanted <= {j['sample_id'] for j in jobs}:
        raise ValueError('sample selection contains unknown IDs')
    groups = defaultdict(list)
    for job in jobs:
        if wanted is None or job['sample_id'] in wanted:
            groups[job['bundle_path']].append(job)
    owned = [p for i, p in enumerate(sorted(groups)) if i % args.shards == args.rank]
    remaining, encoders = args.limit, None
    if remaining == 0:
        print(json.dumps(dict(event='recipe_verified', fingerprint=recipe['fingerprint'])), flush=True)
        return
    if args.device == 'cuda':
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    for bundle in owned:
        if remaining == 0:
            break
        with lock(out / '.locks' / (digest(bundle) + '.lock'), blocking=False):
            pending = []
            for job in groups[bundle]:
                sidecar = cache / (job['sample_id'] + '.json')
                if sidecar.exists():
                    load_entry(out / 'train.jsonl', json.loads(sidecar.read_text()), verify_hash=True)
                else:
                    pending.append(job)
            if not pending:
                continue
            check_sources(pending[0], checksums=True)
            if encoders is None:
                encoders = FrozenCWMEncoders(args.checkpoint, device=args.device, posterior='mean')
            # A sequential pass per video/bundle; retain only one window in RAM.
            source = groups[bundle][0]
            rgb = iter_video_chunks(Path(bundle) / 'rgb.mkv', start=0, num_frames=source['frame_count'],
                                    height=args.height, width=args.width, chunk_frames=WINDOW)
            proxy = iter_video_chunks(Path(bundle) / 'proxy.mkv', start=0, num_frames=source['frame_count'],
                                      height=args.proxy_height, width=args.proxy_width, chunk_frames=WINDOW, nearest=True)
            by_start = {j['local_start']: j for j in pending}
            for window, (frames, proxies) in enumerate(zip(rgb, proxy)):
                job = by_start.get(window * WINDOW)
                if job is None:
                    continue
                seed = (args.seed + int(digest(job['sample_id'])[:8], 16)) % 2**32
                static = encoders.condition(recipe['system_prompt'], images=[frames[0]], videos=[])
                sample = dict(video=encoders.video(frames, seed=seed), audio=torch.empty(2, 32, 0),
                    references=[dict(kind='image', role='first_frame', video=encoders.image(frames[0], seed=seed)),
                                dict(kind='video', role='semantic', video=encoders.video(proxies, seed=seed))],
                    teacher_condition=static, student_condition=dict(static=static, chunks=[]),
                    metadata=dict(schema=SCHEMA, conditioning_mode=STATIC_CONDITIONING_MODE,
                        sample_id=job['sample_id'], fingerprint=recipe['fingerprint'], take_id=job['take_id'],
                        bundle_path=bundle, local_start=job['local_start'], fps=24, num_frames=WINDOW,
                        source_frames=list(range(job['local_start'], job['local_end'])), normalized=True,
                        condition_seed=seed, system_prompt=recipe['system_prompt'], prompt_sha256=recipe['prompt_sha256'],
                        proxy_style=job['proxy_style'], rgb_resolution=[args.width, args.height],
                        proxy_resolution=[args.proxy_width, args.proxy_height],
                        chunk_raw_ranges=[[a, min(a + 17, WINDOW)] for a in range(0, WINDOW, 17)],
                        chunk_latent_ranges=[[a, min(a + 5, 47)] for a in range(0, 47, 5)]))
                validate_sample(sample)
                check_sources(job)
                target = cache / (job['sample_id'] + '.pt')
                save_sample(target, sample)
                record = dict(schema=SCHEMA, conditioning_mode=STATIC_CONDITIONING_MODE,
                    sample_id=job['sample_id'], take_id=job['take_id'], split=job['split'],
                    fingerprint=recipe['fingerprint'], path=str(target.relative_to(out)), sha256=digest_file(target),
                    latent_shape=list(sample['video'].shape), references=['first_frame', 'semantic'],
                    lengths={s: packed_length(sample, s) for s in ('sft', 'tf', 'dmd')})
                validate_sample(load_entry(out / 'train.jsonl', record, verify_hash=True))
                atomic_json(target.with_suffix('.json'), record)
                print(json.dumps(dict(event='encoded', sample_id=job['sample_id'], lengths=record['lengths'])), flush=True)
                del sample, static
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        break
            rgb.close(); proxy.close()


def finalize(args):
    out = Path(args.output).resolve()
    with lock(out / '.finalize.lock'):
        _finalize(args, out)


def _finalize(args, out):
    recipe = json.loads((out / 'recipe.json').read_text())
    if recipe['index_sha256'] != digest_file(args.index):
        raise ValueError('index changed after encoding')
    expected = {j['sample_id']: j for j in rows(args.index)}
    cache = out / 'samples' / recipe['fingerprint'][:16]
    records = []
    for path in sorted(cache.glob('ref2va_*.json')):
        r = json.loads(path.read_text())
        j = expected.get(r['sample_id'])
        if j is None or r['fingerprint'] != recipe['fingerprint'] or any(r[k] != j[k] for k in ('take_id', 'split')):
            raise ValueError('cache entry is not part of this snapshot')
        validate_sample(load_entry(out / 'train.jsonl', r, verify_hash=True))
        records.append(r)
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('empty or duplicate cache')
    if len(records) != len(expected) and not args.allow_partial:
        raise ValueError(f'{len(records)}/{len(expected)} complete; explicit --allow-partial is required for experiments')
    for split in ('train', 'val'):
        atomic_json(out / f'{split}.jsonl', [r for r in records if r['split'] == split], lines=True)
    status = dict(complete=len(records) == len(expected), encoded=len(records), expected=len(expected),
                  fingerprint=recipe['fingerprint'], splits=dict(Counter(r['split'] for r in records)))
    atomic_json(out / 'status.json', status)
    print(json.dumps(status), flush=True)


def verify(args):
    path, entries, _ = read_manifest(args.manifest)
    for entry in entries:
        sample = load_entry(path, entry, verify_hash=True)
        validate_sample(sample)
        if sample['metadata'].get('conditioning_mode') != STATIC_CONDITIONING_MODE:
            raise ValueError('not a Ref2VA static-image/proxy-VAE sample')
        if any(entry['lengths'][s] != packed_length(sample, s) for s in ('sft', 'tf', 'dmd')):
            raise ValueError('packed token budget mismatch')
    print(json.dumps(dict(verified=len(entries), manifest=str(path))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('index'); p.add_argument('--data-root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026); p.add_argument('--val-fraction', type=float, default=.1)
    p.add_argument('--skip-incomplete', action='store_true', help='Record and explicitly exclude bundles failing metadata validation')
    p = sub.add_parser('encode'); p.add_argument('--index', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--prompt-file', default=str(PROMPT))
    p.add_argument('--height', type=int, default=768); p.add_argument('--width', type=int, default=1344)
    p.add_argument('--proxy-height', type=int, default=384); p.add_argument('--proxy-width', type=int, default=672)
    p.add_argument('--seed', type=int, default=2026); p.add_argument('--device', default='cuda')
    p.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)))
    p.add_argument('--shards', type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    p.add_argument('--sample-ids'); p.add_argument('--limit', type=int)
    p = sub.add_parser('finalize'); p.add_argument('--index', required=True); p.add_argument('--output', required=True)
    p.add_argument('--allow-partial', action='store_true')
    p = sub.add_parser('verify'); p.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if getattr(args, 'limit', None) is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    dict(index=index_dataset, encode=encode_dataset, finalize=finalize, verify=verify)[args.command](args)


if __name__ == '__main__':
    main()
