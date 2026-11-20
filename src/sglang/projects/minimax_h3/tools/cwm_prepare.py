"""CWM indexing, frozen-encoder caching and integrity checks.

Run with ``python -m projects.minimax_h3.tools.cwm_prepare --help``. Indexing and
verification need no model/GPU. Encoding shards by take, decodes each take once
into temporary memmaps, and respects every window's independent VAE boundary.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from projects.minimax_h3.data.ref_latent import SCHEMA, packed_length, validate_sample, read_manifest, load_entry


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value, *, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        if lines:
            for row in value:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        else:
            json.dump(value,f,ensure_ascii=False,indent=2)
        f.flush(); os.fsync(f.fileno())
        name = f.name
    os.replace(name,path)


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def index_dataset(args):
    root, out = Path(args.data_root).resolve(),Path(args.output).resolve()
    contract = json.loads((root/'rigid/metadata/reader_contract.json').read_text())
    if contract['fps'] != 24 or contract['clip_frames'] != 124:
        raise ValueError('unsupported CWM reader contract')
    fields = ('job_ordinal','take_id','take_local_start_frame_inclusive','take_local_end_frame_exclusive')
    captions = {}
    for caption in rows(root/contract['captions']):
        key = tuple(caption[k] for k in fields)
        if key in captions:
            raise ValueError(f'duplicate caption {key}')
        captions[key] = caption
    frame_map = defaultdict(dict)
    with gzip.open(root/contract['frame_index'],'rt') as f:
        for line in f:
            row = json.loads(line)
            take,local,source = row['take_id'],row['take_local_frame'],row['source_video_frame']
            if local in frame_map[take] or min(local,source)<0:
                raise ValueError('duplicate/negative frame ordinal')
            frame_map[take][local] = source
    system = (root/contract['system_prompt']).read_text().strip()
    jobs = rows(root/contract['jobs'])
    take_ids = sorted({j['take_id'] for j in jobs})
    if not 0 < args.val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    order = sorted(take_ids,key=lambda t:hashlib.sha256(f'{args.seed}:{t}'.encode()).digest())
    val = set(order[:max(1,round(len(order)*args.val_fraction))])
    if args.split_file:
        split_map = json.loads(Path(args.split_file).read_text())
        if set(split_map) != set(take_ids) or not set(split_map.values()) <= {'train','val'}:
            raise ValueError('split_file must map every take to train or val')
        val = {k for k,v in split_map.items() if v=='val'}
    source_digest = {p:digest_file(root/p) for p in (contract['jobs'],contract['frame_index'],contract['captions'],contract['system_prompt'],contract['semantic_codebook'])}
    result = []
    for job in jobs:
        key = tuple(job[k] for k in fields)
        caption = captions[key]
        take,a,b = key[1:]
        if b-a != 124:
            raise ValueError(f'job {key}: expected 124 frames')
        mapping = [frame_map[take][local] for local in range(a,b)]
        if mapping != list(range(mapping[0],mapping[0]+124)):
            raise ValueError(f'job {key}: non-contiguous RGB mapping; never bridge gaps')
        if mapping[0] != job['source_video_frame_start_inclusive'] or mapping[-1]+1 != job['source_video_frame_end_exclusive']:
            raise ValueError('jobs/frame-index mapping mismatch')
        if caption.get('source_video_frame_first',mapping[0]) != mapping[0] or caption.get('source_video_frame_last',mapping[-1]) != mapping[-1]:
            raise ValueError('caption/frame-index mapping mismatch')
        body = caption.get('caption_body_en',caption.get('caption'))
        if not body or system in body:
            raise ValueError('caption must be body-only and nonempty')
        rgb = root/contract['rgb_path_template'].format(take_id=take)
        proxy = root/contract['proxy_archive_template'].format(take_id=take)
        media = {str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (rgb,proxy)}
        result.append(dict(sample_id=f'job_{key[0]:06d}',take_id=take,job_ordinal=key[0],
            local_start=a,local_end=b,source_frames=mapping,rgb_path=str(rgb),proxy_path=str(proxy),
            source_files=media,prompt=system+'\n\n'+body,system_prompt=system,
            split='val' if take in val else 'train',source_metadata=source_digest,data_root=str(root)))
    if len({r['sample_id'] for r in result}) != len(result):
        raise ValueError('duplicate jobs')
    atomic_json(out/'index.jsonl',result,lines=True)
    atomic_json(out/'split.json',{t:('val' if t in val else 'train') for t in take_ids})
    print(json.dumps(dict(windows=len(result),takes=len(take_ids),val_takes=len(val),index=str(out/'index.jsonl'))))


def prepare_take(jobs, scratch, height, width, roles, stack):
    """One sequential RGB decode and proxy archive pass; no extractall/seeks."""
    import av
    import zstandard
    job = jobs[0]
    mapping = {}
    for j in jobs:
        for local,source in zip(range(j['local_start'],j['local_end']),j['source_frames']):
            if local in mapping and mapping[local] != source:
                raise ValueError('overlapping jobs disagree about frame mapping')
            mapping[local] = source
    local_frames = sorted(mapping)
    local_to_row = {f:i for i,f in enumerate(local_frames)}
    values = {role:np.lib.format.open_memmap(Path(scratch)/f'{role}.npy',mode='w+',dtype=np.uint8,
              shape=(len(local_frames),height,width,3)) for role in ['rgb',*roles]}
    for value in values.values():
        stack.callback(value._mmap.close)
    found = {role:set() for role in values}
    source_to_local = {source:local for local,source in mapping.items()}
    if len(source_to_local) != len(mapping):
        raise ValueError('RGB mapping is not one-to-one')
    last_source = max(source_to_local)
    with av.open(job['rgb_path']) as container:
        if container.streams.audio:
            raise ValueError('CWM audio contract changed; handle supervision explicitly')
        stream = container.streams.video[0]
        if abs(float(stream.average_rate)-24)>1e-4:
            raise ValueError('RGB FPS differs from 24')
        stream.thread_type = 'AUTO'
        for ordinal,frame in enumerate(container.decode(stream)):
            if ordinal > last_source:
                break
            if ordinal in source_to_local:
                local = source_to_local[ordinal]
                values['rgb'][local_to_row[local]] = np.asarray(frame.to_image().resize((width,height),Image.Resampling.LANCZOS))
                found['rgb'].add(local)
    codebook = json.loads((Path(job['data_root'])/'rigid/metadata/semantic_codebook.json').read_text())
    palette = np.array([c['visualization_rgb'] for c in sorted(codebook['classes'],key=lambda c:c['id'])],dtype=np.uint8)
    with open(job['proxy_path'],'rb') as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader,mode='r|') as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    if len(parts) < 2 or not member.isfile():
                        continue
                    folder,filename = parts[-2:]
                    role = 'semantic' if folder=='semantic_id' else ('depth' if folder=='depth' else None)
                    if role not in roles:
                        continue
                    try:
                        local = int(filename.split('.')[0])
                    except ValueError:
                        continue
                    if local not in local_to_row:
                        continue
                    if local in found[role]:
                        raise ValueError('duplicate proxy archive member')
                    raw = archive.extractfile(member).read()
                    if role == 'semantic':
                        ids = np.asarray(Image.open(io.BytesIO(raw)))
                        if ids.ndim != 2 or ids.max() >= len(palette):
                            raise ValueError('unknown semantic ID')
                        image = Image.fromarray(palette[ids])
                    else:
                        depth = np.frombuffer(raw,dtype='<f4').reshape(96,168)
                        if not np.isfinite(depth).all() or (depth<0).any():
                            raise ValueError('invalid metric depth')
                        # Fixed physical mapping; zero/no-hit remains black.
                        d = np.where(depth>0,1-np.log1p(np.clip(depth,0,1000))/np.log1p(1000),0)
                        image = Image.fromarray(np.repeat((d*255).round().astype(np.uint8)[...,None],3,axis=-1))
                    values[role][local_to_row[local]] = np.asarray(image.resize((width,height),Image.Resampling.NEAREST))
                    found[role].add(local)
    for role,frames in found.items():
        if len(frames) != len(local_frames):
            raise ValueError(f'{job["take_id"]}: missing {len(local_frames)-len(frames)} {role} frames')
    return values,local_to_row


def encoder_fingerprint(args):
    root = Path(args.checkpoint).resolve()
    # Content-addressed model identity. Expensive once per new snapshot; the
    # Stamps invalidate cached content hashes, including same-size replacements.
    stamp_path = Path(args.output)/'encoder_hashes.json'
    saved = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    model_files = sorted(p for folder in ('text_encoder','video_vae/source','processor')
                         for p in (root/folder).glob('*') if p.is_file())
    hashes,stamps = {},{}
    for path in model_files:
        stat = path.stat()
        stamp = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        old = saved.get(str(path),{})
        digest = old.get('sha256') if old.get('stamp')==stamp else digest_file(path)
        after = path.stat()
        if stamp != [after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]:
            raise ValueError(f'encoder file changed while hashing: {path}')
        hashes[str(path.relative_to(root))] = digest
        stamps[str(path)] = dict(stamp=stamp,sha256=digest)
    atomic_json(stamp_path,stamps)
    config = dict(schema=SCHEMA,height=args.height,width=args.width,roles=args.roles,posterior=args.posterior,
        first_frame=not args.no_first_frame,
        model_hashes=hashes,seed=args.seed,depth_mapping='inverse_log_0_1000m_zero_invalid',
        caption_mode='planned' if not args.static_prompt_file else 'provided_static',
        static_prompt=None if not args.static_prompt_file else Path(args.static_prompt_file).read_text(),
        qwen_layer=50,qwen_fps=2,vae_chunk=17,vae_eos_drop=3,
        encoder_code=digest_file(Path(__file__).parents[1]/'modeling/cwm_encoders.py'),
        processor_code=digest_file(__file__))
    if getattr(args,'guidance_drop',None):
        config['guidance_drop'] = args.guidance_drop
    config['fingerprint'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return config


def save_sample(target, sample):
    """Atomic sample commit; run scripts may swap in a paced/locked writer for shared storage."""
    with tempfile.NamedTemporaryFile(dir=Path(target).parent,delete=False) as f:
        torch.save(sample,f); f.flush(); os.fsync(f.fileno()); temp = f.name
    os.replace(temp,target)


def encode_guidance_condition(encoders, drop, *, prompt, picture, videos, raw_ranges, static):
    """Independent weak Qwen documents; no latent re-encoding or embedding slicing."""
    if drop not in ('proxy','prompt','prompt_proxy'):
        raise ValueError('invalid guidance drop')
    if not picture:
        raise ValueError('partial guidance requires a first-frame image')
    weak_prompt = prompt if drop == 'proxy' else ''
    weak_videos = videos if drop == 'prompt' else []
    teacher = encoders.condition(weak_prompt,weak_videos,images=picture)
    weak_static = static if drop == 'proxy' else (
        teacher if drop == 'prompt_proxy' else encoders.condition('',images=picture))
    chunks = ([encoders.condition('',[v[a:b] for v in videos]) for a,b in raw_ranges]
              if drop == 'prompt' else [])
    return dict(teacher_condition=teacher,student_condition=dict(static=weak_static,chunks=chunks))


def encode_dataset(args):
    from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
    if args.height%32 or args.width%32 or not (0<=args.rank<args.shards):
        raise ValueError('resolution must be divisible by 32; rank must be in [0,shards)')
    if args.roles not in (['semantic'],['semantic','depth']):
        raise ValueError('roles must be semantic or semantic depth, in that order')
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    config = encoder_fingerprint(args)
    jobs = rows(args.index)
    # Include the exact source mapping/captions in the shared cache identity.
    config['index_sha256'] = digest_file(args.index)
    config['fingerprint'] = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    fingerprint = config['fingerprint']
    cache = out/'samples'/fingerprint[:16]
    cache.mkdir(parents=True,exist_ok=True)
    atomic_json(cache/'config.json',config)
    takes = sorted({j['take_id'] for j in jobs})
    owned = set(takes[args.rank::args.shards])
    selected = [j for j in jobs if j['take_id'] in owned]
    if args.sample_ids:
        wanted = set(Path(args.sample_ids).read_text().split())
        selected = [j for j in selected if j['sample_id'] in wanted]
    if args.limit is not None:
        selected = selected[:args.limit]
    pending = []
    for job in selected:
        sidecar = cache/f'{job["sample_id"]}.json'
        if sidecar.exists():
            record = json.loads(sidecar.read_text())
            load_entry(out/'train.jsonl',record,verify_hash=True)
        else:
            pending.append(job)
    if not pending:
        print('All selected samples are complete.'); return
    encoders = FrozenCWMEncoders(args.checkpoint,device=args.device,posterior=args.posterior)
    groups = defaultdict(list)
    for job in pending:
        groups[job['take_id']].append(job)
    static_prompt = config['static_prompt']
    role_prompt = 'Video 1 is a semantic segmentation proxy.'
    if 'depth' in args.roles:
        role_prompt += ' Video 2 is aligned inverse log metric depth, with invalid surfaces black.'
    role_prompt += ' Generate the corresponding natural RGB video following the supplied appearance and action plan.'
    # Only the documents that carry the picture describe it; per-chunk proxy documents never do.
    plan_prompt = ('Picture 1 is the first RGB frame of the target video. ' if config['first_frame'] else '') + role_prompt
    for take,take_jobs in groups.items():
        for path,stamp in take_jobs[0]['source_files'].items():
            stat = Path(path).stat()
            if dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns) != stamp:
                raise ValueError(f'source changed after indexing: {path}')
        with tempfile.TemporaryDirectory(prefix='cwm_take_',dir=args.scratch) as scratch, ExitStack() as stack:
            media,lookup = prepare_take(take_jobs,scratch,args.height,args.width,args.roles,stack)
            for job in take_jobs:
                select = [lookup[f] for f in range(job['local_start'],job['local_end'])]
                frames = {role:np.asarray(value[select]) for role,value in media.items()}
                seed = args.seed + job['job_ordinal']*101
                picture = [frames['rgb'][0]] if config['first_frame'] else []
                refs = [dict(kind='image',role='first_frame',video=encoders.image(p,seed=seed+len(args.roles)+1))  # after the role seeds
                        for p in picture]
                refs += [dict(kind='video',role=role,video=encoders.video(frames[role],seed=seed+j+1))
                         for j,role in enumerate(args.roles)]
                prompt = plan_prompt+'\n'+job['prompt']
                teacher = encoders.condition(prompt,[frames[r] for r in args.roles],images=picture)
                static = encoders.condition(prompt if static_prompt is None else plan_prompt+'\n'+job['system_prompt']+'\n'+static_prompt,images=picture)
                raw_ranges = [[a,min(a+17,124)] for a in range(0,124,17)]
                chunks = [encoders.condition(role_prompt,[frames[r][a:b] for r in args.roles]) for a,b in raw_ranges]
                sample = dict(video=encoders.video(frames['rgb'],seed=seed),audio=torch.empty(2,32,0),
                    references=refs,teacher_condition=teacher,student_condition=dict(static=static,chunks=chunks),
                    metadata=dict(schema=SCHEMA,sample_id=job['sample_id'],fingerprint=fingerprint,take_id=take,
                        fps=24,num_frames=124,normalized=True,condition_seed=seed,
                        chunk_raw_ranges=raw_ranges,chunk_latent_ranges=[[a,min(a+5,37)] for a in range(0,37,5)],
                        local_start=job['local_start'],source_frames=job['source_frames'],caption_mode=config['caption_mode']))
                if config.get('guidance_drop'):
                    drop = config['guidance_drop']
                    sample['guidance_conditions'] = {drop:encode_guidance_condition(encoders,drop,
                        prompt=prompt,picture=picture,videos=[frames[r] for r in args.roles],
                        raw_ranges=raw_ranges,static=static)}
                validate_sample(sample)
                target = cache/f'{job["sample_id"]}.pt'
                save_sample(target,sample)
                record = dict(schema=SCHEMA,sample_id=job['sample_id'],take_id=take,split=job['split'],
                    path=str(target.relative_to(out)),sha256=digest_file(target),fingerprint=fingerprint,
                    references=[r['role'] for r in sample['references']],
                    latent_shape=list(sample['video'].shape),lengths={s:packed_length(sample,s) for s in ('sft','tf','dmd')})
                if config.get('guidance_drop'):
                    record['guidance_drop'] = config['guidance_drop']
                atomic_json(target.with_suffix('.json'),record)
                print(json.dumps(dict(completed=job['sample_id'],lengths=record['lengths'])),flush=True)


def finalize(args):
    out = Path(args.output).resolve()
    expected = {j['sample_id']:j for j in rows(args.index)}
    records = []
    for path in sorted((out/'samples').glob('*/job_*.json')):
        record = json.loads(path.read_text())
        if record['sample_id'] in expected:
            records.append(record)
    if not records or len({r['fingerprint'] for r in records}) != 1:
        raise ValueError('expected exactly one completed encoder cache version; use a separate output directory per recipe')
    recipe = json.loads((out/'samples'/records[0]['fingerprint'][:16]/'config.json').read_text())
    if recipe['index_sha256'] != digest_file(args.index):
        raise ValueError('source index changed after encoding; rebuild the cache from this index')
    completed = {r['sample_id'] for r in records}
    if len(completed) != len(records) or (completed != set(expected) and not args.allow_partial):
        raise ValueError(f'{len(completed)}/{len(expected)} jobs complete; partial manifests require --allow-partial')
    for record in records:
        if record['take_id'] != expected[record['sample_id']]['take_id'] or record['split'] != expected[record['sample_id']]['split']:
            raise ValueError('cache split differs from authoritative index')
        load_entry(out/'train.jsonl',record)
    for split in ('train','val'):
        atomic_json(out/f'{split}.jsonl',[r for r in records if r['split']==split],lines=True)
    print(json.dumps(dict(completed=len(records),expected=len(expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('index'); p.add_argument('--data-root',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--val-fraction',type=float,default=.1); p.add_argument('--split-file')
    p = sub.add_parser('encode'); p.add_argument('--index',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--height',type=int,default=768); p.add_argument('--width',type=int,default=1344)  # MiniMax-H3 native 1344x768
    p.add_argument('--roles',nargs='+',default=['semantic'],help='Proxy reference roles in order: semantic, optionally depth')
    p.add_argument('--no-first-frame',action='store_true',help='Omit the first RGB frame <Picture 1> reference (default: include it)')
    p.add_argument('--posterior',choices=['mean','sample'],default='mean')
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--device',default='cuda'); p.add_argument('--scratch')
    p.add_argument('--rank',type=int,default=0); p.add_argument('--shards',type=int,default=1); p.add_argument('--limit',type=int)
    p.add_argument('--sample-ids',help='Whitespace-separated sample_id file; encode only this subset of the index')
    p.add_argument('--static-prompt-file',help='Known online text; default treats full caption as a preprovided plan')
    p.add_argument('--guidance-drop',choices=['proxy','prompt','prompt_proxy'],
                   help='Cache independent weak Qwen documents for CFG/GFT; use a separate output directory')
    p = sub.add_parser('finalize'); p.add_argument('--index',required=True); p.add_argument('--output',required=True); p.add_argument('--allow-partial',action='store_true')
    p = sub.add_parser('verify'); p.add_argument('--manifest',required=True)
    p = sub.add_parser('decode'); p.add_argument('--input',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True); p.add_argument('--mode',choices=['native','strict'],default='native')
    p.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if args.command == 'index': index_dataset(args)
    elif args.command == 'encode': encode_dataset(args)
    elif args.command == 'finalize': finalize(args)
    elif args.command == 'decode':
        from common.media import save_video
        from projects.minimax_h3.modeling.cwm_encoders import FrozenCWMEncoders
        sample = torch.load(args.input,map_location='cpu',weights_only=True)
        encoder = FrozenCWMEncoders(args.checkpoint,device=args.device,load_text_encoder=False)
        frames = encoder.decode_video(sample['video'],num_frames=sample['metadata']['num_frames'],mode=args.mode)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        save_video(frames.permute(3,0,1,2).unsqueeze(0).float()/255,args.output,fps=24,nrow=1,value_range=(0,1))
    else:
        path,entries,_ = read_manifest(args.manifest)
        for entry in entries: validate_sample(load_entry(path,entry,verify_hash=True))
        print(json.dumps(dict(verified=len(entries))))


if __name__ == '__main__':
    main()
