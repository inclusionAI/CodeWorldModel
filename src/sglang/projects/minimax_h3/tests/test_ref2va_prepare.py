"""Small CPU fixtures for Ref2VA indexing, identity, streaming and resume safety."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from fractions import Fraction
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from projects.minimax_h3.tools import ref2va_prepare as prep


def jsonlines(values):
    return ''.join(json.dumps(value)+'\n' for value in values).encode()


def make_bundle(path, source_id='source', mutate=None):
    path=Path(path); path.mkdir(parents=True)
    n=158
    frames=[dict(record_type='frame',frame=i,rgb_frame=i,proxy_frame=i,camera_frame=i,
                 video_seconds=i/24,camera_seconds=10+i/24) for i in range(n)]
    cameras=[dict(frame=i,rgb_frame=i,proxy_frame=i,action_frame=i,
                  video_seconds=i/24,camera_seconds=10+i/24) for i in range(n)]
    chunks=[dict(record_type='chunk',frame_start=a,frame_end_exclusive=min(a+17,n),
                 window_complete=True,chunk_complete=True) for a in range(0,n,17)]
    if mutate:
        mutate(frames,cameras,chunks)
    camera_bytes=jsonlines(cameras)
    payload=jsonlines(frames+chunks)
    for name,data in [('rgb.mkv',b'RGB!'),('proxy.mkv',b'PRXY'),('camera.jsonl',camera_bytes)]:
        (path/name).write_bytes(data)
    header=dict(record_type='header',schema='rgb-proxy-actions/v2',fps=24,frame_count=n,
                window_frames=n,stride_frames=n,tail_padding_frames=0,
                actions_payload_sha256=hashlib.sha256(payload).hexdigest(),
                camera_file=dict(path='camera.jsonl',frame_count=n,bytes=len(camera_bytes),
                                 sha256=hashlib.sha256(camera_bytes).hexdigest()),
                source_render=dict(source_id=source_id),rgb_resolution=[1344,768],proxy_resolution=[672,384],
                proxy_video=dict(style_version='proxy-depth-semantic-rgb/v1'),
                configured_scene=dict(scene='fixture',behavior='walking'),
                media=[dict(path=name,bytes=(path/name).stat().st_size,sha256=prep.digest_file(path/name))
                       for name in ('rgb.mkv','proxy.mkv')])
    (path/'actions.jsonl').write_bytes(jsonlines([header])+payload)
    return path


class Ref2VAPrepareTests(unittest.TestCase):
    def test_complete_bundle_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            path=make_bundle(Path(directory)/'bundle')
            result=prep.inspect_bundle(path)
            self.assertEqual(result['frame_count'],158)
            prep.check_sources(result,checksums=True)
            raw=(path/'actions.jsonl').read_bytes()
            (path/'actions.jsonl').write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'payload checksum'):
                prep.inspect_bundle(path)

    def test_invalid_time_mapping_and_chunk_bounds(self):
        changes=[lambda f,c,k:f[20].update(video_seconds=0),
                 lambda f,c,k:c[20].update(action_frame=21),
                 lambda f,c,k:k[-1].update(frame_end_exclusive=157)]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                path=make_bundle(Path(directory)/'bundle',mutate=change)
                with self.assertRaises(ValueError):
                    prep.inspect_bundle(path)

    def test_camera_checksum_and_media_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path=make_bundle(Path(directory)/'bundle')
            (path/'camera.jsonl').write_bytes((path/'camera.jsonl').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'camera checksum'):
                prep.inspect_bundle(path)
        with tempfile.TemporaryDirectory() as directory:
            path=make_bundle(Path(directory)/'bundle')
            (path/'rgb.mkv').write_bytes(b'longer RGB')
            with self.assertRaisesRegex(ValueError,'file size mismatch'):
                prep.inspect_bundle(path)

    def test_same_source_id_stays_in_one_split_and_snapshot_is_frozen(self):
        seed=2026
        ids={}
        for i in range(100):
            identity=f'source_{i}'
            split='val' if int(prep.digest([seed,identity])[:16],16)/2**64 < .5 else 'train'
            ids.setdefault(split,identity)
            if len(ids)==2:
                break
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'data'; out=Path(directory)/'index'
            make_bundle(root/'group_a'/'take_a',ids['train'])
            make_bundle(root/'group_b'/'take_b',ids['train'])
            make_bundle(root/'group_a'/'take_c',ids['val'])
            args=SimpleNamespace(data_root=root,output=out,seed=seed,val_fraction=.5,skip_incomplete=False)
            prep.index_dataset(args)
            index=prep.rows(out/'index.jsonl')
            self.assertEqual(len(index),3)
            self.assertEqual({r['split'] for r in index if r['take_id']==ids['train']},{'train'})
            before=(out/'index.jsonl').read_bytes()
            make_bundle(root/'group_b'/'later_arrival','later')
            prep.index_dataset(args)
            self.assertEqual((out/'index.jsonl').read_bytes(),before)

    def test_video_iterator_never_requests_the_next_chunk(self):
        consumed=[]; closed=[]
        stream=SimpleNamespace(average_rate=24,time_base=Fraction(1,24),codec_context=SimpleNamespace(thread_count=0))
        class Frame:
            def __init__(self,i): self.pts=i
            def to_image(self): return Image.new('RGB',(2,2))
        class Container:
            streams=SimpleNamespace(video=[stream],audio=[])
            def __enter__(self): return self
            def __exit__(self,*args): closed.append(True)
            def decode(self,selected):
                for i in range(30):
                    consumed.append(i)
                    yield Frame(i)
        with patch.dict('sys.modules',{'av':SimpleNamespace(open=lambda _:Container())}):
            iterator=prep.iter_video_chunks('proxy.mkv',start=3,num_frames=8,height=2,width=2,chunk_frames=5)
            self.assertEqual(consumed,[])
            self.assertEqual(next(iterator).shape[0],5)
            self.assertEqual(consumed,list(range(8)))
            self.assertEqual(next(iterator).shape[0],3)
            self.assertEqual(consumed,list(range(11)))
            with self.assertRaises(StopIteration): next(iterator)
            self.assertEqual(consumed,list(range(11)))
            self.assertEqual(closed,[True])

    def test_encoder_replacement_with_preserved_mtime_changes_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); checkpoint=root/'checkpoint'
            for folder in ('text_encoder','video_vae/source','processor'):
                target=checkpoint/folder/'fixture.bin'; target.parent.mkdir(parents=True)
                target.write_bytes(b'abcd')
            (root/'prompt.txt').write_text('根据首帧和代理视频生成一致的视频。')
            (root/'index.jsonl').write_text('{}\n')
            args=SimpleNamespace(output=root/'cache',checkpoint=checkpoint,prompt_file=root/'prompt.txt',
                                 index=root/'index.jsonl',height=32,width=32,proxy_height=32,proxy_width=32,seed=2026)
            prep.encoder_recipe(args)
            target=checkpoint/'text_encoder'/'fixture.bin'; old=target.stat()
            target.write_bytes(b'wxyz')
            os.utime(target,ns=(old.st_atime_ns,old.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                prep.encoder_recipe(args)

    def test_concurrent_finalize_cannot_publish_an_older_snapshot_last(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); fingerprint='a'*64
            expected=[dict(sample_id=f'ref2va_{i}',take_id=f'source_{i}',split='train' if i==0 else 'val') for i in range(2)]
            index=out/'index.jsonl'; index.write_bytes(jsonlines(expected))
            (out/'recipe.json').write_text(json.dumps(dict(fingerprint=fingerprint,index_sha256=prep.digest_file(index))))
            cache=out/'samples'/fingerprint[:16]; cache.mkdir(parents=True)
            records=[dict(row,fingerprint=fingerprint) for row in expected]
            (cache/'ref2va_0.json').write_text(json.dumps(records[0]))
            args=SimpleNamespace(output=out,index=index,allow_partial=True)
            scanned=threading.Event(); resume=threading.Event(); second_done=threading.Event(); errors=[]
            def load(*args,**kwargs):
                if threading.current_thread().name=='stale-finalizer' and not scanned.is_set():
                    scanned.set()
                    if not resume.wait(5): raise RuntimeError('test synchronization timeout')
            def run(done=None):
                try: prep.finalize(args)
                except BaseException as error: errors.append(error)
                finally:
                    if done: done.set()
            # Isolate transaction ordering from tensor/hash checks covered separately.
            with patch.object(prep,'load_entry',side_effect=load), patch.object(prep,'validate_sample'):
                first=threading.Thread(target=run,name='stale-finalizer',daemon=True)
                second=threading.Thread(target=run,args=(second_done,),name='new-finalizer',daemon=True)
                first.start()
                try:
                    self.assertTrue(scanned.wait(3))
                    (cache/'ref2va_1.json').write_text(json.dumps(records[1]))
                    second.start()
                    # A correctly scoped lock blocks the second reader until first commits.
                    second_done.wait(.5)
                finally:
                    resume.set()
                    first.join(3)
                    if second.ident is not None: second.join(3)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors,[])
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['encoded'],2)
            self.assertTrue(status['complete'])


if __name__=='__main__': unittest.main()
