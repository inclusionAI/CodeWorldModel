"""Package five reviewed camera studies; lossless inputs are verified against PNGs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
CASES=ROOT/'docs/demo-brainstorm/camera-director'
VERSION='natural-20261108-v1'
IDS=('01-neon-car-ad','02-cliff-hero','04-mechanical-ad','05-crystal-stage','06-motorcycle-tempo')
STYLES=('wire','depth','coarse','fine','subject_bg')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
HELPER=CASES/'03-rainy-chase-natural-v1/assets/scene'
encode=load_module('rainy_package',HELPER/'package_videos.py').encode
quant=load_module('rainy_quantize',HELPER/'quantize_coarse.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def previews(case):
    root=CASES/case;scene=root/'assets/scene'/VERSION
    routes=json.loads((scene/'camera-paths.json').read_text(encoding='utf8'))['routes']
    ref=Image.open(root/'assets/first-frame/natural-first-20261108-v1.png').convert('RGB').resize((672,384))
    alignment=Image.new('RGB',(1344,816),'#111820');d=ImageDraw.Draw(alignment)
    alignment.paste(ref,(0,24));alignment.paste(Image.open(scene/'aligned-fine.png'),(672,24))
    alignment.paste(Image.blend(ref,Image.open(scene/'aligned-fine.png').convert('RGB'),.45),(0,432))
    alignment.paste(Image.open(scene/'aligned-coarse.png'),(672,432))
    for xy,label in [((0,3),'APPROVED REFERENCE'),((672,3),'FINE BLOCKOUT - SAME OPENING'),((0,410),'45% OVERLAY - APPROXIMATE GEOMETRY'),((672,410),'COARSE SEMANTIC')]:d.text(xy,label,fill='white')
    alignment.save(scene/'alignment-check.jpg',quality=93)
    strip=Image.new('RGB',(1680,420),'#111820');d=ImageDraw.Draw(strip)
    for j,name in enumerate(routes):
        for i,f in enumerate([1,48,72,96,144]):
            strip.paste(Image.open(scene/'review'/f'{name}-{f:03}.png').resize((336,192)),(i*336,j*210+18))
            d.text((i*336,j*210),f'{name} / {f}',fill='white')
    strip.save(scene/'route-filmstrip.jpg',quality=93)

def package(case):
    root=CASES/case;scene=root/'assets/scene'/VERSION
    validation=json.loads((scene/'collision-validation.json').read_text(encoding='utf8'))
    assert not any(validation[k] for k in ('violations','center_ray_occlusions','framing_violations')),case
    wire=json.loads((scene/'wire-framing-validation.json').read_text(encoding='utf8'))
    assert not wire['violations'],case
    routes=list(json.loads((scene/'camera-paths.json').read_text(encoding='utf8'))['routes'])
    quant.quantize(scene/'aligned-coarse.png')
    rows=[];prefixes={}
    for style in STYLES:
        for route in routes:
            directory=root/'assets/proxy'/VERSION/style;frames=directory/f'.{route}_frames'
            pngs=sorted(frames.glob('frame_*.png'));assert len(pngs)==144,(case,style,route,len(pngs))
            if style=='coarse':
                for p in pngs:quant.quantize(p)
            for suffix in ('.mp4','.mkv'):encode(frames,directory/(route+suffix))
            prefix=hashlib.sha256()
            with av.open(str(directory/(route+'.mkv'))) as source:
                assert source.streams.video[0].average_rate==24
                assert source.streams.video[0].codec_context.name=='ffv1'
                count=0
                for i,(frame,png) in enumerate(zip(source.decode(video=0),pngs,strict=True)):
                    rgb=frame.to_ndarray(format='rgb24');original=np.asarray(Image.open(png).convert('RGB'))
                    assert np.array_equal(rgb,original),(case,style,route,i)
                    if i<24:prefix.update(rgb.tobytes())
                    if style=='coarse':assert np.all(np.min(np.max(np.abs(rgb.astype(np.int16)[:,:,None,:]-quant.PALETTE[None,None,:,:]),axis=3),axis=2)==0)
                    count+=1
                assert count==144
            assert np.array_equal(np.asarray(Image.open(pngs[0])),np.asarray(Image.open(scene/f'aligned-{style}.png')))
            if style in prefixes:assert prefixes[style]==prefix.hexdigest()
            else:prefixes[style]=prefix.hexdigest()
            with av.open(str(directory/(route+'.mp4'))) as source:
                assert source.streams.video[0].average_rate==24
                assert sum(1 for _ in source.decode(video=0))==144
            rows.append(dict(style=style,route=route,frames=144,fps=24,width=672,height=384,
                proxy=str((directory/(route+'.mkv')).relative_to(root)).replace('\\','/'),
                video=str((directory/(route+'.mp4')).relative_to(root)).replace('\\','/'),
                sha256=sha(directory/(route+'.mkv')),video_sha256=sha(directory/(route+'.mp4')),
                prefix_rgb_sha256=prefix.hexdigest()))
            print('PACKAGED',case,style,route,flush=True)
    receipt=dict(case=case,version=VERSION,reference_sha256=sha(root/'assets/first-frame/natural-first-20261108-v1.png'),
                 source_sha256=sha(ROOT/'tools/blender_proxy/natural_five.py'),
                 validation_sha256=sha(scene/'collision-validation.json'),rows=rows)
    if (scene/'camera-repair.json').exists():receipt['camera_repair_sha256']=sha(scene/'camera-repair.json')
    (scene/'media-validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');previews(case)
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--case',choices=IDS);p.add_argument('--previews',action='store_true');a=p.parse_args()
    for case in ([a.case] if a.case else IDS):
        if a.previews:previews(case)
        else:package(case)
