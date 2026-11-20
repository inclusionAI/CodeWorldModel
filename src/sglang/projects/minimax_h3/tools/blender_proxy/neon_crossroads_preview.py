"""Encode and check the rendered crossroads preview frames."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1]).resolve()
count, fps, size = 209,24,(336,192)
names = ('left','right')
videos = {name:root/f'{name}-proxy-preview.mp4' for name in names}
videos['compare'] = root/'left-right-preview.mp4'
sheet = root/'first-frame-comparison.png'
receipt_path = root/'preview-receipt.json'
if any(path.exists() for path in (*videos.values(),sheet,receipt_path)):
    raise FileExistsError('preview outputs already exist')
paths = {name:sorted((root/'frames'/name).glob('*.png')) for name in names}
for name in names:
    if len(paths[name]) != count or any(p.name != f'{i:04}.png' for i,p in enumerate(paths[name])):
        raise ValueError(f'{name}: expected {count} sequential frames')

def open_stream(path, width, height):
    container = av.open(str(path),mode='w')
    stream = container.add_stream('libx264',rate=fps)
    stream.width, stream.height, stream.pix_fmt = width,height,'yuv420p'
    stream.options = {'crf':'18','preset':'medium'}
    return container,stream

outputs = {name:open_stream(path,*(size if name != 'compare' else (672,224))) for name,path in videos.items()}
try:
    for index in range(count):
        images = {name:Image.open(paths[name][index]).convert('RGB') for name in names}
        if any(image.size != size for image in images.values()):
            raise ValueError(f'unexpected frame size at {index}')
        if index < 49 and not np.array_equal(np.asarray(images['left']),np.asarray(images['right'])):
            raise ValueError(f'common path differs at frame {index}')
        compare = Image.new('RGB',(672,224),(17,23,31))
        compare.paste(images['left'],(0,32))
        compare.paste(images['right'],(336,32))
        draw = ImageDraw.Draw(compare)
        draw.text((12,9),'LEFT: ARCADE',fill='white')
        draw.text((348,9),'RIGHT: BRIDGE',fill='white')
        images['compare'] = compare
        for name,(container,stream) in outputs.items():
            frame = av.VideoFrame.from_ndarray(np.asarray(images[name]),format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
    for container,stream in outputs.values():
        for packet in stream.encode():
            container.mux(packet)
finally:
    for container,_ in outputs.values():
        container.close()

reference = Image.open(root/'reference.png').convert('RGB')
if reference.size != (672,384):
    raise ValueError('reference must be 672x384')
proxy = Image.open(paths['left'][0]).convert('RGB').resize((672,384),Image.Resampling.NEAREST)
comparison = Image.new('RGB',(1344,420),(17,23,31))
comparison.paste(reference,(0,36)); comparison.paste(proxy,(672,36))
draw = ImageDraw.Draw(comparison)
draw.text((14,11),'REFERENCE IMAGE',fill='white')
draw.text((686,11),'COARSE PROXY - FIRST FRAME',fill='white')
comparison.save(sheet)

for name,path in videos.items():
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        decoded=list(container.decode(stream))
        if len(decoded) != count or stream.average_rate != Fraction(fps,1):
            raise ValueError(f'{name}: encoded frame count or fps mismatch')
        if any(frame.pts is None or abs(float(frame.pts*stream.time_base)-i/fps) > .002 for i,frame in enumerate(decoded)):
            raise ValueError(f'{name}: frame timestamps do not match {fps} fps')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt = {'fps':fps,'frames_per_route':count,'render_size':list(size),
           'reference_size':[672,384],'common_frames_verified':49,
           'reference_sha256':digest(root/'reference.png'),
           'reference_original_sha256':digest(root/'reference-original.png'),
           'scene_sha256':digest(root/'crossroads.blend'),
           'scene_script_sha256':digest(Path(__file__).with_name('neon_crossroads.py')),
           'camera_paths_sha256':digest(root/'camera-paths.json'),
           'videos':{name:{'file':path.name,'sha256':digest(path)} for name,path in videos.items()},
           'limitations':['preview is Blender coarse proxy, not model output',
                          'reference alignment is approximate and needs visual review',
                          '336x192 preview rendered directly; not the verified 672x384 input pipeline',
                          'H.264/YUV420 previews are lossy; PNG frames preserve rendered RGB']}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print('CROSSROADS_PREVIEW',json.dumps(receipt))
