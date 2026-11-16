"""Package synchronized hero PNG masters as a padded, single-clock H.264 atlas."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np
from PIL import Image

STAGES = ('rgb', 'fine', 'coarse', 'depth')
PADDING = 16
PROXY_GEOMETRY = {
    'fine':'RGB geometry. Buildings and ground use camera-axis depth gray; other categories use semantic colors.',
    'coarse':'Simplified tree and person geometry. The person is rigid apart from whole-body motion; silhouettes differ from RGB and fine.',
    'depth':'RGB/fine geometry with a fixed linear camera-axis 7..65m near-white/far-black map. It does not exactly trace coarse silhouettes.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atlas_layout(width, height):
    cell_width, cell_height = width + 2*PADDING, height + 2*PADDING
    return {'width':cell_width*2, 'height':cell_height*2, 'padding':PADDING,
            'rects':{stage:[(index%2)*cell_width+PADDING, (index//2)*cell_height+PADDING, width, height]
                     for index,stage in enumerate(STAGES)}}


def make_atlas(images, layout):
    canvas = np.zeros((layout['height'],layout['width'],3),dtype=np.uint8)
    for stage, image in images.items():
        x,y,width,height = layout['rects'][stage]
        if image.shape != (height,width,3):
            raise ValueError(f'Wrong source size for {stage}: {image.shape}')
        canvas[y-PADDING:y+height+PADDING,x-PADDING:x+width+PADDING] = np.pad(image,((PADDING,PADDING),(PADDING,PADDING),(0,0)),mode='edge')
    return canvas


def package(output):
    out = Path(output).resolve()
    info = json.loads((out/'scene-info.json').read_text(encoding='utf8'))
    if info['schema'] != 'hero-loop-source/1' or info['fps'] != 24 or info['frameCount'] != 48:
        raise ValueError('First hero slice requires 48 frames at 24 fps')
    width,height = info['tileSize']
    layout = atlas_layout(width,height)
    count = info['frameCount']
    sources = {stage:[out/'frames'/stage/f'{index:04}.png' for index in range(1,count+2)] for stage in STAGES}
    for files in sources.values():
        for path in files:
            if not path.is_file():
                raise ValueError(f'Missing synchronized master {path}')
    endpoint = {}
    for stage,files in sources.items():
        first,last = (np.asarray(Image.open(path).convert('RGB')) for path in (files[0],files[-1]))
        endpoint[stage] = {'equal':bool(np.array_equal(first,last)), 'maxError':int(np.abs(first.astype(int)-last).max())}
    if not all(result['equal'] for result in endpoint.values()):
        raise ValueError(f'Frame N+1 must equal frame 1 before packaging: {endpoint}')
    web = out/'web'
    web.mkdir()
    target = web/'atlas.mp4'
    with av.open(str(target),'w',options={'movflags':'+faststart'}) as container:
        stream = container.add_stream('libx264',rate=info['fps'])
        stream.width,stream.height,stream.pix_fmt = layout['width'],layout['height'],'yuv420p'
        stream.options = {'crf':'18','preset':'medium','g':str(count),'bf':'0'}
        for index in range(count):
            images = {stage:np.asarray(Image.open(files[index]).convert('RGB')) for stage,files in sources.items()}
            frame = av.VideoFrame.from_ndarray(make_atlas(images,layout),format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = []
    timestamps = []
    with av.open(str(target)) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_ndarray(format='rgb24'))
            timestamps.append(frame.pts*frame.time_base)
    if len(decoded) != count or any(time != Fraction(index,info['fps']) for index,time in enumerate(timestamps)):
        raise ValueError('Decoded atlas frame count or presentation timestamps differ from masters')
    alignment = {}
    for stage,(x,y,w,h) in layout['rects'].items():
        errors = []
        for index,pixels in enumerate(decoded):
            master = np.asarray(Image.open(sources[stage][index]).convert('RGB'),dtype=np.int16)
            errors.append(float(np.abs(pixels[y:y+h,x:x+w].astype(np.int16)-master).mean()))
        if max(errors) > 12:
            raise ValueError(f'Decoded {stage} does not match its source tile: MAE {max(errors):.3f}')
        alignment[stage] = {'maxMeanAbsoluteError':round(max(errors),4)}
    seam = {}
    for stage,(x,y,w,h) in layout['rects'].items():
        first,last = (pixels[y:y+h,x:x+w].astype(np.int16) for pixels in (decoded[0],decoded[-1]))
        seam[stage] = {'meanAbsoluteDifference':round(float(np.abs(first-last).mean()),4),
                       'note':'Adjacent last/first frames differ by one motion step. Visual seam acceptance is separate.'}
    posters = {}
    for stage,files in sources.items():
        destination = web/f'{stage}.png'
        shutil.copyfile(files[0],destination)
        posters[stage] = {'src':destination.name,'sha256':digest(destination)}
    Image.fromarray(decoded[0]).save(out/'decoded-first.png')
    Image.fromarray(decoded[-1]).save(out/'decoded-last.png')
    manifest = {'schema':'hero-loop/1','fps':info['fps'],'frameCount':count,'tileSize':[width,height],
                'atlas':layout,'media':{'src':target.name,'sha256':digest(target),'codec':'h264-yuv420p','bytes':target.stat().st_size},
                'posters':posters,'depth':info['depth'],'source':info['source'],'blenderVersion':info['blenderVersion'],
                'limitation':info['limitation'],'proxyGeometry':PROXY_GEOMETRY,
                'loopVerification':{'endpoint':endpoint,'duplicateEndpointExported':False,
                    'decodedFrames':len(decoded),'timestamps':'exact index/fps','decodedTileAlignment':alignment,
                    'decodedLastFirst':seam,'visualAcceptance':'pending actual decoded last/first review'}}
    (web/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'web':str(web),'atlas':[layout['width'],layout['height']],'frames':count,'bytes':target.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    package(parser.parse_args().output)
