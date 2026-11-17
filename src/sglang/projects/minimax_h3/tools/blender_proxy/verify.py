"""Check packaged videos against the existing Ref2VA input boundary."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

from inference.ref2va import input_spec
from tools.blender_proxy.package import digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    out = parser.parse_args().directory.resolve()
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
    if digest(out/manifest['reference']) != manifest['reference_sha256'] or digest(out/manifest['scene_file']) != manifest['scene_sha256']:
        raise ValueError('reference or scene checksum mismatch')
    if 'scene_info' in manifest and digest(out/manifest['scene_info']) != manifest['scene_info_sha256']:
        raise ValueError('scene info checksum mismatch')
    num_frames = manifest['frames']
    for name in ('fine', 'coarse'):
        variant = manifest['variants'][name]
        if digest(out/variant['prompt']) != variant['prompt_sha256']:
            raise ValueError(f'{name} prompt checksum mismatch')
        w,h = variant['historical_model_size']
        for kind in ('lossless','preview'):
            path = out/variant[kind]
            if digest(path) != variant[kind+'_sha256']:
                raise ValueError(f'{name} {kind} checksum mismatch')
            args = argparse.Namespace(num_frames=num_frames, start_frame=0, window_index=None, bundle=None,
                                      proxy_video=str(path), first_frame=str(out/'reference.png'))
            input_spec(args, {'height':384, 'width':672, 'proxy_height':h, 'proxy_width':w,
                              'system_prompt':'placeholder for input-only validation'})
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                count = 0
                for count,frame in enumerate(container.decode(stream), 1):
                    if frame.pts is None or abs(float(frame.pts * stream.time_base) - (count-1)/24) > .002:
                        raise ValueError(f'{name} {kind} frame {count} has an invalid timestamp')
                    if kind == 'lossless':
                        source = out/('coarse-categorical' if name == 'coarse' else 'fine')/f'{count-1:04}.png'
                        if not np.array_equal(frame.to_ndarray(format='rgb24'), np.asarray(Image.open(source).convert('RGB'))):
                            raise ValueError(f'{name} lossless frame {count} differs from source')
                if count != num_frames:
                    raise ValueError(f'{name} {kind} expected {num_frames} frames, got {count}')
        print(f'{name}: existing input_spec, FPS, PTS and {num_frames}-frame RGB round-trip passed')

if __name__ == '__main__':
    main()
