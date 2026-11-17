"""Encode and audit a rendered Blender proxy sample."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def video(files, target, codec, pix_fmt):
    with av.open(str(target), 'w') as container:
        stream = container.add_stream(codec, rate=24)
        stream.width, stream.height, stream.pix_fmt = 672, 384, pix_fmt
        if codec == 'libx264':
            stream.options = {'crf': '18', 'preset': 'medium'}
        for path in files:
            pixels = np.asarray(Image.open(path).convert('RGB'))
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--standard', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=39)
    parser.add_argument('--coarse-action', required=True)
    parser.add_argument('--sample-check', action='store_true')
    args = parser.parse_args()
    if args.frames < 5 or args.frames % 17 != 5 or not args.coarse_action.strip():
        parser.error('frames must be 17*n+5 and coarse action must be nonempty')
    out = args.directory.resolve()
    scene_info_path = out/'scene-info.json'
    scene_info = json.loads(scene_info_path.read_text(encoding='utf8'))
    if scene_info.get('schema') != 'blender-proxy-scene/1' or scene_info.get('fps') != 24 or scene_info.get('frames',0) < args.frames:
        raise ValueError('scene-info does not cover the requested 24 FPS frames')
    ref = out/'reference.png'
    if Image.open(ref).size != (672,384):
        raise ValueError('reference must be 672x384')
    variants = {}
    for mode, profile, prompt_name, dimensions, resample in (
        ('fine','depth_semantic_fine_v1','fine-inference-original.txt',(448,256),Image.Resampling.BOX),
        ('coarse','legacy_semantic_coarse_v1','old-coarse-training-fixed.txt',(336,192),Image.Resampling.NEAREST)):
        folder = out/mode
        files = sorted(folder.glob('*.png'))
        if len(files) < args.frames:
            raise ValueError(f'{mode} requires {args.frames} source frames, got {len(files)}')
        files = files[:args.frames]
        if any(path.name != f'{i:04}.png' for i,path in enumerate(files)):
            raise ValueError(f'{mode} source frames must be contiguous from 0000.png')
        reassigned = 0
        if mode == 'coarse':
            palette = np.array([(105,145,225),(60,62,78),(130,100,190),(80,150,80),(60,170,240)], dtype=np.int32)
            destination = out/'coarse-categorical'
            destination.mkdir()
            for path in files:
                original = np.asarray(Image.open(path).convert('RGB'), dtype=np.int32)
                distance = ((original[:,:,None,:] - palette[None,None,:,:])**2).sum(axis=3)
                quantized = palette[distance.argmin(axis=2)].astype(np.uint8)
                reassigned += int(np.any(quantized != original, axis=2).sum())
                Image.fromarray(quantized).save(destination/path.name)
            files = sorted(destination.glob('*.png'))
        lossless, preview = out/f'{mode}-lossless.mkv', out/f'{mode}-preview.mp4'
        video(files, lossless, 'ffv1', 'bgr0')
        video(files, preview, 'libx264', 'yuv420p')
        with av.open(str(lossless)) as container:
            decoded = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        if len(decoded) != len(files):
            raise ValueError(f'{mode} lossless frame count mismatch')
        for path, pixels in zip(files, decoded):
            if not np.array_equal(np.asarray(Image.open(path).convert('RGB')), pixels):
                raise ValueError(f'lossless RGB mismatch: {path}')
        first = Image.open(files[0]).convert('RGB')
        first.resize(dimensions, resample).save(out/f'{mode}-model-input-example.png')
        prompt = (args.standard/'prompts'/prompt_name).read_text(encoding='utf8')
        if mode == 'coarse':
            prompt += '\n\nAction: ' + args.coarse_action.strip() + '\n'
        (out/f'{mode}-prompt.txt').write_text(prompt, encoding='utf8')
        variants[mode] = {'profile':profile, 'source_size':[672,384], 'historical_model_size':list(dimensions),
                          'historical_resize':resample.name.lower(), 'first_frame_sha256':digest(files[0]),
                          'lossless':lossless.name, 'lossless_sha256':digest(lossless), 'lossless_rgb_verified':True,
                          'preview':preview.name, 'preview_sha256':digest(preview), 'preview_codec':'h264-yuv420p-lossy',
                          'prompt':f'{mode}-prompt.txt', 'prompt_sha256':digest(out/f'{mode}-prompt.txt'),
                          'categorical_pixels_reassigned':reassigned if mode == 'coarse' else None}
    depth_check_path = out/'depth-expectations.json'
    if depth_check_path.exists():
        depth_check = json.loads(depth_check_path.read_text(encoding='utf8'))
        image = Image.open(out/'fine/0000.png').convert('RGB')
        for item in depth_check:
            pixel = image.getpixel((item['x'], item['y']))
            if max(abs(channel-item['expected_gray']) for channel in pixel) > 2:
                raise ValueError(f'depth check failed: {item}, {pixel}')
    def bounds(path, color):
        pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
        y,x = np.where(np.max(np.abs(pixels-np.array(color)), axis=2) <= 2)
        if not len(x):
            raise ValueError(f'person mask missing: {path}')
        return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    ref_bounds = bounds(ref, [75,70,81]) if args.sample_check else None
    fine_bounds = bounds(out/'fine/0000.png', [60,169,239]) if args.sample_check else None
    if args.sample_check:
        if max(abs(a-b) for a,b in zip(ref_bounds, fine_bounds)) > 2:
            raise ValueError(f'sample first-frame bounds differ: {ref_bounds}, {fine_bounds}')
    metadata = {'schema':'blender-proxy-demo/1', 'fps':24, 'frames':args.frames, 'frame_range':[0,args.frames-1],
                'scene':'scene.blend', 'reference':'reference.png', 'reference_sha256':digest(ref),
                'scene_file':'scene.blend', 'scene_sha256':digest(out/'scene.blend'), 'variants':variants,
                'scene_info':'scene-info.json', 'scene_info_sha256':digest(scene_info_path),
                'alignment':'same Blender scene, camera and first frame; reference and proxy role geometry can differ',
                'reference_person_bounds':ref_bounds, 'fine_person_bounds':fine_bounds,
                'depth_check':'depth-expectations.json; three known surfaces agree within two code values' if depth_check_path.exists() else 'not performed',
                'deviations':['scene mesh provenance and human geometry require separate verification',
                              'coarse pixels are snapped to nearest palette after render; edge labels are approximate',
                              'training equivalence, checkpoint-specific preprocessing and model quality not verified',
                              'unconditional/no-text encoder behavior not verified'],
                'text_condition':'historical matched prompts provided; no-text requires recipe-specific verification'}
    (out/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'status':'ok','frames':args.frames,'lossless_rgb_verified':True,'manifest':str(out/'manifest.json')}))

if __name__ == '__main__':
    main()
