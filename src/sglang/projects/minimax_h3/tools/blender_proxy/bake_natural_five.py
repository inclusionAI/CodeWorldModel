"""Save inspectable animated Blender projects from the frozen camera samples."""
import hashlib
import json
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
IDS=('01-neon-car-ad','02-cliff-hero','04-mechanical-ad','05-crystal-stage','06-motorcycle-tempo')
for case in IDS:
    directory=ROOT/'docs/demo-brainstorm/camera-director'/case/'assets/scene/natural-20261108-v1'
    paths=json.loads((directory/'camera-paths.json').read_text(encoding='utf8'))
    outputs=[]
    for name,items in paths['routes'].items():
        bpy.ops.wm.open_mainfile(filepath=str(directory/'scene.blend'))
        scene=bpy.context.scene;cam=scene.camera;scene.frame_start=1;scene.frame_end=144;scene.render.fps=24
        cam.animation_data_clear();cam.data.animation_data_clear();previous=None
        for item in items:
            frame=item['frame'];scene.frame_set(frame);cam.location=item['position']
            cam.rotation_quaternion=(Vector(item['target'])-cam.location).to_track_quat('-Z','Y')
            if previous and previous.dot(cam.rotation_quaternion)<0:cam.rotation_quaternion.negate()
            previous=cam.rotation_quaternion.copy();cam.data.lens=item['lens'];cam.data.shift_x,cam.data.shift_y=item['shift']
            for prop in ('location','rotation_quaternion'):cam.keyframe_insert(data_path=prop,frame=frame)
            for prop in ('lens','shift_x','shift_y'):cam.data.keyframe_insert(data_path=prop,frame=frame)
        for obj in (cam,cam.data):
            for layer in obj.animation_data.action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for fc in bag.fcurves:
                            for key in fc.keyframe_points:key.interpolation='LINEAR'
        for item in items:
            scene.frame_set(item['frame'])
            assert (cam.location-Vector(item['position'])).length<1e-5
            assert abs(cam.data.lens-item['lens'])<1e-4
        scene.frame_set(1);path=directory/f'{name}.blend';bpy.ops.wm.save_as_mainfile(filepath=str(path))
        outputs.append(dict(route=name,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (directory/'animation-projects.json').write_text(json.dumps(dict(projects=outputs,
        camera_paths_sha256=hashlib.sha256((directory/'camera-paths.json').read_bytes()).hexdigest(),
        baker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validation='144 baked poses match camera samples'),indent=2),encoding='utf8')
print('BAKED 10 animation projects')
