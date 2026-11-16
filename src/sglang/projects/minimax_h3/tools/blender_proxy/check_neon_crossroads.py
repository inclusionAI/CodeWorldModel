"""Check the saved crossroads cameras and their paths through the scene."""
import json

import bpy
from bpy_extras.object_utils import world_to_camera_view

scene = bpy.context.scene
cameras = [bpy.data.objects['Camera left'],bpy.data.objects['Camera right']]
solids = [o for o in scene.objects if o.type == 'MESH' and
          o.get('proxy_category') in ('building','infrastructure','vehicle') and
          len(o.data.vertices) == 8]
if scene.frame_start != 1 or scene.frame_end != 209 or scene.render.fps != 24:
    raise ValueError('unexpected timeline')
collisions = []
shared_max = 0.0
for frame in range(1,210):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    if frame <= 49:
        shared_max = max(shared_max, max(abs(a-b) for row_a,row_b in zip(cameras[0].matrix_world,cameras[1].matrix_world)
                                         for a,b in zip(row_a,row_b)))
    for camera in cameras:
        for obj in solids:
            point = obj.matrix_world.inverted() @ camera.matrix_world.translation
            if all(-.5 < value < .5 for value in point):
                collisions.append((frame,camera.name,obj.name))
if shared_max > 1e-6 or collisions:
    raise AssertionError(f'common camera mismatch={shared_max}, collisions={collisions[:8]}')
scene.frame_set(1)
scene.camera = cameras[0]
anchors = {}
for name in ('Tower circular crown','Parked vehicle','Street lamp'):
    position = world_to_camera_view(scene,cameras[0],bpy.data.objects[name].matrix_world.translation)
    if position.z <= 0 or not (0 <= position.x <= 1 and 0 <= position.y <= 1):
        raise AssertionError(f'{name} is outside the initial camera view: {position}')
    anchors[name] = [round(position.x,4),round(1-position.y,4)]
print('CROSSROADS_CHECK',json.dumps({'frames':209,'fps':24,'shared_camera_frames':49,
      'shared_matrix_max_error':shared_max,'camera_center_collisions':len(collisions),
      'initial_anchor_pixels_normalized':anchors}))
