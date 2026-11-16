"""Check the actual wire envelope separately from detailed subject geometry."""
import json
from pathlib import Path
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'docs/demo-brainstorm/camera-director').glob('*/assets/scene/natural-20261108-v1/scene.blend'):
    bpy.ops.wm.open_mainfile(filepath=str(p));scene=bpy.context.scene;cam=scene.camera
    wire=bpy.data.objects['L1Wire_Cuboid'];corners=[wire.matrix_world@Vector(v) for v in wire.bound_box]
    paths=json.loads((p.parent/'camera-paths.json').read_text())['routes'];bad=[]
    for route,poses in paths.items():
        for item in poses:
            cam.location=item['position'];cam.rotation_quaternion=(Vector(item['target'])-cam.location).to_track_quat('-Z','Y')
            cam.data.lens=item['lens'];cam.data.shift_x,cam.data.shift_y=item['shift'];bpy.context.view_layer.update()
            q=[world_to_camera_view(scene,cam,v) for v in corners]
            # Portrait framing deliberately drops lower legs; the top must still stay visible.
            if any(v.x<.02 or v.x>.98 or (v.y<.02 and route!='portrait_push') or v.y>.98 or v.z<=0 for v in q):bad.append((route,item['frame']))
    (p.parent/'wire-framing-validation.json').write_text(json.dumps(dict(violations=bad,margin_normalized=.02,scope='all 8 envelope corners, all144 poses per route',
        exceptions='portrait_push permits lower-leg/bottom cropping by design; no exemption for top or sides'),indent=2),encoding='utf8')
    print('WIRE_FRAMING',p.parts[-5],len(bad),bad[:2],flush=True)
