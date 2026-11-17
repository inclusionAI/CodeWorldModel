"""Compare rendered fine grayscale with independent camera-ray geometry."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

out = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
bpy.ops.wm.open_mainfile(filepath=str(out/'scene.blend'))
scene = bpy.context.scene
scene.frame_set(1)
camera = scene.camera
view = camera.data.view_frame(scene=scene)
left, right = min(v.x for v in view), max(v.x for v in view)
bottom, top = min(v.y for v in view), max(v.y for v in view)
result = []
for x,y in ((336,370),(200,200),(400,210)):
    local = Vector((left+(right-left)*(x+.5)/672, top-(top-bottom)*(y+.5)/384, view[0].z))
    direction = (camera.matrix_world.to_3x3() @ local).normalized()
    hit, point, normal, face, obj, matrix = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), camera.location, direction)
    if not hit or obj.name not in ('Road','Building left','Building right'):
        raise ValueError(f'No known gray surface at {(x,y)}: {obj.name if obj else None}')
    depth = -(camera.matrix_world.inverted() @ point).z
    expected = max(45, min(230, int(230-36*math.log1p(max(depth,0)))))
    result.append({'x':x,'y':y,'object':obj.name,'depth_m':depth,'expected_gray':expected})
(out/'depth-expectations.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
print(json.dumps(result))
