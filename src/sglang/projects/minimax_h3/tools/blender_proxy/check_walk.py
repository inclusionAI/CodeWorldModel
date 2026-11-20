"""Check shared joint endpoints in the sample walk animation."""
import json
import sys

import bpy
from mathutils import Vector

path = sys.argv[sys.argv.index('--')+1]
bpy.ops.wm.open_mainfile(filepath=path)
scene = bpy.context.scene
radii = [.048,.044,.115,.044,.038,.038,.044,.042,.042,.044,.038,.038,.044,.042,.042]
pairs = [((0,1),(1,0)),((0,1),(3,0)),((0,1),(9,0)),((0,0),(6,0)),((0,0),(12,0)),
         ((3,1),(4,0)),((4,1),(5,0)),((6,1),(7,0)),((7,1),(8,0)),
         ((9,1),(10,0)),((10,1),(11,0)),((12,1),(13,0)),((13,1),(14,0))]
max_error = 0.0
roots = []
for frame in range(1,40):
    scene.frame_set(frame)
    graph = bpy.context.evaluated_depsgraph_get()
    points = []
    for i,radius in enumerate(radii):
        obj = bpy.data.objects[f'Fine articulated capsules-{i}'].evaluated_get(graph)
        t = max(0,1-radius/obj.scale.z)
        points.append((obj.matrix_world @ Vector((0,0,-t)),obj.matrix_world @ Vector((0,0,t))))
    max_error = max(max_error, max((points[a][ai]-points[b][bi]).length for (a,ai),(b,bi) in pairs))
    roots.append(tuple(bpy.data.objects['Fine articulated capsules'].location))
if max_error > 1e-4 or Vector(roots[-1]).y - Vector(roots[0]).y < .5:
    raise ValueError(f'walk continuity or movement failed: {max_error}, {roots[0]}, {roots[-1]}')
print(json.dumps({'frames':39,'max_joint_gap_m':max_error,'root_start':roots[0],'root_end':roots[-1]}))
