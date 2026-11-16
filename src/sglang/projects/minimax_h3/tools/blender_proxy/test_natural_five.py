"""Blender-only regression: translated geometry, wall occlusion, and containment."""
import importlib.util
from pathlib import Path
import bpy
spec=importlib.util.spec_from_file_location('natural_five',Path(__file__).with_name('natural_five.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def run(camera):
    m.SOLIDS.clear();m.SUBJECT.clear();m.MATS.clear();bpy.ops.wm.read_factory_settings(use_empty=True)
    m.box('TranslatedBlocker',(10,0,1),(2,2,2))
    m.sphere('Subject',(10,5,1),(.3,.3,.3),'vehicle',True)
    bpy.ops.object.camera_add(location=camera);cam=bpy.context.object;cam.rotation_mode='QUATERNION';bpy.context.scene.camera=cam
    bpy.context.view_layer.update()
    return m.validate(cam,{'fixture':[dict(frame=i+1,position=camera,target=(10,5,1),lens=35,shift=(0,0)) for i in range(144)]})

blocked=run((10,-5,1));assert not blocked['violations'];assert len(blocked['center_ray_occlusions'])==144
inside=run((10,0,1));assert any(row[-1] for row in inside['violations'])
clear=run((0,-5,1));assert not clear['violations'] and not clear['center_ray_occlusions']
print('REGRESSION_OK: world-space translated blocker, line-of-sight, inside volume, clear path')
