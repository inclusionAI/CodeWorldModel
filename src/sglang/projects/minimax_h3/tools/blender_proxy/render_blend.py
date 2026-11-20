"""Render tagged Blender scenes into reference and proxy frame sequences."""
import json
import math
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index('--') + 1:]
if len(args) != 2:
    raise ValueError('usage: blender -b input.blend --python render_blend.py -- output_dir frame_count')
out = Path(args[0]).resolve()
count = int(args[1])
scene = bpy.context.scene
if not scene.camera or not scene.world or count < 1 or scene.frame_start + count - 1 > scene.frame_end:
    raise ValueError('scene needs a camera, world and the requested frame range')
palette = {'road':(60,62,78),'building':(130,100,190),'vegetation':(80,150,80),'human':(60,170,240)}
fine_colors = {'vegetation':(79,149,80),'human':(60,169,239)}
objects = [o for o in scene.objects if o.type == 'MESH' and not o.hide_render]
objects += [o for o in scene.objects if o.type == 'MESH' and o.hide_render and o.get('proxy_role') in ('fine','coarse','reference')]
for obj in objects:
    if obj.get('proxy_category') not in palette or obj.get('proxy_role','all') not in ('all','fine','coarse','reference'):
        raise ValueError(f'object {obj.name} needs valid proxy_category and proxy_role')
if not objects:
    raise ValueError('no tagged renderable mesh objects')
human_roles = {o.get('proxy_role','all') for o in objects if o['proxy_category'] == 'human'}
if human_roles and (not {'fine','coarse'} <= human_roles or 'all' in human_roles):
    raise ValueError('human proxy requires separate fine and rigid coarse objects')
out.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'), copy=True)
scene.render.engine = 'CYCLES'
scene.cycles.device, scene.cycles.samples = 'CPU', 1
scene.render.threads_mode, scene.render.threads = 'FIXED', 2
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 672,384,100
scene.render.fps, scene.render.fps_base = 24,1
scene.render.image_settings.file_format, scene.render.image_settings.color_mode = 'PNG','RGB'
scene.render.dither_intensity = 0
scene.view_settings.view_transform = 'Raw'

def material(name, color, depth=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    links = m.node_tree.links
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*[x/255 for x in color],1)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    if depth:
        geo = nodes.new('ShaderNodeNewGeometry')
        trans = nodes.new('ShaderNodeVectorTransform')
        trans.vector_type, trans.convert_from, trans.convert_to = 'POINT','WORLD','CAMERA'
        sep = nodes.new('ShaderNodeSeparateXYZ')
        links.new(geo.outputs['Position'], trans.inputs['Vector'])
        links.new(trans.outputs['Vector'], sep.inputs['Vector'])
        prev = sep.outputs['Z']
        for op,value in [('MAXIMUM',0),('ADD',1),('LOGARITHM',math.e),('MULTIPLY',-36),
                         ('ADD',230),('MAXIMUM',45),('MINIMUM',230),('DIVIDE',255)]:
            n = nodes.new('ShaderNodeMath'); n.operation = op; n.inputs[1].default_value = value
            links.new(prev,n.inputs[0]); prev = n.outputs[0]
        links.new(prev,emission.inputs['Color'])
    return m

materials = {mode:{category:material(f'export-{mode}-{category}',
             fine_colors.get(category,(0,0,0)) if mode == 'fine' else color,
             mode == 'fine' and category not in fine_colors)
             for category,color in palette.items()} for mode in ('fine','coarse')}
for obj in objects:
    obj.data = obj.data.copy()
    reference = obj.get('proxy_reference_material')
    if reference:
        if reference not in bpy.data.materials:
            raise ValueError(f'{obj.name}: missing reference material {reference}')
        obj.data.materials.clear()
        obj.data.materials.append(bpy.data.materials[reference])
    elif not obj.data.materials:
        raise ValueError(f'{obj.name}: reference material is missing')
scene.frame_set(scene.frame_start)
reference_categories = {o['proxy_category'] for o in objects if o.get('proxy_role') == 'reference'}
for obj in objects:
    role = obj.get('proxy_role','all')
    obj.hide_render = role == 'coarse' or (role == 'fine' and obj['proxy_category'] in reference_categories)
scene.render.filepath = str(out/'reference.png')
bpy.ops.render.render(write_still=True)
scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get('Background')
if background is None:
    raise ValueError('world needs a Background shader')
background.inputs['Strength'].default_value = 1
for mode in ('fine','coarse'):
    folder = out/mode
    folder.mkdir()
    sky = (105,143,224) if mode == 'fine' else (105,145,225)
    background.inputs['Color'].default_value = (*[x/255 for x in sky],1)
    for obj in objects:
        role = obj.get('proxy_role','all')
        obj.hide_render = role not in ('all',mode)
        obj.data.materials.clear()
        obj.data.materials.append(materials[mode][obj['proxy_category']])
    for frame in range(scene.frame_start,scene.frame_start+count):
        scene.frame_set(frame)
        scene.render.filepath = str(folder/f'{frame-scene.frame_start:04}.png')
        bpy.ops.render.render(write_still=True)
info = {'schema':'blender-proxy-scene/1','source':str(Path(bpy.data.filepath).resolve()),
        'frame_start':scene.frame_start,'frames':count,'fps':24,'camera':scene.camera.name,
        'categories':sorted({o['proxy_category'] for o in objects}),
        'reference_and_proxy':'same camera, frame and object animation; role-specific geometry may differ',
        'blender_version':bpy.app.version_string}
(out/'scene-info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
print('BLENDER_PROXY_RENDER_DONE',out)
