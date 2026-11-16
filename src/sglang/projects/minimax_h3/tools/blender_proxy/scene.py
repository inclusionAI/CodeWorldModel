"""Create the fixed third-person calibration scene in Blender."""
import math
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
out = Path(args[0]).resolve()
frame_count = int(args[1]) if len(args) > 1 else 49
if not 1 <= frame_count <= 49:
    raise ValueError('frame count must be 1..49')
out.mkdir(parents=True, exist_ok=False)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 1
scene.render.threads_mode = 'FIXED'
scene.render.threads = 2
scene.render.resolution_x, scene.render.resolution_y = 672, 384
scene.render.resolution_percentage = 100
scene.render.fps = 24
scene.frame_start, scene.frame_end = 1, 49
scene.view_settings.view_transform = 'Raw'
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.dither_intensity = 0

def mat(name, rgb, depth=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    links = m.node_tree.links
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*[x / 255 for x in rgb], 1)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    if depth:
        geo = nodes.new('ShaderNodeNewGeometry')
        transform = nodes.new('ShaderNodeVectorTransform')
        transform.vector_type = 'POINT'
        transform.convert_from, transform.convert_to = 'WORLD', 'CAMERA'
        separate = nodes.new('ShaderNodeSeparateXYZ')
        links.new(geo.outputs['Position'], transform.inputs['Vector'])
        links.new(transform.outputs['Vector'], separate.inputs['Vector'])
        prev = separate.outputs['Z']
        for op, value in [('MAXIMUM', 0), ('ADD', 1),
                          ('LOGARITHM', math.e), ('MULTIPLY', -36), ('ADD', 230),
                          ('MAXIMUM', 45), ('MINIMUM', 230), ('DIVIDE', 255)]:
            node = nodes.new('ShaderNodeMath')
            node.operation = op
            node.inputs[1].default_value = value
            links.new(prev, node.inputs[0])
            prev = node.outputs[0]
        links.new(prev, emission.inputs['Color'])
    return m

fine = {'road': mat('fine-depth-road', (0, 0, 0), True), 'building': mat('fine-depth-building', (0, 0, 0), True),
        'vegetation': mat('fine-vegetation', (79,149,80)), 'human': mat('fine-human', (60,169,239))}
coarse = {'road': mat('coarse-road', (60,62,78)), 'building': mat('coarse-building', (130,100,190)),
          'vegetation': mat('coarse-vegetation', (80,150,80)), 'human': mat('coarse-human', (60,170,240))}
appearance = {'road': mat('appearance-road', (96,93,86)), 'building': mat('appearance-building', (167,151,126)),
              'vegetation': mat('appearance-vegetation', (52,113,57)), 'human': mat('appearance-human', (75,70,81))}
for reference_material in appearance.values():
    reference_material.use_fake_user = True
objects = []
def cube(name, loc, scale, category):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    objects.append((obj, category))
    return obj

cube('Road', (0,4,-.15), (12,25,.3), 'road')
cube('Building left', (-6,5,2), (3,14,4), 'building')
cube('Building right', (6,10,2.5), (3,13,5), 'building')
cube('Tree', (3.7,5,1.8), (1.4,1.4,3.6), 'vegetation')
bpy.ops.object.camera_add(location=(0,-7,3.1))
camera = bpy.context.object
camera.rotation_euler = (Vector((0,1.4,1.0)) - camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.lens = 35
scene.camera = camera
world = bpy.data.worlds.new('Sky')
world.use_nodes = True
scene.world = world
background = world.node_tree.nodes.get('Background')
background.inputs['Strength'].default_value = 1

def body(name, parts):
    root = bpy.data.objects.new(name, None)
    scene.collection.objects.link(root)
    for i, (a, b, radius) in enumerate(parts):
        a, b = Vector(a), Vector(b)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8)
        ob = bpy.context.object
        ob.name = name + '-' + str(i)
        ob.location = (a+b)/2
        ob.scale = (radius, radius, max(radius, (b-a).length/2 + radius))
        if (b-a).length:
            ob.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
        ob.parent = root
        objects.append((ob, 'human'))
    return root

torso = [((0,0,0.9),(0,0,1.5),.048),((0,0,1.5),(0,0,1.65),.044),((0,0,1.7),(0,0,1.7),.115)]
limbs=[]
for side in (-1,1):
    limbs += [((0,0,1.5),(.16*side,0,1.48),.044), ((.16*side,0,1.48),(.27*side,.1,1.18),.038),
              ((.27*side,.1,1.18),(.30*side,.2,.94),.038), ((0,0,.9),(.1*side,0,.9),.044),
              ((.1*side,0,.9),(.1*side,.18,.48),.042), ((.1*side,.18,.48),(.1*side,-.08,.08),.042)]
fine_root = body('Fine articulated capsules', torso + limbs)
fixed = [((0,0,.75),(0,0,1.48),.048), ((-.105,0,.75),(.105,0,.75),.044),
         ((0,0,1.28),(-.3,0,1.12),.038), ((-.3,0,1.12),(-.49,0,.865),.038),
         ((0,0,1.28),(.3,0,1.12),.038), ((.3,0,1.12),(.49,0,.865),.038),
         ((-.105,0,.75),(-.105,0,.005),.042), ((.105,0,.75),(.105,0,.005),.042),
         ((0,0,1.55),(0,0,1.63),.115)]
coarse_root = body('Coarse rigid approximate', fixed)
for root in (fine_root, coarse_root):
    for frame, y, yaw in ((1,0,0),(25,1.0,0),(49,2.0,-math.pi/3)):
        root.location = (0,y,0)
        root.rotation_euler.z = yaw
        root.keyframe_insert(data_path='location', frame=frame)
        root.keyframe_insert(data_path='rotation_euler', frame=frame)
for frame in range(1,50):
    step = math.sin((frame-1)*math.pi/6)
    animated = list(torso)
    for side in (-1, 1):
        stride = side*step
        shoulder = (.16*side, 0, 1.48)
        elbow = (.27*side, -.12*stride, 1.18)
        hand = (.30*side, -.22*stride, .94)
        hip = (.1*side, 0, .9)
        knee = (.1*side, .19*stride, .48)
        foot = (.1*side, .32*stride, .08 + .08*max(0, stride))
        animated += [((0,0,1.5),shoulder,.044), (shoulder,elbow,.038),
                     (elbow,hand,.038), ((0,0,.9),hip,.044), (hip,knee,.042), (knee,foot,.042)]
    for idx,(a,b,radius) in enumerate(animated):
        child = bpy.data.objects[f'Fine articulated capsules-{idx}']
        a,b = Vector(a),Vector(b)
        child.location = (a+b)/2
        child.scale = (radius,radius,max(radius,(b-a).length/2+radius))
        if (b-a).length:
            child.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
        for key in ('location','rotation_euler','scale'):
            child.keyframe_insert(data_path=key,frame=frame)

for obj,category in objects:
    obj['proxy_category'] = category
    obj['proxy_role'] = ('fine' if obj.parent == fine_root else
                         'coarse' if obj.parent == coarse_root else 'all')
    obj['proxy_reference_material'] = appearance[category].name

def select_style(mode):
    palette = {'fine':fine, 'coarse':coarse, 'reference':appearance}[mode]
    for obj, category in objects:
        obj.data.materials.clear()
        obj.data.materials.append(palette[category])
        if obj.parent in (fine_root, coarse_root):
            obj.hide_render = obj.parent == (coarse_root if mode != 'coarse' else fine_root)
    sky = (105,143,224) if mode == 'fine' else (105,145,225) if mode == 'coarse' else (138,171,209)
    background.inputs['Color'].default_value = (*[x/255 for x in sky], 1)

scene.frame_set(1)
select_style('reference')
scene.render.filepath = str(out/'reference.png')
bpy.ops.render.render(write_still=True)
for mode in ('fine','coarse'):
    folder = out/mode
    folder.mkdir()
    select_style(mode)
    for frame in range(1,frame_count+1):
        scene.frame_set(frame)
        scene.render.filepath = str(folder/f'{frame-1:04}.png')
        bpy.ops.render.render(write_still=True)
select_style('fine')
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
(out/'scene-info.json').write_text(json.dumps({'schema':'blender-proxy-scene/1', 'source':'generated sample',
    'frame_start':1, 'frames':frame_count, 'fps':24, 'camera':scene.camera.name,
    'categories':['building','human','road','vegetation'], 'blender_version':bpy.app.version_string},indent=2)+'\n',encoding='utf8')
print('BLENDER_PROXY_RENDER_DONE', out)
