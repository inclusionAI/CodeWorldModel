"""Create and render a coarse semantic crossroads with two camera paths."""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

COLORS = {
    'sky': (105,145,225), 'road': (60,62,78), 'building': (130,100,190),
    'infrastructure': (235,135,30), 'vehicle': (210,60,50), 'vegetation': (80,150,80),
}
FRAMES = 209


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*(x/255 for x in rgb),1)
    mat.node_tree.links.new(emission.outputs['Emission'],out.inputs['Surface'])
    return mat


def build(out):
    blend = out/'crossroads.blend'
    if blend.exists():
        raise FileExistsError(blend)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device, scene.cycles.samples = 'CPU', 1
    scene.render.threads_mode, scene.render.threads = 'FIXED', 2
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 672,384,100
    scene.render.fps, scene.render.fps_base = 24,1
    scene.frame_start, scene.frame_end = 1,FRAMES
    scene.view_settings.view_transform = 'Raw'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.dither_intensity = 0
    world = bpy.data.worlds.new('Semantic sky')
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (*(x/255 for x in COLORS['sky']),1)
    background.inputs['Strength'].default_value = 1
    scene.world = world
    mats = {key: material(key,value) for key,value in COLORS.items() if key != 'sky'}

    def box(name, center, size, kind):
        bpy.ops.mesh.primitive_cube_add(size=1,location=center)
        obj = bpy.context.object
        obj.name = name
        obj.scale = size
        obj['proxy_category'] = kind
        obj.data.materials.append(mats[kind])
        return obj

    def beam(name, a, b, width, kind):
        a,b = Vector(a),Vector(b)
        obj = box(name,(a+b)/2,(width,width,(b-a).length),kind)
        obj.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
        return obj

    box('Street', (0,13,-.15), (10,100,.3), 'road')
    box('Cross street', (0,8,-.14), (100,12,.28), 'road')
    box('Left sidewalk', (-6.5,-3,-.1), (3,34,.2), 'road')
    box('Right sidewalk', (6.5,-3,-.1), (3,34,.2), 'road')
    box('Left foreground building',(-11.4,.5,7),(13.2,7,14),'building')
    box('Right foreground building',(14,.5,7),(12,7,14),'building')
    box('Left corner block',(-13,29,7),(15,30,14),'building')
    box('Right corner block',(35,29,7),(15,30,14),'building')
    box('Distant tower',(0,72,8),(7,7,16),'building')
    bpy.ops.mesh.primitive_torus_add(major_segments=32,minor_segments=8,location=(0,72,16.5),major_radius=4.8,minor_radius=.4)
    crown = bpy.context.object
    crown.name = 'Tower circular crown'
    crown['proxy_category'] = 'building'
    crown.data.materials.append(mats['building'])
    for x,y in ((-4.0,.5),(4.8,14),(-4.8,27),(4.8,31)):
        box('Street lamp', (x,y,3.25),(.22,.22,6.5),'infrastructure')
        box('Lamp arm', (x+(0.6 if x<0 else -0.6),y,6.45),(1.4,.25,.2),'infrastructure')
    box('Parked vehicle', (6.7,-1.0,.7), (1.9,4.3,1.4), 'vehicle')
    box('Vehicle cabin', (6.7,-1.3,1.45), (1.6,2.2,.7), 'vehicle')
    for x in (-9,-14,-19,-24):
        box('Arcade near pillar',(x,3,2.2),(.7,.7,4.4),'building')
        box('Arcade far pillar',(x,13,2.2),(.7,.7,4.4),'building')
        box('Arcade cross beam',(x,8,4.3),(.7,10,.6),'building')
    box('Arcade side wall',(-17,14,4),(20,1,8),'building')
    for x in (9,22):
        box('Bridge tower near',(x,3,4),(.8,.8,8),'building')
        box('Bridge tower far',(x,13,4),(.8,.8,8),'building')
        box('Bridge cross beam',(x,8,7.7),(.8,10,.6),'building')
        for y in (3,13):
            beam('Bridge cable',(x,y,7.5),(x-6,y,1.0),.16,'building')
            beam('Bridge cable',(x,y,7.5),(x+6,y,1.0),.16,'building')
    for y in (3,13):
        box('Bridge guardrail',(20,y,.75),(28,.25,1.5),'infrastructure')

    paths = {}
    for direction, sign in (('left',-1),('right',1)):
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = 'Camera '+direction
        camera.data.lens, camera.data.sensor_width = 28,36
        paths[direction] = []
        for frame in range(1,FRAMES+1):
            if frame <= 49:
                t = (frame-1)/48
                x,y,yaw = 0,-14+17*t,0
            elif frame <= 133:
                t = (frame-49)/84
                angle = (math.pi/2)*(t*t*(3-2*t))
                x = sign*5*(1-math.cos(angle))
                y = 3+5*math.sin(angle)
                yaw = -sign*angle
            else:
                t=(frame-133)/(FRAMES-133)
                x,y,yaw=sign*(5+11*t),8,-sign*math.pi/2
            camera.location = (x,y,1.6)
            camera.rotation_euler = (math.pi/2,0,yaw)
            camera.keyframe_insert(data_path='location',frame=frame)
            camera.keyframe_insert(data_path='rotation_euler',frame=frame)
            paths[direction].append({'frame':frame,'position_m':[x,y,1.6],'yaw_radians':yaw})
    scene.camera = bpy.data.objects['Camera left']
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    (out/'camera-paths.json').write_text(json.dumps({'fps':24,'frames':FRAMES,'paths':paths},indent=2)+'\n',encoding='utf8')
    print('CROSSROADS_BUILT',blend)


def render(out, sample):
    scene=bpy.context.scene
    target=out/('samples' if sample else 'frames')
    if target.exists():
        raise FileExistsError(target)
    target.mkdir()
    frames = (1,49,91,133,175,209) if sample else range(1,FRAMES+1)
    for direction in ('left','right'):
        directory=target/direction
        directory.mkdir()
        scene.camera=bpy.data.objects['Camera '+direction]
        for frame in frames:
            scene.frame_set(frame)
            scene.render.filepath=str(directory/f'{frame-1:04}.png')
            bpy.ops.render.render(write_still=True)
    print('CROSSROADS_RENDERED',target)


args=sys.argv[sys.argv.index('--')+1:]
parser=argparse.ArgumentParser()
parser.add_argument('operation',choices=('build','sample','render'))
parser.add_argument('output',type=Path)
options=parser.parse_args(args)
out=options.output.resolve()
if options.operation=='build':
    build(out)
else:
    render(out,options.operation=='sample')
