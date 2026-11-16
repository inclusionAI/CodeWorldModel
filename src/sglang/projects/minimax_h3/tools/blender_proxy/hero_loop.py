"""Build and render a synthetic harbor loop for the project-page hero.

This exporter deliberately has no dependency on the model-input proxy contract.
In live Blender, import this file and call build_scene(new_external_directory).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

STAGES = ('rgb', 'fine', 'coarse', 'depth')
LIMITATION = 'Synthetic stylized Blender concept; not a model result or training-standard proxy.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enum_value(owner, property_name, candidates):
    values = {item.identifier for item in owner.bl_rna.properties[property_name].enum_items}
    for value in candidates:
        try:
            setattr(owner, property_name, value)
            return value
        except (TypeError, ValueError):
            continue
    raise ValueError(f'No supported {property_name}; RNA reported {values}, tried {candidates}')


def build_scene(output, frames=48, fps=24, width=640, height=360):
    """Create a new named scene and save a copy, preserving all existing scenes."""
    if frames < 4 or fps < 1 or width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError('Need >=4 frames, positive fps, and positive even tile dimensions')
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite output: {out}')
    out.mkdir(parents=True)
    scene = bpy.data.scenes.new('Hero Harbor Loop')
    if bpy.context.window:
        bpy.context.window.scene = scene
    enum_value(scene.render, 'engine', ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'))
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    scene.frame_start, scene.frame_end = 1, frames
    enum_value(scene.view_settings, 'view_transform', ('Standard',))
    enum_value(scene.view_settings, 'look', ('None', 'Medium High Contrast', 'AgX - Medium High Contrast'))
    scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
    world = bpy.data.worlds.new('Hero Harbor Sky')
    world.use_nodes = True
    scene.world = world
    background = next(node for node in world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs[0].default_value = (.38, .61, .78, 1)
    background.inputs[1].default_value = .65
    palette = {
        'ground': ((.69, .61, .46), (.57, .49, .37)),
        'building': ((.91, .68, .43), (.54, .43, .7)),
        'vegetation': ((.13, .36, .23), (.22, .56, .31)),
        'water': ((.07, .43, .55), (.24, .51, .69)),
        'human': ((.88, .24, .12), (.1, .72, .86)),
        'detail': ((.22, .28, .3), (.42, .48, .57)),
        'boat': ((.94, .89, .72), (.74, .58, .36)),
    }
    materials = {}

    def material(name, color, emission=False):
        mat = bpy.data.materials.new('Hero ' + name)
        mat.use_fake_user = True
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        surface = nodes.new('ShaderNodeEmission' if emission else 'ShaderNodeBsdfPrincipled')
        surface.inputs[0].default_value = (*color, 1)
        if not emission:
            surface.inputs['Roughness'].default_value = .68
        target = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(surface.outputs[0], target.inputs[0])
        return mat

    for category, (rgb, proxy) in palette.items():
        materials[category] = (material(category, rgb), material(category + ' fine', proxy, True),
                               material(category + ' coarse', tuple(v * .85 for v in proxy), True))
    for category in ('ground', 'building'):
        mat = materials[category][1]
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        surface = next(node for node in nodes if node.type == 'EMISSION')
        camera_data = nodes.new('ShaderNodeCameraData')
        mapping = nodes.new('ShaderNodeMapRange')
        mapping.clamp = True
        for socket,value in (('From Min',7),('From Max',65),('To Min',.8),('To Max',.18)):
            mapping.inputs[socket].default_value = value
        links.new(camera_data.outputs['View Z Depth'], mapping.inputs['Value'])
        links.new(mapping.outputs[0], surface.inputs[0])
    depth = material('depth', (1, 1, 1), True)
    nodes, links = depth.node_tree.nodes, depth.node_tree.links
    emission = next(node for node in nodes if node.type == 'EMISSION')
    camera_data = nodes.new('ShaderNodeCameraData')
    mapping = nodes.new('ShaderNodeMapRange')
    mapping.clamp = True
    mapping.inputs['From Min'].default_value = 7
    mapping.inputs['From Max'].default_value = 65
    mapping.inputs['To Min'].default_value = 1
    mapping.inputs['To Max'].default_value = 0
    links.new(camera_data.outputs['View Z Depth'], mapping.inputs['Value'])
    links.new(mapping.outputs[0], emission.inputs[0])

    def mesh(name, vertices, faces, category, role='all'):
        data = bpy.data.meshes.new('Hero ' + name)
        data.from_pydata(vertices, [], faces)
        data.update()
        obj = bpy.data.objects.new('Hero ' + name, data)
        scene.collection.objects.link(obj)
        obj['hero_role'], obj['hero_category'] = role, category
        for stage, mat in zip(STAGES[:3], materials[category]):
            obj['hero_' + stage] = mat.name
        obj['hero_depth'] = depth.name
        data.materials.append(materials[category][0])
        return obj

    def box(name, location, size, category, role='all'):
        vertices = [(x/2, y/2, z/2) for x, y, z in
                    [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]]
        obj = mesh(name, vertices, [(0,4,6,2),(1,3,7,5),(0,1,5,4),(2,6,7,3),(0,2,3,1),(4,5,7,6)], category, role)
        obj.location, obj.scale = location, size
        return obj

    def ellipsoid(name, location, scale, category, role='detail', segments=12, rings=8):
        vertices = []
        for row in range(rings + 1):
            theta = math.pi * row / rings
            for col in range(segments):
                phi = math.tau * col / segments
                vertices.append((math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)))
        faces = []
        for row in range(rings):
            for col in range(segments):
                a, b = row*segments+col, row*segments+(col+1)%segments
                faces.append((a,b,b+segments,a+segments))
        obj = mesh(name, vertices, faces, category, role)
        obj.location, obj.scale = location, scale
        return obj

    box('promenade', (0, 7, -.3), (18, 42, .6), 'ground')
    box('seawall', (5.3, 8, .1), (.5, 42, .8), 'ground')
    box('sea', (23, 15, -.55), (35, 60, .15), 'water')
    for index, (x,y,w,h) in enumerate([(-9,6,4,6),(-11,14,5,9),(-10,23,6,12),(-6,30,5,7),(7,43,6,9),(17,47,8,12),(29,51,6,8)]):
        box(f'building {index}', (x,y,h/2), (w,5,h), 'building')
        box(f'roof {index}', (x,y,h+.12), (w+.3,5.3,.24), 'detail', 'detail')
        for z in range(2, int(h), 2):
            for offset in (-.8,.8):
                box(f'window {index} {z} {offset}', (x+offset,y-2.52,z), (.65,.06,.9), 'detail', 'detail')
    for index, (x,y) in enumerate([(-4,3),(-4,13),(-4,25),(3.7,19)]):
        box(f'trunk {index}', (x,y,1.8), (.32,.32,3.6), 'detail', 'detail')
        ellipsoid(f'canopy {index}', (x,y,4), (1.7,1.6,2.3), 'vegetation')
        box(f'tree proxy {index}', (x,y,2.8), (2.6,2.5,5.6), 'vegetation', 'coarse')
    for y in (1,7,13,19,25):
        box(f'rail post {y}', (4.8,y,.8), (.12,.12,1.6), 'detail', 'detail')
    box('railing', (4.8,13,1.4), (.12,25,.12), 'detail', 'detail')
    for y in (6,18):
        box(f'bench seat {y}', (-2.8,y,.65), (1.6,.6,.16), 'boat', 'detail')
        box(f'bench back {y}', (-2.8,y+.3,1), (1.6,.12,.6), 'boat', 'detail')
    boat = box('boat hull', (13,14,.05), (2.4,6,.8), 'boat')
    cabin = box('boat cabin', (13,14,1), (1.5,2,1.2), 'boat', 'detail')
    mast = box('boat mast', (13,15,2.7), (.1,.1,4.8), 'detail', 'detail')
    ripples = [box(f'water glint {i}', (8+i*1.6,4+(i%5)*4,-.44), (1.1,.08,.012), 'boat', 'detail') for i in range(15)]
    root = bpy.data.objects.new('Hero walker motion', None)
    scene.collection.objects.link(root)
    body = ellipsoid('walker coat', (0,0,1.1), (.25,.17,.4), 'human')
    head = ellipsoid('walker head', (0,0,1.72), (.15,.15,.19), 'boat')
    hair = ellipsoid('walker hair', (0,.02,1.82), (.16,.15,.11), 'detail')
    limbs = [ellipsoid('walker limb ' + str(i), (0,0,0), (.075,.075,.3), 'human') for i in range(8)]
    coarse_body = box('walker coarse torso', (0,0,1.12), (.5,.34,.7), 'human', 'coarse')
    coarse_head = box('walker coarse head', (0,0,1.72), (.3,.3,.35), 'human', 'coarse')
    coarse_limbs = [box('walker coarse limb ' + str(i), (x,0,z), (.16,.22,h), 'human', 'coarse')
                    for i,(x,z,h) in enumerate([(-.16,.4,.8),(.16,.4,.8),(-.36,1.05,.6),(.36,1.05,.6)])]
    for obj in [body,head,hair,*limbs,coarse_body,coarse_head,*coarse_limbs]:
        obj.parent = root
    for frame in range(1, frames + 2):
        phase = math.tau * ((frame - 1) % frames) / frames
        root.location = (.55*math.sin(phase), 2+.65*math.cos(phase), .025*math.cos(phase*4))
        root.rotation_euler.z = .55*math.sin(phase)
        for attr in ('location','rotation_euler'):
            root.keyframe_insert(data_path=attr, frame=frame)
        for side_index, side in enumerate((-1,1)):
            swing = side * math.sin(phase*2)
            hip, knee, foot = (side*.13,0,.82), (side*.13,.2*swing,.46), (side*.13,.35*swing,.09+.12*max(0,-swing))
            shoulder, elbow, hand = (side*.29,0,1.36), (side*.33,-.16*swing,1.06), (side*.34,-.3*swing,.84)
            for limb, (a,b) in zip(limbs[side_index*4:side_index*4+4], [(hip,knee),(knee,foot),(shoulder,elbow),(elbow,hand)]):
                a, b = Vector(a), Vector(b)
                limb.location = (a+b)/2
                limb.scale = (.072,.072,(b-a).length/2+.04)
                limb.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
                for attr in ('location','scale','rotation_euler'):
                    limb.keyframe_insert(data_path=attr, frame=frame)
        for obj in (boat,cabin,mast):
            obj.location.z += .035*math.sin(phase)
            obj.keyframe_insert(data_path='location', frame=frame)
            obj.location.z -= .035*math.sin(phase)
        for i,obj in enumerate(ripples):
            obj.scale.x = 1.1 + .3*math.sin(phase+i)
            obj.keyframe_insert(data_path='scale', frame=frame)
    camera = bpy.data.objects.new('Hero fixed camera', bpy.data.cameras.new('Hero camera'))
    scene.collection.objects.link(camera)
    camera.location = (8,-12,6.8)
    camera.rotation_euler = (Vector((0.5,9,2.1))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.lens = 34
    scene.camera = camera
    sun = bpy.data.objects.new('Hero warm sun', bpy.data.lights.new('Hero sun', 'SUN'))
    scene.collection.objects.link(sun)
    sun.rotation_euler = (.5,-.6,-.5)
    sun.data.energy, sun.data.angle = 2.2, .15
    sun.data.color = (1,.83,.64)
    scene['hero_output'], scene['hero_frames'] = str(out), frames
    scene['hero_depth_near'], scene['hero_depth_far'] = 7.0, 65.0
    set_stage(scene, 'rgb')
    scene.frame_set(1)
    info = {'schema':'hero-loop-source/1', 'scene':scene.name, 'fps':fps, 'frameCount':frames,
            'tileSize':[width,height], 'depth':{'near':7,'far':65,'nearValue':1,'farValue':0,'measure':'camera view Z'},
            'blenderVersion':bpy.app.version_string, 'source':{'script':Path(__file__).name,'sha256':digest(__file__)},
            'limitation':LIMITATION}
    (out/'scene-info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'hero-loop.blend'), copy=True)
    return scene


def set_stage(scene, stage):
    if stage not in STAGES:
        raise ValueError(stage)
    enum_value(scene.view_settings, 'view_transform', ('Standard',) if stage == 'rgb' else ('Raw',))
    for obj in scene.objects:
        if 'hero_role' not in obj:
            continue
        role = obj['hero_role']
        obj.hide_render = role == 'coarse' if stage != 'coarse' else role == 'detail'
        obj.hide_viewport = obj.hide_render
        if obj.type == 'MESH':
            obj.data.materials[0] = bpy.data.materials[obj['hero_' + stage]]
    background = next(node for node in scene.world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs[0].default_value = (.38,.61,.78,1) if stage == 'rgb' else ((0,0,0,1) if stage == 'depth' else (.3,.47,.65,1))


def render_scene(output, sample=False):
    out = Path(output).resolve()
    info = json.loads((out/'scene-info.json').read_text(encoding='utf8'))
    scene = bpy.data.scenes[info['scene']]
    target = out/('sample' if sample else 'frames')
    target.mkdir()
    frames = info['frameCount']
    indices = [1,frames//2+1,frames,frames+1] if sample else range(1,frames+2)
    try:
        for stage in STAGES:
            folder = target/stage
            folder.mkdir()
            set_stage(scene, stage)
            for index in indices:
                scene.frame_set(index)
                scene.render.filepath = str(folder/f'{index:04}.png')
                bpy.ops.render.render(write_still=True, scene=scene.name)
    finally:
        set_stage(scene, 'rgb')
        scene.frame_set(1)


def main():
    args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build','render','sample'))
    parser.add_argument('output')
    options = parser.parse_args(args)
    if options.action == 'build':
        build_scene(options.output)
    else:
        render_scene(options.output, options.action == 'sample')


if __name__ == '__main__':
    main()
