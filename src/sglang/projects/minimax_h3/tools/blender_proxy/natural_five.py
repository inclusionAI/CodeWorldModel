"""First-frame-first coarse camera studies; run inside Blender 5.2 (EEVEE).

The manual image bounds constrain the opening projection, not unseen geometry.
Render stills/filmstrips first; --render exports full 144-frame image sequences.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'docs/demo-brainstorm/camera-director'
VERSION = 'natural-20261108-v1'
STYLES = ('wire', 'depth', 'coarse', 'fine', 'subject_bg')
IDS = ('01-neon-car-ad','02-cliff-hero','04-mechanical-ad','05-crystal-stage','06-motorcycle-tempo')
PALETTE = {'sky':(105,145,225),'terrain':(85,85,90),'road':(60,62,78),'vegetation':(80,150,80),
           'building':(130,100,190),'infrastructure':(235,135,30),'human':(60,170,240),
           'vehicle':(210,60,50),'prop':(165,140,95)}
SOLIDS = []
SUBJECT = []
MATS = {}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def mat(name, rgb):
    if name in MATS: return MATS[name]
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
    e=n.new('ShaderNodeEmission');o=n.new('ShaderNodeOutputMaterial')
    e.inputs[0].default_value=tuple((v/255/12.92 if v/255<=.04045 else ((v/255+.055)/1.055)**2.4) for v in rgb)+(1,)
    e.inputs[1].default_value=1;m.node_tree.links.new(e.outputs[0],o.inputs[0]);MATS[name]=m
    return m

def register(obj, category, subject=False, collider=True):
    obj['category']=category;obj['subject']=subject
    obj.data.materials.append(mat(category,PALETTE[category]))
    if subject: SUBJECT.append(obj)
    if collider: SOLIDS.append(obj)
    return obj

def box(name, loc, size, cat='building', subject=False, collider=True):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o=bpy.context.object;o.name=name;o.scale=size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return register(o,cat,subject,collider)

def cyl(name, loc, radius, depth, cat='prop', subject=False, rotation=None, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=radius,depth=depth,location=loc)
    o=bpy.context.object;o.name=name
    if rotation:o.rotation_euler=rotation
    return register(o,cat,subject)

def sphere(name, loc, scale, cat='prop', subject=False):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return register(o,cat,subject)

def beam(name, a, b, width, cat='infrastructure', subject=False):
    a,b=Vector(a),Vector(b)
    o=box(name,(a+b)/2,(width,width,(b-a).length),cat,subject)
    o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
    return o

def human(center=(0,0,0), height=1.8, gesture=False):
    x,y,z=center;h=height
    sphere('Head',(x,y,z+.93*h),(.075*h,.07*h,.075*h),'human',True)
    box('Torso',(x,y,z+.66*h),(.24*h,.14*h,.32*h),'human',True)
    for s in (-1,1):
        beam('Leg',(x+s*.075*h,y,z+.52*h),(x+s*.085*h,y,z+.04*h),.075*h,'human',True)
        box('Foot',(x+s*.085*h,y-.03*h,z+.025*h),(.08*h,.16*h,.05*h),'human',True)
        reach=.30 if gesture and s<0 else .19
        beam('Arm',(x+s*.145*h,y,z+.78*h),(x+s*reach*h,y,z+.48*h),.065*h,'human',True)

def mountains(y, scale=1):
    for i in range(-4,5):
        depth=(12+(i%4)*2)*scale
        bpy.ops.mesh.primitive_cone_add(vertices=5,radius1=(9+(i%3)*2)*scale,depth=depth,
                                      location=(i*12*scale,y+(i%2)*8*scale,5*scale-depth/2))
        o=register(bpy.context.object,'terrain');o.name=f'Mountain_{i}'

def torus(name, loc, major, minor, rotation):
    bpy.ops.mesh.primitive_torus_add(major_segments=64,minor_segments=8,location=loc,
                                   major_radius=major,minor_radius=minor,rotation=rotation)
    o=bpy.context.object;o.name=name;return register(o,'infrastructure',True)

def scene_geometry(case):
    # Sizes in metres, smooth camera motion; background details only carry major occlusions.
    if case=='01-neon-car-ad':
        box('Plaza',(0,8,-.15),(65,80,.3),'road',collider=False)
        box('Body',(0,0,.62),(4.65,1.87,.65),'vehicle',True)
        box('Cabin',(-.3,0,1.12),(2.35,1.5,.48),'vehicle',True)
        box('Nose',(2.2,0,.62),(.25,1.8,.3),'vehicle',True)
        for x in (-1.5,1.5):
            for y in (-.95,.95):cyl('Wheel',(x,y,.36),.36,.18,'vehicle',True,(math.pi/2,0,0))
        box('PavilionRoof',(14,8,5.5),(16,13,.65))
        box('PavilionBack',(14,14.5,2.6),(16,.3,5.2))
        for x in (7,11,15,19):cyl('PavilionColumn',(x,3,2.6),.22,5.2,'infrastructure')
        for i in range(12):box('Skyline',(-49+i*3.2,44,3+i%4),(2.4,3,6+(i%4)*2))
        for x,y in ((-10,2),(-13,9),(20,12)):cyl('Planter',(x,y,.35),1.2,.7,'vegetation')
        return dict(camera=(6,-10,1.5),target=(0,0,.65),bounds=(.232,.383,.702,.674),
                    route_names=('low_orbit','crane_reveal'),subject_center=(0,0,.7),end_angle=-100,crane=(-2,-16,9))
    if case=='02-cliff-hero':
        box('Terrace',(0,-5,-.25),(15,13,.5),'terrain',collider=False)
        for x in (-3.4,3.4):
            cyl('Column',(x,1,2.3),.38,4.6,'building');box('ColumnBase',(x,1,.22),(1.1,1.1,.44),'building')
        human()
        box('BackCloak',(0,-.14,.95),(.52,.08,.95),'human',True)
        # Solid cliff only behind the platform edge; the valley stays open.
        box('Cliff',(0,1.9,-13),(15,.8,26),'terrain')
        box('FarCliff',(23,48,-7),(25,12,22),'terrain')
        box('Monastery',(24,48,6),(13,8,4),'building')
        for x in (19,24,29):
            cyl('TempleTower',(x,48,9),1,4,'building')
            bpy.ops.mesh.primitive_cone_add(vertices=8,radius1=1.4,depth=2,location=(x,48,12));register(bpy.context.object,'building')
        # Deck plus sparse piers conveys a spanning bridge, with actual empty space underneath.
        for i in range(14):
            x=-25+i*3.5;y=35+i*.1
            box('BridgeDeck',(x,y,5.1),(3.6,2,.6),'infrastructure')
            if i%2==0:box('BridgePier',(x,y,-5),(1,1.8,20),'infrastructure')
        mountains(88,2)
        return dict(camera=(0,-7,1.6),target=(0,0,.9),bounds=(.40,.416,.515,.811),
                    route_names=('hero_push','valley_crane'),subject_center=(0,0,1),end_angle=0,crane=(0,-14,11))
    if case=='04-mechanical-ad':
        box('HallFloor',(0,8,-.18),(60,75,.36),'road',collider=False)
        cyl('Pedestal',(0,0,.4),4.2,.8,'infrastructure',True)
        sphere('Core',(0,0,3.1),(1.15,1.15,1.15),'prop',True)
        torus('GimbalOuter',(0,0,3.1),3.8,.18,(math.radians(40),math.radians(18),math.radians(20)))
        torus('GimbalEquator',(0,0,3.1),3.55,.15,(math.radians(18),math.radians(-8),math.radians(-15)))
        torus('GimbalInner',(0,0,3.1),3.15,.14,(math.radians(60),math.radians(-35),math.radians(30)))
        for s in (-1,1):beam('Support',(s*4,0,.8),(s*2.8,0,3.3),.45,'infrastructure',True)
        for x in (-27,27):
            box('HallWall',(x,20,8),(.6,60,16))
            for y in range(-4,45,7):box('HallRib',(x,y,8),(1,1,16),'infrastructure')
        box('HallBack',(0,48,8),(54,.5,16))
        box('HallRoof',(0,18,20),(55,60,.7))
        box('ForegroundFrame',(-12,-8,5),(1,1,10),'infrastructure')
        return dict(camera=(4,-19,4.6),target=(0,0,3),bounds=(.292,.098,.837,.770),
                    route_names=('core_push','outer_half_orbit'),subject_center=(0,0,3.5),end_angle=115,crane=None)
    if case=='05-crystal-stage':
        cyl('Stage',(0,0,-.2),2.4,.4,'prop')
        bpy.ops.mesh.primitive_torus_add(major_segments=64,minor_segments=6,major_radius=2.35,minor_radius=.045,location=(0,0,-.01))
        register(bpy.context.object,'infrastructure')
        human(height=1.8,gesture=True)
        # Flat five-sided occupancy shards, no glass shading or decorative facets.
        for i,(x,y,r,h) in enumerate(((-4.8,2,1.3,5),(-4,5,1,4),(-7,8,1.2,8),(5,5,1.1,4),(7,9,1,7),
                                     (2,9,.5,2.2),(-7,-2,.8,2),(6,-1,.8,1.4),(10,13,1,5))):
            bpy.ops.mesh.primitive_cone_add(vertices=5,radius1=r,radius2=.08,depth=h,location=(x,y,h/2))
            o=register(bpy.context.object,'prop');o.name=f'Crystal_{i}';o.rotation_euler[1]=(-.18 if x<0 else .1)
        box('DistantGround',(0,25,-1),(100,70,.5),'terrain',collider=False)
        mountains(110,1.8)
        for o in bpy.context.scene.objects:
            if o.name.startswith('Mountain_'):o.location.z-=8
        return dict(camera=(1,-7,2),target=(0,0,.9),bounds=(.414,.255,.565,.762),
                    route_names=('stage_arc','portrait_push'),subject_center=(0,0,1),end_angle=-40,crane=None)
    if case=='06-motorcycle-tempo':
        box('Asphalt',(0,8,-.1),(70,90,.2),'road',collider=False)
        for x in (-.92,.92):cyl('Wheel',(x,0,.40),.4,.15,'vehicle',True,(math.pi/2,0,0))
        beam('Fork',(-.92,0,.4),(-.55,0,1.08),.12,'vehicle',True)
        box('Engine',(0,0,.58),(.65,.4,.55),'vehicle',True)
        sphere('Tank',(-.22,0,.97),(.46,.3,.22),'vehicle',True)
        box('Seat',(.50,0,1.03),(.8,.4,.12),'vehicle',True)
        cyl('Headlamp',(-.66,0,1.15),.14,.16,'vehicle',True,(0,math.pi/2,0))
        beam('Handlebar',(-.5,-.43,1.2),(-.5,.43,1.2),.07,'vehicle',True)
        beam('Frame',(-.92,0,.4),(.92,0,.4),.1,'vehicle',True)
        for y in (9,23,37,51):box('Pier',(4.8,y,2),(2.2,3,4),'infrastructure')
        box('Viaduct',(4.8,25,4.4),(8,90,.8),'infrastructure')
        for y in range(-6,57,5):box('CeilingBeam',(4.8,y,3.9),(8,.25,.3),'infrastructure')
        for i in range(14):box('Shops',(-28+i*4.5,28,1.7+i%3*.3),(3.9,3,3.4+i%3*.6),'building')
        return dict(camera=(-3.7,-6.2,1.35),target=(0,0,.65),bounds=(.307,.412,.750,.864),
                    route_names=('constant_track','speed_ramp'),subject_center=(0,0,.65),end_angle=0,crane=None)
    raise ValueError(case)

def points(objects):
    return [o.matrix_world@v.co for o in objects for v in o.data.vertices]

def project(cam, p):
    q=world_to_camera_view(bpy.context.scene,cam,p)
    return (q.x,1-q.y,q.z)

def bbox(cam, vertices):
    p=[project(cam,v) for v in vertices]
    assert min(v[2] for v in p)>0
    return (min(v[0] for v in p),min(v[1] for v in p),max(v[0] for v in p),max(v[1] for v in p))

def look(cam, target):
    cam.rotation_quaternion=(Vector(target)-cam.location).to_track_quat('-Z','Y')

def calibrate(cam,cfg):
    verts=points(SUBJECT);wanted=cfg['bounds']
    # Match horizontal envelope and centre, plus foot/base height. Vertical extent is reported separately.
    for _ in range(8):
        b=bbox(cam,verts);cam.data.lens*= min((wanted[2]-wanted[0])/(b[2]-b[0]),(wanted[3]-wanted[1])/(b[3]-b[1]))
        b=bbox(cam,verts);cam.data.shift_x+=(b[0]+b[2]-wanted[0]-wanted[2])/2
        cam.data.shift_y-=(b[3]-wanted[3])*384/672
        bpy.context.view_layer.update()
    return {'manual_bounds_normalized':wanted,'projected_bounds_normalized':bbox(cam,verts),
            'method':'manual subject bounds, approximate pinhole; no full-scene reconstruction claim'}

def ground_at(cam,u,v,z=0):
    frame=cam.data.view_frame(scene=bpy.context.scene)
    ray=cam.rotation_quaternion@Vector((min(p.x for p in frame)+(max(p.x for p in frame)-min(p.x for p in frame))*u,
                                      max(p.y for p in frame)-(max(p.y for p in frame)-min(p.y for p in frame))*v,frame[0].z))
    assert ray.z<0
    return cam.location+ray*((z-cam.location.z)/ray.z)

def align_background(cam,case):
    if case=='01-neon-car-ad':
        cols=[o for o in bpy.context.scene.objects if o.name.startswith('PavilionColumn')]
        for o,u in zip(cols,(.63,.76,.89,1.02)):
            foot=ground_at(cam,u,.53);o.location=foot+Vector((0,0,2.6))
        # Pavilion occupies the right half; skyline remains a distant low band.
        p0,p1=cols[0].location,cols[-1].location
        roof=bpy.data.objects['PavilionRoof'];roof.location=(p0+p1)/2+Vector((0,4,2.9))
        roof.dimensions=((p1-p0).length+3,10,.6)
        back=bpy.data.objects['PavilionBack'];back.location=(p0+p1)/2+Vector((0,8,0));back.dimensions=((p1-p0).length+3,.3,5.2)
        for o in bpy.context.scene.objects:
            if o.name.startswith('Skyline'):o.location.z-=2
    bpy.context.view_layer.update()

def ease(t):return t*t*t*(10+t*(-15+6*t))

def poses(cam,cfg,case):
    start=cam.location.copy();target=Vector(cfg['target']);sc=Vector(cfg['subject_center'])
    shift=(cam.data.shift_x,cam.data.shift_y);lens=cam.data.lens
    out={k:[] for k in cfg['route_names']}
    for r,name in enumerate(out):
        for i in range(144):
            t=max(0,(i-23)/120);e=ease(t);p=start.copy();tar=target.copy()
            if case=='06-motorcycle-tempo':
                # Same spatial endpoints and framing: A near-uniform middle speed, B pronounced ramp.
                u=e if r else (t-math.sin(2*math.pi*t)/(2*math.pi)*.25)
                p=start+Vector((5.0*u,0,0));tar=target.copy()
            elif name in ('hero_push','portrait_push','core_push'):
                factor={'hero_push':.65,'portrait_push':.56,'core_push':.65}[name]
                p=start.lerp(sc+(start-sc)*factor,e)
                if name=='portrait_push':tar=target.lerp(Vector((0,0,1.25)),e)
                if name=='hero_push':tar=target.lerp(Vector((0,0,1.2)),e)
            elif r==1 and cfg['crane']:
                p=start.lerp(Vector(cfg['crane']),e)
                tar=target.lerp(Vector((3,28,4)) if case=='02-cliff-hero' else Vector((1,12,3.5)),e*.65)
            else:
                angle=math.radians(cfg['end_angle'])*e;rel=start-sc
                p=sc+Vector((rel.x*math.cos(angle)-rel.y*math.sin(angle),rel.x*math.sin(angle)+rel.y*math.cos(angle),rel.z+(2.3*e if case=='04-mechanical-ad' and r else .5*e)))
            # Recenter gradually so a fitted image shift never sends the subject off-screen later.
            focal=lens
            if name=='crane_reveal':focal*=1-.38*e
            if name=='valley_crane':
                focal*=1-.45*e
                tar=target.lerp(Vector((2,16,0)),e)
            if name=='core_push':
                focal*=1-.28*e
                tar=target.lerp(Vector((0,0,2.5)),e)
            sx,sy=shift[0]*(1-e),shift[1]*(1-e)
            out[name].append(dict(frame=i+1,position=list(p),target=list(tar),lens=focal,shift=[sx,sy]))
    assert list(out.values())[0][:24]==list(out.values())[1][:24]
    return out

def set_pose(cam,p):
    cam.location=p['position'];look(cam,p['target']);cam.data.lens=p['lens']
    cam.data.shift_x,cam.data.shift_y=p['shift'];bpy.context.view_layer.update()

def validate(cam,paths):
    # Triangle BVH tests include hollow rings correctly; sample camera sphere at quarter-frame spacing.
    from mathutils.bvhtree import BVHTree
    dg=bpy.context.evaluated_depsgraph_get()
    trees=[]
    for o in SOLIDS:
        evaluated=o.evaluated_get(dg);mesh=evaluated.to_mesh()
        tree=BVHTree.FromPolygons([o.matrix_world@v.co for v in mesh.vertices],[p.vertices[:] for p in mesh.polygons])
        trees.append((o.name,tree,bool(o.get('subject'))));evaluated.to_mesh_clear()
    minimum=1e9;nearest=None;violations=[];maximum_step=0
    for route,items in paths.items():
        for i in range((len(items)-1)*4+1):
            k=min(i//4,len(items)-2);f=(i-k*4)/4
            p=Vector(items[k]['position']).lerp(Vector(items[k+1]['position']),f)
            for name,tree,_ in trees:
                hit=tree.find_nearest(p)
                if hit and hit[0] is not None:
                    dist=hit[3]
                    # BVH object-space queries and edge-normal signs give false collisions.
                    # Use world triangles and ray parity for actual volume containment.
                    inside=False
                    if (p-hit[0]).dot(hit[1]) < -1e-5:
                        ray=Vector((.937,.241,.251)).normalized();origin=p.copy();crossings=0
                        for _ in range(32):
                            rh=tree.ray_cast(origin,ray)
                            if rh[0] is None:break
                            crossings+=1;origin=rh[0]+ray*.0001
                        inside=bool(crossings%2)
                    if dist<minimum:minimum=dist;nearest=(route,i/4+1,name)
                    if dist<.2 or inside:violations.append((route,i/4+1,name,round(dist,4),inside))
        maximum_step=max(maximum_step,max((Vector(a['position'])-Vector(b['position'])).length for a,b in zip(items,items[1:])))
    occlusions=[];framing=[]
    subject_vertices=points(SUBJECT)
    center=sum(subject_vertices,Vector())/len(subject_vertices)
    for route,items in paths.items():
        for item in items:
            set_pose(cam,item);p=cam.location;ray=(center-p).normalized();distance=(center-p).length
            for name,tree,subject in trees:
                if subject:continue
                hit=tree.ray_cast(p,ray,distance-.05)
                if hit[0] is not None:occlusions.append((route,item['frame'],name))
            b=bbox(cam,subject_vertices)
            # Portrait push may crop legs, but never the head; other studies keep the subject envelope.
            if b[1]<.015 or (not route.startswith('portrait') and (b[0]<.015 or b[2]>.985 or b[3]>.985)):
                framing.append((route,item['frame'],b))
    return dict(camera_radius_m=.2,samples_per_route=573,method='world-space triangle BVH nearest surface + ray-parity containment, including hollow rings',
                minimum_distance_m=minimum,nearest=nearest,violations=violations,max_frame_translation_m=maximum_step,
                center_ray_occlusions=occlusions,framing_violations=framing,
                exclusions='support floor/contact surfaces; static subject components intentionally overlap at joints; no moving subject in this version')

def style(name,wire):
    for o in bpy.context.scene.objects:
        if o.type!='MESH':continue
        if o==wire:o.hide_render=name!='wire';continue
        if o.name=='CoarseSubjectEnvelope':o.hide_render=name!='coarse';continue
        o.hide_render=name=='wire'
        subject=o.get('subject',False);cat=o.get('category','prop')
        if name=='coarse' and subject and cat=='vehicle':o.hide_render=True
        if name=='coarse':rgb=PALETTE[cat]
        elif subject:rgb=(240,38,20)
        elif name=='subject_bg':rgb=(42,62,92)
        elif name=='depth':rgb={'road':(192,207,228),'terrain':(137,158,186),'infrastructure':(101,125,160)}.get(cat,(38,58,92))
        else:rgb={'road':(180,190,204),'terrain':(124,139,162),'infrastructure':(113,140,174),'building':(58,82,116),'prop':(148,161,182)}.get(cat,(95,131,105))
        if name=='fine' and o.name=='Viaduct':rgb=(38,52,72)
        o.data.materials.clear();o.data.materials.append(mat(name+cat+str(subject),rgb))
    w=bpy.context.scene.world;w.use_nodes=True
    rgb=PALETTE['sky'] if name=='coarse' else ((42,62,92) if name=='subject_bg' else (7,12,24))
    w.node_tree.nodes['Background'].inputs[0].default_value=tuple((v/255/12.92 if v/255<=.04045 else ((v/255+.055)/1.055)**2.4) for v in rgb)+(1,)
    w.node_tree.nodes['Background'].inputs[1].default_value=1

def render(scene,path):
    path.parent.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)

def build(case,full):
    SOLIDS.clear();SUBJECT.clear();MATS.clear();bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=672;scene.render.resolution_y=384;scene.render.resolution_percentage=100
    scene.render.fps=24;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.world=bpy.data.worlds.new('FlatWorld')
    cfg=scene_geometry(case);bpy.context.view_layer.update()
    bpy.ops.object.camera_add(location=cfg['camera']);cam=bpy.context.object;cam.rotation_mode='QUATERNION'
    cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';cam.data.lens=35;cam.data.clip_end=1000
    scene.camera=cam;look(cam,cfg['target']);bpy.context.view_layer.update()
    fit=calibrate(cam,cfg);align_background(cam,case);paths=poses(cam,cfg,case)
    verts=points(SUBJECT);mn=Vector(tuple(min(v[k] for v in verts) for k in range(3)));mx=Vector(tuple(max(v[k] for v in verts) for k in range(3)))
    wire=box('L1Wire_Cuboid',(mn+mx)/2,mx-mn,'vehicle',collider=False)
    mod=wire.modifiers.new('OneThinCuboid','WIREFRAME');mod.thickness=.018 if case.startswith(('02','05','06')) else .035;mod.use_replace=True
    wire.data.materials.clear();wire.data.materials.append(mat('WireRed',(245,20,12)))
    if case.startswith(('01','06')):
        envelope=box('CoarseSubjectEnvelope',(mn+mx)/2,mx-mn,'vehicle',collider=False)
    out=CASES/case/'assets/scene'/VERSION;out.mkdir(parents=True,exist_ok=True)
    dump(out/'opening-fit.json',fit);dump(out/'camera-paths.json',dict(fps=24,frames=144,shared_frames=24,routes=paths))
    validation=validate(cam,paths);dump(out/'collision-validation.json',validation)
    # Render diagnostic frames even on a failed clearance check, to make repairs inspectable.
    for name,items in paths.items():
        for frame in (1,24,48,72,96,120,144):
            set_pose(cam,items[frame-1]);style('fine',wire);render(scene,out/'review'/f'{name}-{frame:03}.png')
    set_pose(cam,list(paths.values())[0][0])
    for s in STYLES:style(s,wire);render(scene,out/f'aligned-{s}.png')
    style('fine',wire);bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    if full:
        assert not validation['violations'],validation['violations'][:8]
        assert not validation['center_ray_occlusions'],validation['center_ray_occlusions'][:8]
        assert not validation['framing_violations'],validation['framing_violations'][:8]
        for name,items in paths.items():
            for s in STYLES:
                style(s,wire)
                for p in items:
                    set_pose(cam,p);render(scene,CASES/case/'assets/proxy'/VERSION/s/f'.{name}_frames'/f"frame_{p['frame']:04}.png")
    dump(out/'build-receipt.json',dict(case=case,version=VERSION,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         reference_sha256=hashlib.sha256((CASES/case/'assets/first-frame/natural-first-20261108-v1.png').read_bytes()).hexdigest(),
         camera_validation_pass=not any(validation[k] for k in ('violations','center_ray_occlusions','framing_violations')),rendered=full,engine=scene.render.engine,
         styles=list(STYLES),limitations='manual coarse inverse scene; new hidden views are authored, not reconstruction; all subjects static; depth is categorical layers, not metric depth'))
    print('CASE_DONE',case,'violations',len(validation['violations']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=IDS);p.add_argument('--render',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    for c in ([args.case] if args.case else IDS):build(c,args.render)
