"""Widen both mechanical routes smoothly so the single wire box also fits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location('natural_five',HERE.with_name('natural_five.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
directory=m.CASES/'04-mechanical-ad/assets/scene'/m.VERSION
before=directory/'camera-paths-before-wire-fix.json'
if not before.exists():
    shutil.copy2(directory/'camera-paths.json',before)
    shutil.copy2(directory/'all-styles-filmstrip.jpg',directory/'before-wire-fix.jpg')
original=m.poses
def widened(cam,cfg,case):
    paths=original(cam,cfg,case)
    for items in paths.values():
        for p in items:p['lens']*=1-.25*m.ease(max(0,(p['frame']-24)/120))
    return paths
m.poses=widened
m.build('04-mechanical-ad','--render' in sys.argv)
(directory/'camera-repair.json').write_text(json.dumps(dict(
    base_source_sha256=hashlib.sha256(HERE.with_name('natural_five.py').read_bytes()).hexdigest(),
    repair_source_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
    reason='independent reviewer found wire box clipping despite detailed mesh being in frame',
    effect='same camera for all five styles; focal multiplier 1 -> 0.75 via smooth quintic after shared opening'),indent=2),encoding='utf8')
