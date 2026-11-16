"""Run a module from the vendored H3 inference tree without submodules."""
import argparse
import importlib.machinery
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SGLANG = ROOT.parents[1]
RAVEN = ROOT/'frameworks/raven'
OMNI = ROOT/'frameworks/sglang-omni'


def bootstrap():
    if not (RAVEN/'common/engine.py').is_file():
        raise RuntimeError('vendored RAVEN source is missing')
    if not (OMNI/'sglang_omni/__init__.py').is_file():
        raise RuntimeError('vendored SGLang-Omni source is missing')
    for path in (ROOT, RAVEN, OMNI, SGLANG/'python', SGLANG):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for name, path in (('projects',RAVEN/'projects'),('projects.minimax_h3',ROOT)):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
        sys.modules[name] = package
    sys.modules['projects'].minimax_h3 = sys.modules['projects.minimax_h3']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', help='Python module, e.g. projects.minimax_h3.tools.infer_ref_fixed_history')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    bootstrap()
    sys.argv = [args.module,*args.args]
    runpy.run_module(args.module,run_name='__main__',alter_sys=True)


if __name__ == '__main__':
    main()
