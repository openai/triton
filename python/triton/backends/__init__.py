import os
import importlib.util
import inspect
from dataclasses import dataclass
from .driver import DriverBase
from .compiler import BaseBackend


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_concrete_subclasses(module, base_class):
    ret = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: BaseBackend = None
    driver: DriverBase = None


def _discover_backends_from_par():
    import sys
    import zipfile

    backends = {}
    par_path = sys.argv[0]
    with zipfile.ZipFile(par_path) as z:
        backends_dir = zipfile.Path(z, "triton/backends/")
        for path in backends_dir.iterdir():
            name = path.name
            if path.is_dir():
                compiler = importlib.import_module(f"triton.backends.{name}.compiler")
                driver = importlib.import_module(f"triton.backends.{name}.driver")
                backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                         _find_concrete_subclasses(driver, DriverBase))
    return backends


def _discover_backends():
    try:
        backends = dict()
        root = os.path.dirname(__file__)
        for name in os.listdir(root):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            if name.startswith('__'):
                continue
            compiler = _load_module(name, os.path.join(root, name, 'compiler.py'))
            driver = _load_module(name, os.path.join(root, name, 'driver.py'))
            backends[name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),
                                     _find_concrete_subclasses(driver, DriverBase))
        return backends
    except OSError:
        return _discover_backends_from_par()


backends = _discover_backends()
