import functools
import hashlib
import importlib
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import setuptools
import torch

import triton
import triton.language as tl
from triton.common.backend import (BaseBackend, compute_core_version_key, register_backend)
from triton.common.build import quiet
from triton.compiler.make_launcher import make_so_cache_key
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import DriverBase


def build_for_backend(name, src, srcdir):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cflags = os.environ.get("CFLAGS", "")

    ret = subprocess.check_call([cc, src, f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-o", so, cflags])
    if ret == 0:
        return so
    # fallback on setuptools
    # Currently, this should never happen, since check_call raises an exception when the called process returns a non-zero exit status.
    extra_compile_args = []
    library_dirs = []
    include_dirs = [srcdir]
    libraries = []
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3'],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so


class ExtensionUtils:

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExtensionUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "extension_backend.c")).read_text()
        key = hashlib.sha256(src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        fname = "ext_utils.so"
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_for_backend("ext_utils", src_path, tmpdir)
                with open(so, "rb") as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location("ext_utils", cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


class ExtensionDriver(DriverBase):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExtensionDriver, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.utils = ExtensionUtils()


class ExtensionBackend(BaseBackend):
    stub_so_path = ""

    def __init__(self, device_type: str) -> None:
        super(ExtensionBackend, self).__init__(device_type)
        self.driver = ExtensionDriver()
        self.version_key = None

    def add_stages(self, arch, extern_libs, stages):
        filter_in_stages = ["ast", "ttir", "ttgir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        metadata["name"] = "extension_backend_name"

    def get_driver(self):
        return self.driver

    def get_stream(self):
        return ""

    @functools.lru_cache(None)
    def get_device_properties(self, device):
        return self.driver.utils.get_device_properties()

    def get_current_device(self):
        return torch.device("cpu")

    def set_current_device(self, device):
        pass

    def get_load_binary_fn(self):
        return self.driver.utils.load_binary

    def get_kernel_bin(self):
        return "ttgir"

    def get_architecture_descriptor(self, **kwargs):
        return ""

    def get_version_key(self):
        if self.version_key is None:
            self.version_key = compute_core_version_key()
        return self.version_key

    def make_launcher_stub(self, name, signature, constants):
        # name of files that are cached
        so_cache_key = make_so_cache_key(self.get_version_key(), signature, constants)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = self._generate_launcher(constants, signature)
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_for_backend(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    so_path = so_cache_manager.put(f.read(), so_name, binary=True)
                    type(self).stub_so_path = so_path
                    return so_path
        else:
            type(self).stub_so_path = cache_path
            return cache_path

    def _generate_launcher(self, constants, signature):
        # generate glue code
        src = """
        #define __EXTENSION_BACKEND__
        #include <Python.h>
        #include <stdio.h>

        static PyObject* launch_counter(PyObject* self, PyObject* args) {
        static int64_t launch_counter = 0;
        launch_counter += 1;
        return PyLong_FromLong(launch_counter);
        }

        static PyObject* launch(PyObject* self, PyObject* args) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        launch_counter(self, args);
        // return None
        Py_INCREF(Py_None);
        return Py_None;
        }

        static PyMethodDef ModuleMethods[] = {
        {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
        {"launch_counter", launch_counter, METH_VARARGS, "Entry point to get launch counter"},
        {NULL, NULL, 0, NULL} // sentinel
        };

        static struct PyModuleDef ModuleDef = {
        PyModuleDef_HEAD_INIT,
        \"__triton_launcher\",
        NULL, //documentation
        -1, //size
        ModuleMethods
        };

        PyMODINIT_FUNC PyInit___triton_launcher(void) {
        PyObject *m = PyModule_Create(&ModuleDef);
        if(m == NULL) {
            return NULL;
        }
        PyModule_AddFunctions(m, ModuleMethods);
        return m;
        }
        """

        return src


def test_dummy_backend():
    register_backend("cpu", ExtensionBackend)

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    inp = torch.randn(10)
    out = torch.randn(10)
    kernel[(10, )](inp, out, 10, XBLOCK=16)
    spec = importlib.util.spec_from_file_location("__triton_launcher", ExtensionBackend.stub_so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    launch_counter = getattr(mod, "launch_counter")

    for _ in range(100):
        kernel[(10, )](inp, out, 10, XBLOCK=16)

    assert launch_counter() > 0
