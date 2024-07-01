import contextlib
import sys
import io
import sysconfig
import os
import shutil
import subprocess
import setuptools


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
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
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-o", so]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    ret = subprocess.check_call(cc_cmd)
    if ret == 0:
        return so
    # fallback on setuptools
    extra_compile_args = []
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
