import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request

from bs4 import BeautifulSoup
from distutils.command.clean import clean
from pathlib import Path
from typing import NamedTuple

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from dataclasses import dataclass

from distutils.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel


@dataclass
class Backend:
    name: str
    package_data: dict
    src_dir: str
    backend_dir: str
    install_dir: str
    is_external: bool


class BackendInstaller:

    @staticmethod
    def prepare(backend_name: str, backend_src_dir: str = None, is_external: bool = False):
        # Initialize submodule if there is one for in-tree backends.
        if not is_external:
            root_dir = os.path.join(os.pardir, "third_party")
            assert backend_name in os.listdir(
                root_dir), f"{backend_name} is requested for install but not present in {root_dir}"

            try:
                subprocess.run(["git", "submodule", "update", "--init", f"{backend_name}"], check=True,
                               stdout=subprocess.DEVNULL, cwd=root_dir)
            except subprocess.CalledProcessError:
                pass
            except FileNotFoundError:
                pass

            backend_src_dir = os.path.join(root_dir, backend_name)

        backend_path = os.path.abspath(os.path.join(backend_src_dir, "backend"))
        assert os.path.exists(backend_path), f"{backend_path} does not exist!"

        for file in ["compiler.py", "driver.py"]:
            assert os.path.exists(os.path.join(backend_path, file)), f"${file} does not exist in ${backend_path}"

        install_dir = os.path.join(os.path.dirname(__file__), "triton", "backends", backend_name)
        package_data = [f"{os.path.relpath(p, backend_path)}/*" for p, _, _, in os.walk(backend_path)]
        return Backend(name=backend_name, package_data=package_data, src_dir=backend_src_dir, backend_dir=backend_path,
                       install_dir=install_dir, is_external=is_external)

    # Copy all in-tree backends under triton/third_party.
    @staticmethod
    def copy(active):
        return [BackendInstaller.prepare(backend) for backend in active]

    # Copy all external plugins provided by the `TRITON_PLUGIN_DIRS` env var.
    # TRITON_PLUGIN_DIRS is a semicolon-separated list of paths to the plugins.
    # Expect to find the name of the backend under dir/backend/name.conf
    @staticmethod
    def copy_externals():
        backend_dirs = os.getenv("TRITON_PLUGIN_DIRS")
        if backend_dirs is None:
            return []
        backend_dirs = backend_dirs.strip().split(";")
        backend_names = [Path(os.path.join(dir, "backend", "name.conf")).read_text().strip() for dir in backend_dirs]
        return [
            BackendInstaller.prepare(backend_name, backend_src_dir=backend_src_dir, is_external=True)
            for backend_name, backend_src_dir in zip(backend_names, backend_dirs)
        ]


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    elif check_env_flag("TRITON_REL_BUILD_WITH_ASSERTS"):
        return "TritonRelBuildWithAsserts"
    else:
        # TODO: change to release when stable enough
        return "TritonRelBuildWithAsserts"


# --- third party packages -----


class Package(NamedTuple):
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str


# pybind11


def get_pybind11_package_info():
    name = "pybind11-2.11.1"
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz"
    return Package("pybind11", name, url, "PYBIND11_INCLUDE_DIR", "", "PYBIND11_SYSPATH")


# llvm


def get_llvm_package_info():
    # added statement for Apple Silicon
    system = platform.system()
    arch = platform.machine()
    if arch == 'aarch64':
        arch = 'arm64'
    if system == "Darwin":
        arch = platform.machine()
        if arch == "x86_64":
            arch = "x64"
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == 'arm64':
            system_suffix = 'ubuntu-arm64'
        else:
            vglibc = tuple(map(int, platform.libc_ver()[1].split('.')))
            vglibc = vglibc[0] * 100 + vglibc[1]
            system_suffix = 'ubuntu-x64' if vglibc > 217 else 'centos-x64'
    else:
        return Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_file = open("../cmake/llvm-hash.txt", "r")
    rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    url = f"https://tritonlang.blob.core.windows.net/llvm-builds/{name}.tar.gz"
    return Package("llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")


def open_url(url):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers = {
        'User-Agent': user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forver
    return urllib.request.urlopen(request, timeout=300)


# ---- package data ---


def get_triton_cache_path():
    user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or \
            os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def get_thirdparty_packages():
    triton_cache_path = get_triton_cache_path()
    packages = [get_pybind11_package_info(), get_llvm_package_info()]
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if p.syspath_var_name in os.environ:
            package_dir = os.environ[p.syspath_var_name]
        version_file_path = os.path.join(package_dir, "version.txt")
        if p.syspath_var_name not in os.environ and\
           (not os.path.exists(version_file_path) or Path(version_file_path).read_text() != p.url):
            try:
                shutil.rmtree(package_root_dir)
            except Exception:
                pass
            os.makedirs(package_root_dir, exist_ok=True)
            print(f'downloading and extracting {p.url} ...')
            file = tarfile.open(fileobj=open_url(p.url), mode="r|*")
            file.extractall(path=package_root_dir)
            # write version url to package_dir
            with open(os.path.join(package_dir, "version.txt"), "w") as f:
                f.write(p.url)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args


def download_and_copy(src_path, variable, version, url_func):
    triton_cache_path = get_triton_cache_path()
    if variable in os.environ:
        return
    base_dir = os.path.dirname(__file__)
    system = platform.system()
    arch = {"x86_64": "64", "arm64": "aarch64", "aarch64": "aarch64"}[platform.machine()]
    url = url_func(arch, version)
    tmp_path = os.path.join(triton_cache_path, "nvidia")  # path to cache the download
    dst_path = os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", src_path)  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if os.path.exists(dst_path) and system == "Linux":
        curr_version = subprocess.check_output([dst_path, "--version"]).decode("utf-8").strip()
        curr_version = re.search(r"V([.|\d]+)", curr_version).group(1)
        download = download or curr_version != version
    if download:
        print(f'downloading and extracting {url} ...')
        file = tarfile.open(fileobj=open_url(url), mode="r|*")
        file.extractall(path=tmp_path)
    print(f'copy {src_path} to {dst_path} ...')
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    shutil.copy(src_path, dst_path)


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        version_line = next(line for line in output.split("\n") if "release" in line)
        cuda_version = version_line.split("release")[-1].strip().split(",")[0]
        if cuda_version.count(".") < 2:
            cuda_version+=".0"
        return cuda_version
    except Exception as e:
        raise RuntimeError("Failed to determine CUDA version") from e


def fetch_download_links(url):
    arch = {"x86_64": "64", "arm64": "aarch64", "aarch64": "aarch64"}[platform.machine()]
    response = open_url(url)
    if response is None:
        return []  # Return an empty list if there was an error opening the URL

    html = response.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a", href=True)

    download_url = ''
    for link in links:
        href = link['href']
        if '/download/' in href and arch in href:  # Filter for download links
            download_url = f"https://anaconda.org{href}"
            break

    if download_url == '':
        raise RuntimeError(f"Failed to find download link for {url}")
    version_pattern = re.compile(r"\d+\.\d+\.\d+")
    match = version_pattern.search(download_url)
    if match:
        version = match.group()
        return version
    else:
        raise RuntimeError(f"Failed to find version in {download_url}")


# ---- cmake extension ----


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "python" / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


class CMakeClean(clean):

    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = get_cmake_dir()


class CMakeBuildPy(build_py):

    def run(self) -> None:
        self.run_command('build_ext')
        return super().run()


class CMakeExtension(Extension):

    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + \
        [('base-dir=', None, 'base directory of Triton')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        match = re.search(r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode())
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        lit_dir = shutil.which('lit')
        ninja_dir = shutil.which('ninja')
        # lit is used by the test suite
        thirdparty_cmake_args = get_thirdparty_packages()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dir = sysconfig.get_path("platinclude")
        cmake_args = [
            "-G", "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DLLVM_ENABLE_WERROR=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir, "-DTRITON_BUILD_TUTORIALS=OFF",
            "-DTRITON_BUILD_PYTHON_MODULE=ON", "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON", "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
            "-DTRITON_CODEGEN_BACKENDS=" + ';'.join([b.name for b in backends if not b.is_external]),
            "-DTRITON_PLUGIN_DIRS=" + ';'.join([b.src_dir for b in backends if b.is_external])
        ]
        if lit_dir is not None:
            cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
        cmake_args.extend(thirdparty_cmake_args)

        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
            build_args += ['-j' + max_jobs]

        if check_env_flag("TRITON_BUILD_WITH_CLANG_LLD"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        # Note that asan doesn't work with binaries that use the GPU, so this is
        # only useful for tools like triton-opt that don't run code on the GPU.
        #
        # I tried and gave up getting msan to work.  It seems that libstdc++'s
        # std::string does not play nicely with clang's msan (I didn't try
        # gcc's).  I was unable to configure clang to ignore the error, and I
        # also wasn't able to get libc++ to work, but that doesn't mean it's
        # impossible. :)
        if check_env_flag("TRITON_BUILD_WITH_ASAN"):
            cmake_args += [
                "-DCMAKE_C_FLAGS=-fsanitize=address",
                "-DCMAKE_CXX_FLAGS=-fsanitize=address",
            ]

        if check_env_flag("TRITON_BUILD_WITH_CCACHE"):
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        env = os.environ.copy()
        cmake_dir = get_cmake_dir()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=cmake_dir)
        subprocess.check_call(["cmake", "--build", ".", "--target", "mlir-doc"], cwd=cmake_dir)


cuda_version = get_cuda_version()

ptxas_version = fetch_download_links(f"https://anaconda.org/nvidia/cuda-nvcc/files?channel=cuda-{cuda_version}")
cuobjdump_version = fetch_download_links(
    f"https://anaconda.org/nvidia/cuda-cuobjdump/files?channel=cuda-{cuda_version}")
nvdisasm_version = fetch_download_links(f"https://anaconda.org/nvidia/cuda-nvdisasm/files?channel=cuda-{cuda_version}")

download_and_copy(
    src_path="bin/ptxas",
    variable="TRITON_PTXAS_PATH",
    version=ptxas_version,
    url_func=lambda arch, version:
    f"https://anaconda.org/nvidia/cuda-nvcc/{version}/download/linux-{arch}/cuda-nvcc-{version}-0.tar.bz2",
)
download_and_copy(
    src_path="bin/cuobjdump",
    variable="TRITON_CUOBJDUMP_PATH",
    version=cuobjdump_version,
    url_func=lambda arch, version:
    f"https://anaconda.org/nvidia/cuda-cuobjdump/{version}/download/linux-{arch}/cuda-cuobjdump-{version}-0.tar.bz2",
)
download_and_copy(
    src_path="bin/nvdisasm",
    variable="TRITON_NVDISASM_PATH",
    version=nvdisasm_version,
    url_func=lambda arch, version:
    f"https://anaconda.org/nvidia/cuda-nvdisasm/{version}/download/linux-{arch}/cuda-nvdisasm-{version}-0.tar.bz2",
)

backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()]


def add_link_to_backends():
    for backend in backends:
        if os.path.islink(backend.install_dir):
            os.unlink(backend.install_dir)
        if os.path.exists(backend.install_dir):
            shutil.rmtree(backend.install_dir)
        os.symlink(backend.backend_dir, backend.install_dir)


class plugin_install(install):

    def run(self):
        add_link_to_backends()
        install.run(self)


class plugin_develop(develop):

    def run(self):
        add_link_to_backends()
        develop.run(self)


class plugin_bdist_wheel(bdist_wheel):

    def run(self):
        add_link_to_backends()
        bdist_wheel.run(self)


class plugin_egginfo(egg_info):

    def run(self):
        add_link_to_backends()
        egg_info.run(self)


package_data = dict()
package_data["triton/tools"] = ["compile.h", "compile.c"]
package_data.update({f"triton/backends/{b.name}": b.package_data for b in backends})

setup(
    name=os.environ.get("TRITON_WHEEL_NAME", "triton"),
    version="3.0.0" + os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""),
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    packages=[
        "triton",
        "triton/_C",
        "triton/compiler",
        "triton/language",
        "triton/language/extra",
        "triton/language/extra/cuda",
        "triton/ops",
        "triton/ops/blocksparse",
        "triton/runtime",
        "triton/backends",
        "triton/tools",
    ] + [f'triton/backends/{backend.name}' for backend in backends],
    install_requires=["filelock"],
    package_data=package_data,
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/_C/")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
        "clean": CMakeClean,
        "install": plugin_install,
        "develop": plugin_develop,
        "bdist_wheel": plugin_bdist_wheel,
        "egg_info": plugin_egginfo,
    },
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://github.com/openai/triton/",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    test_suite="tests",
    extras_require={
        "build": ["cmake>=3.20", "lit", "bs4"],
        "tests": [
            "autopep8",
            "flake8",
            "isort",
            "numpy",
            "pytest",
            "scipy>=1.7.1",
        ],
        "tutorials": [
            "matplotlib",
            "pandas",
            "tabulate",
        ],
    },
)
