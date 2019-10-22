import os
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test


def get_readme(file):
    with open(file) as f:
        return f.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildExt(build_ext):
    def run(self):
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError as e:
            ext_names = ", ".join(e.name for e in self.extensions)
            raise RuntimeError(
                f"CMake must be installed to build the following extensions: {ext_names}"
            ) from e

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extension_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )
        cmake_args = [
            "-DPython_TARGET_VERSION=3.6",
            "-DCMAKE_CXX_COMPILER=g++",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
        ]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        env = os.environ.copy()
        subprocess.check_call(
            ["make", f"-j{os.cpu_count()}"], cwd=self.build_temp, env=env
        )


class CTestTestCommand(test):
    @staticmethod
    def _get_distutils_dirname(name):
        return f"{name}.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"

    def run_tests(self):
        env = os.environ.copy()
        subprocess.check_call(
            ["ctest", f"-j{4*os.cpu_count()}", "--output-on-failure"],
            cwd=os.path.join("build", self._get_distutils_dirname("temp")),
            env=env,
        )


setup(
    name="pyspiel",
    version="0.0.1rc2",
    author="Marc Lanctot",
    author_email="lanctot@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=get_readme("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=[
        "absl-py == 0.7.1",
        "tensorflow < 1.15.0, >= 1.14.0",
        "dm-sonnet == 1.32",
        "IPython == 5.8.0",
        "tensorflow-probability < 0.8.0, >= 0.7.0",
        "cvxopt == 1.2.3",
        "networkx == 2.2",
        "mock == 3.0.5",
        "matplotlib == 3.1.1",
        "nashpy == 0.0.19",
        "scipy == 1.1.0",
        "attrs == 19.1.0",
    ],
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt, "test": CTestTestCommand},
    zip_safe=False,
)
