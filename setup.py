import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def get_readme(file):
    with open(file) as f:
        return f.read()


class CMakeExtension(Extension):
    """An extension with no sources.

    We do not want distutils to handle any of the compilation (instead we rely on 
    CMake), so we always pass an empty list to the constructor.
    """

    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildExt(build_ext):
    """Our custom build_ext command.

    Uses CMake to build extensions instead of a bare compiler (e.g. gcc, clang).
    """

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


def get_requirements(requirements_file):
    """Parse the requirements.txt file and produce a list of dependencies for setup()

    Currently a requirements.txt is being used to specify dependencies. In order to
    avoid specifying it in two places, we're going to use that file as the source
    of truth.
    """
    with open(requirements_file) as f:
        return list(filter(lambda s: len(s) > 0, (parse_requirement(l) for l in f)))


def parse_requirement(s):
    """Parse a line of a requirements.txt file"""
    requirement, *_ = s.split("#")
    return requirement.strip()


setup(
    name="pyspiel",
    version="0.0.1rc2",
    author="Marc Lanctot",
    author_email="lanctot@google.com",
    description="A Framework for Reinforcement Learning in Games",
    long_description=get_readme("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/open_spiel",
    install_requires=get_requirements("requirements.txt"),
    ext_modules=[CMakeExtension("pyspiel", sourcedir="open_spiel")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
