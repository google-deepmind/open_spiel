import os
import sys
import sysconfig

import nox


def get_distutils_tempdir():
    return (
        f"temp.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"
    )


@nox.session
def tests(session):
    session.install("-r", "requirements.txt")
    session.run("python", "setup.py", "build")
    session.run("python", "setup.py", "install")
    session.cd(os.path.join("build", get_distutils_tempdir()))
    session.run(
        "ctest", f"-j{4*os.cpu_count()}", "--output-on-failure", external=True
    )
