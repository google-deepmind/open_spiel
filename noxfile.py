# Lint as: python3
"""An integration test building and testing open_spiel wheel."""
import os
import sys
import sysconfig

import nox


def get_distutils_tempdir():
  return (
      f"temp.{sysconfig.get_platform()}-{sys.version_info[0]}.{sys.version_info[1]}"
  )


@nox.session(python="3")
def tests(session):
  session.install("-r", "requirements.txt")
  child_env = os.environ.copy()
  child_env["OPEN_SPIEL_BUILD_ALL"] = "ON"
  session.run("python3", "setup.py", "build", env=child_env)
  session.run("python3", "setup.py", "install", env=child_env)
  session.cd(os.path.join("build", get_distutils_tempdir()))
  session.run(
      "ctest", f"-j{4*os.cpu_count()}", "--output-on-failure", external=True)
