# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type-checked Python rules for OpenSpiel BUILD files.

Static type checking only runs in Google's own build; here the rules are
plain rules_python targets.
"""

load(":py_binary.bzl", "py_binary")
load(":py_library.bzl", "py_library")
load(":py_test.bzl", "py_test")

def pytype_strict_library(**kwargs):
    """A `py_library`."""
    py_library(**kwargs)

def pytype_strict_binary(**kwargs):
    """A `py_binary`."""
    py_binary(**kwargs)

def pytype_strict_contrib_test(**kwargs):
    """A `py_test`."""
    py_test(**kwargs)
