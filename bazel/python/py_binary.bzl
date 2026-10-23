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

"""`py_binary` for OpenSpiel BUILD files: rules_python's, with adapted attributes."""

load("@rules_python//python:py_binary.bzl", _py_binary = "py_binary")
load(":py_common.bzl", "adapt_py_executable_kwargs")

def py_binary(**kwargs):
    """rules_python `py_binary`; see `adapt_py_executable_kwargs`."""
    _py_binary(**adapt_py_executable_kwargs(kwargs))
