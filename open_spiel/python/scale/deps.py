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

"""The optional packages the framework can use but does not install.

GAMES_PACKAGE is the import name of OpenSpiel's pure-JAX games package
(`open_spiel/python/jax/games`), which every game adapter but the built-in
`kuhn_poker_jax` runs on. Code that names the package as a string (the
optional-dependency checks, the test setup) reads it from here; adapters
import the package directly.
"""

# The games package sits under the framework's import root ("" when the
# framework is top-level or this file is loaded by path, as conftest does).
# The two names below are the one place the framework spells a package name
# as a string: the games package is optional and imports JAX, so the
# JAX-free control plane cannot import it to read its __name__, and the
# framework's own name is the anchor that finds the shared import root when
# both are nested under another package.
_FRAMEWORK = "open_spiel.python.scale"
_HEAD, _FOUND, _ = f".{__name__}".partition(f".{_FRAMEWORK}.")
_ROOT = _HEAD[1:] + "." if _FOUND and _HEAD else ""
GAMES_PACKAGE = _ROOT + "open_spiel.python.jax.games"
PYSPIEL = "pyspiel"
# Every optional package (the games package, pyspiel and the tensorboardX and
# wandb metric sinks): a test that needs an absent one is skipped
# (conftest.py), and a test that names them reads them here, never from
# conftest.py, whose import turns x64 on.
OPTIONAL = (GAMES_PACKAGE, PYSPIEL, "tensorboardX", "wandb")
