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

"""Guards against caching JAX-array builders."""

import ast
import pathlib

from absl.testing import absltest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def _is_cache_decorator(decorator):
  """Whether a decorator is functools.cache or lru_cache.

  Args:
    decorator: The decorator node.

  Returns:
    True for a cache decorator.
  """
  if isinstance(decorator, ast.Name):
    return decorator.id == "lru_cache"
  if isinstance(decorator, ast.Attribute):
    return decorator.attr in ("cache", "lru_cache")
  if isinstance(decorator, ast.Call):
    return _is_cache_decorator(decorator.func)
  return False


def _contains_jnp_reference(node):
  """Whether `node` references jnp anywhere.

  Args:
    node: An AST node.

  Returns:
    True when a jnp attribute appears under it.
  """
  for child in ast.walk(node):
    if (
        isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "jnp"
    ):
      return True
  return False


class CachedJaxArrayLintTest(absltest.TestCase):
  """Cached functions build numpy arrays, never jnp ones."""

  def test_cached_functions_do_not_use_jnp(self):
    """No cached function in the package calls jnp."""
    violations = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
      tree = ast.parse(path.read_text(), filename=str(path))
      for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
          continue
        if not any(
            _is_cache_decorator(decorator) for decorator in node.decorator_list
        ):
          continue
        if _contains_jnp_reference(node):
          relpath = path.relative_to(_PACKAGE_ROOT)
          violations.append(f"{relpath}:{node.lineno} {node.name}")

    self.assertEmpty(
        violations,
        msg=(
            "Cached constant-builders must return NumPy arrays, not JAX arrays."
            " JAX bakes NumPy arrays into XLA constants when traced; caching"
            " JAX arrays can leak tracers. Offending functions: "
            + ", ".join(violations)
        ),
    )


if __name__ == "__main__":
  absltest.main()
