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

"""Plugins: modules whose import registers extensions into the library.

The library is extended without editing it: a deployment writes a module that
calls the registration functions below, and names it in the environment
variable ENV ($OPEN_SPIEL_SCALE_PLUGINS, comma-separated) or passes it to
`load`.
Every entry point (the trainer `train`, the launch supervisor, the metrics
writer, the config loader) calls `load_env()`, so every process of a run gets
the same extensions; a plugin that fails to import raises.

  # acme_plugin.py
  from open_spiel.python.scale import metrics
  metrics.register_sink("acme", lambda ctx: AcmeSink(ctx.logdir))

  export OPEN_SPIEL_SCALE_PLUGINS=acme_plugin
  python -m open_spiel.python.scale.train --config=... --metric_sinks=acme

Every environment variable the library reads starts with ENV_PREFIX.

Extension points, by package (tests/extension_points_test checks that the list
is complete):

  (root)           registry.register_path_resolver; plugins.register_flags
  metrics          register_sink, register_auto_sink, register_params_publisher,
                   register_series_emitter, register_tensorboard_writer
  league.config_common  register_config_package, register_config_resolver,
                   register_hash_transform
  core             frameworks.register_framework, auxiliary.register_aux_key,
                   decide.set_scope
  envs             registry.register (a game), registry.set_rules_sha_tag,
                   gin.set_default_engine
  encoders         registry.register, register_module, register_family
  networks         catalog.register, actor_critic.register_torso,
                   actor_critic.register_policy_head
  policies         registry.register, registry.register_lazy,
                   registry.register_kwargs_translator
  belief           registry.register, registry.register_lazy
  learners         pg.register_loss
  sources          deals.register_canon
  league           kinds.register_kind, runner.register_controller,
                   runner.register_on_commit, runner.register_reference_eval,
                   refs.register_policy_format, population.register_reader,
                   pool_params.register, commit.register_layout_reader,
                   loop.register_state_translation,
                   slot_factories.register, slot_factories.register_lazy
  ckpt             formats.register_format, formats.set_default_writer,
                   codecs.register_params_codec, codecs.register_state_codec,
                   artifact.register_meta_extension,
                   records.register_field_name,
                   run_state.register_record_migration,
                   snapshot.register_leaf_codec,
                   provenance.set_attempt_id_env, provenance.set_code_repo,
                   provenance.register_provenance_extra
  eval             registry.register
  interop          host.register_array_backend
  launch           config.register_backend, preemption.register_notice_source,
                   preemption.set_default_sources,
                   hardware.register_machine_info,
                   hardware.register_accelerator_namer,
                   compile_cache.register_shared_dir
  parallel         collectives_check.register_preflight
  serve            controller.register_checkpoint_resolver,
                   controller.register_into
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Iterable, Mapping, Optional

from absl import logging

#: The prefix of every environment variable the library reads.
ENV_PREFIX = "OPEN_SPIEL_SCALE_"
#: The plugin modules every entry point loads (comma-separated).
ENV = ENV_PREFIX + "PLUGINS"
_LOADED: list[str] = []
_FLAG_EXTENSIONS: list[Callable[[Any], None]] = []


def load(names: Iterable[str]) -> tuple[str, ...]:
  """Imports each named plugin module once, in order.

  Args:
    names: The plugin module names; blank names are skipped.

  Returns:
    Every plugin loaded in this process so far.

  Raises:
    ValueError: a name is not a dotted module name.
    ImportError: a plugin fails to import.
  """
  for name in names:
    name = name.strip()
    if not name:
      continue
    if name in _LOADED:
      continue
    if not all(part.isidentifier() for part in name.split(".")):
      raise ValueError(f"plugin {name!r} is not a dotted module name")
    importlib.import_module(name)
    _LOADED.append(name)
    logging.info("plugin loaded: %s", name)
  return tuple(_LOADED)


def load_env(environ: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
  """`load` of the modules named in $OPEN_SPIEL_SCALE_PLUGINS.

  Args:
    environ: The environment to read ENV (comma-separated) from; defaults to
      os.environ.

  Returns:
    Every plugin loaded in this process so far.
  """
  env = os.environ if environ is None else environ
  return load(env.get(ENV, "").split(","))


def loaded() -> tuple[str, ...]:
  """The plugins loaded in this process so far, in load order.

  Returns:
    The plugin module names.
  """
  return tuple(_LOADED)


def register_flags(fn: Callable[[Any], None]) -> None:
  """Adds fn(flag_values), which defines extra flags on the entry points.

  The flags (aliases included) go on the entry points that parse flags. The
  trainer (`train`) calls it at import, so only a plugin loaded through ENV
  adds flags there.

  Args:
    fn: Called with the entry point's flag values; defines its flags.
      Registering the same function twice has no effect.

  Raises:
    TypeError: fn is not callable.
  """
  if not callable(fn):
    raise TypeError(f"{fn!r} is not callable")
  if fn not in _FLAG_EXTENSIONS:
    _FLAG_EXTENSIONS.append(fn)


def flag_extensions() -> tuple[Callable[[Any], None], ...]:
  """The registered flag extensions (`register_flags`), in order.

  Returns:
    The functions.
  """
  return tuple(_FLAG_EXTENSIONS)
