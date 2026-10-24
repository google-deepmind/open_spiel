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

"""Serving: answer one decision from an artifact as in-graph evaluation would.

On CPU the reply is bit-identical to the Arena's for tabular, uniform and
host-computed policies; for every other JAX policy (networks, search over a
network) the action is the same, the logits agree within 1e-6 absolute plus
1e-6 relative and the distribution correspondingly (core.types.Policy). The
tests establish this on XLA:CPU only; on GPU or TPU, or against an Arena batch
shape the tests do not cover, the reply can differ from the Arena's by
rounding.

  reply = PolicyServer().act(pyspiel_state, checkpoint_dir=ckpt_dir, step=step)
  reply.action, reply.probs, reply.decider

  server     PolicyServer and Reply: pyspiel state in, one move out.
  host_agent HostAgent: one seat's decision over a replayed history.
  replay     a recorded match to the SeatSteps one seat saw.
  wire       pyspiel histories to raw env histories, and the node check.
  games      which games are served and the env that plays given parameters.
"""

# Re-exported as part of the package API.
from open_spiel.python.scale.serve.server import PolicyServer  # noqa: F401

# Re-exported as part of the package API.
from open_spiel.python.scale.serve.server import Reply  # noqa: F401
