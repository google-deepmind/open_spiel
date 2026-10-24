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

"""Running training somewhere, surviving preemptions.

backend.py        the Backend interface (storage, ordered fallback
provisioning, launch, liveness and preemption, teardown)
and RunSpec, one run's identity and trainer arguments
storage.py        Storage: LocalStorage and GcsStorage
supervisor.py     one run, many attempts; its state lives in the run's own
storage; a preempted attempt is relaunched and the
trainer resumes from its checkpoint
local.py          the local backend (subprocesses on this machine)
gcp.py            the reference GCP backend (Compute Engine VMs through
gcloud, configured by one YAML file)
config.py         the backend config file and the backend registry
run.py            the command line
                  (`python -m open_spiel.python.scale.launch.run`)
cloud.py          the one-command Google Cloud launcher: defaults,
preflight checks with their fixes, --dry_run, --down
(`python -m open_spiel.python.scale.launch.cloud`; cloud_example.yaml)
preflight.py      before any machine exists: refuses a run whose game
needs a game package the launch does not ship
preemption.py     the trainer's side: the preemption notice and its sources
hardware.py       the hardware a run got
compile_cache.py  the persistent XLA compilation cache
subprocess_env.py the package import root and the one interpreter setting;
                  a child Python process that finds the package
"""
