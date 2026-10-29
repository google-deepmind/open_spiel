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

"""Anonymous public reads: public Cloud Storage HTTPS and gs:// (anon).

Pins: a public HTTPS URL normalises and keeps its scheme; its artifact lists,
loads and verifies through a local cache exactly as the local folder does,
recording the URL as its path; a second load fetches nothing; a listing
pages; a corrupt object, a refused listing and a write are refused loudly;
gs:// without credentials reads through gcsfs with token="anon", and with
credentials through etils epath. Every request goes to a local server
(testing.public_bucket); no test touches the network.
"""

import os
import sys
import types as pytypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import public
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.testing import public_bucket

KUHN_INDEX = artifact.__name__.rpartition(".ckpt.")[0] + ".envs.kuhn:info_index"
BUCKET = "example-bucket"
RUN = "runs/kuhn-a"


def _write(folder, steps=(3, 7)):
  """Writes a Kuhn tabular artifact at each of `steps`.

  Args:
    folder: The artifact folder.
    steps: The steps.

  Returns:
    The params written at the last step.
  """
  pol = tabular.TabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  params = None
  for i, step in enumerate(steps):
    params = pol.params_from_logits(jnp.arange(24.0).reshape(12, 2) + float(i))
    artifact.write_artifact(
        str(folder),
        step,
        pol,
        params,
        rules_sha="r" * 8,
        config_hash="c" * 8,
        algo="test",
        env_name="kuhn_poker",
    )
  return params


@pytest.fixture(name="bucket")
def _bucket(tmp_path, monkeypatch):
  """A served public bucket holding runs/kuhn-a@{3,7} in BUCKET.

  Args:
    tmp_path: pytest's temporary folder.
    monkeypatch: pytest's monkeypatch.

  Yields:
    (server, the run's public URL, the params of step 7).
  """
  root = tmp_path / "served"
  params = _write(root / BUCKET / RUN)
  server = public_bucket.FakeBucketServer(str(root)).start()
  monkeypatch.setattr(public, "ENDPOINT", server.url)
  monkeypatch.setenv(public.CACHE_ENV, str(tmp_path / "cache"))
  try:
    yield server, f"{server.url}/{BUCKET}/{RUN}", params
  finally:
    server.stop()


def test_norm_path_keeps_a_public_url():
  """A public HTTPS URL normalises with its scheme; other schemes refused."""
  url = public.ENDPOINT
  assert url.startswith("https://")
  assert artifact.norm_path(f"{url}/b//runs/a/") == f"{url}/b/runs/a"
  assert public.split(f"{url}/b/runs/a") == ("b", "b/runs/a")
  assert public.split("gs://b/runs/a/") == ("b", "b/runs/a")
  for bad in (f"{url}/", f"{url}/b/../c", f"{url}/b/a?x=1"):
    with pytest.raises(ValueError):
      artifact.norm_path(bad)
  with pytest.raises(ValueError, match="scheme"):
    artifact.norm_path("https://example.com/b/runs/a")
  with pytest.raises(ValueError, match="scheme"):
    artifact.norm_path(url.replace("https://", "http://") + "/b/runs/a")


def test_public_url_loads_like_the_local_folder(bucket, tmp_path):
  """Steps, latest, exists and load through the cache match the local copy."""
  server, url, params = bucket
  local = str(tmp_path / "served" / BUCKET / RUN)
  assert artifact.steps(url) == artifact.steps(local) == (3, 7)
  assert artifact.latest_step(url) == 7
  assert artifact.exists(url) and artifact.exists(url, 3)
  assert not artifact.exists(url, 5)
  # listing and step discovery read only the steps' manifests
  assert all("/params" not in r for r in server.object_reads())
  art = artifact.load_artifact(url, 7)
  assert art.path == url
  assert art.manifest == artifact.load_manifest(local, 7)
  assert artifact.load_manifest(url, 7) == art.manifest
  for a, b in zip(jax.tree.leaves(art.params), jax.tree.leaves(params)):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
  cached = os.path.join(os.environ[public.CACHE_ENV], *url.split("/")[2:])
  assert not os.path.exists(cached)  # the cache keeps <bucket>/<path> only
  assert os.path.isdir(
      os.path.join(os.environ[public.CACHE_ENV], BUCKET, *RUN.split("/"), "7")
  )
  # a second load fetches nothing: every cached file has the listed md5
  before = len(server.object_reads())
  artifact.load_artifact(url, 7)
  assert len(server.object_reads()) == before


def test_listing_pages(bucket):
  """A listing longer than a page is read to its end."""
  server, url, _ = bucket
  server.page_size = 2
  assert artifact.latest_step(url) == 7
  assert artifact.load_artifact(url, 3).manifest.step == 3
  assert sum("pageToken" in r for r in server.requests) >= 2


def test_corrupt_object_refused(bucket):
  """An object that is not its listed size and md5 is refused, not cached."""
  server, url, _ = bucket
  server.corrupt.add(f"{RUN}/7/manifest.json")
  with pytest.raises(public.FetchError, match="not the listed object"):
    artifact.load_artifact(url, 7)
  root = os.path.join(os.environ[public.CACHE_ENV], BUCKET, *RUN.split("/"))
  assert not os.path.exists(os.path.join(root, "7", "manifest.json"))


def test_private_folder_refused_with_the_fix(bucket):
  """A refused listing raises, naming the status and what to check."""
  server, url, _ = bucket
  server.deny.add(RUN)
  with pytest.raises(public.FetchError, match="HTTP 403.*public"):
    artifact.latest_step(url)


def test_empty_public_folder_holds_no_step(bucket):
  """A public folder with no object holds no step."""
  server, _, _ = bucket
  url = f"{server.url}/{BUCKET}/runs/none"
  assert not artifact.steps(url) and not artifact.exists(url)
  with pytest.raises(FileNotFoundError):
    artifact.latest_step(url)


def test_public_url_is_read_only(bucket):
  """A write to a public URL raises before anything is written."""
  _, url, _ = bucket
  pol = tabular.TabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX)
  with pytest.raises(ValueError, match="read-only"):
    artifact.ckpt_root(url)
  with pytest.raises(ValueError, match="read-only"):
    artifact.write_artifact(
        url,
        9,
        pol,
        pol.params_from_logits(jnp.zeros((12, 2))),
        rules_sha="r" * 8,
        config_hash="c" * 8,
        algo="test",
        env_name="kuhn_poker",
    )


def _fake_gcsfs(monkeypatch, root, default_method):
  """Installs a gcsfs stand-in; returns the filesystems it makes.

  Args:
    monkeypatch: pytest's monkeypatch.
    root: the folder it serves (<root>/<bucket>/<object>).
    default_method: what a filesystem made without a token connects with.

  Returns:
    The list the made filesystems are appended to.
  """
  made = []

  def make(token=None):
    """A FakeGcsfs, recorded.

    Args:
      token: the token.

    Returns:
      The filesystem.
    """
    fs = public_bucket.FakeGcsfs(root, token, default_method)
    made.append(fs)
    return fs

  monkeypatch.setitem(
      sys.modules, "gcsfs", pytypes.SimpleNamespace(GCSFileSystem=make)
  )
  public.gcs_has_credentials.cache_clear()
  return made


def test_gs_without_credentials_reads_anonymously(tmp_path, monkeypatch):
  """No credentials: gs:// is read through gcsfs with token="anon"."""
  root = tmp_path / "served"
  params = _write(root / BUCKET / RUN)
  monkeypatch.setenv(public.CACHE_ENV, str(tmp_path / "cache"))
  made = _fake_gcsfs(monkeypatch, str(root), default_method="anon")
  try:
    assert not public.gcs_has_credentials()
    url = public.GCS_PREFIX + f"{BUCKET}/{RUN}"
    assert artifact.latest_step(url) == 7
    art = artifact.load_artifact(url + "/", 7)
    assert art.path == url
    for a, b in zip(jax.tree.leaves(art.params), jax.tree.leaves(params)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert [fs.token for fs in made][0] is None  # the credentials check
    assert {fs.token for fs in made[1:]} == {"anon"}
  finally:
    public.gcs_has_credentials.cache_clear()


def test_gs_with_credentials_is_read_as_before(tmp_path, monkeypatch):
  """Credentials: gs:// is returned as is (etils epath reads it)."""
  made = _fake_gcsfs(monkeypatch, str(tmp_path), default_method="cloud")
  try:
    assert public.gcs_has_credentials()
    assert public.readable("gs://b/runs/a", 7) == "gs://b/runs/a"
    assert [fs.token for fs in made] == [None]
  finally:
    public.gcs_has_credentials.cache_clear()
  assert (
      public.readable("gs://b/runs/a", has_credentials=lambda: True)
      == "gs://b/runs/a"
  )


def test_gs_without_gcsfs_is_read_as_before(monkeypatch):
  """Without gcsfs installed, gs:// is left to etils epath (which names it)."""
  monkeypatch.setitem(sys.modules, "gcsfs", None)
  public.gcs_has_credentials.cache_clear()
  try:
    assert public.gcs_has_credentials()
  finally:
    public.gcs_has_credentials.cache_clear()


def test_cache_root(tmp_path):
  """The cache: CACHE_ENV, else XDG_CACHE_HOME, else ~/.cache."""
  assert public.cache_root({public.CACHE_ENV: str(tmp_path)}) == str(tmp_path)
  pkg = public.__name__.split(".", maxsplit=1)[0]
  assert public.cache_root({"XDG_CACHE_HOME": "/x"}) == os.path.join(
      "/x", pkg, "artifacts"
  )


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v"]))
