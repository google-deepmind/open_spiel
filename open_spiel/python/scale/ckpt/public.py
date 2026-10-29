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

"""Anonymous reads of public artifacts, into a local cache.

Two kinds of path read a public Cloud Storage folder without an account:

  https://storage.googleapis.com/<bucket>/<path>
      read over plain HTTPS with the standard library: the folder is listed
      through Cloud Storage's JSON API and each object fetched from its public
      URL. Nothing beyond Python is needed.
  gs://<bucket>/<path>
      read through etils epath when Google credentials exist; when none
      do and gcsfs is installed, read anonymously through gcsfs with
      token="anon".

An anonymous read copies the objects it needs into a local cache (CACHE_ENV,
default ~/.cache/<package>/artifacts) and the artifact is loaded from there:
the folder's own files and each step's top-level files (a manifest, which is
how a step is found), and a step's whole folder when that step is loaded.
A file already in the cache with the listed size and MD5 is not fetched
again; every fetched file is checked against the listed size and MD5. Public
paths are read-only: writing to one raises.

Both need the folder to be readable by anyone, objects and listing (Cloud
Storage's "Storage Object Viewer" role for allUsers grants both).
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Optional, Protocol
import urllib.error
import urllib.parse
import urllib.request

from absl import logging

from open_spiel.python.scale import plugins

#: The public HTTPS endpoint of Cloud Storage. Tests point it at a local
#: server.
ENDPOINT = "https://storage.googleapis.com"
#: Cloud Storage's JSON API; its path, under ENDPOINT, lists a bucket.
JSON_API = "https://storage.googleapis.com/storage/v1"
GCS_PREFIX = "gs://"
#: Where anonymous reads are cached (default ~/.cache/<package>/artifacts).
CACHE_ENV = plugins.ENV_PREFIX + "ARTIFACT_CACHE"
#: Seconds an HTTPS request may take.
TIMEOUT_S = 60.0


class FetchError(OSError):
  """A public folder could not be listed or read."""


class Transport(Protocol):
  """Lists and reads the objects of one bucket."""

  def list(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Every object under `prefix`.

    Args:
      bucket: the bucket.
      prefix: the object name prefix (ends with "/").

    Returns:
      [{"name", "size", "md5Hash"}] ("md5Hash": base64, or None).
    """

  def read(self, bucket: str, name: str) -> bytes:
    """The bytes of one object.

    Args:
      bucket: the bucket.
      name: the object name.

    Returns:
      Its bytes.
    """


def is_https(path: str) -> bool:
  """Whether `path` is a public Cloud Storage HTTPS URL (module doc).

  Args:
    path: a path.

  Returns:
    True for "<ENDPOINT>/<bucket>/...".
  """
  return str(path).startswith(ENDPOINT.rstrip("/") + "/")


def norm_https(path: str) -> str:
  """A public HTTPS URL, normalised (duplicate and trailing slashes dropped).

  Args:
    path: "<ENDPOINT>/<bucket>/<path>".

  Returns:
    The normalised URL.

  Raises:
    ValueError: no bucket, a query or fragment, or ".." in the path.
  """
  _, rest = split(path)
  return ENDPOINT.rstrip("/") + "/" + rest


def split(path: str) -> tuple[str, str]:
  """(bucket, "<bucket>/<path>") of a public HTTPS or gs:// URL.

  Args:
    path: a public HTTPS URL or a gs:// URL.

  Returns:
    (bucket, the normalised "<bucket>/<path>").

  Raises:
    ValueError: not such a URL, no bucket, a query or fragment, or "..".
  """
  path = str(path)
  if is_https(path):
    rest = path[len(ENDPOINT.rstrip("/")) + 1 :]
  elif path.startswith(GCS_PREFIX):
    rest = path[len(GCS_PREFIX) :]
  else:
    raise ValueError(f"not a public Cloud Storage URL: {path!r}")
  if "?" in rest or "#" in rest:
    raise ValueError(f"a query or fragment in {path!r}")
  parts = [p for p in rest.split("/") if p]
  if not parts:
    raise ValueError(f"no bucket in {path!r}")
  if any(p in (".", "..") for p in parts):
    raise ValueError(f"'.' or '..' in {path!r}")
  return parts[0], "/".join(parts)


class HttpsTransport:
  """Anonymous HTTPS: Cloud Storage's JSON API lists, public URLs read."""

  def __init__(self, endpoint: Optional[str] = None):
    self.endpoint = (endpoint or ENDPOINT).rstrip("/")

  def _get(self, url: str) -> bytes:
    """The body of an anonymous GET.

    Args:
      url: the URL.

    Returns:
      The body.

    Raises:
      FetchError: the request fails (an HTTP error names its status).
    """
    req = urllib.request.Request(url, headers={"User-Agent": "artifact-read"})
    try:
      with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:  # noqa: S310
        return r.read()
    except urllib.error.HTTPError as e:
      raise FetchError(
          f"GET {url}: HTTP {e.code} {e.reason} (is the folder public: "
          "objects and listing readable by anyone?)"
      ) from e
    except urllib.error.URLError as e:
      raise FetchError(f"GET {url}: {e.reason}") from e

  def list(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Every object under `prefix` (Transport.list).

    Args:
      bucket: the bucket.
      prefix: the object name prefix.

    Returns:
      [{"name", "size", "md5Hash"}].

    Raises:
      FetchError: a listing fails or is not the JSON API's.
    """
    out = []
    token = None
    while True:
      query = {
          "prefix": prefix,
          "fields": "items(name,size,md5Hash),nextPageToken",
      }
      if token:
        query["pageToken"] = token
      url = (
          f"{self.endpoint}{urllib.parse.urlsplit(JSON_API).path}/b/"
          f"{urllib.parse.quote(bucket, safe='')}/o?"
          f"{urllib.parse.urlencode(query)}"
      )
      try:
        page = json.loads(self._get(url))
      except ValueError as e:
        raise FetchError(f"GET {url}: not a JSON listing") from e
      for item in page.get("items", ()):
        out.append(
            dict(
                name=item["name"],
                size=int(item["size"]),
                md5Hash=item.get("md5Hash"),
            )
        )
      token = page.get("nextPageToken")
      if not token:
        return out

  def read(self, bucket: str, name: str) -> bytes:
    """The bytes of one object (Transport.read).

    Args:
      bucket: the bucket.
      name: the object name.

    Returns:
      Its bytes.
    """
    return self._get(
        f"{self.endpoint}/{urllib.parse.quote(bucket, safe='')}/"
        f"{urllib.parse.quote(name)}"
    )


class GcsfsAnonTransport:
  """Anonymous gs:// through gcsfs, token="anon"."""

  def __init__(self, fs: Any = None):
    if fs is None:
      # Lazy: gcsfs is an optional extra (gs:// paths).
      import gcsfs  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

      fs = gcsfs.GCSFileSystem(token="anon")
    self.fs = fs

  def list(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Every object under `prefix` (Transport.list).

    Args:
      bucket: the bucket.
      prefix: the object name prefix.

    Returns:
      [{"name", "size", "md5Hash"}].
    """
    found = self.fs.find(f"{bucket}/{prefix}", detail=True)
    out = []
    for key, info in sorted(found.items()):
      if info.get("type") == "directory":
        continue
      name = key.split("/", 1)[1]
      out.append(
          dict(
              name=name,
              size=int(info["size"]),
              md5Hash=info.get("md5Hash"),
          )
      )
    return out

  def read(self, bucket: str, name: str) -> bytes:
    """The bytes of one object (Transport.read).

    Args:
      bucket: the bucket.
      name: the object name.

    Returns:
      Its bytes.
    """
    return self.fs.cat_file(f"{bucket}/{name}")


@functools.cache
def gcs_has_credentials() -> bool:
  """Whether gcsfs finds Google credentials (checked once per process).

  gcsfs, given no token, tries Google's default credentials, its token cache
  and the Compute Engine metadata server, and connects anonymously when none
  works; this reports which.

  Returns:
    False when gcsfs is installed and found no credentials; True otherwise
    (including when gcsfs is not installed: gs:// is then read through
    etils epath and fails there with gcsfs named).
  """
  try:
    # Lazy: gcsfs is an optional extra (gs:// paths).
    import gcsfs  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
  except ImportError:
    return True
  method = gcsfs.GCSFileSystem().credentials.method
  logging.info("gcsfs connects to gs:// with method %r", method)
  return method != "anon"


def cache_root(environ: Optional[dict[str, str]] = None) -> str:
  """The local cache of anonymous reads (module doc).

  Args:
    environ: the environment (default os.environ).

  Returns:
    The cache directory.
  """
  env = os.environ if environ is None else environ
  if env.get(CACHE_ENV):
    return os.path.abspath(env[CACHE_ENV])
  base = env.get("XDG_CACHE_HOME") or os.path.join(
      os.path.expanduser("~"), ".cache"
  )
  pkg = __name__.split(".", maxsplit=1)[0]
  return os.path.join(base, pkg, "artifacts")


def _md5_ok(data: bytes, md5: Optional[str]) -> bool:
  """Whether `data` has the listed MD5 (base64), or none is listed.

  Args:
    data: the bytes.
    md5: the listed MD5, base64, or None.

  Returns:
    True when it matches or none is listed.
  """
  if not md5:
    return True
  return base64.b64encode(hashlib.md5(data).digest()).decode() == md5


def _cached_ok(local: str, obj: dict[str, Any]) -> bool:
  """Whether the cached file `local` is the listed object.

  Args:
    local: the cached file.
    obj: the listed object.

  Returns:
    True when it exists with the listed size and MD5.
  """
  if not os.path.isfile(local) or os.path.getsize(local) != obj["size"]:
    return False
  with open(local, "rb") as f:
    return _md5_ok(f.read(), obj["md5Hash"])


def _fetch(
    transport: Transport, bucket: str, obj: dict[str, Any], local: str
) -> None:
  """Copies one object to `local`, checked, atomically.

  Args:
    transport: the transport.
    bucket: the bucket.
    obj: the listed object.
    local: the destination file.

  Raises:
    FetchError: the bytes are not the listed size and MD5.
  """
  data = transport.read(bucket, obj["name"])
  if len(data) != obj["size"] or not _md5_ok(data, obj["md5Hash"]):
    raise FetchError(
        f"gs://{bucket}/{obj['name']}: read {len(data)} bytes that are not the "
        f"listed object ({obj['size']} bytes, md5 {obj['md5Hash']})"
    )
  os.makedirs(os.path.dirname(local), exist_ok=True)
  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(local), prefix=".part-")
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
    os.replace(tmp, local)
  except BaseException:
    if os.path.exists(tmp):
      os.remove(tmp)
    raise


def _wanted(rel: str, step: Optional[int]) -> bool:
  """Whether a folder-relative object name is copied for `step`.

  The folder's own files and every step's top-level files always; the
  whole folder of `step`.

  Args:
    rel: the object name relative to the artifact folder.
    step: the step being loaded, or None.

  Returns:
    True when it is copied.
  """
  head, sep, tail = rel.partition("/")
  if not sep or not head.isdigit():
    return True
  if step is not None and int(head) == int(step):
    return True
  return "/" not in tail


def localize(
    path: str,
    step: Optional[int] = None,
    *,
    transport: Optional[Transport] = None,
    environ: Optional[dict[str, str]] = None,
) -> str:
  """The local cached copy of a public artifact folder (module doc).

  Args:
    path: a public HTTPS URL or a gs:// URL.
    step: the step being loaded (its whole folder is copied), or None.
    transport: lists and reads objects (default: HTTPS for an HTTPS URL,
      gcsfs with token="anon" for gs://).
    environ: the environment (cache_root).

  Returns:
    The local folder holding the copy (empty when no object is under it).

  Raises:
    FetchError: a listing or read fails, or a read is not the listed
      object.
  """
  bucket, rest = split(path)
  if transport is None:
    transport = HttpsTransport() if is_https(path) else GcsfsAnonTransport()
  prefix = rest[len(bucket) + 1 :]
  prefix = prefix + "/" if prefix else ""
  objects = transport.list(bucket, prefix)
  if not objects:
    logging.warning("no object under %s", path)
  root = os.path.join(cache_root(environ), *rest.split("/"))
  fetched = 0
  for obj in objects:
    rel = obj["name"][len(prefix) :]
    if not rel or rel.endswith("/") or not _wanted(rel, step):
      continue
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
      raise FetchError(f"gs://{bucket}/{obj['name']}: unsafe object name")
    local = os.path.join(root, *parts)
    if _cached_ok(local, obj):
      continue
    _fetch(transport, bucket, obj, local)
    fetched += 1
  logging.info(
      "public artifact %s%s: %d objects fetched into %s",
      path,
      "" if step is None else f"@{step}",
      fetched,
      root,
  )
  os.makedirs(root, exist_ok=True)
  return root


def readable(
    path: str,
    step: Optional[int] = None,
    *,
    has_credentials: Callable[[], bool] = gcs_has_credentials,
) -> str:
  """A path the artifact readers can open for `path`@`step`.

  A public HTTPS URL, or a gs:// URL with no credentials, is copied into the
  cache (localize) and its local copy returned; any other path is returned
  as it is.

  Args:
    path: a normalised checkpoint path.
    step: the step being read, or None.
    has_credentials: whether gs:// has credentials (gcs_has_credentials).

  Returns:
    The path to read.
  """
  if is_https(path):
    return localize(path, step)
  if path.startswith(GCS_PREFIX) and not has_credentials():
    logging.info("no Google credentials: reading %s anonymously", path)
    return localize(path, step)
  return path
