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

"""A local stand-in for public Cloud Storage, for tests (no network).

FakeBucketServer serves a local folder over HTTP on 127.0.0.1 the way
public Cloud Storage serves a bucket: <root>/<bucket>/<object> is
the object gs://<bucket>/<object>, listed by the JSON API
(GET /storage/<version>/b/<bucket>/o?prefix=...&pageToken=...,
`page_size` items a page, with size and base64 MD5) and read at
GET /<bucket>/<object>. Point
ckpt.public.ENDPOINT at `url` to read through it.

FakeGcsfs is a gcsfs.GCSFileSystem stand-in over the same folder layout,
recording the token it was made with.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import threading
from typing import Any, Optional
import urllib.parse


def _md5(data: bytes) -> str:
  """The base64 MD5 Cloud Storage lists.

  Args:
    data: the bytes.

  Returns:
    The digest.
  """
  return base64.b64encode(hashlib.md5(data).digest()).decode()


def _objects(root: str, bucket: str, prefix: str) -> list[tuple[str, str]]:
  """(object name, local file) of every object under `prefix`, sorted.

  Args:
    root: the served folder.
    bucket: the bucket.
    prefix: the object name prefix.

  Returns:
    The objects.
  """
  base = os.path.join(root, bucket)
  out = []
  for dirpath, _, filenames in os.walk(base):
    for fn in filenames:
      local = os.path.join(dirpath, fn)
      name = os.path.relpath(local, base).replace(os.sep, "/")
      if name.startswith(prefix):
        out.append((name, local))
  return sorted(out)


class FakeBucketServer:
  """Public Cloud Storage over HTTP on 127.0.0.1 (module doc).

  Attributes:
    root: the served folder (<root>/<bucket>/<object>).
    page_size: listing items per page.
    requests: the paths requested, in order.
    deny: object name prefixes answered 403 (listing and reads).
    corrupt: object names served with one byte flipped.
    url: the server's base URL.
  """

  def __init__(self, root: str, page_size: int = 1000):
    self.root = root
    self.page_size = page_size
    self.requests: list[str] = []
    self.deny: set[str] = set()
    self.corrupt: set[str] = set()
    fake = self

    class Handler(http.server.BaseHTTPRequestHandler):
      """Answers listing and object GETs."""

      def log_message(self, *args: Any) -> None:  # pylint: disable=arguments-differ
        del args

      def _send(self, code: int, body: bytes, ctype: str) -> None:
        """Sends one response.

        Args:
          code: the status.
          body: the body.
          ctype: the content type.
        """
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def do_GET(self) -> None:  # pylint: disable=invalid-name
        """One GET."""
        fake.requests.append(self.path)
        url = urllib.parse.urlsplit(self.path)
        listing = re.fullmatch(r"/storage/[^/]+/b/([^/]+)/o", url.path)
        if listing:
          bucket = urllib.parse.unquote(listing.group(1))
          q = urllib.parse.parse_qs(url.query)
          prefix = q.get("prefix", [""])[0]
          if any(prefix.startswith(d) for d in fake.deny):
            self._send(403, b'{"error": "forbidden"}', "application/json")
            return
          start = int(q.get("pageToken", ["0"])[0])
          objs = _objects(fake.root, bucket, prefix)
          page = objs[start : start + fake.page_size]
          items = []
          for name, local in page:
            with open(local, "rb") as f:
              data = f.read()
            items.append(
                dict(name=name, size=str(len(data)), md5Hash=_md5(data))
            )
          body: dict[str, Any] = {"items": items} if items else {}
          if start + fake.page_size < len(objs):
            body["nextPageToken"] = str(start + fake.page_size)
          self._send(200, json.dumps(body).encode(), "application/json")
          return
        bucket, _, name = urllib.parse.unquote(url.path[1:]).partition("/")
        local = os.path.join(fake.root, bucket, *name.split("/"))
        if any(name.startswith(d) for d in fake.deny):
          self._send(403, b"forbidden", "text/plain")
          return
        if not name or not os.path.isfile(local):
          self._send(404, b"not found", "text/plain")
          return
        with open(local, "rb") as f:
          data = f.read()
        if name in fake.corrupt and data:
          data = bytes([data[0] ^ 1]) + data[1:]
        self._send(200, data, "application/octet-stream")

    self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
    self._thread = threading.Thread(
        target=self._httpd.serve_forever, daemon=True
    )

  def start(self) -> FakeBucketServer:
    """Starts serving.

    Returns:
      self.
    """
    self._thread.start()
    return self

  def stop(self) -> None:
    """Stops serving."""
    self._httpd.shutdown()
    self._httpd.server_close()
    self._thread.join()

  def object_reads(self) -> list[str]:
    """The object GETs so far (listings left out).

    Returns:
      Their paths.
    """
    return [r for r in self.requests if not r.startswith("/storage/")]


class _Credentials:
  """What gcsfs reports of its connection."""

  def __init__(self, method: str):
    self.method = method


class FakeGcsfs:
  """A gcsfs.GCSFileSystem stand-in over a local folder (module doc).

  Attributes:
    root: the folder (<root>/<bucket>/<object>).
    token: the token it was made with.
    credentials: .method is "anon" for token None or "anon" when
      `default_method` is "anon", else that method.
  """

  def __init__(
      self, root: str, token: Optional[str], default_method: str = "anon"
  ):
    self.root = root
    self.token = token
    self.credentials = _Credentials(default_method if token is None else token)

  def find(self, path: str, detail: bool = False) -> Any:
    """Every object under `path` ("<bucket>/<prefix>").

    Args:
      path: the bucket and prefix.
      detail: return {key: info} (else the keys).

    Returns:
      The objects.
    """
    bucket, _, prefix = path.partition("/")
    out = {}
    for name, local in _objects(self.root, bucket, prefix):
      with open(local, "rb") as f:
        data = f.read()
      out[f"{bucket}/{name}"] = dict(
          type="file", size=len(data), md5Hash=_md5(data)
      )
    return out if detail else sorted(out)

  def cat_file(self, path: str) -> bytes:
    """The bytes of "<bucket>/<object>".

    Args:
      path: the object.

    Returns:
      Its bytes.
    """
    bucket, _, name = path.partition("/")
    with open(os.path.join(self.root, bucket, *name.split("/")), "rb") as f:
      return f.read()
