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


"""Serving the built browser client.

The client is a single-page application: its `index.html` answers every path
that does not name a file, and the client routes from there.
"""

import dataclasses
import pathlib
import urllib.parse

# Where the client's build writes it, relative to this package.
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "gui" / "dist"

# The files a client build contains, by extension. Anything else is refused.
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}

_INDEX = "index.html"


@dataclasses.dataclass(frozen=True)
class Asset:
  """A file of the client.

  Attributes:
    content: The file's bytes.
    content_type: The `Content-Type` to serve it with.
  """

  content: bytes
  content_type: str


class ClientFiles:
  """The files of a built client."""

  def __init__(self, root: pathlib.Path) -> None:
    """Initializes the files.

    Args:
      root: The directory the client's build wrote. It need not exist yet;
        files are read on each request, so a rebuild needs no restart.
    """
    self._root = root.resolve()

  @property
  def root(self) -> pathlib.Path:
    """The directory the client is served from."""
    return self._root

  @property
  def built(self) -> bool:
    """Whether the directory holds a client build."""
    return (self._root / _INDEX).is_file()

  def find(self, url_path: str) -> Asset | None:
    """Returns the asset that answers a request path, or None.

    Args:
      url_path: The path of the request URL, percent-encoded.

    Returns:
      The named file when the path has a file extension, and `index.html`
      when it has none. None when there is no such file, or the path leaves
      the root, names a hidden file, or has an extension not in
      `CONTENT_TYPES`.
    """
    parts = [part for part in urllib.parse.unquote(url_path).split("/") if part]
    if any(
        part.startswith(".") or "\\" in part or "\0" in part for part in parts
    ):
      return None
    if not parts or "." not in parts[-1]:
      parts = [_INDEX]
    path = self._root.joinpath(*parts).resolve()
    content_type = CONTENT_TYPES.get(path.suffix)
    if (
        content_type is None
        or not path.is_relative_to(self._root)
        or not path.is_file()
    ):
      return None
    return Asset(path.read_bytes(), content_type)
