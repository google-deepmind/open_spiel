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


"""Tests for serving the built client."""

import pathlib

from absl.testing import absltest

from open_spiel.python.gui import static


class ClientFilesTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.root = pathlib.Path(self.create_tempdir("dist").full_path)
    (self.root / "index.html").write_text("<html></html>")
    (self.root / "assets").mkdir()
    (self.root / "assets" / "app.js").write_text("run();")
    (self.root / ".env").write_text("SECRET=1")
    (self.root / "notes.py").write_text("print()")
    self.files = static.ClientFiles(self.root)

  def test_serves_a_named_file_with_its_content_type(self) -> None:
    asset = self.files.find("/assets/app.js")
    self.assertIsNotNone(asset)
    assert asset is not None
    self.assertEqual(asset.content, b"run();")
    self.assertEqual(asset.content_type, "text/javascript; charset=utf-8")

  def test_serves_the_index_for_paths_without_an_extension(self) -> None:
    for path in ("/", "", "/games/chess", "/games/chess/"):
      with self.subTest(path=path):
        asset = self.files.find(path)
        assert asset is not None
        self.assertEqual(asset.content, b"<html></html>")

  def test_a_missing_file_is_not_found(self) -> None:
    self.assertIsNone(self.files.find("/assets/missing.js"))

  def test_refuses_paths_that_leave_the_root(self) -> None:
    outside = self.root.parent / "secret.txt"
    outside.write_text("secret")
    for path in (
        "/../secret.txt",
        "/assets/../../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets%2f..%2f..%2fsecret.txt",
    ):
      with self.subTest(path=path):
        self.assertIsNone(self.files.find(path))

  def test_refuses_symlinks_out_of_the_root(self) -> None:
    outside = self.root.parent / "linked.txt"
    outside.write_text("secret")
    (self.root / "link.txt").symlink_to(outside)
    self.assertIsNone(self.files.find("/link.txt"))

  def test_refuses_hidden_files(self) -> None:
    self.assertIsNone(self.files.find("/.env"))

  def test_refuses_extensions_a_build_does_not_contain(self) -> None:
    self.assertIsNone(self.files.find("/notes.py"))

  def test_reports_whether_the_client_is_built(self) -> None:
    self.assertTrue(self.files.built)
    empty = static.ClientFiles(self.root / "missing")
    self.assertFalse(empty.built)
    self.assertIsNone(empty.find("/"))


if __name__ == "__main__":
  absltest.main()
