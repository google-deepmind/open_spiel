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


from absl.testing import absltest

from open_spiel.scripts.bazel_to_cmake import label_map


class LabelMapTest(absltest.TestCase):
  """The external label table and label helpers."""

  def test_normalize(self):
    self.assertEqual(label_map.normalize("//a/b"), "//a/b:b")
    self.assertEqual(label_map.normalize("//a/b:c"), "//a/b:c")
    self.assertEqual(label_map.normalize("@x"), "@x//:x")
    self.assertEqual(label_map.normalize("@x//y"), "@x//y:y")

  def test_lookup_accepts_both_forms(self):
    for entry in label_map.EXTERNAL_LABELS:
      self.assertIs(label_map.lookup(entry.source), entry)
      self.assertIs(label_map.lookup(entry.exported), entry)
    self.assertEqual(label_map.lookup("//third_party/absl/strings:strings"),
                     label_map.lookup("@abseil-cpp//absl/strings"))
    self.assertIsNone(label_map.lookup("//third_party/unknown"))

  def test_entries_are_well_formed(self):
    for entry in label_map.EXTERNAL_LABELS:
      with self.subTest(entry.source):
        self.assertFalse(label_map.is_open_spiel(entry.source))
        self.assertFalse(label_map.is_open_spiel(entry.exported))
        self.assertNotEqual(entry.cmake is None, entry.skip is None)
        if entry.option:
          self.assertStartsWith(entry.option, "OPEN_SPIEL_BUILD_WITH_")

  def test_open_spiel_labels(self):
    for prefix in ("//third_party/open_spiel", "//open_spiel"):
      self.assertTrue(label_map.is_open_spiel(prefix))
      self.assertEqual(label_map.open_spiel_package_and_name(prefix),
                       ("", "open_spiel"))
      self.assertEqual(
          label_map.open_spiel_package_and_name(prefix + ":all_games"),
          ("", "all_games"))
      self.assertEqual(
          label_map.open_spiel_package_and_name(prefix + "/games/chess"),
          ("games/chess", "chess"))
      self.assertEqual(
          label_map.open_spiel_package_and_name(prefix + "/games:all_games"),
          ("games", "all_games"))
    self.assertFalse(label_map.is_open_spiel("//third_party/open_spiel_x"))


if __name__ == "__main__":
  absltest.main()
