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


"""Tests for the command line."""

import contextlib
import io
import pathlib
import queue
import subprocess
import sys
import threading
import unittest

from absl.testing import absltest

from open_spiel.python.gui import cli
from open_spiel.python.gui import static


class ParseArgsTest(absltest.TestCase):

  def test_defaults_to_loopback_and_the_built_client(self) -> None:
    args = cli.parse_args([])
    self.assertEqual((args.host, args.port), ("127.0.0.1", 8080))
    self.assertEqual(args.client_dir, static.DEFAULT_ROOT)
    self.assertEqual(args.allow_host, [])

  def test_reads_every_flag(self) -> None:
    args = cli.parse_args([
        "--host=::1",
        "--port=0",
        "--client-dir=/tmp/dist",
        "--allow-host=box",
        "--allow-host=box.local",
    ])
    self.assertEqual((args.host, args.port), ("::1", 0))
    self.assertEqual(args.client_dir, pathlib.Path("/tmp/dist"))
    self.assertEqual(args.allow_host, ["box", "box.local"])


class AllowedHostsTest(absltest.TestCase):

  def test_loopback_is_always_allowed(self) -> None:
    self.assertEqual(
        cli.allowed_hosts("127.0.0.1", []),
        frozenset({"localhost", "127.0.0.1", "::1"}),
    )

  def test_a_specific_bind_address_is_allowed(self) -> None:
    self.assertIn("192.168.1.5", cli.allowed_hosts("192.168.1.5", []))

  def test_a_wildcard_bind_address_is_not_a_host_name(self) -> None:
    self.assertNotIn("0.0.0.0", cli.allowed_hosts("0.0.0.0", []))
    self.assertNotIn("::", cli.allowed_hosts("::", []))

  def test_extra_names_are_allowed_in_lower_case(self) -> None:
    self.assertIn("box.local", cli.allowed_hosts("0.0.0.0", ["Box.Local"]))


class DisclaimerTest(absltest.TestCase):

  def test_help_carries_the_disclaimer(self) -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit):
      cli.parse_args(["--help"])
    help_text = " ".join(stdout.getvalue().split())
    self.assertIn(cli.DISCLAIMER, help_text)

  def test_loopback_addresses_and_localhost_are_loopback(self) -> None:
    for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost", "LocalHost"):
      self.assertTrue(cli.is_loopback(host), host)

  def test_other_addresses_and_names_are_not_loopback(self) -> None:
    for host in ("0.0.0.0", "::", "192.168.1.5", "box.local"):
      self.assertFalse(cli.is_loopback(host), host)

  def test_logs_the_disclaimer_on_loopback(self) -> None:
    with self.assertLogs(level="INFO") as logs:
      cli.log_disclaimer("127.0.0.1")
    self.assertLen(logs.records, 1)
    self.assertEqual(logs.records[0].levelname, "INFO")
    self.assertEqual(logs.records[0].getMessage(), cli.DISCLAIMER)

  def test_warns_with_the_disclaimer_beyond_loopback(self) -> None:
    with self.assertLogs(level="INFO") as logs:
      cli.log_disclaimer("0.0.0.0")
    self.assertLen(logs.records, 1)
    self.assertEqual(logs.records[0].levelname, "WARNING")
    message = logs.records[0].getMessage()
    self.assertIn("0.0.0.0", message)
    self.assertIn(cli.DISCLAIMER, message)
    self.assertIn("no authentication", message)


class MainTest(absltest.TestCase):

  @unittest.skipIf(not sys.executable, "No Python interpreter to start.")
  def test_says_where_it_serves_at_once_with_stdout_a_pipe(self) -> None:
    client_dir = self.create_tempdir().full_path
    lines: queue.Queue[str] = queue.Queue()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "open_spiel.python.gui",
            "--port=0",
            f"--client-dir={client_dir}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as process:
      assert process.stdout is not None
      stdout = process.stdout
      threading.Thread(
          target=lambda: lines.put(stdout.readline()), daemon=True
      ).start()
      try:
        # The server runs until stopped, so a line held in its buffer would
        # never arrive.
        line = lines.get(timeout=120)
      finally:
        process.terminate()
    self.assertRegex(
        line, r"^Serving OpenSpiel games at http://127\.0\.0\.1:\d+/\n$"
    )


if __name__ == "__main__":
  absltest.main()
