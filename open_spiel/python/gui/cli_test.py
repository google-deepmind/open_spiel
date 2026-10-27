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
import http
import http.client
import io
import pathlib
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import unittest
import urllib.parse

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


# How long the crash test waits for the server to answer, then to settle.
_HEALTH_PATIENCE_SECONDS = 120.0


def _served_url(serving: str) -> str:
  """Returns the URL in the server's "Serving ... at <url>" line."""
  match = re.fullmatch(r"Serving OpenSpiel games at (\S+)\n?", serving)
  if match is None:
    raise AssertionError(f"Expected the Serving line, read {serving!r}.")
  return match.group(1)


def _wait_for_health(url: str, process: subprocess.Popen[str]) -> None:
  """Waits until `url`'s /api/health answers 200, failing loud on timeout."""
  parts = urllib.parse.urlsplit(url)
  assert parts.hostname is not None and parts.port is not None, url
  deadline = time.monotonic() + _HEALTH_PATIENCE_SECONDS
  last: object = None
  while True:
    if process.poll() is not None:
      raise AssertionError(
          f"The server exited ({process.returncode}) before /api/health"
          " answered."
      )
    # A plain connection: no HTTP proxy from the environment applies.
    connection = http.client.HTTPConnection(
        parts.hostname, parts.port, timeout=10
    )
    try:
      connection.request("GET", "/api/health")
      response = connection.getresponse()
      response.read()
      if response.status == http.HTTPStatus.OK:
        return
      last = response.status
    except OSError as error:  # Refused, reset or timed out: not up yet.
      last = error
    finally:
      connection.close()
    if time.monotonic() > deadline:
      raise AssertionError(
          f"{url}api/health did not answer 200 within"
          f" {_HEALTH_PATIENCE_SECONDS:g} s; last: {last!r}."
      )
    time.sleep(0.05)


def _thread_count(process: subprocess.Popen[str]) -> int:
  """Returns how many threads `process` runs, from /proc."""
  return len(list(pathlib.Path(f"/proc/{process.pid}/task").iterdir()))


def _wait_for_thread_count(process: subprocess.Popen[str], count: int) -> None:
  """Waits until `process` runs at most `count` threads, failing loud."""
  if process.poll() is not None:
    raise AssertionError(
        f"The server exited ({process.returncode}) before its threads were"
        " counted."
    )
  deadline = time.monotonic() + _HEALTH_PATIENCE_SECONDS
  while (now := _thread_count(process)) > count:
    if process.poll() is not None:
      raise AssertionError(
          f"The server exited ({process.returncode}) while the test waited"
          " for its threads to settle."
      )
    if time.monotonic() > deadline:
      raise AssertionError(
          f"The server still runs {now} threads, not {count}, after"
          f" {_HEALTH_PATIENCE_SECONDS:g} s."
      )
    time.sleep(0.01)


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

  @unittest.skipUnless(
      pathlib.Path("/proc/self/task").is_dir(),
      "Needs /proc/<pid>/task to see the server's threads exit.",
  )
  def test_a_crash_prints_every_threads_stack(self) -> None:
    # A fault in native code kills the process before Python logs anything;
    # the server must still leave the stacks of its threads on stderr. A
    # SIGSEGV sent to it takes the path a segmentation fault takes.
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
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
      assert process.stdout is not None
      stdout = process.stdout
      threading.Thread(
          target=lambda: lines.put(stdout.readline()), daemon=True
      ).start()
      try:
        serving = lines.get(timeout=120)
        idle = _thread_count(process)
        # "Serving" is printed before `serve_forever` starts; under load the
        # signal could land while the main thread is still printing. An
        # answered request proves `serve_forever` is running.
        _wait_for_health(_served_url(serving), process)
        # The request's handler thread outlives the response by a moment;
        # a fault while it exits cuts the stack dump short (faulthandler
        # writes "File ???" and stops), so wait for it to be gone.
        _wait_for_thread_count(process, idle)
        process.send_signal(signal.SIGSEGV)
        _, stderr = process.communicate(timeout=120)
      finally:
        process.kill()
    self.assertEqual(process.returncode, -signal.SIGSEGV, stderr)
    self.assertIn("Fatal Python error: Segmentation fault", stderr)
    self.assertRegex(stderr, r"in serve_forever")


if __name__ == "__main__":
  absltest.main()
