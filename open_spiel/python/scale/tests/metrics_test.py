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

"""metrics.

the writer (absl logging and the JSONL record), the MetricSink interface (named,
auto and explicit sinks; params publishers; series records), held settings,
resume and validation.
"""

import builtins
import json
import sys
import types as pytypes

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale import metrics


def test_writes_json_lines_with_step_and_prefix(tmp_path):
  """Rows are JSON lines with their step and the train/ prefix."""
  w = metrics.MetricsWriter(str(tmp_path))
  w.write(1, {"loss": 0.5, "lr": np.float32(1e-3)})
  w.write(
      2, {"loss": jnp.float32(0.25), "frames": np.int64(2**40)}, prefix="eval"
  )
  w.close()
  rows = metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))
  assert [r["step"] for r in rows] == [1, 2]
  assert rows[0]["train/loss"] == 0.5 and rows[0]["train/lr"] == pytest.approx(
      1e-3
  )
  assert rows[1]["eval/frames"] == 2**40 and isinstance(
      rows[1]["eval/frames"], int
  )
  assert all("time" in r for r in rows)
  with open(tmp_path / metrics.JSONL_NAME, encoding="utf-8") as f:
    assert all(json.loads(line) for line in f)


def test_rejects_non_scalars_non_finite_and_bad_steps(tmp_path):
  """Write refuses non-scalars, non-finite values and bad steps."""
  w = metrics.MetricsWriter(str(tmp_path))
  with pytest.raises(ValueError, match="scalar"):
    w.write(1, {"x": np.zeros(3)})
  with pytest.raises(ValueError, match="finite"):
    w.write(1, {"x": float("nan")})
  with pytest.raises(ValueError, match="step"):
    w.write(-1, {"x": 1.0})
  w.write(5, {"x": 1.0})
  with pytest.raises(ValueError, match="step"):
    w.write(4, {"x": 1.0})  # steps never go back within one writer


def test_logging_only_writer_and_tensorboard_fails_loud(tmp_path):
  """A logging-only writer works; tensorboard without its package fails loud."""
  w = metrics.MetricsWriter(None)
  w.write(1, {"x": 1.0})
  w.close()
  if not metrics.tensorboard_available():
    with pytest.raises(ImportError):
      metrics.MetricsWriter(str(tmp_path), tensorboard=True)


def test_reopen_continues_and_resume_drops_rows_past_the_checkpoint(tmp_path):
  """A run preempted at 230, checkpointed at 200, resumes at 200.

  Rows 201..230 are dropped, not doubled.

  Args:
    tmp_path: pytest's temporary directory.
  """
  w = metrics.MetricsWriter(str(tmp_path))
  for s in (1, 2, 3, 4, 5):
    w.write(s, {"x": float(s)})
  w.close()
  again = metrics.MetricsWriter(
      str(tmp_path)
  )  # no resume: the floor is the last step
  with pytest.raises(ValueError, match="goes back"):
    again.write(2, {"x": 0.0})
  resumed = metrics.MetricsWriter(str(tmp_path), resume_step=3)
  resumed.write(4, {"x": 40.0})
  resumed.close()
  rows = metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))
  assert [r["step"] for r in rows] == [1, 2, 3, 4]
  assert rows[-1]["train/x"] == 40.0


def test_held_keys_are_params_until_they_move(tmp_path):
  """A value that never changes is a param; once it moves it is a row."""
  w = metrics.MetricsWriter(str(tmp_path))
  for step, lr in ((1, 1e-3), (2, 1e-3), (3, 5e-4), (4, 1e-3)):
    w.write(
        step, {"loss": float(step), "lr": lr, "rows": 64}, held=("lr", "rows")
    )
  w.close()
  rows = metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))
  assert [r["step"] for r in rows] == [1, 2, 3, 4]
  assert [r.get("train/lr") for r in rows] == [
      None,
      None,
      5e-4,
      1e-3,
  ]  # a series once it moved
  assert all("train/rows" not in r for r in rows)
  doc = json.loads((tmp_path / metrics.PARAMS_NAME).read_text())
  assert doc == {
      "train/lr": {"initial": 1e-3, "moved_at_step": 3},
      "train/rows": 64,
  }
  # A resumed writer keeps the record: rows stays a param, lr stays a series.
  w = metrics.MetricsWriter(str(tmp_path))
  w.write(5, {"lr": 1e-3, "rows": 64}, held=("lr", "rows"))
  w.close()
  last = metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))[-1]
  assert (
      last["step"] == 5
      and last["train/lr"] == 1e-3
      and "train/rows" not in last
  )


def test_an_unreadable_params_record_is_logged_and_restarted(tmp_path):
  """An unreadable params record is logged and started afresh."""
  (tmp_path / metrics.PARAMS_NAME).write_text(
      '{"train/rows": 6'
  )  # cut mid-write
  w = metrics.MetricsWriter(str(tmp_path))
  w.write(1, {"rows": 64}, held=("rows",))
  w.close()
  assert json.loads((tmp_path / metrics.PARAMS_NAME).read_text()) == {
      "train/rows": 64
  }
  assert not (tmp_path / (metrics.PARAMS_NAME + ".tmp")).exists()


class _Sink(metrics.MetricSink):
  """A sink that records every call, and can fail on params."""

  def __init__(self, fail_params=False):
    self.calls, self.fail_params = [], fail_params

  def write(self, step, values):
    """Records the row.

    Args:
      step: the row's step.
      values: the row's values.
    """
    self.calls.append(("write", step, dict(values)))

  def params(self, values):
    """Records the settings, or fails when `fail_params`.

    Args:
      values: the settings.

    Raises:
      RuntimeError: when `fail_params`.
    """
    if self.fail_params:
      raise RuntimeError("tracking service down")
    self.calls.append(("params", dict(values)))

  def flush(self):
    """Records the flush."""
    self.calls.append(("flush",))

  def close(self):
    """Records the close."""
    self.calls.append(("close",))


def test_every_row_and_setting_goes_to_every_sink_and_the_jsonl_record(
    tmp_path,
):
  """Every row and setting reaches every sink and the JSONL record."""
  s = _Sink()
  w = metrics.MetricsWriter(str(tmp_path), sinks=[s])
  w.write(1, {"loss": 0.5, "frames": np.int64(64)})
  w.write(2, {"win": 0.25}, prefix="eval")
  w.params({"run/seed": 3})
  w.close()
  assert s.calls[0] == ("write", 1, {"train/loss": 0.5, "train/frames": 64})
  assert s.calls[1] == ("write", 2, {"eval/win": 0.25})
  assert ("params", {"run/seed": 3}) in s.calls and s.calls[-1] == ("close",)
  assert [
      r["step"] for r in metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))
  ] == [1, 2]
  assert json.loads((tmp_path / metrics.PARAMS_NAME).read_text()) == {
      "run/seed": 3
  }
  # A logdir-less writer still feeds its sinks.
  s2 = _Sink()
  w = metrics.MetricsWriter(None, sinks=[s2])
  w.write(1, {"x": 1.0})
  w.close()
  assert s2.calls[0] == ("write", 1, {"train/x": 1.0})


def test_named_and_auto_sinks_resolve_through_the_registries(
    tmp_path, monkeypatch
):
  """Named and automatic sinks resolve through the sink registries."""
  made = []

  def named(ctx):
    """A registered named sink factory.

    Args:
      ctx: the sink context (unused).

    Returns:
      a recording sink.
    """
    made.append(("named", ctx.logdir))
    return _Sink()

  def auto(ctx):
    made.append(("auto", ctx.tensorboard))
    return None  # does not apply here

  # Reaches private _NAMED of the code under test.
  monkeypatch.setattr(metrics, "_NAMED", dict(metrics._NAMED))  # pylint: disable=protected-access
  # Reaches private _AUTO of the code under test.
  monkeypatch.setattr(metrics, "_AUTO", list(metrics._AUTO))  # pylint: disable=protected-access
  metrics.register_sink("test_sink", named)
  metrics.register_auto_sink(auto)
  assert "test_sink" in metrics.sink_names() and {
      "tensorboard",
      "wandb",
  } <= set(metrics.sink_names())
  w = metrics.MetricsWriter(str(tmp_path), sinks=["test_sink"])
  w.close()
  assert made == [("auto", False), ("named", str(tmp_path))]
  with pytest.raises(KeyError, match="unknown metric sink"):
    metrics.MetricsWriter(str(tmp_path), sinks=["nope"])
  with pytest.raises(ValueError, match="already registered"):
    metrics.register_sink("test_sink", lambda ctx: None)


def test_a_failing_sink_params_is_logged_not_raised(tmp_path):
  """A sink whose params call fails is logged, not raised."""
  w = metrics.MetricsWriter(str(tmp_path), sinks=[_Sink(fail_params=True)])
  w.params({"x": 1})
  w.close()
  assert json.loads((tmp_path / metrics.PARAMS_NAME).read_text()) == {"x": 1}
  with pytest.raises(ValueError, match="not finite"):
    w.params({"bad": float("nan")})


def test_a_failing_sink_write_flush_or_close_is_logged_not_raised(tmp_path):
  """A sink failing in write, flush or close is logged, not raised."""

  class Down(_Sink):
    """A sink whose every call raises."""

    def write(self, step, values):
      raise RuntimeError("down")

    def flush(self):
      raise RuntimeError("down")

    def close(self):
      raise RuntimeError("down")

  ok = _Sink()
  w = metrics.MetricsWriter(str(tmp_path), sinks=[Down(), ok])
  w.write(1, {"loss": 0.5})
  w.close()
  assert ok.calls[0] == ("write", 1, {"train/loss": 0.5}) and ok.calls[-1] == (
      "close",
  )
  assert [
      r["step"] for r in metrics.read_jsonl(str(tmp_path / metrics.JSONL_NAME))
  ] == [1]


def test_a_resumed_writer_hands_the_settings_record_to_its_sinks(tmp_path):
  """A resumed writer hands the saved settings record to its sinks."""
  w = metrics.MetricsWriter(str(tmp_path))
  w.write(1, {"loss": 1.0, "rows": 64}, held=("rows",))
  w.close()
  s = _Sink()
  w = metrics.MetricsWriter(
      str(tmp_path), sinks=[s]
  )  # a relaunch reporting elsewhere
  w.write(
      2, {"loss": 0.5, "rows": 64}, held=("rows",)
  )  # unchanged: not written again
  w.close()
  assert s.calls[0] == ("params", {"train/rows": 64})
  assert ("write", 2, {"train/loss": 0.5}) in s.calls


def test_register_params_goes_to_every_publisher_and_survives_a_failing_one(
    monkeypatch,
):
  """register_params reaches every publisher, past a failing one."""
  got = []
  monkeypatch.setattr(metrics, "_PARAMS_PUBLISHERS", [])
  # No publisher: logs only.
  assert not metrics.register_params({"hw/devices/n": np.int64(8)})

  def boom(vals):
    raise RuntimeError("down")

  metrics.register_params_publisher(boom)
  metrics.register_params_publisher(lambda vals: got.append(dict(vals)) or True)
  assert metrics.register_params({"hw/devices/n": np.int64(8)})
  assert got == [{"hw/devices/n": 8}]
  with pytest.raises(ValueError, match="no params"):
    metrics.register_params({})


def test_wandb_sink_is_optional_and_fails_loud_without_the_package(
    tmp_path, monkeypatch
):
  """The wandb sink is optional; asked for without wandb it fails loud."""
  real = builtins.__import__

  def no_wandb(name, *a, **kw):
    """__import__ with wandb absent.

    Args:
      name: the module name.
      *a: passed through.
      **kw: passed through.

    Returns:
      the imported module.

    Raises:
      ImportError: name is wandb.
    """
    if name == "wandb":
      raise ImportError("No module named 'wandb'")
    return real(name, *a, **kw)

  monkeypatch.setattr(builtins, "__import__", no_wandb)
  with pytest.raises(ImportError, match="pip install wandb"):
    metrics.MetricsWriter(str(tmp_path), sinks=["wandb"])


def test_wandb_sink_logs_rows_and_settings_through_the_wandb_api(
    tmp_path, monkeypatch
):
  """The wandb sink logs rows and settings through the wandb API."""
  calls = []

  class _Run:
    """A stand-in wandb run that records its calls."""

    config = pytypes.SimpleNamespace(
        update=lambda d, allow_val_change: calls.append(("config", dict(d)))
    )

    def log(self, d, step):
      """Records a log call.

      Args:
        d: the row.
        step: its step.
      """
      calls.append(("log", step, dict(d)))

    def finish(self):
      """Records the finish call."""
      calls.append(("finish",))

  fake = pytypes.ModuleType("wandb")
  fake.init = lambda **kw: calls.append(("init", kw)) or _Run()
  monkeypatch.setitem(sys.modules, "wandb", fake)
  w = metrics.MetricsWriter(str(tmp_path), sinks=["wandb"])
  w.write(3, {"loss": 0.5})
  w.params({"seed": 1})
  w.close()
  assert calls[0] == ("init", {"dir": str(tmp_path)})
  assert ("log", 3, {"train/loss": 0.5}) in calls and (
      "config",
      {"seed": 1},
  ) in calls
  assert calls[-1] == ("finish",)


# -- series records
# --------------------------------------------------------------------------
def test_series_record_known_answer():
  """series_record on known inputs."""
  assert metrics.series_record(
      120,
      {"x/decide_mismatches": np.int64(0), "tests_passed": 3},
      step_unit="plies",
  ) == {
      "step": 120,
      "step_unit": "plies",
      "tests_passed": 3,
      "x/decide_mismatches": 0,
  }
  assert metrics.series_record(
      0, {}, mean={"name": "r", "n": 2, "sum": 3.5, "sumsq": 6.25}
  ) == {"mean": {"n": 2, "name": "r", "sum": 3.5, "sumsq": 6.25}, "step": 0}


@pytest.mark.parametrize(
    "kw",
    [
        dict(step=-1, values={"a": 1}),
        dict(step=1.5, values={"a": 1}),
        dict(step=True, values={"a": 1}),
        dict(step=0, values={"step_unit": 1}),
        dict(step=0, values={" a": 1}),
        dict(step=0, values={"a": float("nan")}),
        dict(step=0, values={"a": "text"}),
        dict(step=0, values={}),
        dict(step=0, values={"a": 1}, step_unit="no/slash"),
        dict(
            step=0,
            values={},
            mean={"name": "r", "n": 0, "sum": 0.0, "sumsq": 0.0},
        ),
        dict(step=0, values={}, mean={"name": "r", "n": 1}),
    ],
)
def test_series_record_refusals(kw):
  """series_record refuses malformed steps, values and means."""
  with pytest.raises(ValueError):
    metrics.series_record(kw.pop("step"), kw.pop("values"), **kw)


def test_emit_series_reaches_every_emitter(monkeypatch):
  """emit_series hands its record to every registered emitter."""
  got = []
  monkeypatch.setattr(metrics, "_SERIES_EMITTERS", [])
  metrics.register_series_emitter(got.append)
  rec = metrics.emit_series(7, {"a/b": 1.5}, step_unit="repeats")
  assert got == [rec] == [{"step": 7, "step_unit": "repeats", "a/b": 1.5}]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
