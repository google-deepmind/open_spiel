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

"""The metrics writer: one validated stream of scalars, many destinations.

Learners, training loops and evaluators only emit metrics; sinks consume
them, so a run reports to files, TensorBoard or a tracking service without
code changes.

  writer = metrics.MetricsWriter("/tmp/run/metrics", sinks=["tensorboard"])
  writer.params({"lr": 3e-4})                 # a run setting, not a series
  writer.write(12, {"loss_total": 0.31, "frames": 65536})
  writer.close()

MetricsWriter.write(step, {name: scalar}, prefix="train") sends one line to
absl logging and the row to every sink:

  JsonlSink        the default record: with a logdir, one JSON object per row
                   appended to <logdir>/metrics.jsonl
                     {"step": 12, "time": 1760486400.1,
                      "train/loss_total": 0.31, "train/frames": 65536}
                   and the run's settings in <logdir>/params.json
  TensorBoardSink  scalars through tensorboardX, or a writer module a plugin
                   registers (register_tensorboard_writer); tensorboard=True
                   or the sink name "tensorboard"
  WandbSink        Weights & Biases (the sink name "wandb"; the optional wandb
                   package, imported only when asked for: a missing package
                   raises ImportError, never a silent no-op)

A sink is any object with write(step, values), params(values), flush() and
close() (MetricSink). MetricsWriter(sinks=[...]) takes sink objects or
registered names (register_sink(name, factory)); register_auto_sink(factory)
adds a factory every writer calls (it returns a sink, or None when it does
not apply, e.g. a tracking service's sink outside that service's jobs).
Deployments register theirs from a plugin (plugins); the writer loads the
plugins named in plugins.ENV first.

Integers stay integers (int64 counters such as frames are written exactly),
floats are float64. Values must be finite scalars and steps must not go back:
a NaN or a rewound step raises ValueError instead of being written.

Reopening a logdir continues its metrics.jsonl: the last step already in it
is the floor for new writes (a rewound step raises). A resumed run passes
resume_step (its checkpoint's step): rows after it, written by the preempted
run past its last checkpoint, are dropped (logged) so the resumed run does
not duplicate them. A gs:// logdir is supported: lines are kept in memory and
the whole file is rewritten every `flush_every` writes and on close (GCS has
no append), so the bytes written grow with the square of the run's rows; a
long run on gs:// should use a larger flush_every.

Run settings are not metrics. MetricsWriter.params({name: value}) records
constants of the run (identity, hardware, a flat schedule's value, a static
batch size) in <logdir>/params.json and in every sink's params(), never as a
series. write(..., held=keys) routes the named keys of a row through the same
record: a held key's first value goes to params, later equal values are not
written, and the first value that differs makes the key a series from that
write on (its params entry becomes {"initial": v0, "moved_at_step": s},
logged), so a schedule that moves is still a curve and a setting that never
moves is one params entry.

Failures. The logdir's record (metrics.jsonl, params.json) is the writer's
own: a failure to write it raises. Every other sink is telemetry, and
telemetry never stops a run: a sink whose write, params, flush or close
raises is logged with its traceback, not raised, and the writer goes on
with the other sinks and the record.

register_params(values) publishes settings outside any writer (e.g. the
hardware a process got, before its run starts) to every registered params
publisher (register_params_publisher); with none it only validates and logs.

Series records: emit_series(step, values, step_unit=, mean=) is one
validated record of a script that is not a training run, such as a benchmark:
{"step", ["step_unit"], name: value..., ["mean": {"name", "n", "sum",
"sumsq"}]}, where a mean carries what its confidence band needs. It is
logged and handed to every registered series emitter
(register_series_emitter), for example one that forwards records to a
tracking service.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from absl import logging
from etils import epath
import numpy as np

from open_spiel.python.scale import plugins

JSONL_NAME = "metrics.jsonl"
PARAMS_NAME = "params.json"


def _scalar(name: str, v: Any) -> Any:
  """A finite Python int or float from a scalar (np/jax/python); else raise.

  Args:
    name: The metric name.
    v: The value.

  Returns:
    The int or float.

  Raises:
    ValueError: The value is not a finite numeric scalar.
  """
  arr = np.asarray(v)
  if arr.shape:
    raise ValueError(f"metric {name!r}: want a scalar, got shape {arr.shape}")
  if arr.dtype == np.bool_:
    return int(arr)
  if np.issubdtype(arr.dtype, np.integer):
    return int(arr)
  if not np.issubdtype(arr.dtype, np.floating):
    raise ValueError(f"metric {name!r}: want a numeric scalar, got {arr.dtype}")
  f = float(arr)
  if not math.isfinite(f):
    raise ValueError(f"metric {name!r} is not finite: {f}")
  return f


def _param_value(name: str, v: Any) -> Any:
  """A JSON value for params.

  Scalars as _scalar (finite), str / bool / None as is, dicts and lists
  element-wise.

  Args:
    name: the param's name, for the error message.
    v: the value.

  Returns:
    the JSON value.
  """
  if v is None or isinstance(v, (str, bool)):
    return v
  if isinstance(v, Mapping):
    return {str(k): _param_value(f"{name}.{k}", x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [_param_value(f"{name}[{i}]", x) for i, x in enumerate(v)]
  return _scalar(name, v)


# -- the sink interface --------------------------------------------------------
class MetricSink:
  """Where rows go.

  write gets {prefixed name: finite int | float}; params gets {name: JSON
  value}. The defaults of params / flush / close do nothing.
  """

  def write(self, step: int, values: Mapping[str, Union[int, float]]) -> None:
    """Records one row.

    Args:
      step: The row's step.
      values: {prefixed name: finite int | float}.
    """
    raise NotImplementedError

  def params(self, values: Mapping[str, Any]) -> None:
    """Records run settings.

    Args:
      values: {name: JSON value}.
    """
    del values

  def flush(self) -> None:
    """Flushes buffered rows."""

  def close(self) -> None:
    """Flushes and releases the sink."""


@dataclasses.dataclass(frozen=True)
class SinkContext:
  """What a sink factory knows about the writer it serves.

  Attributes:
    logdir: the writer's log directory, or None.
    tensorboard: whether the writer was asked for TensorBoard.
    resume_step: the step a resumed run continues from, or None.
  """

  logdir: Optional[str]
  tensorboard: bool = False
  resume_step: Optional[int] = None


SinkFactory = Callable[[SinkContext], Optional[MetricSink]]
_NAMED: dict[str, SinkFactory] = {}
_AUTO: list[SinkFactory] = []
_PARAMS_PUBLISHERS: list[Callable[[Mapping[str, Any]], bool]] = []
_SERIES_EMITTERS: list[Callable[[Mapping[str, Any]], None]] = []


def register_sink(name: str, factory: SinkFactory) -> None:
  """Registers a sink chosen by name.

  The name is what MetricsWriter(sinks=[name]) and train --metric_sinks take.

  Args:
    name: The sink name.
    factory: fn(SinkContext) -> the sink.

  Raises:
    ValueError: another factory is registered under the name.
  """
  assert isinstance(name, str) and name and callable(factory), (name, factory)
  if name in _NAMED and _NAMED[name] is not factory:
    raise ValueError(f"metric sink {name!r} is already registered")
  _NAMED[name] = factory


def register_auto_sink(factory: SinkFactory) -> None:
  """Registers a factory every MetricsWriter calls.

  Args:
    factory: fn(SinkContext) -> a sink, or None when it does not apply.
  """
  assert callable(factory), factory
  if factory not in _AUTO:
    _AUTO.append(factory)


def register_params_publisher(fn: Callable[[Mapping[str, Any]], bool]) -> None:
  """Registers a params publisher (register_params).

  Args:
    fn: fn(values) -> whether it recorded them.
  """
  assert callable(fn), fn
  if fn not in _PARAMS_PUBLISHERS:
    _PARAMS_PUBLISHERS.append(fn)


def register_series_emitter(fn: Callable[[Mapping[str, Any]], None]) -> None:
  """Registers a series emitter.

  Args:
    fn: Called as fn(record) for every emit_series record.
  """
  assert callable(fn), fn
  if fn not in _SERIES_EMITTERS:
    _SERIES_EMITTERS.append(fn)


def sink_names() -> tuple[str, ...]:
  """The registered sink names, plugins loaded first.

  Returns:
    The names, sorted.
  """
  plugins.load_env()
  return tuple(sorted(_NAMED))


# -- the built-in sinks --------------------------------------------------------
class JsonlSink(MetricSink):
  """<logdir>/metrics.jsonl and <logdir>/params.json (module doc).

  Attributes:
    logdir: the log directory.
    last_step: the last step in the record, or None.
  """

  def __init__(
      self,
      logdir: str,
      *,
      flush_every: int = 50,
      resume_step: Optional[int] = None,
  ):
    assert flush_every >= 1, flush_every
    assert resume_step is None or int(resume_step) >= 0, resume_step
    self.logdir = str(logdir).rstrip("/")
    self._gcs = self.logdir.startswith("gs://")
    self._flush_every = flush_every
    self._pending = 0
    self._lines: list[str] = []
    self.last_step: Optional[int] = None
    root = epath.Path(self.logdir)
    root.mkdir(parents=True, exist_ok=True)
    self._path = root / JSONL_NAME
    self._params_path = root / PARAMS_NAME
    if self._path.exists():
      self._reopen(resume_step)

  def _reopen(self, resume_step: Optional[int]) -> None:
    """Picks up an existing record, dropping its rows after `resume_step`.

    Args:
      resume_step: the step a resumed run continues from, or None to keep every
        row.
    """
    lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
    steps = [int(json.loads(ln)["step"]) for ln in lines]
    if resume_step is not None:
      keep = [i for i, st in enumerate(steps) if st <= int(resume_step)]
      if len(keep) != len(lines):
        logging.warning(
            "metrics: dropping %d rows after resume step %d from %s",
            len(lines) - len(keep),
            int(resume_step),
            self._path,
        )
        lines, steps = [lines[i] for i in keep], [steps[i] for i in keep]
        self._write_atomic(self._path, "".join(ln + "\n" for ln in lines))
    self.last_step = max(steps) if steps else None
    if self._gcs:
      self._lines = lines

  def read_params(self) -> dict[str, Any]:
    """params.json's record ({} when absent).

    A params.json that is not valid JSON or not a JSON object is logged and the
    record starts empty: telemetry never stops a run.

    Returns:
      The record.
    """
    path = self._params_path
    if not path.exists():
      return {}
    try:
      doc = json.loads(path.read_text())
    except ValueError:
      logging.exception(
          "metrics: %s is not valid JSON; starting an empty params record", path
      )
      return {}
    if not isinstance(doc, dict):
      logging.error(
          "metrics: %s is not a JSON object; starting an empty params record",
          path,
      )
      return {}
    return doc

  def write(self, step, values):
    """Records one row (MetricSink.write).

    Args:
      step: The row's step.
      values: {prefixed name: finite int | float}.
    """
    row = {"step": int(step), "time": time.time(), **values}
    line = json.dumps(row, allow_nan=False, sort_keys=True)
    if self._gcs:
      self._lines.append(line)
      self._pending += 1
      if self._pending >= self._flush_every:
        self.flush()
    else:
      with open(str(self._path), "a", encoding="utf-8") as f:
        f.write(line + "\n")

  def params(self, values):
    """Records run settings (MetricSink.params).

    Args:
      values: {name: JSON value}.
    """
    doc = self.read_params()
    doc.update(values)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    self._write_atomic(self._params_path, text)

  def _write_atomic(self, path, text: str) -> None:
    """Writes `path` whole: a preemption never leaves a half-written file."""
    if self._gcs:
      path.write_text(text)  # a GCS object write is atomic
    else:  # write then rename
      tmp = path.parent / (path.name + ".tmp")
      tmp.write_text(text)
      os.replace(str(tmp), str(path))

  def flush(self):
    """Flushes buffered rows (MetricSink.flush)."""
    if self._gcs and self._pending:
      self._path.write_text("\n".join(self._lines) + "\n")
      self._pending = 0

  def close(self):
    """Flushes and releases the sink (MetricSink.close)."""
    self.flush()


#: Modules providing a tensorboardX-compatible SummaryWriter, tried in order.
_TENSORBOARD_WRITERS: list[str] = ["tensorboardX"]


def register_tensorboard_writer(module: str) -> None:
  """Adds a module whose SummaryWriter the tensorboard sink may use.

  It is tried after the ones already registered.

  Args:
    module: The module's import name.
  """
  assert isinstance(module, str) and module, module
  if module not in _TENSORBOARD_WRITERS:
    _TENSORBOARD_WRITERS.append(module)


def _summary_writer_class():
  """The first importable tensorboard SummaryWriter.

  Returns:
    The class, or None.
  """
  for mod in _TENSORBOARD_WRITERS:
    try:
      return importlib.import_module(mod).SummaryWriter
    except ImportError:
      continue
  return None


def tensorboard_available() -> bool:
  """Whether a tensorboard writer module imports, plugins loaded first.

  Returns:
    True when the tensorboard sink can be built.
  """
  plugins.load_env()
  return _summary_writer_class() is not None


class TensorBoardSink(MetricSink):
  """TensorBoard scalars under logdir, through the first importable writer.

  The writer module is tensorboardX, or one registered with
  register_tensorboard_writer.
  """

  def __init__(self, logdir: str):
    writer = _summary_writer_class()
    if writer is None:
      raise ImportError(
          f"the tensorboard sink needs one of {_TENSORBOARD_WRITERS} "
          "(pip install tensorboardX); none is importable"
      )
    self._tb = writer(str(logdir))

  def write(self, step, values):
    """Records one row (MetricSink.write).

    Args:
      step: The row's step.
      values: {prefixed name: finite int | float}.
    """
    for k, v in values.items():
      self._tb.add_scalar(k, v, int(step))

  def flush(self):
    """Flushes buffered rows (MetricSink.flush)."""
    self._tb.flush()

  def close(self):
    """Flushes and releases the sink (MetricSink.close)."""
    self._tb.close()


class WandbSink(MetricSink):
  """Weights & Biases: one wandb run.

  Rows go through wandb.log(step=), settings into its config. Needs the
  optional wandb package (pip install wandb); project, entity and mode come
  from wandb's own settings (WANDB_PROJECT, ...).
  """

  def __init__(self, logdir: Optional[str] = None, **init_kwargs: Any):
    try:
      # Lazy: optional dependency (the wandb sink).
      import wandb  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
    except ImportError as e:
      raise ImportError(
          "the wandb sink needs the optional wandb package (pip install wandb)"
      ) from e
    self._wandb = wandb
    kw = dict(init_kwargs)
    if logdir is not None and not str(logdir).startswith("gs://"):
      kw.setdefault("dir", str(logdir))
    self._run = wandb.init(**kw)

  def write(self, step, values):
    """Records one row (MetricSink.write).

    Args:
      step: The row's step.
      values: {prefixed name: finite int | float}.
    """
    self._run.log(dict(values), step=int(step))

  def params(self, values):
    """Records run settings (MetricSink.params).

    Args:
      values: {name: JSON value}.
    """
    self._run.config.update(dict(values), allow_val_change=True)

  def close(self):
    """Flushes and releases the sink (MetricSink.close)."""
    self._run.finish()


def _tensorboard_factory(ctx: SinkContext) -> MetricSink:
  """A tensorboard sink.

  Args:
    ctx: The sink context.

  Returns:
    The sink.

  Raises:
    ValueError: There is no logdir.
  """
  if ctx.logdir is None:
    raise ValueError("the tensorboard sink needs a logdir")
  return TensorBoardSink(ctx.logdir)


def _wandb_factory(ctx: SinkContext) -> MetricSink:
  """A wandb sink.

  Args:
    ctx: The sink context.

  Returns:
    The sink.
  """
  return WandbSink(ctx.logdir)


register_sink("tensorboard", _tensorboard_factory)
register_sink("wandb", _wandb_factory)


# -- params outside a writer ---------------------------------------------------
def register_params(values: Mapping[str, Any]) -> bool:
  """Validates, logs and publishes run settings.

  The settings go to every params publisher (register_params_publisher). A
  publisher that raises is logged, not raised (telemetry never stops a run).

  Args:
    values: {name: value}.

  Returns:
    Whether a publisher recorded them.

  Raises:
    ValueError: there are no values, or a value is not a finite scalar or
      JSON value.
  """
  plugins.load_env()
  vals = {str(k): _param_value(str(k), v) for k, v in values.items()}
  if not vals:
    raise ValueError("register_params: no params")
  logging.info("[params] %s", json.dumps(vals, sort_keys=True))
  wrote = False
  for fn in list(_PARAMS_PUBLISHERS):
    try:
      wrote = bool(fn(vals)) or wrote
    # Telemetry never stops the run: any publisher failure is logged.
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "[params] publishing %s through %r failed (telemetry never stops the "
          "run)",
          sorted(vals),
          fn,
      )
  return wrote


_MOVED = (
    object()
)  # a held key whose value moved: written as a series from then on


class MetricsWriter:
  """Validates metric rows and sends them to logging, a record and sinks.

  The record is a JSONL file; see the module doc.

  Attributes:
    logdir: the log directory, or None.
  """

  def __init__(
      self,
      logdir: Optional[str],
      *,
      tensorboard: bool = False,
      flush_every: int = 50,
      resume_step: Optional[int] = None,
      sinks: Sequence[Union[str, MetricSink]] = (),
  ):
    """Opens the writer; the registered auto sinks are added (plugins first).

    Args:
      logdir: The JsonlSink's dir (None: logging plus the given sinks only).
      tensorboard: Also the "tensorboard" sink.
      flush_every: The JsonlSink's flush_every.
      resume_step: The resumed run's checkpoint step (module doc).
      sinks: Sink objects or registered names.

    Raises:
      KeyError: a sink name is not registered.
    """
    plugins.load_env()
    assert resume_step is None or int(resume_step) >= 0, resume_step
    self._held: dict[str, Any] = {}  # held key -> its recorded value, or _MOVED
    self.logdir = None if logdir is None else str(logdir).rstrip("/")
    ctx = SinkContext(self.logdir, bool(tensorboard), resume_step)
    self._jsonl = (
        JsonlSink(self.logdir, flush_every=flush_every, resume_step=resume_step)
        if self.logdir is not None
        else None
    )
    self._last_step = None if self._jsonl is None else self._jsonl.last_step
    names = list(sinks)
    if tensorboard and "tensorboard" not in names:
      names.append("tensorboard")
    self._sinks: list[MetricSink] = []
    for factory in list(_AUTO):
      got = factory(ctx)
      if got is not None:
        self._sinks.append(got)
    for s in names:
      if isinstance(s, str):
        if s not in _NAMED:
          raise KeyError(
              f"unknown metric sink {s!r}; registered: {sorted(_NAMED)}"
          )
        self._sinks.append(_NAMED[s](ctx))
      else:
        assert isinstance(s, MetricSink) or all(
            callable(getattr(s, m, None))
            for m in ("write", "params", "flush", "close")
        ), s
        self._sinks.append(s)
    if self._jsonl is not None:
      self._reopen_params()

  def _reopen_params(self) -> None:
    """A resumed run keeps its settings.

    The held values are the ones params.json records (a moved key stays a
    series), so an unchanged setting is not re-recorded.
    """
    doc = self._jsonl.read_params()
    for k, v in doc.items():
      self._held[k] = (
          _MOVED if isinstance(v, dict) and "moved_at_step" in v else v
      )
    if doc:
      # A resumed attempt may report to a different sink destination: the
      # sinks get the record too, settings that never moved included (they
      # are not written again on this attempt).
      self._sink_params(doc)

  def _each_sink(self, method: str, *args: Any) -> None:
    """Calls sink.method(*args) on every sink.

    A failure is logged, not raised (module doc, Failures).

    Args:
      method: the sink method's name.
      *args: its arguments.
    """
    for s in self._sinks:
      try:
        getattr(s, method)(*args)
      # Telemetry never stops the run: any sink failure is logged.
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "metrics: sink %r failed in %s (telemetry never stops the run; "
            "the logdir's record is unaffected)",
            s,
            method,
        )

  def _sink_params(self, vals: Mapping[str, Any]) -> None:
    self._each_sink("params", vals)

  def params(self, values: Mapping[str, Any]) -> None:
    """Records run constants (module doc).

    They go to <logdir>/params.json (merged) and every sink's params; never a
    series.

    Args:
      values: {name: value}.

    Raises:
      ValueError: there are no values, or a value is not a finite scalar or
        JSON value.
    """
    vals = {str(k): _param_value(str(k), v) for k, v in values.items()}
    if not vals:
      raise ValueError("params: no values")
    logging.info("[params] %s", json.dumps(vals, sort_keys=True))
    self._sink_params(vals)
    if self._jsonl is not None:
      self._jsonl.params(vals)

  def _split_held(
      self, step: int, vals: dict[str, Any], held: frozenset[str]
  ) -> dict[str, Any]:
    """Vals minus the held keys that are still settings (module doc).

    Args:
      step: The step.
      vals: The values.
      held: The keys held as settings.

    Returns:
      The values to write as rows.
    """
    first, moved = {}, {}
    out = {}
    for k, v in vals.items():
      if k not in held:
        out[k] = v
        continue
      rec = self._held.get(k)
      if rec is _MOVED:
        out[k] = v
      elif rec is None:
        self._held[k] = v
        first[k] = v
      elif rec != v:
        logging.info(
            "metrics: %s changed from %r to %r at step %d: a series from here "
            "on",
            k,
            rec,
            v,
            step,
        )
        self._held[k] = _MOVED
        moved[k] = {"initial": rec, "moved_at_step": step}
        out[k] = v
    if first or moved:
      self.params({**first, **moved})
    return out

  def write(
      self,
      step: int,
      metrics: Mapping[str, Any],
      prefix: str = "train",
      held: Any = (),
  ) -> None:
    """Writes one row of metrics.

    Args:
      step: The row's step, a non-negative integer that never goes back.
      metrics: {name: finite scalar}, unprefixed.
      prefix: The prefix of every name ("train/...").
      held: Keys of `metrics` (unprefixed) that are run settings (module doc):
        recorded in params until their value moves.

    Raises:
      ValueError: the step is negative or goes back, or a value is not a
        finite scalar.
    """
    step_v = _scalar("step", step)
    if not isinstance(step_v, int) or step_v < 0:
      raise ValueError(f"step must be a non-negative integer, got {step!r}")
    if self._last_step is not None and step_v < self._last_step:
      raise ValueError(
          f"step {step_v} goes back (last written {self._last_step})"
      )
    assert isinstance(prefix, str) and prefix and "/" not in prefix, prefix
    assert not isinstance(
        held, str
    ), "held is a collection of keys, not one string"
    vals = {f"{prefix}/{k}": _scalar(k, v) for k, v in metrics.items()}
    if held:
      vals = self._split_held(
          step_v, vals, frozenset(f"{prefix}/{k}" for k in held)
      )
      if not vals:
        self._last_step = step_v
        return
    self._last_step = step_v
    shown = " ".join(
        f"{k.split('/', 1)[1]}={v:.6g}"
        if isinstance(v, float)
        else f"{k.split('/', 1)[1]}={v}"
        for k, v in sorted(vals.items())
    )
    logging.info("[%s %d] %s", prefix, step_v, shown)
    self._each_sink("write", step_v, vals)
    if self._jsonl is not None:
      self._jsonl.write(step_v, vals)

  def flush(self) -> None:
    """Flushes every sink and the record."""
    self._each_sink("flush")
    if self._jsonl is not None:
      self._jsonl.flush()

  def close(self) -> None:
    """Flushes, then closes every sink and the record."""
    self.flush()
    self._each_sink("close")
    self._sinks = []
    if self._jsonl is not None:
      self._jsonl.close()


def read_jsonl(path: str) -> list[dict[str, Any]]:
  """Every row of a metrics.jsonl.

  Args:
    path: The file (local or gs://).

  Returns:
    The rows.
  """
  text = epath.Path(path).read_text()
  return [json.loads(line) for line in text.splitlines() if line.strip()]


# -- series lines --------------------------------------------------------------
_STEP_UNIT_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-"
)
# The record's own keys: a metric with one of these names would overwrite them.
RESERVED_SERIES_NAMES = ("step", "step_unit", "mean")


def series_record(
    step: int,
    values: Mapping[str, Any],
    *,
    step_unit: Optional[str] = None,
    mean: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
  """One validated series record.

  Raises on anything a consumer would have to drop. An emitter whose backend
  restricts names further checks them itself.

  Args:
    step: An int >= 0.
    values: {name: finite scalar}; names are non-blank, without surrounding
      whitespace, and not one of RESERVED_SERIES_NAMES (the record's own
      keys).
    step_unit: A short word.
    mean: {"name", "n", "sum", "sumsq"}.

  Returns:
    The record.

  Raises:
    ValueError: the step, step_unit, a name, a value or the mean is invalid,
      or there is neither a value nor a mean.
  """
  if (
      isinstance(step, bool)
      or not isinstance(step, (int, np.integer))
      or int(step) < 0
  ):
    raise ValueError(f"series step must be an int >= 0, got {step!r}")
  out: dict[str, Any] = {"step": int(step)}
  if step_unit is not None:
    if (
        not isinstance(step_unit, str)
        or not 1 <= len(step_unit) <= 24
        or not step_unit[0].isalpha()
        or set(step_unit) - _STEP_UNIT_CHARS
    ):
      raise ValueError(
          f"series step_unit must be a short word, got {step_unit!r}"
      )
    out["step_unit"] = step_unit
  for name, v in values.items():
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name in RESERVED_SERIES_NAMES
    ):
      raise ValueError(f"series metric name {name!r} is not allowed")
    out[name] = _scalar(name, v)
  if mean is not None:
    if set(mean) != {"name", "n", "sum", "sumsq"}:
      raise ValueError(
          f"series mean wants exactly name, n, sum, sumsq; got {sorted(mean)}"
      )
    n = _scalar("mean.n", mean["n"])
    if not isinstance(n, int) or n < 1:
      raise ValueError(f"series mean n must be an int >= 1, got {mean['n']!r}")
    out["mean"] = {
        "name": str(mean["name"]),
        "n": n,
        "sum": float(_scalar("mean.sum", mean["sum"])),
        "sumsq": float(_scalar("mean.sumsq", mean["sumsq"])),
    }
  if len(out) == 1 + (step_unit is not None):
    raise ValueError("a series record needs at least one value or a mean")
  return out


def emit_series(step: int, values: Mapping[str, Any], **kw) -> dict[str, Any]:
  """series_record(...), logged and handed to every series emitter.

  An emitter that raises is logged, not raised (telemetry never stops a run).

  Args:
    step: The record's step.
    values: {name: finite scalar}.
    **kw: series_record's keyword arguments (step_unit, mean).

  Returns:
    The record.
  """
  plugins.load_env()
  rec = series_record(step, values, **kw)
  logging.info("[series] %s", json.dumps(rec, sort_keys=True))
  for fn in list(_SERIES_EMITTERS):
    try:
      fn(rec)
    # Telemetry never stops the run: any emitter failure is logged.
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "[series] emitting step %d through %r failed (telemetry never "
          "stops the run)",
          rec["step"],
          fn,
      )
  return rec
