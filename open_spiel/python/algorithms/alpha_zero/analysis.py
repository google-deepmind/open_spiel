# Copyright 2019 DeepMind Technologies Limited
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

# Lint as: python3
"""Output the config and graphs for an experiment.

This reads the config.json and learner.jsonl from an alpha zero experiment.
"""

import datetime
import json
import math
import os

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from open_spiel.python.utils import gfile

X_AXIS = {
    "step": "step",
    "time": "time_rel_h",
    "states": "total_states",
}

flags.DEFINE_string("path", None,
                    "Where to find config.json and learner.jsonl.")
flags.DEFINE_enum("x_axis", "step", list(X_AXIS.keys()),
                  "What should be on the x-axis.")

flags.mark_flag_as_required("path")
FLAGS = flags.FLAGS

MAX_WIDTH = int(os.getenv("COLUMNS", "200"))  # Get your TTY width.
SMOOTHING_RATE = 10
SUBSAMPLING_MAX = 200


def print_columns(strings, max_width=MAX_WIDTH):
  """Prints a list of strings in columns."""
  padding = 2
  shortest = min(len(s) for s in strings)
  max_columns = max(1, math.floor((max_width - 1) / (shortest + 2 * padding)))
  for cols in range(max_columns, 0, -1):
    rows = math.ceil(len(strings) / cols)
    chunks = [strings[i:i + rows] for i in range(0, len(strings), rows)]
    col_widths = [max(len(s) for s in chunk) for chunk in chunks]
    if sum(col_widths) + 2 * padding * len(col_widths) <= max_width:
      break
  for r in range(rows):
    for c in range(cols):
      i = r + c * rows
      if i < len(strings):
        print(" " * padding + strings[i].ljust(col_widths[c] + padding), end="")
    print()


def load_jsonl_data(filename):
  with gfile.Open(filename) as f:
    return [json.loads(l) for l in f.readlines()]


def sub_sample(data, count):
  return data[::(max(1, len(data) // count))]


def smooth(data, count):
  for k in data.keys():
    if not isinstance(k, str) or not k.startswith("time_"):
      data[k] = data[k].rolling(max(1, len(data) // count)).mean()
  return data


def subselect(row, keys):
  for key in keys:
    row = row[key]
  return row


def select(data, keys):
  return [subselect(row, keys) for row in data]


def prepare(data, cols):
  """Given the dataset and a list of columns return a small pandas dataframe."""
  for col in ["step", "total_states", "total_trajectories", "time_rel"]:
    cols[col] = [col]
  subdata = {key: select(data, col) for key, col in cols.items()}
  # subdata = list(zip(*subdata))  # transpose
  df = pd.DataFrame(subdata)
  df = smooth(df, SMOOTHING_RATE)
  df = sub_sample(df, SUBSAMPLING_MAX)
  df["time_rel_h"] = df["time_rel"] / 3600
  df["zero"] = 0
  return df


def subplot(rows, cols, pos, *args, **kwargs):
  ax = plt.subplot(rows, cols, pos, *args, **kwargs)
  ax.tick_params(top=False, right=False)  # Don't interfere with the titles.
  return ax


def plot_avg_stddev(ax, x, data, data_col):
  """Plot stats produced by open_spiel::BasicStats::ToJson."""
  cols = ["avg", "std_dev", "min", "max"]
  df = prepare(data, {v: data_col + [v] for v in cols})
  df.plot(ax=ax, x=x, y="avg", color="b")
  plt.fill_between(
      x=df[x], color="b", alpha=0.2, label="std dev",
      y1=np.nanmax([df["min"], df["avg"] - df["std_dev"]], 0),
      y2=np.nanmin([df["max"], df["avg"] + df["std_dev"]], 0))
  plt.fill_between(
      x=df[x], color="b", alpha=0.2, label="min/max",
      y1=df["min"], y2=df["max"])
  plot_zero(df, ax, x)


def plot_histogram_numbered(ax, x, data, data_col):
  """Plot stats produced by open_spiel::HistogramNumbered::ToJson."""
  x_min, x_max = 0, data[-1][x]
  y_min, y_max = 0, len(subselect(data, [0] + data_col))
  z_min, z_max = 0, 1
  z = np.array([subselect(row, data_col) for row in data], dtype=float)
  z = np.concatenate((z, np.zeros((x_max, 1))), axis=1)  # Don't cut off the top
  # TODO(author7): smoothing
  z = sub_sample(z, SUBSAMPLING_MAX).transpose()
  p = np.percentile(z, 99)
  if p > 0:
    z /= p
    z[z > 1] = 1
  ax.grid(False)
  ax.imshow(z, cmap="Reds", vmin=z_min, vmax=z_max,
            extent=[x_min, x_max, y_min, y_max + 1],
            interpolation="nearest", origin="lower", aspect="auto")


def plot_histogram_named(ax, x, data, data_col, normalized=True):
  """Plot stats produced by open_spiel::HistogramNamed::ToJson."""
  names = subselect(data, [0] + data_col + ["names"])
  df = prepare(data, {name: data_col + ["counts", i]
                      for i, name in enumerate(names)})
  if normalized:
    total = sum(df[n] for n in names)
    for n in names:
      df[n] /= total
  df.plot.area(ax=ax, x=x, y=names)


def plot_zero(df, ax, x):
  df.plot(ax=ax, x=x, y="zero", label="", visible=False)


def plot_data(config, data):
  """Plot a bunch of graphs from an alphazero experiment."""
  num_rows, num_cols = 3, 4
  x = X_AXIS[FLAGS.x_axis]

  fig = plt.figure(figsize=(num_cols * 7, num_rows * 6))
  fig.suptitle(
      ("Game: {}, Model: {}({}, {}), training time: {}, training steps: {}, "
       "states: {}, games: {}").format(
           config["game"], config["nn_model"], config["nn_width"],
           config["nn_depth"],
           datetime.timedelta(seconds=int(data[-1]["time_rel"])),
           int(data[-1]["step"]), int(data[-1]["total_states"]),
           int(data[-1]["total_trajectories"])))

  cols = ["value", "policy", "l2reg", "sum"]
  df = prepare(data, {v: ["loss", v] for v in cols})
  ax = subplot(num_rows, num_cols, 1, title="Training loss")
  for y in cols:
    df.plot(ax=ax, x=x, y=y)

  cols = list(range(len(data[0]["value_accuracy"])))
  df = prepare(data, {i: ["value_accuracy", i, "avg"] for i in cols})
  ax = subplot(num_rows, num_cols, 2,  # ylim=(0, 1.05),
               title="MCTS value prediction accuracy")
  for y in cols:
    df.plot(ax=ax, x=x, y=y)

  cols = list(range(len(data[0]["value_prediction"])))
  df = prepare(data, {i: ["value_prediction", i, "avg"] for i in cols})
  ax = subplot(num_rows, num_cols, 3,  # ylim=(0, 1.05),
               title="MCTS absolute value prediction")
  for y in cols:
    df.plot(ax=ax, x=x, y=y)

  cols = list(range(len(data[0]["eval"]["results"])))
  df = prepare(data, {i: ["eval", "results", i] for i in cols})
  ax = subplot(num_rows, num_cols, 4, ylim=(-1, 1),
               title="Evaluation returns vs MCTS+Solver with x10^(n/2) sims")
  ax.axhline(y=0, color="black")
  for y in cols:
    df.plot(ax=ax, x=x, y=y)

  df = prepare(data, {"states_per_s": ["states_per_s"]})
  ax = subplot(num_rows, num_cols, 5, title="Speed of actor state/s")
  df.plot(ax=ax, x=x, y="states_per_s")
  plot_zero(df, ax, x)

  cols = ["requests_per_s", "misses_per_s"]
  df = prepare(data, {v: ["cache", v] for v in cols})
  ax = subplot(num_rows, num_cols, 6, title="Cache requests/s")
  for y in cols:
    df.plot(ax=ax, x=x, y=y)
  plot_zero(df, ax, x)

  cols = ["hit_rate", "usage"]
  df = prepare(data, {v: ["cache", v] for v in cols})
  ax = subplot(num_rows, num_cols, 7, title="Cache usage and hit rate.",
               ylim=(0, 1.05))
  for y in cols:
    df.plot(ax=ax, x=x, y=y)

  ax = subplot(num_rows, num_cols, 8, title="Outcomes", ylim=(0, 1))
  plot_histogram_named(ax, x, data, ["outcomes"])

  ax = subplot(num_rows, num_cols, 9,
               title="Inference batch size + stddev + min/max")
  plot_avg_stddev(ax, x, data, ["batch_size"])

  ax = subplot(num_rows, num_cols, 10, title="Inference batch size")
  plot_histogram_numbered(ax, x, data, ["batch_size_hist"])

  ax = subplot(num_rows, num_cols, 11, title="Game length + stddev + min/max")
  plot_avg_stddev(ax, x, data, ["game_length"])

  ax = subplot(num_rows, num_cols, 12, title="Game length histogram")
  plot_histogram_numbered(ax, x, data, ["game_length_hist"])

  plt.show()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with gfile.Open(os.path.join(FLAGS.path, "config.json")) as f:
    config = json.load(f)
  data = load_jsonl_data(os.path.join(FLAGS.path, "learner.jsonl"))

  print("config:")
  print_columns(sorted("{}: {}".format(k, v) for k, v in config.items()))
  print()
  print("data keys:")
  print_columns(sorted(data[0].keys()))
  print()
  print("training time:", datetime.timedelta(seconds=int(data[-1]["time_rel"])))
  print("training steps: %d" % (data[-1]["step"]))
  print("total states: %d" % (data[-1]["total_states"]))
  print("total trajectories: %d" % (data[-1]["total_trajectories"]))
  print()

  try:
    plot_data(config, data)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
