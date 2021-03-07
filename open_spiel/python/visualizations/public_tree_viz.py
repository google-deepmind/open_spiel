# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualizing public trees with graphviz.

GamePublicTree builds a `pygraphviz.AGraph` reprensentation of the public tree.
It extends the module `treeviz`, look there for more details on how to install
dependencies and advanced usage.

GamePublicTree allows to draw relationships between public and world trees
as well. This can be useful in visualizing games that implement factored
observations and PublicState API.
"""

from absl import logging
from open_spiel.python.visualizations.treeviz import GameTree

# pylint: disable=g-import-not-at-top
try:
  import pygraphviz
except (ImportError, Exception) as e:
  raise ImportError(
      str(e) + "\nPlease make sure to install the following dependencies:\n"
      "sudo apt-get install graphviz libgraphviz-dev\n"
      "pip install pygraphviz")
# pylint: enable=g-import-not-at-top

_PLAYER_SHAPES = {0: "square", 1: "ellipse"}
_PLAYER_COLORS = {-1: "black", 0: "blue", 1: "red"}
_FONTSIZE = 8
_WIDTH = _HEIGHT = 0.25
_ARROWSIZE = .5
_MARGIN = 0.01


def default_public_state_decorator(public_state):
  """Decorates a public state - node of the public tree.

  This method can be called by a custom decorator to prepopulate the attributes
  dictionary. Then only relevant attributes need to be changed, or added.

  Args:
    public_state: The public_state.

  Returns:
    `dict` with graphviz node style attributes.
  """
  attrs = {
      "label": "",
      "fontsize": _FONTSIZE,
      "width": _WIDTH,
      "height": _HEIGHT,
      "margin": _MARGIN
  }
  if public_state.is_terminal():
    attrs["shape"] = "diamond"
  else:
    attrs["label"] = str(public_state.get_public_observation_history())
    attrs["shape"] = "point"
    attrs["width"] = _WIDTH / 2.
    attrs["height"] = _HEIGHT / 2.

  if public_state.is_root():
    attrs["label"] = public_state.get_public_observation_history()[0]
    attrs["shape"] = "circle"
  return attrs


# The decorator can be "overriden", so pylint: disable=unused-argument
def default_transition_decorator(parent, child, transition):
  """Decorates a public transition - an edge of the public tree.

  This method can be called by a custom decorator to prepopulate the attributes
  dictionary. Then only relevant attributes need to be changed, or added.

  Args:
    parent: The parent public state.
    child: The child public state.
    transition: `string` the selected transition in the parent state.

  Returns:
    `dict` with graphviz node style attributes.
  """
  attrs = {
      "label": " " + transition,
      "fontsize": _FONTSIZE,
      "arrowsize": _ARROWSIZE
  }
  attrs["color"] = "black"
  return attrs


def default_public_to_base_decorator(public_state, state):
  """Decorates an edge going from public state to the public set.

  This method can be called by a custom decorator to prepopulate the attributes
  dictionary. Then only relevant attributes need to be changed, or added.

  Args:
    public_state: The public state.
    state: A state within the public set (i.e. the cluster of states)

  Returns:
    `dict` with graphviz node style attributes.
  """
  attrs = {
      "label": "",
      "fontsize": _FONTSIZE,
      "arrowsize": _ARROWSIZE,
      "penwidth": 7,
      "color": "#88888888",
      "style": "solid",
      "splines": False,
      "constraint": False,
  }
  return attrs


class GamePublicTree(GameTree, pygraphviz.AGraph):
  """Builds `pygraphviz.AGraph` of the game tree.

  Attributes:
    public_game: A `pyspiel.GameWithPublicStates` object.
    draw_world: Should we draw the world tree? This is accomplished by calling
      drawing of treeviz.GameTree
    target_public_to_base: If we draw the world tree, we can draw an arrow that
      connects the public tree and world tree. This is done only in one place,
      because it would clutter the graph.
    depth_limit: Maximum depth of the tree. Optional, default=-1 (no limit).
    public_state_decorator: Decorator function for nodes (public states).
      Optional, default=`public_tree_viz.default_public_state_decorator`.
    transition_decorator: Decorator function for edges (public transitions).
      Optional, default=`public_tree_viz.default_transition_decorator`.
    public_to_base_decorator: Decorator function for arrows (between public tree
      and the world tree). Optional,
      default=`public_tree_viz.default_public_to_base_decorator`.
    kwargs: Keyword arguments passed on to `pygraphviz.AGraph.__init__` and to
      `pyspiel.treeviz.GameTree.__init__`.
  """

  def __init__(self,
               public_game=None,
               draw_world=True,
               target_public_to_base=None,
               depth_limit=-1,
               public_state_decorator=default_public_state_decorator,
               transition_decorator=default_transition_decorator,
               public_to_base_decorator=default_public_to_base_decorator,
               **kwargs):

    kwargs["directed"] = kwargs.get("directed", True)
    if draw_world:
      super(GamePublicTree, self).__init__(
          public_game.get_base_game() if public_game else None,
          depth_limit=depth_limit,
          target_pubset=target_public_to_base,
          **kwargs)
    else:
      super(GamePublicTree, self).__init__()

    # We use pygraphviz.AGraph.add_subgraph to cluster nodes, and it requires a
    # default constructor. Thus game needs to be optional.
    if public_game is None:
      return

    self.public_game = public_game
    self._public_state_decorator = public_state_decorator
    self._transition_decorator = transition_decorator
    self._public_to_base_decorator = public_to_base_decorator

    root_public_state = public_game.new_initial_public_state()
    self.add_node(
        self.public_state_to_str(root_public_state),
        **self._public_state_decorator(root_public_state))
    self._build_public_tree(root_public_state, depth_limit)

    if target_public_to_base:
      found = False
      for pubset, sibblings in self._pubsets.items():
        if target_public_to_base == pubset:
          found = True
          # Let's find that public state
          transitions = pubset.split(",")
          public_state = public_game.new_initial_public_state()
          for transition in transitions[1:]:
            public_state.apply_public_transition(transition)

          state_str = str(public_state.get_public_observation_history())
          state = sibblings[-1]

          self.add_edge(state_str, state,
                        **self._public_to_base_decorator(public_state, state))
      if not found:
        logging.warning(
            "Could not find target public state '%s' "
            "Did you mean one of these?\n%s", target_public_to_base,
            "\n".join(self._pubsets.keys()))

  def public_state_to_str(self, public_state):
    """Unique string representation of a public state.

    Args:
      public_state: The public state.

    Returns:
      String representation of public state.
    """
    return str(public_state.get_public_observation_history())

  def _build_public_tree(self, public_state, depth_limit):
    """Recursively builds the game tree."""
    state_str = self.public_state_to_str(public_state)

    if public_state.is_terminal():
      return
    if public_state.move_number() > depth_limit >= 0:
      return

    for transition in public_state.legal_transitions():
      child = public_state.child(transition)
      child_str = self.public_state_to_str(child)
      self.add_node(child_str, **self._public_state_decorator(child))
      self.add_edge(
          state_str, child_str,
          **self._transition_decorator(public_state, child, transition))

      self._build_public_tree(child, depth_limit)

  def _repr_svg_(self):
    """Allows to render directly in Jupyter notebooks and Google Colab."""
    if not self.has_layout:
      self.layout(prog="dot")
    return self.draw(format="svg").decode(self.encoding)
