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

"""Tests for a bot's output: what a remote bot's reply gives, who receives it.

A seat's bot output (its policy, value and what decided its move) goes with
its move in full: to omniscient viewers and the seat's own, and to no other
viewer, so in a game that hides information it cannot show the bot's hand
(bot_output_privacy_test.py checks that with twin deals). These tests read
the JSON the API answers each viewer with, not only the frames.
"""

from collections.abc import Mapping
import http
import json
import threading
from typing import Any

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import api
from open_spiel.python.gui import builds
from open_spiel.python.gui import catalogue as catalogue_lib
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
from open_spiel.python.gui import health as health_lib
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_http
from open_spiel.python.gui import remote_bot_server
from open_spiel.python.gui import session as session_lib
from open_spiel.python.gui import store as store_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
# Text only a bot output carries, so its presence in a body shows it was sent.
_DECIDED_BY = "scripted-endgame-7f3a"
_VALUE = 0.4375


def _seat(index: int) -> contract.Viewer:
  """The viewer of seat `index`."""
  return contract.Viewer(contract.ViewerKind.SEAT, index)


class _Thinking:
  """Plays its likeliest action and replies with a policy over the legal ones.

  The highest action gets the most probability, so the move played is the
  first of the policy.
  """

  state_modes = ("seat_view", "full_state")
  games = None

  def __init__(self) -> None:
    self.extra: dict[str, Any] = {}

  def step(self, mode: str, request: dict[str, Any]) -> dict[str, Any]:
    """Replies with the highest legal action and the policy that chose it."""
    legal = sorted(remote_bot_server.legal_actions(mode, request))
    weights = [index + 1 for index in range(len(legal))]
    total = sum(weights)
    return {
        "action": legal[-1],
        "policy": [
            {"action": action, "label": "bot text", "prob": weight / total}
            for action, weight in zip(legal, weights)
        ],
        "value": _VALUE,
        "decided_by": _DECIDED_BY,
        **self.extra,
    }


def _catalogue() -> catalogue_lib.Catalogue:
  """A catalogue the API does not read for these requests."""
  return catalogue_lib.Catalogue(games=())


def _health() -> health_lib.Health:
  """A health report the API does not read for these requests."""
  return health_lib.Health(
      version="0.0.0",
      revision=None,
      capabilities=health_lib.Capabilities(
          controllers=controllers.registered(),
          chance_policies=tuple(contract.ChancePolicy),
          cpp_mcts_limits=controllers.mcts_limits(),
      ),
  )


class _BotOutputTestCase(absltest.TestCase):
  """Serves a thinking bot and opens sessions against it."""

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    self.addCleanup(builds.set_wait_seconds, builds.wait_seconds())
    builds.set_wait_seconds(10.0)
    self.addCleanup(remote_bot.configure)
    self.bot = _Thinking()
    server = remote_bot_server.Server(("127.0.0.1", 0), self.bot, name="test")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    self.url = server.url

  def seat_config(self, state: str = "seat_view") -> dict[str, Any]:
    """A remote seat's config, playing the thinking bot."""
    return {"url": self.url, "state": state}


class ParseBotOutputTest(absltest.TestCase):

  def parse(self, reply: Mapping[str, Any]) -> contract.BotOutput | None:
    """Parses `reply` for a seat whose legal actions are 0, 1 and 2."""
    return remote_bot_http.parse_bot_output(
        {"action": 0, **reply}, [0, 1, 2], lambda action: f"move {action}"
    )

  def test_a_reply_without_one_gives_none(self) -> None:
    self.assertIsNone(self.parse({}))
    self.assertIsNone(
        self.parse({"policy": None, "value": None, "decided_by": None})
    )

  def test_the_policy_is_sorted_and_labelled_by_the_game(self) -> None:
    output = self.parse({
        "policy": [
            {"action": 0, "prob": 0.2, "label": "the bot's"},
            {"action": 2, "prob": 0.5},
            {"action": 1, "prob": 0.3},
        ],
        "value": -1,
        "decided_by": "network",
    })
    assert output is not None
    self.assertEqual(
        [(entry.action, entry.label) for entry in output.policy],
        [(2, "move 2"), (1, "move 1"), (0, "move 0")],
    )
    self.assertEqual((output.value, output.decided_by), (-1.0, "network"))

  def test_a_partial_policy_and_a_value_alone_are_accepted(self) -> None:
    output = self.parse({"policy": [{"action": 1, "prob": 0.6}]})
    assert output is not None
    self.assertEqual(len(output.policy), 1)
    output = self.parse({"value": 0.5})
    assert output is not None
    self.assertEqual((output.policy, output.value), ((), 0.5))

  def test_a_reply_that_is_not_the_protocol_s_is_refused(self) -> None:
    bad: dict[str, dict[str, Any]] = {
        "not a list": {"policy": {"0": 1.0}},
        "an entry not an object": {"policy": [3]},
        "an illegal action": {"policy": [{"action": 7, "prob": 0.5}]},
        "a bool action": {"policy": [{"action": True, "prob": 0.5}]},
        "an action twice": {
            "policy": [
                {"action": 1, "prob": 0.4},
                {"action": 1, "prob": 0.4},
            ]
        },
        "a probability above 1": {"policy": [{"action": 1, "prob": 1.5}]},
        "a negative probability": {"policy": [{"action": 1, "prob": -0.1}]},
        "no probability": {"policy": [{"action": 1}]},
        "a sum above 1": {
            "policy": [
                {"action": 1, "prob": 0.7},
                {"action": 2, "prob": 0.7},
            ]
        },
        "a label not text": {
            "policy": [{"action": 1, "prob": 0.5, "label": 3}]
        },
        "a value not a number": {"value": "high"},
        "an infinite value": {"value": float("inf")},
        "a blank decided_by": {"decided_by": " "},
        "a long decided_by": {"decided_by": "x" * 41},
        "a decided_by not text": {"decided_by": 3},
    }
    for name, reply in bad.items():
      with self.subTest(name):
        with self.assertRaises(remote_bot_http.RemoteBotReplyError):
          self.parse(reply)


class BotOutputContractTest(absltest.TestCase):

  def test_a_bot_output_holds_its_invariants(self) -> None:
    entry = contract.PolicyEntry
    with self.assertRaisesRegex(ValueError, "every action's id or none"):
      contract.BotOutput(
          policy=(entry(0, "a", 0.5), entry(None, "b", 0.4)),
          value=None,
          decided_by=None,
      )
    hidden = contract.BotOutput(
        policy=(entry(None, "Hidden action", 0.5), entry(None, "Call", 0.4)),
        value=None,
        decided_by=None,
    )
    with self.assertRaisesRegex(ValueError, "policy ids with its action"):
      contract.HistoryEntry(
          seat=0, action=1, label="x", rewards=(0.0,), bot_output=hidden
      )
    with self.assertRaisesRegex(ValueError, "policy ids with its action"):
      contract.HistoryEntry(
          seat=0,
          action=None,
          label=None,
          rewards=None,
          bot_output=contract.BotOutput(
              policy=(entry(0, "a", 0.5),), value=None, decided_by=None
          ),
      )
    with self.assertRaisesRegex(ValueError, "most probable"):
      contract.BotOutput(
          policy=(entry(0, "a", 0.2), entry(1, "b", 0.8)),
          value=None,
          decided_by=None,
      )
    with self.assertRaisesRegex(ValueError, "says something"):
      contract.BotOutput(policy=(), value=None, decided_by=None)
    with self.assertRaisesRegex(ValueError, "probability"):
      entry(0, "a", 1.5)

  def test_only_a_seat_s_entry_shown_in_full_has_one(self) -> None:
    output = contract.BotOutput(policy=(), value=0.0, decided_by=None)
    with self.assertRaisesRegex(ValueError, "bot output"):
      contract.HistoryEntry(
          seat=None, action=1, label="x", rewards=(0.0,), bot_output=output
      )
    with self.assertRaisesRegex(ValueError, "tells of its hand"):
      contract.HistoryEntry(
          seat=0, action=None, label=None, rewards=None, bot_output=output
      )
    # The policy ids are withheld with the action, as a public bot output
    # once had them; the entry still may not carry it.
    with self.assertRaisesRegex(ValueError, "tells of its hand"):
      contract.HistoryEntry(
          seat=0,
          action=None,
          label=None,
          rewards=None,
          bot_output=contract.BotOutput(
              policy=(contract.PolicyEntry(None, "Call", 0.5),),
              value=0.25,
              decided_by="network",
          ),
      )
    shown = contract.HistoryEntry(
        seat=0, action=1, label="x", rewards=(0.0,), bot_output=output
    )
    self.assertFalse(shown.withheld)
    self.assertEqual(shown.bot_output, output)


class SessionBotOutputTest(_BotOutputTestCase):

  def open(
      self, game: str, state: str = "seat_view", seat: int = 1
  ) -> session_lib.Session:
    """Opens `game`, every seat human but `seat`, the thinking bot."""
    num = pyspiel.load_game(game).num_players()
    specs = [controllers.HUMAN_SPEC] * num
    specs[seat] = controllers.Spec(remote_bot.NAME, self.seat_config(state))
    return session_lib.Session.create(
        game,
        {},
        seed=3,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=specs,
    )

  def bot_outputs(
      self, frame: Any
  ) -> list[tuple[int | None, contract.BotOutput | None]]:
    """Each history entry's seat and bot output."""
    return [(entry.seat, entry.bot_output) for entry in frame.history]

  def test_the_bot_s_ply_keeps_its_output(self) -> None:
    session = self.open("tic_tac_toe")
    session.apply_action(_seat(0), 4)
    session.advance(_ORACLE)
    frame = session.frame(_ORACLE)
    human, bot = frame.history
    self.assertIsNone(human.bot_output)
    assert bot.bot_output is not None
    output = bot.bot_output
    self.assertEqual(output.policy[0].action, bot.action)
    self.assertEqual(output.policy[0].label, bot.label)
    probs = [entry.prob for entry in output.policy]
    self.assertEqual(probs, sorted(probs, reverse=True))
    self.assertEqual(len(output.policy), 8)
    self.assertEqual((output.value, output.decided_by), (_VALUE, _DECIDED_BY))

  def test_every_bot_ply_keeps_its_own_and_a_replay_shows_them(self) -> None:
    session = self.open("tic_tac_toe")
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    session.apply_action(_seat(0), 1)
    session.advance(_ORACLE)
    outputs = [output for _, output in self.bot_outputs(session.frame(_ORACLE))]
    self.assertEqual(
        [output is None for output in outputs], [True, False, True, False]
    )
    first, second = outputs[1], outputs[3]
    assert first is not None and second is not None
    self.assertEqual(len(first.policy), 8)
    self.assertEqual(len(second.policy), 6)
    replay = session.replay_frame(_ORACLE, 2)
    self.assertEqual(replay.ply, 2)
    self.assertEqual(replay.history[1].bot_output, first)

  def test_an_undo_keeps_the_outputs_of_the_plies_it_keeps(self) -> None:
    session = self.open("tic_tac_toe")
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    kept = session.frame(_ORACLE).history[1].bot_output
    session.apply_action(_seat(0), 1)
    session.advance(_ORACLE)
    session.undo(_seat(0))
    frame = session.frame(_ORACLE)
    self.assertEqual(len(frame.history), 2)
    self.assertIsNotNone(kept)
    self.assertEqual(frame.history[1].bot_output, kept)

  def test_a_reset_s_plies_have_none(self) -> None:
    session = self.open("tic_tac_toe")
    session.apply_action(_seat(0), 4)
    session.advance(_ORACLE)
    session.reset(_ORACLE, history=session.history)
    self.assertEqual(
        {output for _, output in self.bot_outputs(session.frame(_ORACLE))},
        {None},
    )

  def test_an_output_does_not_change_play(self) -> None:
    quiet = session_lib.Session.create(
        "tic_tac_toe",
        {},
        seed=3,
        chance_policy=contract.ChancePolicy.AUTO,
        controllers=[controllers.HUMAN_SPEC, controllers.HUMAN_SPEC],
    )
    thinking = self.open("tic_tac_toe")
    thinking.apply_action(_seat(0), 4)
    thinking.advance(_ORACLE)
    quiet.apply_action(_seat(0), 4)
    quiet.apply_action(_seat(1), thinking.history[1])
    self.assertEqual(quiet.history, thinking.history)

  def test_the_viewers_who_see_the_move_receive_it(self) -> None:
    for game in ("kuhn_poker", "tic_tac_toe"):
      with self.subTest(game):
        session = self.open(game)
        if game == "kuhn_poker":
          session.apply_action(_seat(0), 0)
        else:
          session.apply_action(_seat(0), 4)
        session.advance(_ORACLE)
        expected = [
            output for _, output in self.bot_outputs(session.frame(_ORACLE))
        ]
        self.assertTrue(any(output is not None for output in expected))
        for name, viewer in (
            ("spectator", _SPECTATOR),
            ("seat 0", _seat(0)),
            ("seat 1", _seat(1)),
        ):
          with self.subTest(name):
            shown = [
                output for _, output in self.bot_outputs(session.frame(viewer))
            ]
            if game == "tic_tac_toe" or viewer == _seat(1):
              self.assertEqual(shown, expected)
            else:
              self.assertEqual(shown, [None] * len(expected))

  def test_a_withheld_move_s_output_is_withheld(self) -> None:
    # Leduc's betting is public, and its adapter labels each action, but the
    # bot's probabilities over "Call" and "Raise" follow its private card.
    session = self.open("leduc_poker")
    session.apply_action(_seat(0), 1)
    session.advance(_ORACLE)
    self.assertIsNotNone(session.frame(_seat(1)).history[3].bot_output)
    for viewer in (_seat(0), _SPECTATOR):
      with self.subTest(viewer=viewer):
        entry = session.frame(viewer).history[3]
        self.assertTrue(entry.withheld)
        self.assertIsNotNone(entry.public_label)
        self.assertIsNone(entry.bot_output)

  def test_a_full_state_seat_s_output_reaches_only_its_seat(self) -> None:
    session = self.open("kuhn_poker", state="full_state")
    session.apply_action(_seat(0), 0)
    session.advance(_ORACLE)
    for viewer in (_ORACLE, _seat(1)):
      self.assertIsNotNone(session.frame(viewer).history[3].bot_output)
    for viewer in (_seat(0), _SPECTATOR):
      self.assertIsNone(session.frame(viewer).history[3].bot_output)

  def test_a_bad_output_fails_the_seat(self) -> None:
    self.bot.extra = {"policy": [{"action": 99, "prob": 0.5}]}
    session = self.open("tic_tac_toe")
    session.apply_action(_seat(0), 4)
    with self.assertRaises(errors.ControllerFailedError):
      session.advance(_ORACLE)
    failure = session.frame(_ORACLE).seats[1].failure
    assert failure is not None
    self.assertEqual(failure.error, "RemoteBotReplyError")


class WireBotOutputTest(_BotOutputTestCase):
  """What the API sends each viewer, as the bytes a client receives."""

  def setUp(self) -> None:
    super().setUp()
    self.api = api.Api(
        store_lib.SessionStore(),
        catalogue=_catalogue,
        health=_health,
    )

  def call(
      self, method: str, target: str, body: object = None
  ) -> tuple[http.HTTPStatus, str]:
    """Calls the API; returns the status and the body as it is sent."""
    raw = b"" if body is None else json.dumps(body).encode()
    response = self.api.handle(method, target, raw)
    return response.status, json.dumps(response.body)

  def play(self, game: str, opening: int, state: str = "seat_view") -> str:
    """Opens `game` with the bot in seat 1, plays `opening`; returns the id."""
    human = {"controller": "human", "config": {}, "autoplay": False}
    bot = {
        "controller": remote_bot.NAME,
        "config": self.seat_config(state),
        "autoplay": True,
    }
    status, body = self.call(
        "POST",
        "/api/sessions?viewer=oracle",
        {
            "game": game,
            "parameters": {},
            "seats": [human, bot],
            "chance_policy": "auto",
            "seed": 5,
        },
    )
    self.assertEqual(status, http.HTTPStatus.OK, body)
    session = json.loads(body)["session"]["id"]
    status, body = self.call(
        "POST",
        f"/api/sessions/{session}/actions?viewer=0",
        {"action": opening, "ply": None},
    )
    self.assertEqual(status, http.HTTPStatus.OK, body)
    return str(session)

  def bodies(self, session: str, viewer: str) -> list[str]:
    """Every frame `viewer` may fetch: the live one and each replay."""
    status, live = self.call("GET", f"/api/sessions/{session}?viewer={viewer}")
    self.assertEqual(status, http.HTTPStatus.OK, live)
    bodies = [live]
    for ply in range(json.loads(live)["num_plies"] + 1):
      status, body = self.call(
          "GET", f"/api/sessions/{session}/plies/{ply}?viewer={viewer}"
      )
      self.assertEqual(status, http.HTTPStatus.OK, body)
      bodies.append(body)
    return bodies

  def test_the_bodies_of_the_viewers_who_see_the_move_carry_it(self) -> None:
    for game, opening, state in (
        ("kuhn_poker", 0, "seat_view"),
        ("kuhn_poker", 0, "full_state"),
        ("tic_tac_toe", 4, "seat_view"),
    ):
      session = self.play(game, opening, state)
      for viewer in ("0", "spectator", "oracle", "1"):
        with self.subTest(game=game, state=state, viewer=viewer):
          _, live = self.call(
              "GET", f"/api/sessions/{session}?viewer={viewer}"
          )
          if game == "kuhn_poker" and viewer in ("0", "spectator"):
            # Kuhn hides the cards: the bot's output would tell of its own.
            for body in self.bodies(session, viewer):
              self.assertNotIn(_DECIDED_BY, body)
            self.assertEqual(
                [entry["bot_output"] for entry in json.loads(live)["history"]],
                [None] * len(json.loads(live)["history"]),
            )
            continue
          self.assertIn(_DECIDED_BY, live)
          outputs = [
              entry["bot_output"]
              for entry in json.loads(live)["history"]
              if entry["bot_output"] is not None
          ]
          self.assertEqual(len(outputs), 1)
          self.assertEqual(
              sorted(outputs[0]), ["decided_by", "policy", "value"]
          )
          self.assertEqual(
              sorted(outputs[0]["policy"][0]), ["action", "label", "prob"]
          )
          self.assertIn(_DECIDED_BY, self.bodies(session, viewer)[-1])


if __name__ == "__main__":
  absltest.main()
