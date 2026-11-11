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

"""Twin tests: a bot's output tells no other viewer anything of its hand.

Each test plays the same game twice, under the `expose` chance policy, with
deals that differ only in the cards of the bot's hand: one card of the bot's
is exchanged with a card of another hidden hand, and neither is played. The
humans and the bot make the same moves in both. A viewer who holds neither
hand (a human seat, a spectator) may then tell the twins apart by nothing,
so every frame it can fetch, live and in replay of every ply, must be the
same bytes in both. The bot answers as a real bot does, from what its seat
sees: its policy lists its legal actions (so, in trick play, its cards) and
its probabilities and value follow its information state.

The twins must differ where they should: the oracle's frames and the bot's
own seat's bot output tell them apart, or the test proves nothing.
"""

from collections.abc import Callable, Sequence
import dataclasses
import json
import threading
from typing import Any
import zlib

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import remote_bot
from open_spiel.python.gui import remote_bot_server
from open_spiel.python.gui import session as session_lib

_ORACLE = contract.Viewer(contract.ViewerKind.ORACLE)
_SPECTATOR = contract.Viewer(contract.ViewerKind.SPECTATOR)
_BOT = 1


def _seat(index: int) -> contract.Viewer:
  """The viewer of seat `index`."""
  return contract.Viewer(contract.ViewerKind.SEAT, index)


class _SeeingBot:
  """Plays its lowest legal action; its output follows what its seat sees.

  Its policy names every legal action, with probabilities and a value drawn
  from its information state, as a bot's output follows its hand.
  """

  state_modes = (remote_bot_server.SEAT_VIEW,)
  games = None

  def step(self, mode: str, request: dict[str, Any]) -> dict[str, Any]:
    """Replies with the lowest legal action and an output from the view."""
    legal = sorted(remote_bot_server.legal_actions(mode, request))
    seen = json.dumps(request["observation"], sort_keys=True).encode()
    digest = zlib.crc32(seen)
    weights = [(digest >> (index % 24)) % 7 + 1 for index in range(len(legal))]
    total = sum(weights)
    return {
        "action": legal[0],
        "policy": [
            {"action": action, "prob": weight / total}
            for action, weight in zip(legal, weights)
        ],
        "value": (digest % 2001) / 1000.0 - 1.0,
        "decided_by": "forced" if len(legal) == 1 else "network",
    }


# Chooses a human seat's action from the seat and the legal actions' labels.
Chooser = Callable[[int, dict[str, int]], int]


@dataclasses.dataclass(frozen=True)
class _Twin:
  """A game and the deal of each twin.

  Attributes:
    game: The game's short name.
    parameters: Its parameters.
    deal: The chance outcomes of the first twin, in order.
    swap: Two outcomes of `deal`, a card the bot is dealt and one another
      hidden hand is dealt; the second twin deals each in the other's place.
    choose: Chooses each human seat's action.
    bot_plies: How many plies the bot plays before the twins are compared.
  """

  game: str
  parameters: dict[str, contract.ParameterValue]
  deal: tuple[int, ...]
  swap: tuple[int, int]
  choose: Chooser
  bot_plies: int

  def deals(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The two twins' chance outcomes."""
    a, b = self.swap
    exchange = {a: b, b: a}
    return self.deal, tuple(exchange.get(card, card) for card in self.deal)


def _lowest_deal(game: str, parameters: dict[str, Any]) -> tuple[int, ...]:
  """The chance outcomes of a deal that always takes the lowest outcome."""
  state = pyspiel.load_game(game, parameters).new_initial_state()
  deal = []
  while state.is_chance_node():
    outcome = min(action for action, _ in state.chance_outcomes())
    deal.append(outcome)
    state.apply_action(outcome)
  return tuple(deal)


def _card(game: str, text: str) -> int:
  """The id of the card `text` names in `game`, as its deal labels it."""
  state = pyspiel.load_game(game).new_initial_state()
  while state.is_chance_node():
    by_label: dict[str, int] = {
        str(state.action_to_string(action)): int(action)
        for action, _ in state.chance_outcomes()
    }
    if text in by_label:
      return by_label[text]
    state.apply_action(min(by_label.values()))
  raise AssertionError(f"{game} deals no {text}.")


def _lowest(seat: int, labels: dict[str, int]) -> int:
  """The lowest legal action."""
  del seat
  return min(labels.values())


def _euchre_choice(seat: int, labels: dict[str, int]) -> int:
  """The dealer (seat 0) orders the upcard up; the others pass."""
  if seat == 0 and "Clubs" in labels:
    return labels["Clubs"]
  if "Pass" in labels:
    return labels["Pass"]
  return min(labels.values())


def _leduc_choice(seat: int, labels: dict[str, int]) -> int:
  """Calls (or checks)."""
  del seat
  return labels["Call"]


def _hearts() -> _Twin:
  """Hearts: the bot's first-trick policy names its diamonds.

  The lowest deal gives seat 0 the clubs, the bot (seat 1) the diamonds,
  seat 2 the hearts and seat 3 the spades, with no pass. Seat 0 leads the
  2C; the bot, void in clubs and barred from hearts on the first trick, may
  play any diamond, and plays the 2D. The second twin gives it the 3H for
  its 3D.
  """
  game, parameters = "hearts", dict[str, contract.ParameterValue]()
  return _Twin(
      game=game,
      parameters=parameters,
      deal=_lowest_deal(game, parameters),
      swap=(_card(game, "3D"), _card(game, "3H")),
      choose=_lowest,
      bot_plies=1,
  )


def _euchre() -> _Twin:
  """Euchre: the bot's lead policy names its whole hand.

  The lowest deal makes seat 0 the dealer, gives it the clubs, the bot (seat
  1) the diamonds, seat 2 the hearts, seat 3 the spades, and turns up the
  CA. The bot and seats 2 and 3 pass, the dealer orders the CA up and
  discards; the bot leads its lowest card, the D9. The second twin gives it
  the HT for its DT.
  """
  game, parameters = "euchre", dict[str, contract.ParameterValue]()
  return _Twin(
      game=game,
      parameters=parameters,
      deal=_lowest_deal(game, parameters),
      swap=(_card(game, "DT"), _card(game, "HT")),
      choose=_euchre_choice,
      bot_plies=2,
  )


def _leduc() -> _Twin:
  """Leduc: the bot's labels are public, its probabilities and value not.

  Leduc's six cards are two of each rank, by id: jacks 0 and 1, queens 2
  and 3, kings 4 and 5. Seat 0 is dealt a jack, the bot a queen in one twin
  and a king in the other. Seat 0 checks and the bot checks behind.
  """
  return _Twin(
      game="leduc_poker",
      parameters={},
      deal=(0, 2),
      swap=(2, 4),
      choose=_leduc_choice,
      bot_plies=1,
  )


def _differences(a: object, b: object, path: str) -> list[str]:
  """The paths at which two JSON values differ, with each side's value."""
  if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
    return [
        found
        for key in sorted(a)
        for found in _differences(a[key], b[key], f"{path}.{key}")
    ]
  if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
    return [
        found
        for index, (x, y) in enumerate(zip(a, b))
        for found in _differences(x, y, f"{path}[{index}]")
    ]
  if a == b:
    return []
  return [f"{path}: {json.dumps(a)[:80]} != {json.dumps(b)[:80]}"]


class BotOutputTwinTest(absltest.TestCase):
  """Twin deals give the bot's opponents and spectators the same bytes."""

  def setUp(self) -> None:
    super().setUp()
    self.addCleanup(controllers.set_move_budget, controllers.move_budget())
    self.addCleanup(builds.set_wait_seconds, builds.wait_seconds())
    builds.set_wait_seconds(10.0)
    self.addCleanup(remote_bot.configure)
    server = remote_bot_server.Server(
        ("127.0.0.1", 0), _SeeingBot(), name="test"
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    self.addCleanup(server.server_close)
    self.addCleanup(server.shutdown)
    self.url = server.url

  def play(self, twin: _Twin, deal: Sequence[int]) -> session_lib.Session:
    """Plays `twin` with `deal` until the bot has played its plies."""
    num = pyspiel.load_game(twin.game, twin.parameters).num_players()
    specs = [controllers.HUMAN_SPEC] * num
    specs[_BOT] = controllers.Spec(
        remote_bot.NAME, {"url": self.url, "state": "seat_view"}
    )
    session = session_lib.Session.create(
        twin.game,
        twin.parameters,
        seed=7,
        chance_policy=contract.ChancePolicy.EXPOSE,
        controllers=specs,
    )
    outcomes = list(deal)
    bot_plies = 0
    for _ in range(400):
      if bot_plies == twin.bot_plies:
        return session
      oracle = session.frame(_ORACLE)
      self.assertNotEqual(
          oracle.status, contract.SessionStatus.TERMINAL, "The game ended."
      )
      seat = oracle.to_move.seat
      if seat is None:
        self.assertTrue(outcomes, "The deal ran out.")
        session.apply_chance(_ORACLE, outcomes.pop(0))
      elif seat == _BOT:
        session.advance(_ORACLE, max_plies=1)
        bot_plies += 1
      else:
        assert oracle.legal_actions is not None
        labels = {action.label: action.id for action in oracle.legal_actions}
        session.apply_action(_seat(seat), twin.choose(seat, labels))
    self.fail("The bot never played its plies.")

  def bodies(
      self, session: session_lib.Session, viewer: contract.Viewer
  ) -> list[str]:
    """Every frame `viewer` may fetch, as wire JSON, the session id blanked."""
    live = session.frame(viewer)
    frames = [live] + [
        session.replay_frame(viewer, step) for step in range(live.num_plies + 1)
    ]
    bodies = []
    for frame in frames:
      body = contract.to_json(frame)
      assert isinstance(body, dict)
      body["session"]["id"] = "<id>"
      bodies.append(json.dumps(body, sort_keys=True))
    return bodies

  def bot_outputs(
      self, session: session_lib.Session, viewer: contract.Viewer
  ) -> list[contract.BotOutput]:
    """The bot outputs `viewer` receives in the live frame."""
    return [
        entry.bot_output
        for entry in session.frame(viewer).history
        if entry.bot_output is not None
    ]

  def check_twins(self, twin: _Twin) -> None:
    """Plays both twins and compares what each viewer may fetch."""
    first, second = (self.play(twin, deal) for deal in twin.deals())
    # The twins differ, and the bot's own output shows it.
    self.assertNotEqual(
        self.bodies(first, _ORACLE), self.bodies(second, _ORACLE)
    )
    own = [self.bot_outputs(s, _seat(_BOT)) for s in (first, second)]
    self.assertEqual(len(own[0]), twin.bot_plies)
    self.assertNotEqual(own[0], own[1])
    self.assertEqual(own[0], self.bot_outputs(first, _ORACLE))
    for name, viewer in (("seat 0", _seat(0)), ("spectator", _SPECTATOR)):
      with self.subTest(game=twin.game, viewer=name):
        firsts, seconds = (self.bodies(s, viewer) for s in (first, second))
        self.assertEqual(len(firsts), len(seconds))
        for step, (a, b) in enumerate(zip(firsts, seconds)):
          if a != b:
            paths = _differences(json.loads(a), json.loads(b), "")
            self.fail(
                f"Frame {step} tells the twins apart at {paths[:5]}"
                f" ({len(paths)} places)."
            )

  def test_hearts(self) -> None:
    self.check_twins(_hearts())

  def test_euchre(self) -> None:
    self.check_twins(_euchre())

  def test_leduc_poker(self) -> None:
    self.check_twins(_leduc())


if __name__ == "__main__":
  absltest.main()
