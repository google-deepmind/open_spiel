// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// STRUCT-GLUE(kuhn_poker StateStruct): the board reads the universal-poker-
// shaped struct the kuhn_poker and leduc_poker adapters build
// (adapters/poker_table.py).
//
// The poker renderer: the poker board, given the frame as the session it
// reads (poker_session.ts), the viewer's moves, and a new hand. It keeps its
// state from one ply to the next, as the board's chips and cards in flight
// need, and says whose turn it is and how the hand ended on its own table.

import {type JSX, useContext, useMemo} from 'react';

import type {
  CreateSession,
  Frame,
  SeatConfig,
} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import {GameContext} from '../../game/game_context.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import type {SessionCommands} from '../../session/session_store.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {PokerGameplay} from './poker_gameplay_panel.tsx';
import {PokerHistoryRenderer} from './poker_history_renderer.tsx';
import {
  type NewHand,
  PokerSessionContext,
  type SessionContextValue,
  pokerSession,
} from './poker_session.ts';

// The controller a new seat gets when the table grows: the viewer's
// opponent's, else the random player.
const DEFAULT_OPPONENT = 'random';

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

/**
 * The game `frame` shows with `hand.seats` seats: seat 0 as it is, each other
 * seat keeping its controller, and a new seat playing as seat 1 does.
 */
export function newHandSession(frame: Frame, hand: NewHand): CreateSession {
  const opponent = frame.seats[1] ?? frame.seats[0];
  const seats: SeatConfig[] = Array.from({length: hand.seats}, (_, slot) => {
    const seat = frame.seats[slot];
    if (seat !== undefined) {
      return {
        controller: seat.controller,
        config: {...seat.config},
        autoplay: seat.autoplay,
      };
    }
    return opponent === undefined
      ? {controller: DEFAULT_OPPONENT, config: {}, autoplay: false}
      : {
          controller: opponent.controller,
          config: {...opponent.config},
          autoplay: opponent.autoplay,
        };
  });
  return {
    game: frame.session.game,
    parameters: {...frame.session.parameters, players: hand.seats},
    seats,
    chance_policy: frame.session.chance_policy,
    seed: null,
  };
}

/** Deals a new hand: the same table again, or a table of another size. */
function dealer(
  frame: Frame,
  commands: SessionCommands | null,
): (hand: NewHand) => void {
  return hand => {
    if (commands === null) {
      throw new Error('A new hand is dealt only on a game page.');
    }
    if (hand.seats === frame.seats.length) commands.reset();
    else commands.create(newHandSession(frame, hand));
  };
}

/** Draws a kuhn_poker or leduc_poker frame with the poker board. */
export function PokerRenderer({
  frame,
  presentation,
  play,
  busy = false,
}: RendererProps): JSX.Element {
  const game = useContext(GameContext);
  const commands = game?.session.commands ?? null;
  const value = useMemo((): SessionContextValue => {
    const session = pokerSession(frame, presentation);
    const {to_move: toMove, viewer} = frame;
    const acting =
      toMove.kind === 'seat' &&
      (viewer.kind === 'oracle' || toMove.seat === viewer.seat);
    return {
      status: 'ready',
      session,
      error: null,
      viewer: {
        participantId: viewer.seat === null ? null : String(viewer.seat),
        canAct: play !== null && acting,
      },
      isBusy: busy,
      submitAction:
        play === null
          ? cannotMove
          : action => {
              play(action);
              return Promise.resolve();
            },
      createSession: dealer(frame, commands),
    };
  }, [frame, presentation, play, busy, commands]);
  return (
    <PokerSessionContext.Provider value={value}>
      <PokerGameplay />
    </PokerSessionContext.Provider>
  );
}

// The board's View section: the Animation speed its chips and cards fly at.
const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'poker_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The poker renderer's registry entry. */
export const POKER_RENDERER: RendererEntry = {
  name: 'poker',
  render: PokerRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
  History: PokerHistoryRenderer,
};
