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

// STRUCT-GLUE(hearts ObservationStruct): the board reads the adapter's
// struct-shaped view (hands by compass, the trick, the last trick, the
// points and the pass), built from the state for the viewer
// (hearts_adapter.py).
//
// The hearts renderer: the hearts board, given the frame as the session it
// reads (hearts_session.ts) and the viewer's moves. It keeps its state from
// one ply to the next, as the pass in flight, the held trick and the staged
// plays need, and says whose turn it is on its own table.

import {type JSX, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {HeartsGameplayConfigExtras} from './hearts_gameplay_config_extras.tsx';
import {HeartsGameplay} from './hearts_gameplay_panel.tsx';
import {
  HeartsSessionContext,
  type SessionContextValue,
  type SessionPayload,
  heartsReplay,
  heartsSession,
  heartsViewer,
} from './hearts_session.ts';

function toMoveIsViewer(frame: Frame): boolean {
  const {to_move: toMove, viewer} = frame;
  return toMove.kind === 'seat' && toMove.seat === viewer.seat;
}

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

/** Draws a hearts frame with the hearts board. */
export function HeartsRenderer({
  frame,
  presentation,
  play,
  playSequence,
  busy = false,
}: RendererProps): JSX.Element {
  const value = useMemo((): SessionContextValue => {
    const session = heartsSession(frame, presentation);
    const sequence = playSequence ?? null;
    const toSession = (next: Frame): SessionPayload => {
      if (next.presentation === null) {
        throw new Error('The frame after the move has no hearts view.');
      }
      return heartsSession(next, next.presentation);
    };
    return {
      session,
      error: null,
      viewer: heartsViewer(frame),
      replay: heartsReplay(frame),
      // A seat to move that cannot play here (a replay, a paused or finished
      // game) is busy to the board, as it is while a move is in flight.
      isBusy: busy || (play === null && toMoveIsViewer(frame)),
      submitAction:
        play === null
          ? cannotMove
          : action => {
              play(action);
              return Promise.resolve();
            },
      submitAtomicSequence:
        sequence === null
          ? cannotMove
          : async actions => toSession(await sequence(actions)),
    };
  }, [frame, presentation, play, playSequence, busy]);
  return (
    <HeartsSessionContext.Provider value={value}>
      <HeartsGameplay />
    </HeartsSessionContext.Provider>
  );
}

const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'hearts', title: 'Hearts', Content: HeartsGameplayConfigExtras},
  {id: 'hearts_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The hearts renderer's registry entry. */
export const HEARTS_RENDERER: RendererEntry = {
  name: 'hearts',
  render: HeartsRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
};
