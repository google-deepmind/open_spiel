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

// STRUCT-GLUE(gin_rummy ObservationStruct): the board reads the adapter's
// struct-shaped view, which adds to the observation struct what the board
// needs (gin_rummy_adapter.py).
//
// The gin_rummy renderer: the gin board, given the frame as the session it
// reads (gin_session.ts) and the viewer's moves. It keeps its state from one
// ply to the next, as the board's hand order, lay-down composition and cards
// in flight need, and says whose turn it is on its own table.

import {type JSX, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {
  GinRummyViewSettings,
  GinRummyVisualsSettings,
} from './gin_rummy_gameplay_config_extras.tsx';
import {GinRummyGameplay} from './gin_rummy_gameplay_panel.tsx';
import {
  GinSessionContext,
  type SessionContextValue,
  type SessionPayload,
  ginSession,
  ginViewer,
} from './gin_session.ts';

function toMoveIsViewer(frame: Frame): boolean {
  const {to_move: toMove, viewer} = frame;
  return toMove.kind === 'seat' && toMove.seat === viewer.seat;
}

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

/** Draws a gin_rummy frame with the gin board. */
export function GinRummyRenderer({
  frame,
  presentation,
  play,
  playSequence,
  busy = false,
}: RendererProps): JSX.Element {
  const value = useMemo((): SessionContextValue => {
    const session = ginSession(frame, presentation);
    const sequence = playSequence ?? null;
    const toSession = (next: Frame): SessionPayload => {
      if (next.presentation === null) {
        throw new Error('The frame after the move has no gin_rummy view.');
      }
      return ginSession(next, next.presentation);
    };
    return {
      session,
      error: null,
      viewer: ginViewer(frame),
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
    <GinSessionContext.Provider value={value}>
      <GinRummyGameplay />
    </GinSessionContext.Provider>
  );
}

const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'gin_visuals', title: 'Visuals', Content: GinRummyVisualsSettings},
  {id: 'gin_rummy', title: 'Gin Rummy', Content: GinRummyViewSettings},
  {id: 'gin_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The gin_rummy renderer's registry entry. */
export const GIN_RUMMY_RENDERER: RendererEntry = {
  name: 'gin_rummy',
  render: GinRummyRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
};
