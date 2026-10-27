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

// STRUCT-GLUE(checkers StateStruct): the board reads the adapter's
// struct-shaped view, built from the state's text (checkers_adapter.py).
//
// The checkers renderer: the checkers board, given the frame as the session
// it reads (checkers_session.ts) and the viewer's moves. It keeps its state
// from one ply to the next, as the board's move slide needs, and says whose
// move it is in its own status row.

import {type JSX, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import type {RendererEntry, RendererProps} from '../../renderers/renderer.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {CheckersConfigSection} from './checkers_config_section.tsx';
import {CheckersGameplay} from './checkers_gameplay_panel.tsx';
import {
  CheckersSessionContext,
  type SessionContextValue,
  checkersSession,
  checkersViewer,
} from './checkers_session.ts';

function toMoveIsViewer(frame: Frame): boolean {
  const {to_move: toMove, viewer} = frame;
  return toMove.kind === 'seat' && toMove.seat === viewer.seat;
}

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

/** Draws a checkers frame with the checkers board. */
export function CheckersRenderer({
  frame,
  presentation,
  play,
  busy = false,
}: RendererProps): JSX.Element {
  const value = useMemo(
    (): SessionContextValue => ({
      session: checkersSession(frame, presentation),
      error: null,
      viewer: checkersViewer(frame),
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
    }),
    [frame, presentation, play, busy],
  );
  return (
    <CheckersSessionContext.Provider value={value}>
      <CheckersGameplay />
    </CheckersSessionContext.Provider>
  );
}

const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'checkers_board', title: 'Board', Content: CheckersConfigSection},
  {id: 'checkers_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The checkers renderer's registry entry. */
export const CHECKERS_RENDERER: RendererEntry = {
  name: 'checkers',
  render: CheckersRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
};
