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

// A renderer entry for a board panel: the panel, given the frame as
// the session it reads (board_session.ts) and the viewer's move. The panel
// draws its own status row, as it does in its own app, and keeps its state
// from one ply to the next, so its measured board size does not reset.

import {type FC, type JSX, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {
  Renderer,
  RendererEntry,
  RendererProps,
} from '../../renderers/renderer.ts';
import {
  BoardSessionContext,
  type SessionContextValue,
  boardSession,
  boardViewer,
} from './board_session.ts';

function toMoveIsViewer(frame: Frame): boolean {
  const {to_move: toMove, viewer} = frame;
  return toMove.kind === 'seat' && toMove.seat === viewer.seat;
}

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

function rendererFor(Panel: FC): Renderer {
  return function BoardRenderer({
    frame,
    presentation,
    play,
    busy = false,
  }: RendererProps): JSX.Element {
    const value = useMemo((): SessionContextValue => {
      const session = boardSession(frame, presentation);
      return {
        session,
        error: null,
        viewer: boardViewer(frame, session.state),
        // A seat to move that cannot play here (a replay, a paused game) is
        // busy to the board, as it is while a move is in flight.
        isBusy: busy || (play === null && toMoveIsViewer(frame)),
        submitAction:
          play === null
            ? cannotMove
            : action => {
                play(action);
                return Promise.resolve();
              },
      };
    }, [frame, presentation, play, busy]);
    return (
      <BoardSessionContext.Provider value={value}>
        <Panel />
      </BoardSessionContext.Provider>
    );
  };
}

/** The registry entry named `name` that draws `Panel`. */
export function boardRendererEntry(name: string, Panel: FC): RendererEntry {
  return {
    name,
    render: rendererFor(Panel),
    keepsState: true,
    drawsStatus: true,
  };
}
