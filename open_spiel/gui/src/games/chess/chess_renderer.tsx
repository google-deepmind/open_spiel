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

// STRUCT-GLUE(chess ObservationStruct): the board reads the chess adapter's
// view (chess_adapter.py), which stands in for the struct chess does not
// have yet.
//
// The chess renderer: draws a chess frame with a chess board through the
// board slot (chess_board_slot.ts). The built-in board keeps its state from
// one ply to the next, so a piece slides to its new square and a piece being
// moved is not dropped, says whose move it is on the players' strips, and
// the renderer draws its own move list in Actions › History. An add-on may
// draw chess with another board: replaceRenderer('chess',
// chessRenderer(TheirBoard)).

import {type JSX, useEffect, useMemo} from 'react';

import type {Frame} from '../../contract/contract.ts';
import type {SectionDefinition} from '../../dock/panel_groups.ts';
import {useAnalysisViewSeat} from '../../game/bottom_seat.ts';
import type {
  Renderer,
  RendererEntry,
  RendererProps,
} from '../../renderers/renderer.ts';
import {AnimationSpeedField} from '../../view/animation_speed_field.tsx';
import {
  type ChessAct,
  type ChessBoardComponent,
  type ChessBoardProps,
  readChessFrame,
  readChessViewer,
} from './chess_board_slot.ts';
import {
  type BoardOrientation,
  ChessConfigSection,
  useBoardOrientation,
} from './chess_config_section.tsx';
import {ChessGameplay} from './chess_gameplay_panel.tsx';
import {ChessHistoryRenderer} from './chess_history_renderer.tsx';
import {
  ChessSessionContext,
  type SessionContextValue,
  useChessSession,
} from './chess_session.ts';

import './chess_board.css';
import './chess.css';

function cannotMove(): Promise<never> {
  return Promise.reject(new Error('You cannot move at this position.'));
}

/** The built-in chess board: the board, the players' strips and status. */
export function KernelChessBoard({
  frame,
  viewer,
  act,
}: ChessBoardProps): JSX.Element {
  const value = useMemo(
    (): SessionContextValue => ({
      status: 'ready',
      session: frame.session,
      error: null,
      viewer,
      // A viewer that cannot move here (a replay, a paused or finished
      // game, a bot to move) is busy to the board, as it is while a move is
      // in flight.
      isBusy: !viewer.canMove,
      submitAction: act,
    }),
    [frame, viewer, act],
  );
  return (
    <ChessSessionContext.Provider value={value}>
      <ChessGameplay useSession={useChessSession} />
    </ChessSessionContext.Provider>
  );
}

/**
 * Turns View › Board › View from to the seat the board is seen from: in
 * analysis the seat chosen in View › Perspective, in a
 * game the viewer's own; seat 0 plays Black. View from still turns it after,
 * until the seat changes or the board mounts again.
 *
 * @returns The side View from now shows.
 */
function useChessPerspective(frame: Frame): BoardOrientation {
  const [analysisViewSeat] = useAnalysisViewSeat();
  const [viewFrom, setViewFrom] = useBoardOrientation();
  const effectiveSeat =
    frame.viewer.kind === 'oracle' ? analysisViewSeat : frame.viewer.seat;
  // The side follows the seat whenever the board mounts or the seat
  // changes, so a board mounted again after a View from flip turns back to
  // the chosen seat's side.
  useEffect(() => {
    if (effectiveSeat !== null) {
      setViewFrom(effectiveSeat === 0 ? 'black' : 'white');
    }
  }, [effectiveSeat, setViewFrom]);
  return viewFrom;
}

/** Draws `props` with `board`, given the frame, the viewer and `act`. */
function ChessSlot({
  board: Board,
  frame,
  presentation,
  play,
  busy = false,
}: RendererProps & {readonly board: ChessBoardComponent}): JSX.Element {
  const chessFrame = useMemo(
    () => readChessFrame(frame, presentation),
    [frame, presentation],
  );
  const canMove = !busy && play !== null;
  const viewFrom = useChessPerspective(frame);
  const viewer = useMemo(
    () => readChessViewer(frame, canMove, viewFrom),
    [frame, canMove, viewFrom],
  );
  const act = useMemo(
    (): ChessAct =>
      play === null || busy
        ? cannotMove
        : action => {
            play(action);
            return Promise.resolve();
          },
    [play, busy],
  );
  return <Board frame={chessFrame} viewer={viewer} act={act} />;
}

/**
 * The "chess" renderer drawing with `board`: the component to pass to
 * replaceRenderer('chess', ...). Call it once, at module level or in an
 * add-on's registerInto, so the renderer keeps its identity.
 */
export function chessRenderer(board: ChessBoardComponent): Renderer {
  return function ChessBoardRenderer(props: RendererProps): JSX.Element {
    return <ChessSlot {...props} board={board} />;
  };
}

/** Draws a chess frame with the built-in board. */
export const ChessRenderer: Renderer = chessRenderer(KernelChessBoard);

const VIEW_SECTIONS: readonly SectionDefinition[] = [
  {id: 'chess_board', title: 'Board', Content: ChessConfigSection},
  {id: 'chess_animation', title: 'Animation', Content: AnimationSpeedField},
];

/** The chess renderer's registry entry. */
export const CHESS_RENDERER: RendererEntry = {
  name: 'chess',
  render: ChessRenderer,
  keepsState: true,
  drawsStatus: true,
  viewSections: VIEW_SECTIONS,
  History: ChessHistoryRenderer,
};
