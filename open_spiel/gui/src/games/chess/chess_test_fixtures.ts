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

// Frames of a chess game for the chess board's tests: the adapter's view and
// move hints, as chess_adapter.py sends them.

import type {
  Frame,
  HistoryEntry,
  Presentation,
} from '../../contract/contract.ts';
import {frame} from '../../testing/frames.ts';

/** The position after 1. e4 d5. */
export const AFTER_E4_D5 =
  'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2';

/** White's moves the tests offer: e4xd5, e4-e5 and Ng1-f3, as action ids. */
export const MOVES = [
  {id: 10, label: 'exd5', from: 'e4', to: 'd5'},
  {id: 11, label: 'e5', from: 'e4', to: 'e5'},
  {id: 12, label: 'Nf3', from: 'g1', to: 'f3'},
] as const;

/** The history entries of `labels`, white (seat 1) first. */
export function chessHistory(labels: readonly string[]): HistoryEntry[] {
  return labels.map((label, index) => ({
    seat: index % 2 === 0 ? 1 : 0,
    action: 100 + index,
    label,
    rewards: [0, 0],
    public_label: null,
  }));
}

/** The chess adapter's presentation of a position. */
export function chessPresentation(
  view: Record<string, unknown> = {},
  hints: ReadonlyArray<{
    id: number;
    from: string;
    to: string;
    promotion?: string;
  }> = MOVES,
): Presentation {
  return {
    renderer: 'chess',
    view: {
      fen: AFTER_E4_D5,
      chessCheck: null,
      chessLastMove: {from: 'd7', to: 'd5'},
      chessOutcome: {checkmate: false, stalemate: false, threefold: false},
      ...view,
    },
    actions: hints.map(({id, from, to, promotion}) => ({
      id,
      hint: promotion === undefined ? {from, to} : {from, to, promotion},
    })),
  };
}

/**
 * A chess frame after 1. e4 d5, white (seat 1) to move, shown to the oracle;
 * `fields` replace any of its fields.
 */
export function chessFrame(
  fields: Partial<Frame> = {},
  presentation: Presentation = chessPresentation(),
): Frame {
  const history = fields.history ?? chessHistory(['e4', 'd5']);
  return frame({
    history,
    to_move: {kind: 'seat', seat: 1},
    legal_actions: MOVES.map(({id, label}) => ({id, label})),
    state_text: AFTER_E4_D5,
    presentation,
    ...fields,
  });
}
