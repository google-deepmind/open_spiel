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

import type {CSSProperties, JSX} from 'react';

import {ChessPieceArt} from './chess_pieces.tsx';
import {
  GROUP_GAP_REM,
  PIECE_REM,
  PILE_MAX_REM,
  SIDE_NAME,
  other,
  pileStep,
  type Role,
  type Side,
  type SideMaterial,
} from './chess_table.ts';

/** The properties of one side's strip. */
export interface ChessPlayerStripProps {
  side: Side;
  /** Who plays this side: a display name or the controller's name. */
  name: string;
  /** This side is the viewer's own seat (game mode). */
  isViewer?: boolean;
  material: SideMaterial;
  /**
   * Whose move it is, said quietly on that side's strip: "to move", or
   * what a bot is doing. Null when it is not this side's move or the game
   * is over.
   */
  turn: string | null;
}

const ROLE_NAME: Record<Role, string> = {
  pawn: 'pawn',
  knight: 'knight',
  bishop: 'bishop',
  rook: 'rook',
  queen: 'queen',
};

/** The taken pieces, in runs of one kind. */
function pileGroups(taken: readonly Role[]): Array<{role: Role; n: number}> {
  const groups: Array<{role: Role; n: number}> = [];
  for (const role of taken) {
    const last = groups[groups.length - 1];
    if (last?.role === role) last.n += 1;
    else groups.push({role, n: 1});
  }
  return groups;
}

/** What a screen reader says of the strip. */
function stripLabel(
  side: Side,
  name: string,
  isViewer: boolean,
  material: SideMaterial,
  groups: ReadonlyArray<{role: Role; n: number}>,
  turn: string | null,
): string {
  const takenName = material.taken.length
    ? `material ahead: ${groups.map(g => (g.n > 1 ? `${String(g.n)} ${ROLE_NAME[g.role]}s` : ROLE_NAME[g.role])).join(', ')}`
    : '';
  return [
    `${SIDE_NAME[side]}, ${name}${isViewer ? ' (you)' : ''}`,
    takenName,
    material.lead > 0 ? `up ${String(material.lead)}` : '',
    turn ?? '',
  ]
    .filter(Boolean)
    .join(', ');
}

/**
 * One side of the table, above or below the board as the board is turned:
 * who plays it, its material ahead (the difference in each kind of piece,
 * drawn in the opponent's colour as taken pieces lie beside a real board,
 * lichess's material count) and how many points it is up, and, when it is
 * this side's move, a quiet word saying so. The difference is not a capture
 * record: after a promotion the promoting side is a queen ahead with nothing
 * taken, so the screen reader says "material ahead", never "has taken". The
 * pieces are the board's own (chess_pieces.tsx), so they match the board. When
 * space runs short the player's name shortens first; the pile tightens
 * to stay within PILE_MAX_REM, so it never runs past the board. Its sizes
 * come from chessTable.ts as CSS variables, so the budget and the drawing
 * cannot disagree.
 */
export function ChessPlayerStrip({
  side,
  name,
  isViewer,
  material,
  turn,
}: ChessPlayerStripProps): JSX.Element {
  const groups = pileGroups(material.taken);
  const label = stripLabel(
    side,
    name,
    isViewer === true,
    material,
    groups,
    turn,
  );
  const pieceColour = other(side);
  const step = pileStep(groups.map(g => g.n));
  return (
    <div
      className="chess-strip"
      role="group"
      aria-label={label}
      data-testid={`chess-strip-${side}`}
      data-to-move={turn ? 'true' : undefined}
    >
      <span className="chess-strip__side" aria-hidden="true">
        {SIDE_NAME[side]}
      </span>
      <span className="chess-strip__name" aria-hidden="true">
        {name}
        {isViewer ? ' · you' : ''}
      </span>
      <span
        className="cg-wrap chess-strip__taken"
        aria-hidden="true"
        style={
          {
            '--chess-pile-piece': `${String(PIECE_REM)}rem`,
            '--chess-pile-gap': `${String(GROUP_GAP_REM)}rem`,
            '--chess-pile-step': `${String(step)}rem`,
            '--chess-pile-max': `${String(PILE_MAX_REM)}rem`,
          } as CSSProperties
        }
      >
        {groups.map((g, i) => (
          <span
            key={`${g.role}-${String(i)}`}
            className="chess-strip__group"
            data-role={g.role}
            data-count={g.n}
          >
            {Array.from({length: g.n}, (_, k) => (
              <span key={k} className={`piece ${g.role} ${pieceColour}`}>
                <ChessPieceArt role={g.role} color={pieceColour} />
              </span>
            ))}
          </span>
        ))}
      </span>
      {material.lead > 0 && (
        <span className="chess-strip__lead" aria-hidden="true">
          +{material.lead}
        </span>
      )}
      {turn && (
        <span className="chess-strip__turn" aria-hidden="true">
          {turn}
        </span>
      )}
    </div>
  );
}
