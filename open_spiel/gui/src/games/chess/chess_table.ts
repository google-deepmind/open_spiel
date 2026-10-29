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

/**
 * What the chess table shows around the board, derived from the position:
 * the material each side is up (the pieces it has taken, net, as they would
 * lie beside a real board), the result with its reason, and figurine
 * notation for the move list. Pure, so each rule is tested on its own.
 */
import {HUMAN} from '../../players/seat_draft.ts';
import type {ChessOutcome} from './chess_session.ts';

/** A side of the board. */
export type Side = 'white' | 'black';
/** A kind of piece that can be taken. */
export type Role = 'pawn' | 'knight' | 'bishop' | 'rook' | 'queen';

/** OpenSpiel chess seats: player 0 is Black, player 1 is White. */
export const SLOT_OF: Record<Side, number> = {white: 1, black: 0};
/** Each side's name. */
export const SIDE_NAME: Record<Side, string> = {white: 'White', black: 'Black'};
/** The other side. */
export function other(side: Side): Side {
  return side === 'white' ? 'black' : 'white';
}

/** The order taken pieces lie in, cheapest first (chess.com, lichess). */
const ROLES: Role[] = ['pawn', 'knight', 'bishop', 'rook', 'queen'];
const LETTER: Record<Role, string> = {
  pawn: 'p',
  knight: 'n',
  bishop: 'b',
  rook: 'r',
  queen: 'q',
};
/** The conventional piece values behind the "+N" material count. */
export const PIECE_VALUE: Record<Role, number> = {
  pawn: 1,
  knight: 3,
  bishop: 3,
  rook: 5,
  queen: 9,
};

/** The material one side is up. */
export interface SideMaterial {
  /** The opponent's pieces this side is up, net, cheapest first. */
  taken: Role[];
  /** Points ahead (0 when level or behind). */
  lead: number;
}

function placement(fen: string): string {
  return fen.trim().split(/\s+/)[0] ?? '';
}

/**
 * The material difference, as lichess computes it: for each kind of piece,
 * the side with more of them is shown as having taken the difference. Read
 * from the position alone, so it holds after a promotion and from any start.
 */
export function materialDiff(fen: string): Record<Side, SideMaterial> {
  const board = placement(fen);
  const count = (ch: string) => Array.from(board).filter(c => c === ch).length;
  const white: Role[] = [];
  const black: Role[] = [];
  let score = 0;
  for (const role of ROLES) {
    const d = count(LETTER[role].toUpperCase()) - count(LETTER[role]);
    score += d * PIECE_VALUE[role];
    for (let i = 0; i < Math.abs(d); i += 1) (d > 0 ? white : black).push(role);
  }
  return {
    white: {taken: white, lead: Math.max(score, 0)},
    black: {taken: black, lead: Math.max(-score, 0)},
  };
}

/**
 * The engine's ChessBoard::HasSufficientMaterial (open_spiel chess_board.cc),
 * so the reason given for a draw is the engine's own: any pawn, rook or queen
 * is enough; two knights on one side are; a knight with anything else is;
 * otherwise only bishops on both square colours are.
 */
export function hasSufficientMaterial(fen: string): boolean {
  const rows = placement(fen).split('/');
  const knights = {w: 0, b: 0};
  const dark = {w: 0, b: 0};
  const light = {w: 0, b: 0};
  for (let r = 0; r < rows.length; r += 1) {
    let file = 0;
    for (const c of rows[r] ?? '') {
      if (/\d/.test(c)) {
        file += Number(c);
        continue;
      }
      const side = c === c.toUpperCase() ? 'w' : 'b';
      const p = c.toLowerCase();
      if (p === 'p' || p === 'r' || p === 'q') return true;
      if (p === 'n') knights[side] += 1;
      if (p === 'b') {
        // FEN rank 8 comes first; a1 (file 0, rank 1) is dark.
        const rank = 7 - r;
        if ((file + rank) % 2 === 0) dark[side] += 1;
        else light[side] += 1;
      }
      file += 1;
    }
  }
  if (knights.w > 1 || knights.b > 1) return true;
  const minors = (s: 'w' | 'b') => knights[s] + dark[s] + light[s];
  if (knights.w === 1) return dark.w + light.w > 0 || minors('b') > 0;
  if (knights.b === 1) return dark.b + light.b > 0 || minors('w') > 0;
  return dark.w + dark.b > 0 && light.w + light.b > 0;
}

/** A finished game's result. */
export interface GameResult {
  /** The PGN result: "1-0", "0-1" or "½-½". */
  score: string;
  /** The outcome and its reason, e.g. "White wins by checkmate". */
  text: string;
}

/**
 * The result of a finished game. Who won is the engine's (returns, player 1
 * is White); the reason is the position's, in the engine's own order
 * (MaybeFinalReturns): insufficient material, threefold repetition, no legal
 * move (checkmate or stalemate), then the fifty-move rule. The adapter
 * reads checkmate, stalemate and repetition from the engine's state. A draw
 * whose reason cannot be read from the position says only "Draw".
 */
export function gameResult(
  fen: string,
  outcome: ChessOutcome,
  returns: readonly number[] | null | undefined,
): GameResult {
  let winner: Side | null = null;
  if (returns && returns.length >= 2) {
    if ((returns[SLOT_OF.white] ?? 0) > 0) winner = 'white';
    else if ((returns[SLOT_OF.black] ?? 0) > 0) winner = 'black';
  } else if (outcome.checkmate) {
    winner = fen.trim().split(/\s+/)[1] === 'w' ? 'black' : 'white';
  }
  if (winner) {
    return {
      score: winner === 'white' ? '1-0' : '0-1',
      text: outcome.checkmate
        ? `${SIDE_NAME[winner]} wins by checkmate`
        : `${SIDE_NAME[winner]} wins`,
    };
  }
  const draw = (reason: string | null): GameResult => ({
    score: '½-½',
    text: reason ? `Draw by ${reason}` : 'Draw',
  });
  if (!hasSufficientMaterial(fen)) return draw('insufficient material');
  if (outcome.threefold) return draw('threefold repetition');
  if (outcome.stalemate) return draw('stalemate');
  const halfmoves = Number(fen.trim().split(/\s+/)[4]);
  if (Number.isFinite(halfmoves) && halfmoves >= 100) {
    return draw('the fifty-move rule');
  }
  return draw(null);
}

/** The taken-piece pile on each strip (ChessPlayerStrip): piece size, the
 *  gap between kinds and the usual overlap within a kind, in rem, as
 *  chess.css draws them. */
export const PIECE_REM = 1.25;
/** The gap between kinds of piece in a pile, in rem. */
export const GROUP_GAP_REM = 0.3;
const STEP_REM = 0.7;
/** The narrowest step a pile may take before it is clipped. */
const MIN_STEP_REM = 0.2;
/** The widest the pile may grow: what is left of the narrowest strip (a
 *  390 px phone) once the side's name, the lead and the longest turn word
 *  have their room. */
export const PILE_MAX_REM = 7.5;

/**
 * How far each further piece of a kind sits from the one before it, in rem:
 * the usual overlap, tightened when the pile would grow past PILE_MAX_REM,
 * as a long row of pieces bunches beside a real board. Every piece stays
 * drawn; only the overlap changes.
 */
export function pileStep(groupSizes: readonly number[]): number {
  const pieces = groupSizes.reduce((a, b) => a + b, 0);
  const groups = groupSizes.length;
  const extra = pieces - groups;
  if (extra <= 0) return STEP_REM;
  const room = PILE_MAX_REM - groups * PIECE_REM - (groups - 1) * GROUP_GAP_REM;
  return Math.min(STEP_REM, Math.max(MIN_STEP_REM, room / extra));
}

/** The pile's width in rem at pileStep's overlap. */
export function pileWidth(groupSizes: readonly number[]): number {
  const groups = groupSizes.length;
  if (groups === 0) return 0;
  const pieces = groupSizes.reduce((a, b) => a + b, 0);
  return (
    groups * PIECE_REM +
    (groups - 1) * GROUP_GAP_REM +
    (pieces - groups) * pileStep(groupSizes)
  );
}

// The solid glyphs: the outline ones (U+2654-U+2658) come out of the
// fallback symbol fonts as hairlines, about 2.3:1 against the move text.
const FIGURINE: Record<string, string> = {
  K: '♚',
  Q: '♛',
  R: '♜',
  B: '♝',
  N: '♞',
};
const SAN =
  /^(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?$/;

/**
 * Figurine algebraic notation, the default move list of lichess and
 * chess.com: the piece letters of a SAN move become piece symbols ("Nf3" is
 * "♞f3", "e8=Q+" is "e8=♛+"). Anything that is not a SAN move is returned
 * as it is.
 */
export function toFigurine(label: string): string {
  if (!SAN.test(label)) return label;
  return label.replace(/[KQRBN]/g, letter => FIGURINE[letter] ?? letter);
}

/**
 * Who plays a seat, named as the Players panel names it: the seat's display
 * name, else its controller's name ("Human", "Stockfish Bot").
 */
export function playerName(
  entry:
    | {slot: number; controller?: string; displayName?: string | null}
    | undefined,
  controllers: ReadonlyArray<{id: string; label: string}>,
): string {
  const display = entry?.displayName?.trim();
  if (display) return display;
  const controller = entry?.controller;
  const option = controllers.find(c => c.id === controller);
  if (option) return option.label;
  if (controller === HUMAN) return 'Human';
  return controller !== undefined && controller !== ''
    ? controller
    : 'Open seat';
}
