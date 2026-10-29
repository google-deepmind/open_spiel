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

// The facts the chess table shows around the board, each checked against
// the rule it comes from: lichess's material count, the engine's own draw
// conditions in the engine's order (open_spiel chess.cc MaybeFinalReturns,
// chess_board.cc HasSufficientMaterial), and figurine notation. Checkmate,
// stalemate and repetition are the engine's, read by the adapter.
import {describe, expect, it} from 'vitest';
import {
  gameResult,
  hasSufficientMaterial,
  materialDiff,
  pileStep,
  pileWidth,
  playerName,
  toFigurine,
} from './chess_table.ts';

const NONE = {checkmate: false, stalemate: false, threefold: false};
const MATE = {...NONE, checkmate: true};

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('materialDiff', () => {
  it('is level at the start', () => {
    expect(materialDiff(START)).toEqual({
      white: {taken: [], lead: 0},
      black: {taken: [], lead: 0},
    });
  });

  it('shows a capture beside the side that made it, with its points', () => {
    // 1. e4 d5 2. exd5: White has taken a pawn.
    const fen = 'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2';
    expect(materialDiff(fen).white).toEqual({taken: ['pawn'], lead: 1});
    expect(materialDiff(fen).black).toEqual({taken: [], lead: 0});
  });

  it('nets the pieces both sides have taken, cheapest first', () => {
    // White is a knight and a pawn down, Black is a rook down.
    const fen = 'rnbqkbn1/pppppppp/8/8/8/8/PPPPPPP1/R1BQKBNR w KQq - 0 1';
    const diff = materialDiff(fen);
    expect(diff.white.taken).toEqual(['rook']);
    expect(diff.black.taken).toEqual(['pawn', 'knight']);
    expect(diff.white.lead).toBe(1);
    expect(diff.black.lead).toBe(0);
  });

  it('reads a promotion from the position: a second queen, one pawn fewer', () => {
    // White's a-pawn has queened (the new queen on d5) and nothing was taken:
    // a queen up for White, a pawn up for Black, net 8 (lichess's count).
    const fen = 'rnbqkbnr/pppppppp/8/3Q4/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1';
    const diff = materialDiff(fen);
    expect(diff.white).toEqual({taken: ['queen'], lead: 8});
    expect(diff.black).toEqual({taken: ['pawn'], lead: 0});
  });
});

describe("hasSufficientMaterial (the engine's subset)", () => {
  it.each([
    ['K v K', '8/8/8/4k3/8/8/8/4K3 w - - 0 1', false],
    ['K+N v K', '8/8/8/4k3/8/8/8/1N2K3 w - - 0 1', false],
    ['K+B v K', '8/8/8/4k3/8/8/8/2B1K3 w - - 0 1', false],
    [
      'K+B v K+B, both on dark squares',
      '5b2/8/8/4k3/8/8/8/2B1K3 w - - 0 1',
      false,
    ],
    ['K+B v K+B, opposite colours', '2b5/8/8/4k3/8/8/8/2B1K3 w - - 0 1', true],
    [
      'K+N v K+B (helpmate possible)',
      '8/8/8/4k3/8/2b5/8/1N2K3 w - - 0 1',
      true,
    ],
    ['K+N v K+N', '8/8/8/4k3/8/2n5/8/1N2K3 w - - 0 1', true],
    ['K+N+N v K', '8/8/8/4k3/8/8/8/1N2K1N1 w - - 0 1', true],
    ['a pawn', '8/8/8/4k3/8/8/P7/4K3 w - - 0 1', true],
  ])('%s', (_name, fen, sufficient) => {
    expect(hasSufficientMaterial(fen)).toBe(sufficient);
  });
});

describe('gameResult', () => {
  it('names the winner from the returns (player 1 is White) and checkmate from the position', () => {
    const fen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    expect(gameResult(fen, MATE, [1, -1])).toEqual({
      score: '0-1',
      text: 'Black wins by checkmate',
    });
  });

  it('says only who won when the position is not mate', () => {
    expect(gameResult(START, NONE, [-1, 1])).toEqual({
      score: '1-0',
      text: 'White wins',
    });
  });

  it('stalemate', () => {
    expect(
      gameResult(
        '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
        {...NONE, stalemate: true},
        [0, 0],
      ),
    ).toEqual({score: '½-½', text: 'Draw by stalemate'});
  });

  it('insufficient material', () => {
    expect(
      gameResult('8/8/8/4k3/8/8/8/4K3 w - - 0 60', NONE, [0, 0]).text,
    ).toBe('Draw by insufficient material');
  });

  it('threefold repetition, as the engine reports it', () => {
    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5';
    expect(gameResult(fen, {...NONE, threefold: true}, [0, 0]).text).toBe(
      'Draw by threefold repetition',
    );
    expect(gameResult(fen, NONE, [0, 0]).text).toBe('Draw');
  });

  it('the fifty-move rule at 100 plies without a capture or pawn move ', () => {
    expect(
      gameResult('8/8/8/4k3/8/8/R7/4K3 b - - 100 120', NONE, [0, 0]).text,
    ).toBe('Draw by the fifty-move rule');
  });

  it('checkmate governs a move that also completes the fifty moves ', () => {
    // Back-rank mate with the halfmove clock at 100.
    const fen = '6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 120';
    const after = 'R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 120';
    expect(hasSufficientMaterial(fen)).toBe(true);
    expect(gameResult(after, MATE, [-1, 1])).toEqual({
      score: '1-0',
      text: 'White wins by checkmate',
    });
  });

  it('says "Draw" with no reason when the position shows none', () => {
    expect(gameResult(START, NONE, [0, 0])).toEqual({
      score: '½-½',
      text: 'Draw',
    });
  });
});

describe('toFigurine', () => {
  it.each([
    ['Nf3', '♞f3'],
    ['Qxe7+', '♛xe7+'],
    ['Rad1', '♜ad1'],
    ['Bb5#', '♝b5#'],
    ['Kg1', '♚g1'],
    ['e4', 'e4'],
    ['exd8=Q+', 'exd8=♛+'],
    ['O-O-O', 'O-O-O'],
    ['Chance outcome (hidden)', 'Chance outcome (hidden)'],
  ])('%s -> %s', (san, shown) => {
    expect(toFigurine(san)).toBe(shown);
  });
});

describe('pileStep', () => {
  it('keeps the usual overlap for a short pile', () => {
    expect(pileStep([2, 1])).toBe(0.7);
  });

  it('tightens a long pile to stay within the budget', () => {
    const sizes = [8, 2, 2];
    expect(pileStep(sizes)).toBeLessThan(0.7);
    expect(pileWidth(sizes)).toBeLessThanOrEqual(7.5 + 1e-9);
  });
});

describe('playerName', () => {
  const controllers = [
    {id: 'human', label: 'Human'},
    {id: 'random', label: 'Random'},
  ];

  it("names a seat by its controller's label", () => {
    expect(playerName({slot: 0, controller: 'random'}, controllers)).toBe(
      'Random',
    );
  });

  it('prefers a display name', () => {
    expect(
      playerName(
        {slot: 0, controller: 'random', displayName: 'Ada'},
        controllers,
      ),
    ).toBe('Ada');
  });

  it('calls an unlisted human controller Human', () => {
    expect(playerName({slot: 1, controller: 'human'}, [])).toBe('Human');
    expect(playerName(undefined, [])).toBe('Open seat');
  });
});
