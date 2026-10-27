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

import {describe, expect, it} from 'vitest';

import {readPlacement} from './chess_board_config.ts';
import {moved, track, trackFresh} from './chess_board_pieces.ts';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function idOn(tracked: ReturnType<typeof trackFresh>, square: string): number {
  const piece = tracked.pieces.find(entry => entry.square === square);
  if (piece === undefined) throw new Error(`No piece on ${square}.`);
  return piece.id;
}

describe('track', () => {
  it("keeps a moved piece's id, so it slides", () => {
    const before = trackFresh(readPlacement(START));
    const after = track(
      before,
      readPlacement(
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
      ),
    );
    expect(idOn(after, 'e4')).toBe(idOn(before, 'e2'));
    expect(after.pieces).toHaveLength(32);
    expect(after.nextId).toBe(before.nextId);
  });

  it('drops a captured piece and keeps the capturer', () => {
    const before = trackFresh(
      readPlacement('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1'),
    );
    const after = track(
      before,
      readPlacement('4k3/8/8/3P4/8/8/8/4K3 b - - 0 1'),
    );
    expect(idOn(after, 'd5')).toBe(idOn(before, 'e4'));
    expect(after.pieces).toHaveLength(3);
  });

  it('gives a promoted piece a new id', () => {
    const before = trackFresh(readPlacement('4k3/1P6/8/8/8/8/8/4K3 w - - 0 1'));
    const after = track(
      before,
      readPlacement('1Q2k3/8/8/8/8/8/8/4K3 b - - 0 1'),
    );
    expect(idOn(after, 'b8')).toBe(before.nextId);
    expect(after.nextId).toBe(before.nextId + 1);
  });

  it('moves both pieces of a castling', () => {
    const before = trackFresh(readPlacement('4k3/8/8/8/8/8/8/4K2R w K - 0 1'));
    const after = track(
      before,
      readPlacement('4k3/8/8/8/8/8/8/5RK1 b - - 1 1'),
    );
    expect(idOn(after, 'g1')).toBe(idOn(before, 'e1'));
    expect(idOn(after, 'f1')).toBe(idOn(before, 'h1'));
  });

  it('matches the nearest piece of a kind', () => {
    const before = trackFresh(
      readPlacement('4k3/8/8/8/8/8/8/RN2K1NR w - - 0 1'),
    );
    const after = track(
      before,
      readPlacement('4k3/8/8/8/8/5N2/8/RN2K2R b - - 1 1'),
    );
    expect(idOn(after, 'f3')).toBe(idOn(before, 'g1'));
    expect(idOn(after, 'b1')).toBe(idOn(before, 'b1'));
  });
});

describe('moved', () => {
  it('moves a piece, taking what was on its square', () => {
    const placement = moved(
      readPlacement('4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1'),
      'e4',
      'd5',
    );
    expect(placement.get('d5')).toBe('P');
    expect(placement.has('e4')).toBe(false);
  });

  it('leaves the placement when the square is empty', () => {
    const placement = readPlacement(START);
    expect(moved(placement, 'e4', 'e5')).toBe(placement);
  });
});
