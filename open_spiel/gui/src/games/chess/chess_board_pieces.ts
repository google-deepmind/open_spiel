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

// The pieces the board draws, each with an id that follows it from one
// position to the next, so a piece that moved slides from its old square to
// its new one rather than vanishing and reappearing. Pure, so each rule is
// tested on its own.

import {type Placement, squareIndex} from './chess_board_config.ts';

/** A piece on the board, with the id that follows it. */
export interface TrackedPiece {
  readonly id: number;
  readonly square: string;
  /** Its FEN letter: upper case white. */
  readonly letter: string;
}

/** The pieces of a position and the next free id. */
export interface Tracked {
  /** The placement the pieces were read from, as a key. */
  readonly key: string;
  readonly pieces: readonly TrackedPiece[];
  readonly nextId: number;
}

/** A placement as a key: its squares and letters, in square order. */
export function placementKey(placement: Placement): string {
  return [...placement.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([square, letter]) => `${square}${letter}`)
    .join(',');
}

function distance(a: string, b: string): number {
  const from = squareIndex(a);
  const to = squareIndex(b);
  return Math.abs(from.file - to.file) + Math.abs(from.rank - to.rank);
}

/** Fresh ids for every piece of `placement`. */
export function trackFresh(placement: Placement): Tracked {
  const pieces = [...placement.entries()].map(([square, letter], id) => ({
    id,
    square,
    letter,
  }));
  return {key: placementKey(placement), pieces, nextId: pieces.length};
}

/**
 * The pieces of `placement`, keeping the ids of `previous`: a piece that
 * stayed keeps its id; a piece that appeared takes the id of the nearest
 * piece of the same kind and colour that disappeared (it moved there); any
 * other is new (a promotion's new piece).
 */
export function track(previous: Tracked, placement: Placement): Tracked {
  const left = new Map(previous.pieces.map(piece => [piece.id, piece]));
  const pieces: TrackedPiece[] = [];
  const arrived: Array<[string, string]> = [];
  for (const [square, letter] of placement) {
    const stayed = previous.pieces.find(
      piece => piece.square === square && piece.letter === letter,
    );
    if (stayed === undefined) {
      arrived.push([square, letter]);
    } else {
      pieces.push(stayed);
      left.delete(stayed.id);
    }
  }
  let nextId = previous.nextId;
  for (const [square, letter] of arrived) {
    let nearest: TrackedPiece | null = null;
    for (const piece of left.values()) {
      if (piece.letter !== letter) continue;
      if (
        nearest === null ||
        distance(piece.square, square) < distance(nearest.square, square)
      ) {
        nearest = piece;
      }
    }
    if (nearest === null) {
      pieces.push({id: nextId, square, letter});
      nextId += 1;
    } else {
      left.delete(nearest.id);
      pieces.push({id: nearest.id, square, letter});
    }
  }
  pieces.sort((a, b) => a.id - b.id);
  return {key: placementKey(placement), pieces, nextId};
}

/** `placement` with the piece on `orig` moved to `dest`. */
export function moved(
  placement: Placement,
  orig: string,
  dest: string,
): Placement {
  const letter = placement.get(orig);
  if (letter === undefined) return placement;
  const next = new Map(placement);
  next.delete(orig);
  next.set(dest, letter);
  return next;
}
