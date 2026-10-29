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

// The chess piece-set registry: every set the board, the players' strips and
// the promotion choice may draw their pieces with, by id. A set is one
// module, a label and a component that draws one piece; the Staunton set
// drawn for this client is the default. A set fetched by the user
// (fetched_sets.tsx) is offered only when its files are present.

import {fetchedPieceSets} from './fetched_sets.tsx';
import type {PieceSetModule} from './piece_types.ts';
import {STAUNTON_PIECES} from './staunton.tsx';

/** The id of the set pieces are drawn with until the viewer picks another. */
export const DEFAULT_PIECE_SET = 'staunton';

const sets = new Map<string, PieceSetModule>([
  [DEFAULT_PIECE_SET, STAUNTON_PIECES],
  ...fetchedPieceSets(),
]);
let ids: readonly string[] = [...sets.keys()];

/**
 * Registers `module` as the piece set `id`, offered from then on in the
 * Board settings.
 *
 * @returns A function that removes the set again (for a test).
 * @throws Error when `id` is blank or already registered.
 */
export function registerPieceSet(
  id: string,
  module: PieceSetModule,
): () => void {
  if (id.trim() === '') throw new Error('A piece set has a blank id.');
  if (sets.has(id)) {
    throw new Error(`Piece set "${id}" is already registered.`);
  }
  sets.set(id, module);
  ids = [...sets.keys()];
  return () => {
    if (sets.get(id) !== module) return;
    sets.delete(id);
    ids = [...sets.keys()];
  };
}

/**
 * The piece set `id`.
 *
 * @throws Error naming the registered ids when there is no set `id`.
 */
export function getPieceSet(id: string): PieceSetModule {
  const module = sets.get(id);
  if (module === undefined) {
    throw new Error(
      `There is no piece set "${id}"; the sets are ${ids.join(', ')}.`,
    );
  }
  return module;
}

/** The ids of every registered set, the default first. */
export function pieceSetIds(): readonly string[] {
  return ids;
}
