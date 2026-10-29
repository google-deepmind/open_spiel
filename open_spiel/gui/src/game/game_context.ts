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

import {createContext, useContext} from 'react';

import type {CatalogueEntry, Viewer} from '../contract/contract.ts';
import type {Session} from '../session/use_session.ts';

/** What every panel of a game page shares. */
export interface Game {
  /** The catalogue's entry for the game. */
  readonly entry: CatalogueEntry;
  /** The session the page shows, as `viewer` sees it. */
  readonly session: Session;
  /**
   * Who the page is shown to, for as long as it is open: the oracle in
   * analysis, a seat or a spectator in a game (see bottom_seat.ts).
   */
  readonly viewer: Viewer;
}

/** The game of the page; provided by `GamePage`. */
export const GameContext = createContext<Game | null>(null);

/** The game of the page a panel is in. */
export function useGame(): Game {
  const game = useContext(GameContext);
  if (game === null) throw new Error('useGame is called outside a GamePage.');
  return game;
}
