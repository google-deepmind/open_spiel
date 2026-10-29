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

// The deck the gin board's grids draw: its suits and ranks. A board without
// a provider draws the full 52-card deck.

import {createContext, useContext} from 'react';

import {RANK_CODES, SUIT_ROWS} from './types.ts';

/** The suit rows and rank columns of the deck a gin board deals. */
export interface DeckConfig {
  suitRows: ReadonlyArray<{
    readonly code: string;
    readonly label: string;
    readonly short: string;
  }>;
  rankCodes: readonly string[];
}

/** The full 52-card deck. */
export const FULL_DECK: DeckConfig = {
  suitRows: SUIT_ROWS,
  rankCodes: RANK_CODES,
};

const DeckContext = createContext<DeckConfig>(FULL_DECK);

/** Sets the deck of the gin board under it. */
export const DeckProvider = DeckContext.Provider;

/** The deck of the gin board a component is in. */
export function useDeck(): DeckConfig {
  return useContext(DeckContext);
}
