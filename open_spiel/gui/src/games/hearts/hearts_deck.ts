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

// Hearts on the shared table and card settings: the deck, back and suit
// symbols are the shared ones, with a Hearts-only override.
import type {TableCardScope} from '../../cards/table_cards/index.ts';

/** Hearts' id in the table and card settings. */
export const HEARTS_TABLE_CARDS_GAME = 'hearts';

/** Hearts' scope in the table and card settings. */
export function useHeartsTableCardScope(): TableCardScope {
  return {game: HEARTS_TABLE_CARDS_GAME};
}
