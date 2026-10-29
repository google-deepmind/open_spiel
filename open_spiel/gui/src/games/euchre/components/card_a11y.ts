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

// Accessible names and roles for euchre's cards, from the shared
// card library: the name is shared/cards cardLabel, the role follows
// shared/cards core/a11y.
import {
  cardLabel,
  type Rank as SharedRank,
  type Suit as SharedSuit,
} from '../../../cards/library/index.ts';
import {
  type CardCode,
  type EuchreRank,
  type Suit,
  HIDDEN_CARD,
} from '../types.ts';

const SHARED_SUIT: Record<Suit, SharedSuit> = {h: 'H', d: 'D', c: 'C', s: 'S'};
const SHARED_RANK: Record<EuchreRank, SharedRank> = {
  '9': '9',
  T: '10',
  J: 'J',
  Q: 'Q',
  K: 'K',
  A: 'A',
};

/**
 * The accessible name of a euchre card, from the shared card library's
 * cardLabel (the name gin's cards carry): "Ah" -> "Ace of Hearts",
 * "Td" -> "Ten of Diamonds".
 */
export function euchreCardLabel(card: CardCode): string {
  const rank = card[0] as EuchreRank;
  const suit = card[1] as Suit;
  if (card.length !== 2 || !(rank in SHARED_RANK) || !(suit in SHARED_SUIT)) {
    throw new Error(`euchre: not a card code: ${JSON.stringify(card)}`);
  }
  return cardLabel({rank: SHARED_RANK[rank], suit: SHARED_SUIT[suit]});
}

/**
 * The shared card library's role and name rules (shared/cards core/a11y):
 * an interactive card is a button, a face-up card is an image, both named by
 * the card; a face-down card that cannot be played is decorative (its hand's
 * group names the count).
 */
export function cardA11y(
  card: CardCode | null | undefined,
  faceUp: boolean,
  interactive: boolean,
): {role?: 'button' | 'img'; 'aria-label'?: string; 'aria-hidden'?: true} {
  const known = faceUp && !!card && card !== HIDDEN_CARD;
  const label = known ? euchreCardLabel(card) : 'Face-down card';
  if (interactive) return {role: 'button', 'aria-label': label};
  if (known) return {role: 'img', 'aria-label': label};
  return {'aria-hidden': true};
}
