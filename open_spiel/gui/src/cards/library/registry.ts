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

// Deck registry. Adding or removing a deck is one registerDeck() call in
// decks/index.ts; no game code changes. Unknown ids and duplicate
// registrations fail loud.
import type {Deck} from './types.ts';

const decks = new Map<string, Deck>();

/** Registers `deck`; throws on a repeated id or a malformed deck. */
export function registerDeck(deck: Deck): void {
  if (decks.has(deck.id)) {
    throw new Error(`card deck '${deck.id}' is already registered`);
  }
  if (!deck.provenance) {
    throw new Error(`card deck '${deck.id}' has no provenance line`);
  }
  if (!(deck.aspect > 0)) {
    throw new Error(
      `card deck '${deck.id}' has invalid aspect ${String(deck.aspect)}`,
    );
  }
  const {top, left, width, height} = deck.index;
  for (const v of [top, left, width, height]) {
    if (!(v >= 0 && v <= 1)) {
      throw new Error(
        `card deck '${deck.id}' index geometry out of [0,1]: ${JSON.stringify(deck.index)}`,
      );
    }
  }
  if (left + width > 1 || top + height > 1) {
    throw new Error(
      `card deck '${deck.id}' index box exceeds the card: ${JSON.stringify(deck.index)}`,
    );
  }
  decks.set(deck.id, deck);
}

/** The deck registered as `id`; throws on an unknown id. */
export function getDeck(id: string): Deck {
  const deck = decks.get(id);
  if (!deck) {
    throw new Error(
      `unknown card deck '${id}'; registered: ${[...decks.keys()].join(', ') || '(none)'}`,
    );
  }
  return deck;
}

/** Every registered deck. */
export function listDecks(): Deck[] {
  return [...decks.values()];
}

/** Test-only: remove a deck so specs can register fixtures without leaking. */
export function unregisterDeckForTest(id: string): void {
  decks.delete(id);
}
