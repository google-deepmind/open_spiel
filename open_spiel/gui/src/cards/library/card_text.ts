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

// Card codes in engine text as the player reads them: a player never sees
// developer text. Engines label moves with card codes and, for a
// double pack, instance ids: "Play KS.1", "Deal AH.2 to P0", "Trail TH".
// A player never sees a code or a pack number: every card token reads
// rank and suit glyph ("K♠", "A♥", "10♥"). One formatter for every game's
// visible strings, tooltips and accessible names. Pure.
import {SUIT_GLYPH} from './model.ts';
import type {Suit} from './types.ts';

/**
 * A card token: rank (10 or T for ten, 2-9, J, Q, K, A), suit letter, and an
 * optional instance suffix (".1", ".2"). Case-sensitive and word-bounded, so
 * ordinary words ("As", "Each", "AT hand" with T for an unknown trump) are
 * never touched; "JKR"/"JKB" are jokers.
 */
const CARD_TOKEN = /\b(10|[2-9TJQKA])([SHDC])(?:\.\d+)?\b/g;
const JOKER_TOKEN = /\bJK([RB])\b/g;

/** One card code or instance id as a player reads it: "QS.2" -> "Q♠", "TH" -> "10♥", "JKR" -> "Red joker". */
export function cardText(code: string): string {
  if (!/^(?:(?:10|[2-9TJQKA])[SHDC](?:\.\d+)?|JK[RB])$/.test(code)) {
    throw new Error(`cardText: ${JSON.stringify(code)} is not a card code`);
  }
  return cardTokensText(code);
}

/** Every card token in `text` as rank and suit glyph; the rest of the text unchanged. */
export function cardTokensText(text: string): string {
  return text
    .replace(
      CARD_TOKEN,
      (_m, rank: string, suit: string) =>
        `${rank === 'T' ? '10' : rank}${SUIT_GLYPH[suit as Suit]}`,
    )
    .replace(JOKER_TOKEN, (_m, c: string) =>
      c === 'R' ? 'Red joker' : 'Black joker',
    );
}

/** True when `text` still carries an engine card code or instance id (a test and audit probe). */
export function hasCardCode(text: string): boolean {
  return (
    new RegExp(CARD_TOKEN.source).test(text) ||
    /\b[2-9TJQKA][SHDC]\.\d\b/.test(text)
  );
}
