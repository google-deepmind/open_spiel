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

// Playable cards stand up; the rest are never dimmed. Pure.
//
// The convention:
// on your turn, when only SOME of your cards may go, those cards stand up out
// of the hand (a game-tag lift, which the scene lays out) and the rest keep
// full contrast: no tint, no grayscale or brightness filter, no opacity. A
// tap on an unplayable card still answers (the game says why not). When every
// card may go (a free lead) nothing stands up: a raise that marks every card
// tells the player nothing. Off turn nothing stands up either.
import type {TagStyle} from './core/tags.ts';

/** The game tag a raised playable card carries. */
export const PLAYABLE_TAG = 'playable';

/** How far a playable card stands up out of the hand, px. */
export const PLAYABLE_LIFT_PX = 10;

/** The tag style for PLAYABLE_TAG: a lift and nothing else. Put it in a
 *  game's CardStyleOverride.tags as `[PLAYABLE_TAG]: PLAYABLE_TAG_STYLE`. */
export const PLAYABLE_TAG_STYLE: Readonly<TagStyle> = Object.freeze({
  lift: PLAYABLE_LIFT_PX,
});

/** Whether raising the playable cards of `hand` tells them apart: true when
 *  some card may go and some may not. False on a free lead (all may go) and
 *  when none may (not this seat's turn to play). */
export function raisesPlayable<T>(
  hand: Iterable<T>,
  isPlayable: (card: T) => boolean,
): boolean {
  let some = false;
  let notAll = false;
  for (const c of hand) {
    if (isPlayable(c)) some = true;
    else notAll = true;
    if (some && notAll) return true;
  }
  return false;
}

/** The tags for one card: [PLAYABLE_TAG] when the hand is raised
 *  (raisesPlayable) and this card may go, else none. */
export function playableTags(raise: boolean, playable: boolean): string[] {
  return raise && playable ? [PLAYABLE_TAG] : [];
}
