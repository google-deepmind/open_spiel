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

// The trick's seats and card identity, shared by the trick pile, the play
// staging and the flights.
import type {TrickSeat} from '../../cards/library/index.ts';
import type {PlayerPosition, TrickCard} from './types.ts';

/** A table position as a compass seat of the trick area (shared/cards
 *  trickArea): the viewer, at the bottom, is south; play goes clockwise:
 *  bottom, left, top, right. */
export const TRICK_SEAT: Record<PlayerPosition, TrickSeat> = {
  bottom: 's',
  left: 'w',
  top: 'n',
  right: 'e',
};

/** A trick card's identity on the table: who played it, and which card. */
export function trickKey(c: TrickCard): string {
  return `${c.position}:${c.card}`;
}
