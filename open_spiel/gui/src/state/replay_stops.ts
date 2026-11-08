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

import type {HistoryEntry} from '../contract/contract.ts';

/**
 * The plies a replay control stops at, in ascending order.
 *
 * Every position from the initial one to the live one is a stop. When chance
 * is skipped, the stops are the initial position, each position where a seat
 * is to move, and the live position. `shown`, the position on screen, is
 * always a stop, so that the control can say exactly where it stands.
 *
 * A seat is to move at ply p exactly when `history[p]` is a seat's action:
 * which seat acted is public, so every viewer gets the same stops.
 *
 * @throws RangeError when `shown` is not a ply of `history`.
 */
export function replayStops(
  history: readonly HistoryEntry[],
  shown: number,
  skipChance: boolean,
): number[] {
  const live = history.length;
  if (!Number.isInteger(shown) || shown < 0 || shown > live) {
    throw new RangeError(
      `Ply ${String(shown)} is not between 0 and ${String(live)}.`,
    );
  }
  if (!skipChance) {
    return Array.from({length: live + 1}, (_, ply) => ply);
  }
  const stops = new Set([0, shown, live]);
  history.forEach((entry, ply) => {
    if (entry.seat !== null) stops.add(ply);
  });
  return [...stops].sort((left, right) => left - right);
}
