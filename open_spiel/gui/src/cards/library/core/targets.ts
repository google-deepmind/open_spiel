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

// Targets: what an event is about. Pure.
import type {CardId, TableItem} from './identity.ts';

/** An item on a table, with where it lies. */
export interface ItemRef {
  id: CardId;
  slot: string;
  position: number;
  item: TableItem;
}
/** What an event is about: a card, a slot, a gap, a region or the table. */
export type Target =
  | {kind: 'card'; ref: ItemRef} // also tokens (ref.item.kind === 'token')
  | {kind: 'slot'; slot: string} // includes EMPTY slots
  | {kind: 'insert'; slot: string; index: number} // a gap: 0..len
  | {kind: 'region'; region: string}
  | {kind: 'table'};
/** The kinds of target. */
export type TargetKind = Target['kind'];

/** Stable string key: 'card:QH', 'slot:t3', 'insert:hand:4', 'region:actions', 'table'. */
export function targetKey(t: Target): string {
  switch (t.kind) {
    case 'card':
      return `card:${t.ref.id}`;
    case 'slot':
      return `slot:${t.slot}`;
    case 'insert':
      return `insert:${t.slot}:${String(t.index)}`;
    case 'region':
      return `region:${t.region}`;
    case 'table':
      return 'table';
    default: {
      const unreachable: never = t;
      throw new Error(`Unhandled target: ${JSON.stringify(unreachable)}`);
    }
  }
}

/** Whether two targets (or nulls) name the same thing. */
export function sameTarget(a: Target | null, b: Target | null): boolean {
  if (a === null || b === null) return a === b;
  return targetKey(a) === targetKey(b);
}
