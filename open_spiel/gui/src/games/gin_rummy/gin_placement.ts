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

// Where each physical card on the gin board may be drawn: once. Pure helpers
// for the board (gin_rummy_board_surface.tsx).
import type {TableItem} from '../../cards/library/index.ts';
import type {CardCellState} from './components/index.ts';
import {GIN_SLOT, GRID_PREFIX} from './gin_rummy_layout.ts';

/** The codes a grid's states hold as cards (not ghosts, not empty cells). */
export function heldCodes(
  states: ReadonlyMap<string, CardCellState>,
): Set<string> {
  const out = new Set<string>();
  states.forEach((st, code) => {
    if (st.type === 'card' && st.isGhost !== true) out.add(code);
  });
  return out;
}

/**
 * Is `settled` (the opponent's face-up grid at the board's last rest) the
 * start of the turn whose final grid is `final`? A turn draws one card and
 * discards one, so its start holds as many cards, all from the final hand or
 * the discard, and not the upcard it took (it has not drawn it yet). After a
 * jump (a scrub, a reset, a turn that coalesced over one still staging) the
 * settled grid is another position's and fails this.
 */
export function settledStartFits(
  settled: ReadonlyMap<string, CardCellState>,
  final: ReadonlyMap<string, CardCellState>,
  discarded: string,
  taken: string | null,
): boolean {
  const start = heldCodes(settled);
  const end = heldCodes(final);
  if (start.size !== end.size) return false;
  if (taken !== null && taken !== discarded && start.has(taken)) return false;
  return [...start].every(c => end.has(c) || c === discarded);
}

/**
 * One physical card in one place. The viewer's own slots (the hand, its
 * grid) are kept first, then the public pile; a later item under an id
 * already placed is dropped. A duplicate id would otherwise throw in the card
 * library's assertItems and take the whole page down with it ("duplicate
 * item id 'Ac' (in slots 'opp-Ac' and 'me-Ac')"). Each drop is a
 * fault, never absorbed: it is returned (the board shows it on screen) and
 * logged with console.error.
 */
export function oneCardOnePlace(dealt: Record<string, TableItem[]>): {
  items: Record<string, TableItem[]>;
  faults: string[];
} {
  const rank = (slot: string): number =>
    slot === GIN_SLOT.hand || slot.startsWith(GRID_PREFIX.player)
      ? 0
      : slot === GIN_SLOT.discard
        ? 1
        : 2;
  const ordered = Object.keys(dealt).sort((a, b) => rank(a) - rank(b));
  const placed = new Map<string, string>();
  const out: Record<string, TableItem[]> = {};
  const faults: string[] = [];
  for (const slot of ordered) {
    out[slot] = (dealt[slot] ?? []).filter(item => {
      const prev = placed.get(item.id);
      if (prev !== undefined) {
        const fault = `card '${item.id}' is in both '${prev}' and '${slot}'; drawn in '${prev}' only`;
        console.error(`gin board: ${fault}`);
        faults.push(fault);
        return false;
      }
      placed.set(item.id, slot);
      return true;
    });
  }
  return {items: out, faults};
}
