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

// Slot blocks (SlotSpec.block) and leading holes: which items each slot
// draws, and how far a block's slots move so the block sits where the spec
// says.
import type {SlotBlock, SlotSpec} from '../table_spec.ts';
import {defined} from './defined.ts';
import type {Items, SlotItems, SlotLayout} from './scene_types.ts';

// Slots grouped by block id, for the blocks `wanted` keeps.
function blockGroups(
  slots: readonly SlotSpec[],
  wanted: (s: SlotSpec) => boolean,
): Map<string, SlotSpec[]> {
  const groups = new Map<string, SlotSpec[]>();
  for (const s of slots) {
    if (!s.block || !wanted(s)) continue;
    const g = groups.get(s.block.id);
    if (g) g.push(s);
    else groups.set(s.block.id, [s]);
  }
  return groups;
}

/** The items each slot draws: a packed block's slots that hold cards take
 *  its first slots, in spec order, and its empty ones come last
 *  (SlotSpec.block). Every other slot draws its own items. */
export function packBlocks(slots: readonly SlotSpec[], items: Items): Items {
  const groups = blockGroups(slots, s => s.block?.pack === true);
  if (groups.size === 0) return items;
  const out: Record<string, SlotItems> = {...items};
  for (const group of groups.values()) {
    const lists = group.map(s => items[s.id] ?? []).filter(l => l.length > 0);
    group.forEach((s, i) => {
      out[s.id] = lists[i] ?? [];
    });
  }
  return out;
}

/** SlotSpec.holeRoom 'leading': the positional holes that end a slot's
 *  items are drawn before its cards (after packing, so each list takes
 *  the room side of the slot it is drawn in). */
export function leadHoles(slots: readonly SlotSpec[], items: Items): Items {
  let out: Record<string, SlotItems> | null = null;
  for (const s of slots) {
    if (s.holeRoom !== 'leading') continue;
    const list = items[s.id] ?? [];
    let n = list.length;
    while (n > 0 && list[n - 1] === null) n--;
    if (n === list.length || n === 0) continue;
    out ??= {...items};
    out[s.id] = [...list.slice(n), ...list.slice(0, n)];
  }
  return out ?? items;
}

/** How far each block's slots move, card widths, so that the union of the
 *  drawn boxes of its slots that hold cards (or the room of positional
 *  holes) is centred on the block's x and y, or has its edge there
 *  (alignX, alignY; SlotSpec.block). `lay` lays a slot out where the spec
 *  puts it. */
export function blockShifts(
  slots: readonly SlotSpec[],
  cardWidth: number,
  lay: (s: SlotSpec) => SlotLayout,
): Map<string, {x: number; y: number}> {
  const groups = blockGroups(
    slots,
    s => s.block?.x !== undefined || s.block?.y !== undefined,
  );
  const out = new Map<string, {x: number; y: number}>();
  for (const group of groups.values()) {
    const b = defined(group[0]?.block, 'block of a block group');
    let left = Infinity;
    let right = -Infinity;
    let top = Infinity;
    let bottom = -Infinity;
    for (const s of group) {
      const l = lay(s);
      if (l.positions === 0) continue;
      // Block slots are unturned or half turned about their middle
      // (validateTableSpec): the box is what is drawn.
      left = Math.min(left, l.box.left);
      right = Math.max(right, l.box.left + l.box.width);
      top = Math.min(top, l.box.top);
      bottom = Math.max(bottom, l.box.top + l.box.height);
    }
    if (left === Infinity) continue;
    const at = (lo: number, hi: number, align: SlotBlock['alignX']): number =>
      (align === 'start' ? lo : align === 'end' ? hi : (lo + hi) / 2) /
      cardWidth;
    const d = {
      x: b.x === undefined ? 0 : b.x - at(left, right, b.alignX),
      y: b.y === undefined ? 0 : b.y - at(top, bottom, b.alignY),
    };
    for (const s of group) out.set(s.id, d);
  }
  return out;
}
