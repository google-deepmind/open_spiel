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

// The Scene object layoutScene returns, built from its laid-out slots.
import {defined} from './defined.ts';
import {insertBox, toTableBox} from './scene_geometry.ts';
import type {
  Placement,
  RegionPlacement,
  Scene,
  SlotLayout,
} from './scene_types.ts';

/** The scene's placements in paint order: slot by slot, tag z bumps
 *  reordering only within their slot. */
export function paintOrder(slots: readonly SlotLayout[]): Placement[] {
  let paint = 0;
  for (const lay of slots) {
    for (const e of lay.entries) e.placement.z = paint++;
  }
  const placements: Placement[] = [];
  for (const lay of slots) {
    const sorted = [...lay.entries].sort(
      (a, b) => a.zBump - b.zBump || a.placement.z - b.placement.z,
    );
    const zs = lay.entries.map(e => e.placement.z).sort((a, b) => a - b);
    sorted.forEach((e, k) => {
      e.placement.z = defined(zs[k], `paint slot ${String(k)}`);
    });
    for (const e of sorted) placements.push(e.placement);
  }
  return placements.sort((a, b) => a.z - b.z);
}

function slotOf(slotById: Map<string, SlotLayout>, slot: string): SlotLayout {
  const l = slotById.get(slot);
  if (!l) throw new Error(`scene has no slot '${slot}'`);
  return l;
}

/** The Scene object over laid-out slots and regions. */
export function sceneOf(
  dims: {cardWidth: number; width: number; height: number; variant: number},
  slots: readonly SlotLayout[],
  placements: Placement[],
  regions: RegionPlacement[],
  slotById: Map<string, SlotLayout>,
): Scene {
  const byId = new Map(placements.map(p => [p.id, p] as const));
  const inSlot = new Map<string, Placement[]>(
    slots.map(l => [
      l.spec.id,
      l.entries.map(e => e.placement).sort((a, b) => a.z - b.z),
    ]),
  );
  return {
    ...dims,
    placements,
    regions,
    byId: id => byId.get(id),
    inSlot: slot => {
      const l = inSlot.get(slot);
      if (!l) throw new Error(`scene has no slot '${slot}'`);
      return l;
    },
    slotBox: slot => {
      const l = slotOf(slotById, slot);
      return toTableBox(
        l.box,
        {x: 0, y: 0, width: l.box.width, height: l.box.height},
        0,
      );
    },
    insertBox: (slot, index) => insertBox(slotOf(slotById, slot), index),
  };
}
