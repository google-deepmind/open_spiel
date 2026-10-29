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

// Where a table region (RegionSpec) is drawn: at its own place, or attached
// to a side of a slot (or of the union of several).
import type {RegionSpec} from '../table_spec.ts';
import type {Box} from './identity.ts';
import {boxQuad, norm, toTablePoint, unionBox} from './scene_geometry.ts';
import type {RegionPlacement, SlotBox, SlotLayout} from './scene_types.ts';

type Attach = NonNullable<RegionSpec['attach']>;

// The box an attached region is attached to.
function attachBox(
  r: RegionSpec,
  a: Attach,
  slots: Map<string, SlotLayout>,
): SlotBox {
  const slotOf = (id: string): SlotLayout => {
    const o = slots.get(id);
    if (!o) {
      throw new Error(
        `table region '${r.id}' attaches to unknown slot '${id}'`,
      );
    }
    return o;
  };
  const l = slotOf(a.slot);
  return a.also ? unionBox(r.id, [l, ...a.also.map(slotOf)]) : l.box;
}

// The region's top-left in the attached box's frame, for its local size.
function attachedAt(
  r: RegionSpec,
  a: Attach,
  b: SlotBox,
  lw: number,
  lh: number,
  gap: number,
): {x: number; y: number} {
  const align = a.align ?? 'center';
  const along = (start: number, size: number, own: number): number =>
    align === 'start'
      ? start
      : align === 'end'
        ? start + size - own
        : start + (size - own) / 2;
  switch (a.side) {
    case 'bottom':
      return {x: along(0, b.width, lw), y: b.height + gap};
    case 'top':
      return {x: along(0, b.width, lw), y: -gap - lh};
    case 'left':
      return {x: -gap - lw, y: along(0, b.height, lh)};
    case 'right':
      return {x: b.width + gap, y: along(0, b.height, lh)};
    default: {
      const side: unknown = a.side;
      throw new Error(
        `table region '${r.id}' attach.side must be top/bottom/left/right, got ${String(side)}`,
      );
    }
  }
}

/** Where region `r` is drawn on a table `tableWidth` px wide. */
export function placeRegion(
  r: RegionSpec,
  slots: Map<string, SlotLayout>,
  cardWidth: number,
  tableWidth: number,
): RegionPlacement {
  const w = r.width * cardWidth;
  const h = r.height * cardWidth;
  const occludes = r.occludes ?? true;
  const droppable = r.droppable ?? false;
  const a = r.attach;
  if (!a) {
    if (r.x === undefined || r.y === undefined) {
      throw new Error(`table region '${r.id}' has neither attach nor x, y`);
    }
    return {
      id: r.id,
      box: {
        x: r.x * cardWidth,
        y: r.y * cardWidth,
        width: w,
        height: h,
        rotate: 0,
      },
      upright: true,
      occludes,
      droppable,
    };
  }
  const b = attachBox(r, a, slots);
  const upright = r.upright ?? true;
  // An upright box on a quarter-turned slot is drawn w x h on screen, which
  // in the slot's local frame is h x w: place THAT footprint, so the box
  // sits `gap` from the slot on every seat rather than overlapping it (or
  // floating off it) by half the difference.
  const quarter = upright && (norm(b.rotate) === 90 || norm(b.rotate) === 270);
  const lw = quarter ? h : w;
  const lh = quarter ? w : h;
  const {x, y} = attachedAt(r, a, b, lw, lh, (a.gap ?? 0) * cardWidth);
  const c = toTablePoint(b, {x: x + lw / 2, y: y + lh / 2});
  const box: Box = {
    x: c.x - w / 2,
    y: c.y - h / 2,
    width: w,
    height: h,
    rotate: upright ? 0 : norm(b.rotate),
  };
  const q = boxQuad(box);
  if (q.some(p => p.x < -0.5 || p.x > tableWidth + 0.5 || p.y < -0.5)) {
    throw new Error(
      `table region '${r.id}' (attached ${a.side} of '${a.slot}') lies outside the table: it would clip`,
    );
  }
  return {id: r.id, box, upright, occludes, droppable};
}
