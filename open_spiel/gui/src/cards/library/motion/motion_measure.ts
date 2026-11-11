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

// Measuring what the layout drew: card, slot and region boxes, and the
// frames they are given in (MotionService.measure*).
import type {Box, CardId, Rect} from '../core/identity.ts';
import type {MountedTable} from '../react/runtime.ts';
import type {Frame} from './free_card.ts';
import {
  cssAttr,
  tableScale,
  type Flight,
  type MotionHost,
} from './motion_util.ts';

/** The DOM document, or an error naming what needs one. */
export function doc(): Document {
  if (typeof document === 'undefined') {
    throw new Error('motion needs a DOM document');
  }
  return document;
}

/** Measurement over a card scene's mounted tables (MotionEngine's base). */
export class MotionMeasure {
  readonly flights = new Map<CardId, Flight>();
  readonly host: MotionHost;

  constructor(host: MotionHost) {
    this.host = host;
  }

  protected tableOf(id: CardId): MountedTable | undefined {
    return this.host.tables().find(t => t.scene.byId(id) !== undefined);
  }

  /** The mounted table `tableId`, or the scene's first. */
  pickTable(tableId: string | undefined, what: string): MountedTable {
    const all = this.host.tables();
    if (tableId !== undefined) {
      const t = all.find(x => x.tableId === tableId);
      if (!t) {
        throw new Error(
          `motion.${what}: no mounted table '${tableId}' (mounted: ${all.map(x => x.tableId).join(', ') || 'none'})`,
        );
      }
      return t;
    }
    const first = all[0];
    if (!first) {
      throw new Error(`motion.${what}: the card scene has no mounted table`);
    }
    return first;
  }

  // ---- measurement ----

  private toFrame(
    t: MountedTable | undefined,
    frame: Frame,
    centre: {x: number; y: number},
    size: {w: number; h: number; rotate: number},
    what: string,
  ): Box {
    const {w, h, rotate} = size;
    if (frame === 'viewport') {
      return {
        x: centre.x - w / 2,
        y: centre.y - h / 2,
        width: w,
        height: h,
        rotate,
      };
    }
    const given: unknown = frame;
    if (given !== 'table') {
      throw new Error(`motion.${what}: unknown frame ${JSON.stringify(given)}`);
    }
    if (!t) {
      throw new Error(
        `motion.${what}: frame 'table' needs the target to be in a mounted table`,
      );
    }
    const s = tableScale(t);
    const c = this.host.clientToTable(t.tableId, centre);
    return {
      x: c.x - w / s / 2,
      y: c.y - h / s / 2,
      width: w / s,
      height: h / s,
      rotate,
    };
  }

  /** The measured on-screen box of a card, in `frame`. */
  readonly measure = (id: CardId, frame: Frame = 'viewport'): Box | null => {
    const el = this.host.element(id);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const t = this.tableOf(id);
    const size = {w: r.width, h: r.height, rotate: 0};
    const p = t?.scene.byId(id);
    if (t && p) {
      const s = tableScale(t);
      // A rotated card measures as its bounding box; its own size and
      // rotation are the computed ones, its centre the measured one.
      size.w = p.box.width * s;
      size.h = p.box.height * s;
      size.rotate = this.flights.get(id)?.pose.rotate ?? p.box.rotate;
    }
    const centre = {x: r.left + r.width / 2, y: r.top + r.height / 2};
    return this.toFrame(t, frame, centre, size, `measure('${id}')`);
  };

  /** The measured box of a slot or region, in `frame`. */
  measureMarked(
    kind: 'slot' | 'region',
    name: string,
    frame: Frame,
    tableId: string | undefined,
  ): Box | null {
    const attr = kind === 'slot' ? 'data-slot-id' : 'data-region-id';
    const what = `${kind === 'slot' ? 'measureSlot' : 'measureRegion'}('${name}')`;
    const tables =
      tableId !== undefined
        ? [this.pickTable(tableId, what)]
        : this.host.tables();
    const hits: Array<{t: MountedTable; el: HTMLElement}> = [];
    for (const t of tables) {
      const el = t.root.querySelector<HTMLElement>(
        `[${attr}="${cssAttr(name)}"]`,
      );
      if (el) hits.push({t, el});
    }
    const [hit, ...more] = hits;
    if (!hit) return null;
    if (more.length > 0) {
      throw new Error(
        `motion.${what}: ${kind} '${name}' is in several tables (${hits.map(x => x.t.tableId).join(', ')}); pass a tableId`,
      );
    }
    const {t, el} = hit;
    const r = el.getBoundingClientRect();
    const s = tableScale(t);
    const computed =
      kind === 'slot'
        ? t.scene.slotBox(name)
        : t.scene.regions.find(x => x.id === name)?.box;
    const w =
      el.offsetWidth > 0
        ? el.offsetWidth * s
        : computed
          ? computed.width * s
          : r.width;
    const h =
      el.offsetHeight > 0
        ? el.offsetHeight * s
        : computed
          ? computed.height * s
          : r.height;
    const centre = {x: r.left + r.width / 2, y: r.top + r.height / 2};
    return this.toFrame(
      t,
      frame,
      centre,
      {w, h, rotate: computed?.rotate ?? 0},
      what,
    );
  }

  /** The frame's visible box. */
  readonly viewport = (frame: Frame = 'viewport', tableId?: string): Rect => {
    if (frame === 'viewport') {
      const d = doc().documentElement;
      const width = d.clientWidth || window.innerWidth;
      const height = d.clientHeight || window.innerHeight;
      return {x: 0, y: 0, width, height};
    }
    const given: unknown = frame;
    if (given !== 'table') {
      throw new Error(
        `motion.viewport: unknown frame ${JSON.stringify(given)}`,
      );
    }
    const t = this.pickTable(tableId, 'viewport');
    return {x: 0, y: 0, width: t.scene.width, height: t.scene.height};
  };
}
