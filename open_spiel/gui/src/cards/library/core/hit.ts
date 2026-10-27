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

// Hit-testing on the computed Scene. Pure and rotation-aware:
// works in jsdom, canvas scenes and Node.
import type {Box, Point} from './identity.ts';
import type {Scene} from './scene.ts';
import {boxQuad, sceneInternals} from './scene.ts';
import type {ItemRef, Target, TargetKind} from './targets.ts';
import {targetKey} from './targets.ts';
import {isHandMode} from '../hand_geometry.ts';

/** Whether `p` lies in `b`, rotation included. */
export function pointInBox(b: Box, p: Point): boolean {
  if (!b.rotate) {
    return (
      p.x >= b.x && p.x <= b.x + b.width && p.y >= b.y && p.y <= b.y + b.height
    );
  }
  const cx = b.x + b.width / 2;
  const cy = b.y + b.height / 2;
  const a = (-b.rotate * Math.PI) / 180;
  const dx = p.x - cx;
  const dy = p.y - cy;
  const x = cx + dx * Math.cos(a) - dy * Math.sin(a);
  const y = cy + dx * Math.sin(a) + dy * Math.cos(a);
  return x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height;
}

function refOf(scene: Scene, id: string): ItemRef {
  const p = scene.byId(id);
  if (!p) throw new Error(`scene has no item '${id}'`);
  return {id: p.id, slot: p.slot, position: p.position, item: p.item};
}

/** Topmost target at p (table px). Rotation-aware. Priority: region (if droppable
 *  or content) > interactive card > slot (if slotInteractive) > table.
 *  A non-interactive card on top resolves to its slot when the slot is
 *  interactive, and to null otherwise (it blocks what lies under it). With
 *  `observe` (passive events: hover, context menu, long press), a
 *  non-interactive card that observe() accepts resolves to itself. */
export function hitTest(
  scene: Scene,
  p: Point,
  o?: {
    slotInteractive?: (s: string) => boolean;
    observe?: (ref: ItemRef) => boolean;
  },
): Target | null {
  const slotOk = o?.slotInteractive ?? (() => false);
  for (const r of [...scene.regions].reverse()) {
    if ((r.droppable || r.occludes) && pointInBox(r.box, p)) {
      return {kind: 'region', region: r.id};
    }
  }
  for (const pl of [...scene.placements].reverse()) {
    if (!pointInBox(pl.box, p)) continue;
    if (pl.interactive) return {kind: 'card', ref: refOf(scene, pl.id)};
    if (o?.observe) {
      const ref = refOf(scene, pl.id);
      if (o.observe(ref)) return {kind: 'card', ref};
    }
    return slotOk(pl.slot) ? {kind: 'slot', slot: pl.slot} : null;
  }
  for (const l of [...sceneInternals(scene).slots].reverse()) {
    const id = l.spec.id;
    if (slotOk(id) && pointInBox(scene.slotBox(id), p)) {
      return {kind: 'slot', slot: id};
    }
  }
  if (p.x >= 0 && p.y >= 0 && p.x <= scene.width && p.y <= scene.height) {
    return {kind: 'table'};
  }
  return null;
}

/** Every target under p, top first (includes non-interactive cards; drop logic uses it). */
export function targetsAt(scene: Scene, p: Point): Target[] {
  const out: Target[] = [];
  for (const r of [...scene.regions].reverse()) {
    if (pointInBox(r.box, p)) out.push({kind: 'region', region: r.id});
  }
  for (const pl of [...scene.placements].reverse()) {
    if (pointInBox(pl.box, p)) {
      out.push({kind: 'card', ref: refOf(scene, pl.id)});
    }
  }
  for (const l of [...sceneInternals(scene).slots].reverse()) {
    for (let k = 0; k <= l.items.length && insertable(l.spec.pile); k++) {
      if (pointInBox(scene.insertBox(l.spec.id, k), p)) {
        out.push({kind: 'insert', slot: l.spec.id, index: k});
      }
    }
    if (pointInBox(scene.slotBox(l.spec.id), p)) {
      out.push({kind: 'slot', slot: l.spec.id});
    }
  }
  if (p.x >= 0 && p.y >= 0 && p.x <= scene.width && p.y <= scene.height) {
    out.push({kind: 'table'});
  }
  return out;
}

function insertable(pile: string): boolean {
  return isHandMode(pile) || pile.startsWith('fan-');
}

/** Every candidate of the given kinds (the drag-start and keyboard scan), in slot order. */
export function allTargets(
  scene: Scene,
  kinds: readonly TargetKind[],
): Target[] {
  const out: Target[] = [];
  const want = new Set(kinds);
  const {slots} = sceneInternals(scene);
  for (const l of slots) {
    const id = l.spec.id;
    if (want.has('slot')) out.push({kind: 'slot', slot: id});
    if (want.has('insert') && insertable(l.spec.pile)) {
      for (let k = 0; k <= l.items.length; k++) {
        out.push({kind: 'insert', slot: id, index: k});
      }
    }
    if (want.has('card')) {
      for (const e of [...l.entries].sort(
        (a, b) => a.placement.position - b.placement.position,
      )) {
        out.push({kind: 'card', ref: refOf(scene, e.placement.id)});
      }
    }
  }
  if (want.has('region')) {
    for (const r of scene.regions) out.push({kind: 'region', region: r.id});
  }
  if (want.has('table')) out.push({kind: 'table'});
  return out;
}

/** The box of a target on the scene (null: the target is not there). */
export function targetBox(scene: Scene, t: Target): Box | null {
  switch (t.kind) {
    case 'card':
      return scene.byId(t.ref.id)?.box ?? null;
    case 'slot':
      return scene.slotBox(t.slot);
    case 'insert':
      return scene.insertBox(t.slot, t.index);
    case 'region':
      return scene.regions.find(r => r.id === t.region)?.box ?? null;
    case 'table':
      return {x: 0, y: 0, width: scene.width, height: scene.height, rotate: 0};
    default: {
      const unreachable: never = t;
      throw new Error(`Unhandled target: ${JSON.stringify(unreachable)}`);
    }
  }
}

function centre(b: Box): Point {
  return {x: b.x + b.width / 2, y: b.y + b.height / 2};
}

/** Nearest target in a direction by box centres (keyboard navigation). A
 *  candidate must lie within 45° of the direction; distance along the
 *  direction weighs less than distance across it. */
export function nearestInDirection(
  scene: Scene,
  from: Target,
  dir: 'up' | 'down' | 'left' | 'right',
  among: readonly Target[],
): Target | null {
  const fb = targetBox(scene, from);
  if (!fb) return null;
  const c0 = centre(fb);
  const fromKey = targetKey(from);
  let best: Target | null = null;
  let bestScore = Infinity;
  for (const t of among) {
    if (targetKey(t) === fromKey) continue;
    const b = targetBox(scene, t);
    if (!b) continue;
    const c = centre(b);
    const dx = c.x - c0.x;
    const dy = c.y - c0.y;
    const along =
      dir === 'right' ? dx : dir === 'left' ? -dx : dir === 'down' ? dy : -dy;
    const across =
      dir === 'right' || dir === 'left' ? Math.abs(dy) : Math.abs(dx);
    if (along <= 1e-6 || across > along) continue;
    const score = along + 2 * across;
    if (score < bestScore) {
      bestScore = score;
      best = t;
    }
  }
  return best;
}

/** Corners of a target's box (drop overlap, marks). */
export function targetQuad(scene: Scene, t: Target): Point[] | null {
  const b = targetBox(scene, t);
  return b ? boxQuad(b) : null;
}
