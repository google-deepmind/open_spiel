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

// The drag controller's state types and the geometry it resolves a drag
// with: which legal target the dragged cards are over, the nearest legal
// target in a direction, and where the keyboard preview snaps to.
import type {CardId, Point, TableItem} from '../core/identity.ts';
import {isToken} from '../core/identity.ts';
import type {DragSource, Modifiers} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {Placement, Scene} from '../core/scene.ts';
import {sceneInternals} from '../core/scene.ts';
import type {ItemRef, Target} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import {targetBox, targetsAt} from '../core/hit.ts';
import {defaultLabel} from '../core/a11y.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {
  affineOf,
  clientBounds,
  overlapArea,
  type Affine,
  type Bounds,
  type DragPreview,
} from './drag_preview.tsx';

/** A legal target, with the table it is on and its key. */
export interface Located {
  tableId: string;
  target: Target;
  key: string;
}

/** What a drag is made with. */
export type Via = 'mouse' | 'touch' | 'pen' | 'keyboard';

/** A press on a draggable card, before the pointer has moved far enough. */
export interface Press {
  ref: ItemRef;
  tableId: string;
  ids: readonly CardId[];
  start: Point;
  pointerType: 'mouse' | 'touch' | 'pen';
}

/** A drag in progress. */
export interface Active {
  source: DragSource;
  tableId: string;
  lead: ItemRef;
  via: Via;
  legal: Located[];
  over: Located | null;
  /** Pointer (or, for the keyboard, lead-centre) position at start, viewport px. */
  start: Point;
  last: Point;
  modifiers: Modifiers;
  /** The latest input event (dragOver after an auto-scroll carries it). */
  native: Event;
  preview: DragPreview;
  /** The lead card's viewport bounds at start. */
  leadBounds: Bounds;
}

/** A finished drag whose preview is still on screen: held, or returning. */
export interface Pending {
  kind: 'hold' | 'return';
  ids: readonly CardId[];
  preview: DragPreview;
  /** hold: the holdTimeoutMs timer. */
  timer?: ReturnType<typeof setTimeout>;
  finish(): void;
  /** Ends this pending's share of motion.settled(). */
  settle(): void;
}

function slotName(
  policy: InteractionPolicy | undefined,
  scene: Scene,
  slot: string,
): string {
  return (
    policy?.slotLabel?.(slot) ??
    sceneInternals(scene).slotById.get(slot)?.spec.label ??
    slot
  );
}

/** How a target is named in a keyboard drag's announcements. */
export function describeTarget(
  policy: InteractionPolicy | undefined,
  scene: Scene,
  t: Target,
): string {
  switch (t.kind) {
    case 'card': {
      const p = scene.byId(t.ref.id);
      return policy?.label?.(t.ref) ?? (p ? defaultLabel(p) : t.ref.id);
    }
    case 'slot':
      return slotName(policy, scene, t.slot);
    case 'insert':
      return `position ${String(t.index + 1)} in ${slotName(policy, scene, t.slot)}`;
    case 'region':
      return t.region;
    case 'table':
      return 'the table';
    default: {
      const unreachable: never = t;
      throw new Error(`Unhandled target: ${JSON.stringify(unreachable)}`);
    }
  }
}

function cardSig(it: TableItem): string {
  if (isToken(it)) return 'token';
  const c = it.card;
  const face =
    c === null ? '-' : 'joker' in c ? `J${c.joker}` : `${c.rank}${c.suit}`;
  return `${face}:${String(it.faceUp ?? c !== null)}`;
}

const SIGS = new WeakMap<Scene, string>();

/** What an items change changes: ids, slots, positions, faces. Not tags,
 *  interactivity or geometry (a resize or a highlight is not an items change). */
export function itemsSignature(scene: Scene): string {
  let s = SIGS.get(scene);
  if (s === undefined) {
    s = scene.placements
      .map(p => `${p.id}@${p.slot}#${String(p.position)}=${cardSig(p.item)}`)
      .sort()
      .join('|');
    SIGS.set(scene, s);
  }
  return s;
}

/** The event the controller is handling now, when the platform exposes it. */
export function currentEvent(): Event | null {
  const w =
    typeof window !== 'undefined'
      ? (window as unknown as {event?: Event})
      : undefined;
  return w?.event ?? null;
}

/** The keyboard event behind a keyboard step, or a stand-in. */
export function keyNative(): Event {
  const e = currentEvent();
  if (e) return e;
  return new KeyboardEvent('keydown', {key: ' '});
}

/** The item reference of a placement. */
export function refOf(p: Placement): ItemRef {
  return {id: p.id, slot: p.slot, position: p.position, item: p.item};
}

/** The first legal target under the pointer ('pointer' resolution). */
export function resolvePointerMode(
  rt: SceneRuntime,
  a: Active,
): Located | null {
  const tables = [...rt.tables()].reverse();
  for (const t of tables) {
    const p = rt.clientToTable(t.tableId, a.last);
    if (p.x < 0 || p.y < 0 || p.x > t.scene.width || p.y > t.scene.height) {
      continue;
    }
    for (const target of targetsAt(t.scene, p)) {
      const key = targetKey(target);
      const hit = a.legal.find(l => l.tableId === t.tableId && l.key === key);
      if (hit) return hit;
    }
  }
  return null;
}

// The lead card's viewport bounds now.
function leadNow(a: Active): Bounds {
  const d = {x: a.last.x - a.start.x, y: a.last.y - a.start.y};
  return {
    left: a.leadBounds.left + d.x,
    top: a.leadBounds.top + d.y,
    right: a.leadBounds.right + d.x,
    bottom: a.leadBounds.bottom + d.y,
  };
}

/** The legal target the lead card's preview overlaps most ('overlap'). */
export function resolveOverlapMode(
  rt: SceneRuntime,
  a: Active,
): Located | null {
  const lead = leadNow(a);
  const affines = new Map<string, Affine>();
  const lx = (lead.left + lead.right) / 2;
  const ly = (lead.top + lead.bottom) / 2;
  let best: Located | null = null;
  let bestArea = 0;
  let bestDist = Infinity;
  for (const l of a.legal) {
    let af = affines.get(l.tableId);
    if (!af) {
      af = affineOf(rt, l.tableId);
      affines.set(l.tableId, af);
    }
    const box = targetBox(rt.table(l.tableId).scene, l.target);
    if (!box) continue;
    const b = clientBounds(af, box);
    const area = overlapArea(lead, b);
    if (area <= 0) continue;
    // Equal overlaps (thin insert carets wholly under the card): the
    // target nearest the card's centre wins.
    const dist = Math.hypot(
      (b.left + b.right) / 2 - lx,
      (b.top + b.bottom) / 2 - ly,
    );
    const tie = Math.abs(area - bestArea) <= 1e-6 * Math.max(1, bestArea);
    if ((!tie && area > bestArea) || (tie && dist < bestDist)) {
      bestArea = area;
      bestDist = dist;
      best = l;
    }
  }
  return best;
}

function centreOfLocated(rt: SceneRuntime, l: Located): Point | null {
  const box = targetBox(rt.table(l.tableId).scene, l.target);
  if (!box) return null;
  const b = clientBounds(affineOf(rt, l.tableId), box);
  return {x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2};
}

function sameLocated(a: Located | null, b: Located): boolean {
  return a !== null && a.tableId === b.tableId && a.key === b.key;
}

/** The nearest legal target in a direction (keyboard, viewport space). */
export function nearestFrom(
  rt: SceneRuntime,
  a: Active,
  dir: 'up' | 'down' | 'left' | 'right',
): Located | null {
  const from = a.over
    ? centreOfLocated(rt, a.over)
    : {
        x: (a.leadBounds.left + a.leadBounds.right) / 2,
        y: (a.leadBounds.top + a.leadBounds.bottom) / 2,
      };
  if (!from) return null;
  let best: Located | null = null;
  let bestScore = Infinity;
  for (const l of a.legal) {
    if (sameLocated(a.over, l)) continue;
    const c = centreOfLocated(rt, l);
    if (!c) continue;
    const dx = c.x - from.x;
    const dy = c.y - from.y;
    const along =
      dir === 'right' ? dx : dir === 'left' ? -dx : dir === 'down' ? dy : -dy;
    const across =
      dir === 'right' || dir === 'left' ? Math.abs(dy) : Math.abs(dx);
    if (along <= 1e-6 || across > along) continue;
    const score = along + 2 * across;
    if (score < bestScore) {
      bestScore = score;
      best = l;
    }
  }
  return best;
}

/** The next or previous legal target in the legal list (keyboard). */
export function cycleFrom(a: Active, dir: 'next' | 'prev'): Located | null {
  const n = a.legal.length;
  const over = a.over;
  const i = over ? a.legal.findIndex(l => sameLocated(over, l)) : -1;
  const next =
    dir === 'next'
      ? a.legal[(i + 1) % n]
      : a.legal[i < 0 ? n - 1 : (i - 1 + n) % n];
  return next ?? null;
}

/** Keyboard: the preview snaps onto the target so sighted users see where it goes. */
export function snapPreview(rt: SceneRuntime, a: Active): void {
  if (!a.over) {
    a.preview.setOffset({x: 0, y: 0});
    return;
  }
  const box = targetBox(rt.table(a.over.tableId).scene, a.over.target);
  if (!box) return;
  const tb = clientBounds(affineOf(rt, a.over.tableId), box);
  const lw = a.leadBounds.right - a.leadBounds.left;
  const lh = a.leadBounds.bottom - a.leadBounds.top;
  const cx = (tb.left + tb.right) / 2;
  // On a card: offset down so the target's index stays in view.
  const top =
    a.over.target.kind === 'card'
      ? tb.top + (tb.bottom - tb.top) * 0.25
      : (tb.top + tb.bottom) / 2 - lh / 2;
  a.preview.setOffset({
    x: cx - lw / 2 - a.leadBounds.left,
    y: top - a.leadBounds.top,
  });
  a.last = {
    x: a.start.x + a.preview.offset().x,
    y: a.start.y + a.preview.offset().y,
  };
}
