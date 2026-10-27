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

// The DOM input layer: listeners on the scene root turn DOM events into
// TableEvents. The library decides what an event IS; the policy what it MEANS.
//
// Targets. A pointer event over a card, slot or region element resolves from
// that element (the browser has already hit-tested the real layout, rotation
// included, and it is the only answer for events synthesised on an element);
// anything else falls back to rt.hit on the computed scene. Either way
// the same rule applies: a non-interactive card resolves to its slot when the
// policy makes the slot interactive, and to nothing otherwise. Passive events
// (hoverStart/hoverEnd, contextMenu, longPress) also resolve to a
// non-interactive card the policy observes (policy.observe): a hover peek or a
// menu on a face-down card, which stays inert and keeps its role.
//
// Touch and pen: one long press is ONE longPress event. Browsers that also
// fire a native contextmenu for it (Android Chrome) do not get a second,
// contextMenu, event: that contextmenu fires the long press if the timer has
// not yet, and is swallowed otherwise. The long press stands in for the tap,
// so the click that ends the same gesture (desktop touch) is dropped; a
// browser that sends no click after a long press (Android, iOS) must not
// have the NEXT tap's click dropped instead, so the drop lasts only until the
// next pointerdown.
//
// Click-versus-double-click (no delay): click detail 1 (or 0, a synthetic
// click) emits activate; detail 2 emits doubleActivate INSTEAD of a second
// activate; a dblclick event emits doubleActivate only when no detail-2 click
// already did (environments that send dblclick alone).
import type {Point} from '../core/identity.ts';
import type {Modifiers, TableEvent} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {Scene} from '../core/scene.ts';
import type {ItemRef, Target} from '../core/targets.ts';
import type {SceneRuntimeImpl} from './scene_runtime.ts';

/** How an event arrived: a pointer kind, the keyboard or a dispatch. */
export type Via = TableEvent['via'];

/** The modifier keys held during `e`. */
export function modifiersOf(e: {
  shiftKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  metaKey: boolean;
}): Modifiers {
  return {shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey};
}

/** The pointer kind of `e`; mouse for anything that is not touch or pen. */
export function pointerVia(e: Event): 'mouse' | 'touch' | 'pen' {
  const t = (e as PointerEvent).pointerType;
  return t === 'touch' || t === 'pen' ? t : 'mouse';
}

/** A pointer resolved to its table, target, scene and policy. */
export interface Resolved {
  tableId: string;
  target: Target;
  scene: Scene;
  policy: InteractionPolicy | undefined;
}

/** Elements inside a region that take their own input. */
const REGION_CONTROL =
  'button, a[href], input, select, textarea, [role="button"]:not([data-card-id]), [tabindex]:not([data-card-id])';

/** The keys that only modify others. */
export const MODIFIER_KEYS = new Set([
  'Shift',
  'Control',
  'Alt',
  'Meta',
  'AltGraph',
  'CapsLock',
  'Fn',
]);

/** The scene's target for a DOM element: null when the element is outside
 *  this scene's tables (another scene's, or game UI beside them) or resolves
 *  to nothing; undefined when it is inside one of this scene's table surfaces
 *  but on no library node (then the caller falls back to computed hit-testing). */
export function resolveElement(
  rt: SceneRuntimeImpl,
  el: Element | null,
  passive = false,
): Resolved | null | undefined {
  if (!el) return null;
  const tableEl = el.closest<HTMLElement>('[data-table-id]');
  if (!tableEl) return null;
  const tableId =
    tableEl.getAttribute('data-table-id') ?? missing('data-table-id');
  if (!rt.hasTable(tableId)) return null;
  const {scene} = rt.table(tableId);
  const policy = rt.policy(tableId);
  const slotOk = (s: string) => !!policy?.slotInteractive?.(s);
  const regionEl = el.closest<HTMLElement>('[data-region-id]');
  if (regionEl && tableEl.contains(regionEl)) {
    const region =
      regionEl.getAttribute('data-region-id') ?? missing('data-region-id');
    const placed = scene.regions.find(r => r.id === region);
    if (!placed) {
      throw new Error(
        `input: region '${region}' is drawn but not in the scene`,
      );
    }
    // A region that neither occludes nor takes drops is see-through, as it
    // is in hitTest: a panel drawn behind a grid of cards sits above them in
    // the DOM, and resolving the pointer to it would make every card under
    // it dead to clicks and drags. The computed geometry decides instead.
    // A control the game put in the region (a button, a link, a field)
    // still belongs to the region.
    const control = el.closest(REGION_CONTROL);
    if (
      !placed.occludes &&
      !placed.droppable &&
      !(control && regionEl.contains(control))
    ) {
      return undefined;
    }
    return {tableId, target: {kind: 'region', region}, scene, policy};
  }
  const cardEl = el.closest<HTMLElement>('[data-card-id]');
  if (cardEl && tableEl.contains(cardEl)) {
    const p = scene.byId(
      cardEl.getAttribute('data-card-id') ?? missing('data-card-id'),
    );
    if (!p) return null;
    const ref: ItemRef = {
      id: p.id,
      slot: p.slot,
      position: p.position,
      item: p.item,
    };
    if (p.interactive || (passive && !!policy?.observe?.(ref))) {
      return {tableId, target: {kind: 'card', ref}, scene, policy};
    }
    return slotOk(p.slot)
      ? {tableId, target: {kind: 'slot', slot: p.slot}, scene, policy}
      : null;
  }
  const slotEl = el.closest<HTMLElement>('[data-slot-id]');
  if (slotEl && tableEl.contains(slotEl)) {
    const slot = slotEl.getAttribute('data-slot-id') ?? missing('data-slot-id');
    return slotOk(slot)
      ? {tableId, target: {kind: 'slot', slot}, scene, policy}
      : null;
  }
  return undefined;
}

// An attribute the element was selected by: present, or the DOM is broken.
function missing(name: string): never {
  throw new Error(`input: an element selected by ${name} has none`);
}

/** The table and target under a pointer event, or null. */
export function resolvePointer(
  rt: SceneRuntimeImpl,
  e: MouseEvent,
  passive = false,
): Resolved | null {
  const fromDom = resolveElement(rt, e.target as Element | null, passive);
  if (fromDom !== undefined) return fromDom;
  const h = rt.hit({x: e.clientX, y: e.clientY}, passive);
  if (!h) return null;
  return {
    tableId: h.tableId,
    target: h.target,
    scene: rt.table(h.tableId).scene,
    policy: rt.policy(h.tableId),
  };
}

/** Where `e` is, on the screen and on table `tableId`. */
export function pointOf(
  rt: SceneRuntimeImpl,
  tableId: string,
  e: MouseEvent,
): {point: Point; client: Point} {
  const client = {x: e.clientX, y: e.clientY};
  return {client, point: rt.clientToTable(tableId, client)};
}

/** Whether `r` is a card, slot or region: something an event can act on. */
export function isActionable(r: Resolved): boolean {
  const t = r.target;
  return t.kind === 'card' || t.kind === 'slot' || t.kind === 'region';
}
