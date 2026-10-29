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

// The a11y model: pure. The DOM layer renders these attributes
// and nothing else decides roles.
import {cardLabel} from '../model.ts';
import {isToken} from './identity.ts';
import type {InteractionPolicy} from './policy.ts';
import type {Placement, Scene, SlotEntry, SlotLayout} from './scene.ts';
import {sceneInternals} from './scene.ts';
import type {ItemRef} from './targets.ts';
import {targetKey} from './targets.ts';

/** The accessibility of one target: its role, name and state. */
export interface A11yNode {
  /** targetKey. */
  key: string;
  /** 'button' ONLY when interactive; 'none' is decorative (aria-hidden, no name). */
  role: 'button' | 'img' | 'group' | 'none';
  /** Empty for role 'none' (a decorative node has no name). */
  label: string;
  description?: string;
  tooltip?: string;
  /** Roving: exactly one true per scene (when anything is interactive). */
  tabbable: boolean;
  /** aria-pressed, from policy.pressed, on buttons only (undefined: none). */
  pressed?: boolean;
  /** aria-disabled, from policy.disabled, on buttons only (true or absent). */
  disabled?: true;
  /** aria-hidden (inside a summarised slot, or role 'none'). */
  hidden: boolean;
}

function refOf(p: Placement): ItemRef {
  return {id: p.id, slot: p.slot, position: p.position, item: p.item};
}

/** The neutral accessible name of an item. */
export function defaultLabel(
  p: Pick<Placement, 'item'>,
  drawnFaceUp = true,
): string {
  const it = p.item;
  if (isToken(it)) return it.label;
  if (it.card === null || !drawnFaceUp || it.faceUp === false) {
    return 'Face-down card';
  }
  return cardLabel(it.card);
}

/** The neutral name of a slot when the game gives none: never its id. */
function neutralSlotName(label: string | undefined, empty: boolean): string {
  if (label) return empty ? `Empty: ${label}` : label;
  return empty ? 'Empty pile' : 'Pile';
}

// The node of a slot itself (null: the slot has none).
function slotNode(
  l: SlotLayout,
  policy: InteractionPolicy | undefined,
  live: boolean,
  summary: string | undefined,
): A11yNode | null {
  const slot = l.spec.id;
  const empty = l.items.length === 0;
  const slotInteractive = live && policy?.slotInteractive?.(slot) === true;
  const slotName = policy?.slotLabel?.(slot);
  const key = targetKey({kind: 'slot', slot});
  if (summary !== undefined) {
    return {
      key,
      role: slotInteractive ? 'button' : 'group',
      label: summary,
      tabbable: false,
      hidden: false,
    };
  }
  if (slotInteractive) {
    return {
      key,
      role: 'button',
      label: slotName ?? neutralSlotName(l.spec.label, empty),
      tabbable: false,
      hidden: false,
    };
  }
  if (empty && slotName !== undefined) {
    return {key, role: 'img', label: slotName, tabbable: false, hidden: false};
  }
  return null;
}

// A card's role: its fallback, or the policy's, which must agree with
// whether the card is interactive.
function cardRole(
  p: Placement,
  ref: ItemRef,
  policy: InteractionPolicy | undefined,
  interactive: boolean,
  fallback: A11yNode['role'],
): A11yNode['role'] {
  const asked = policy?.role?.(ref);
  if (asked === undefined) return fallback;
  if (interactive && asked !== 'button') {
    throw new Error(
      `policy.role('${p.id}') = '${asked}', but the card is interactive (an interactive card is a button)`,
    );
  }
  if (!interactive && asked === 'button') {
    throw new Error(
      `policy.role('${p.id}') = 'button', but the card is not interactive (make it interactive, or give 'img' or 'none')`,
    );
  }
  return asked;
}

// The node of one card of a slot.
function cardNode(
  e: SlotEntry,
  policy: InteractionPolicy | undefined,
  live: boolean,
  summarised: boolean,
): A11yNode {
  const p = e.placement;
  const ref = refOf(p);
  const interactive = live && p.interactive && !summarised;
  const own = policy?.label?.(ref);
  const known = !isToken(p.item) && p.item.card !== null && e.drawnFaceUp;
  const fallback: A11yNode['role'] = interactive
    ? 'button'
    : known || isToken(p.item) || own !== undefined
      ? 'img'
      : 'none';
  const role = cardRole(p, ref, policy, interactive, fallback);
  const node: A11yNode = {
    key: targetKey({kind: 'card', ref}),
    role,
    label: role === 'none' ? '' : (own ?? defaultLabel(p, e.drawnFaceUp)),
    tabbable: false,
    hidden: summarised || role === 'none',
  };
  const description = policy?.describe?.(ref);
  if (description !== undefined && role !== 'none') {
    node.description = description;
  }
  const tooltip = policy?.tooltip?.(ref);
  if (tooltip !== undefined) node.tooltip = tooltip;
  if (role === 'button') {
    const pressed = policy?.pressed?.(ref);
    if (pressed !== undefined) node.pressed = pressed;
    if (policy?.disabled?.(ref) === true) node.disabled = true;
  }
  return node;
}

/** Pure. role rules: interactive card or token -> 'button'; a non-interactive
 *  card whose face is drawn -> 'img' labelled with the card; a non-interactive
 *  back -> 'img' when policy.label names it, else 'none' (decorative);
 *  policy.role overrides within those rules. Summarised slot -> 'group' with the
 *  summary, children hidden; interactive slot -> 'button' named by slotLabel;
 *  an EMPTY slot the game names (slotLabel) -> 'img' with that name. Without a
 *  policy, or with an inert one, nothing is a button and nothing is a tab stop;
 *  an inert policy keeps its words. Nodes come in slot order: each slot's node
 *  (when it has one), then its cards bottom first. */
export function a11yModel(
  scene: Scene,
  policy: InteractionPolicy | undefined,
  focus: string | null,
): A11yNode[] {
  const nodes: A11yNode[] = [];
  const live = policy !== undefined && policy.inert !== true;
  for (const l of sceneInternals(scene).slots) {
    const summary = policy?.slotSummary?.(l.spec.id, l.items);
    const own = slotNode(l, policy, live, summary);
    if (own) nodes.push(own);
    const entries = [...l.entries].sort(
      (a, b) => a.placement.position - b.placement.position,
    );
    for (const e of entries) {
      nodes.push(cardNode(e, policy, live, summary !== undefined));
    }
  }
  // Roving tab stop: the focused target if it can take focus, else the first
  // focusable one in slot order.
  const focusable = nodes.filter(n => n.role === 'button' && !n.hidden);
  const stop = focusable.find(n => n.key === focus) ?? focusable[0];
  if (stop) stop.tabbable = true;
  return nodes;
}
