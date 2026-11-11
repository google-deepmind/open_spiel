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

// The motion service's shared types and geometry helpers.
import type {Box, CardId, TableItem} from '../core/identity.ts';
import {isToken} from '../core/identity.ts';
import type {MountedTable, SceneRuntime} from '../react/runtime.ts';
import type {CardStyle} from '../style.ts';
import type {Card} from '../types.ts';
import type {Easing} from './easing.ts';

/** What the motion service needs of the scene runtime: a structural subset
 *  of SceneRuntime, so motion/ depends on the interface only. */
export type MotionHost = Pick<
  SceneRuntime,
  | 'element'
  | 'ticker'
  | 'tables'
  | 'style'
  | 'setLibraryTags'
  | 'onCommit'
  | 'clientToTable'
  | 'tableToClient'
>;

/** z-index of a card while it flies, so it passes over its neighbours. */
export const FLIGHT_Z = '1000';

/** A card pose in one Table's px: centre, rotation, width (height follows). */
export interface Pose2 {
  cx: number;
  cy: number;
  rotate: number;
  width: number;
}

/** The pose of a box. */
export function poseOf(b: Box): Pose2 {
  return {
    cx: b.x + b.width / 2,
    cy: b.y + b.height / 2,
    rotate: b.rotate,
    width: b.width,
  };
}

/** Shortest signed angle from a to b, degrees in (-180, 180]. */
export function angleDelta(a: number, b: number): number {
  let d = (b - a) % 360;
  if (d > 180) d -= 360;
  if (d <= -180) d += 360;
  return d;
}

/** Viewport px per table px of a mounted table. */
export function tableScale(t: MountedTable): number {
  const r = t.root.getBoundingClientRect();
  return r.width > 0 && t.scene.width > 0 ? r.width / t.scene.width : 1;
}

/** A Table that is not laid out (display:none, a hidden panel, no layout
 *  engine) has nothing visible to animate. */
export function hasLayout(t: MountedTable): boolean {
  const r = t.root.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

/** The rotation of the slot a card element sits in (its local frame). */
export function slotRotation(el: HTMLElement): number {
  const s = el.closest('[data-slot-id]');
  const r = Number(s?.getAttribute('data-rotate') ?? 0);
  return Number.isFinite(r) ? r : 0;
}

/** `v` escaped for a CSS attribute selector's quoted value. */
export function cssAttr(v: string): string {
  return v.replace(/["\\]/g, '\\$&');
}

/** The style a laid-out card is drawn with: the scene's, on the card's own
 *  deck (a Table under a nested CardStyleProvider may use another deck). */
export function styleFor(host: MotionHost, el: HTMLElement): CardStyle {
  const style = host.style();
  const deck = el.getAttribute('data-deck');
  return deck && deck !== style.deck ? {...style, deck} : style;
}

/** The side an item shows: its card and whether the face is drawn. */
export function drawnFace(item: TableItem): {
  card: Card | null;
  faceUp: boolean;
} {
  if (isToken(item)) return {card: null, faceUp: false};
  const faceUp = item.card !== null && (item.faceUp ?? true);
  return {card: item.card, faceUp};
}

/** The pixels of an inline CSS length such as '64px'; NaN when it is not one. */
export function inlinePixels(v: string): number {
  return Number(v.replace(/px$/, ''));
}

/** A card in flight: its element, its pose, and how to stop it. */
export interface Flight {
  id: CardId;
  el: HTMLElement;
  tableId: string;
  /** The element's own transform (React's), restored when the flight ends. */
  base: string;
  lastWritten: string;
  zPrev: string;
  /** Current visual pose, table px of `tableId`. */
  pose: Pose2;
  stop(restore: boolean): void;
}

/** Where an id was last laid out, for cards that change Table. */
export interface Seen {
  tableId: string;
  box: Box;
  item: TableItem;
  exitAt?: number;
}

/** Where a card that changes Table flies in from, in viewport px. */
export interface CrossFrom {
  cx: number;
  cy: number;
  rotate: number;
  width: number;
  faceUp: boolean;
  card: Card | null;
}

/** The side a flip turns away from (drawn over the card for the first half). */
export interface FlipSide {
  card: Card | null;
  faceUp: boolean;
}

/** One FLIP: a card's element from one pose to another. */
export interface AnimateRequest {
  id: CardId;
  el: HTMLElement;
  tableId: string;
  from: Pose2;
  to: Pose2;
  slotRotate: number;
  /** Flip from this side to the element's own; null: no flip. */
  oldSide: FlipSide | null;
  durationMs: number;
  delayMs: number;
  easing: Easing;
  /** Paint the card over everything while it moves (FLIGHT_Z): a card
   *  changing slot or flying in. False for a move within its own slot,
   *  which keeps its stacking order (a hovered or lifted card in a fan
   *  never jumps in front of the card that covers it). */
  raise: boolean;
}

/** A clone for the free layer is a picture, not a control. */
export function stripA11y(root: Element): void {
  for (const el of [root, ...root.querySelectorAll('*')]) {
    for (const a of [...el.attributes]) {
      if (
        a.name === 'id' ||
        a.name === 'tabindex' ||
        a.name === 'role' ||
        a.name === 'title' ||
        (a.name.startsWith('aria-') && a.name !== 'aria-hidden')
      ) {
        el.removeAttribute(a.name);
      }
    }
  }
}
