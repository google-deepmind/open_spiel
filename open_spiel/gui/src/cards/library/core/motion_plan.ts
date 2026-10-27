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

// planTransitions — owned by the motion module. Pure: no React, no DOM.
//
// Pairs the items of two computed scenes by id, or by `next.was === prev.id`
// (the same physical card under a new id: a face-down 't3d2' revealed as
// 'QH'), and names what happened to each: it moved, it flipped, both, it
// entered or it left. runFlip (motion/flip.ts) animates the plan.
import type {Box, CardId, TableItem} from './identity.ts';
import {isToken} from './identity.ts';
import type {Placement, Scene} from './scene.ts';
import {DEV} from './dev.ts';

/** What happened to one item between two scenes. */
export interface Transition {
  /** Id in the NEXT scene (an exit: the id it had). */
  id: CardId;
  kind: 'move' | 'flip' | 'move+flip' | 'enter' | 'exit';
  from: Box & {faceUp: boolean};
  to: Box & {faceUp: boolean};
  /** The card stays in the slot it was in (a re-layout of its hand or pile:
   *  a lift, a neighbour leaving, a re-sort). Such a move keeps its place in
   *  the slot's stacking order; only a card changing slot (or entering)
   *  passes over its neighbours. False for enter and exit. */
  sameSlot: boolean;
}

/** Boxes closer than this (table px, degrees) are the same pose. */
const EPS = 0.5;

/** Face shown: a token always "face up"; a card only when its face is known and up. */
function shownFaceUp(item: TableItem): boolean {
  if (isToken(item)) return true;
  return item.card !== null && (item.faceUp ?? true);
}

function faceKnown(item: TableItem): boolean {
  return isToken(item) || item.card !== null;
}

function withFace(p: Placement): Box & {faceUp: boolean} {
  const {x, y, width, height, rotate} = p.box;
  return {x, y, width, height, rotate, faceUp: shownFaceUp(p.item)};
}

/** Shortest signed angle from a to b, degrees in (-180, 180]. */
function angleDelta(a: number, b: number): number {
  let d = (b - a) % 360;
  if (d > 180) d -= 360;
  if (d <= -180) d += 360;
  return d;
}

function moved(a: Box, b: Box): boolean {
  return (
    Math.abs(a.x - b.x) > EPS ||
    Math.abs(a.y - b.y) > EPS ||
    Math.abs(a.width - b.width) > EPS ||
    Math.abs(a.height - b.height) > EPS ||
    Math.abs(angleDelta(a.rotate, b.rotate)) > EPS
  );
}

/** Warned slot pairs: one warning per pair, so a game not yet using `was`
 *  (or on positional ids) does not flood the console. */
const warnedPairs = new Set<string>();

function devWarn(key: string, msg: string): void {
  if (!DEV || warnedPairs.has(key)) return;
  warnedPairs.add(key);
  console.warn(`[cards] ${msg}`);
}

/** Pure. Pairs items by id, or by next.was === prev.id. flip = faceUp or
 *  card-null-ness changed. Dev warning when a card appears to change slot AND
 *  face in one step under a new id without `was` (it would fade instead of
 *  flying and flipping). Unchanged items produce no transition. */
export function planTransitions(prev: Scene, next: Scene): Transition[] {
  const out: Transition[] = [];
  const paired = new Set<CardId>();
  const unpairedEnters: Placement[] = [];
  for (const p of next.placements) {
    let q = prev.byId(p.id);
    if (!q && !isToken(p.item) && p.item.was !== undefined) {
      const was = p.item.was;
      // `was` names a card that is gone: the same physical card, re-identified.
      // If the old id is still on the table it is a different card.
      if (!next.byId(was)) q = prev.byId(was);
    }
    if (q && paired.has(q.id)) {
      throw new Error(
        `planTransitions: card '${q.id}' is claimed twice in one step (by id and by was on '${p.id}')`,
      );
    }
    const to = withFace(p);
    if (!q) {
      out.push({id: p.id, kind: 'enter', from: to, to, sameSlot: false});
      unpairedEnters.push(p);
      continue;
    }
    paired.add(q.id);
    const from = withFace(q);
    const isMove = moved(from, to);
    const isFlip =
      from.faceUp !== to.faceUp || faceKnown(q.item) !== faceKnown(p.item);
    const sameSlot = q.slot === p.slot;
    if (isMove && isFlip) {
      out.push({id: p.id, kind: 'move+flip', from, to, sameSlot});
    } else if (isMove) out.push({id: p.id, kind: 'move', from, to, sameSlot});
    else if (isFlip) out.push({id: p.id, kind: 'flip', from, to, sameSlot});
  }
  const exits: Placement[] = [];
  for (const q of prev.placements) {
    if (paired.has(q.id) || next.byId(q.id)) continue;
    const b = withFace(q);
    out.push({id: q.id, kind: 'exit', from: b, to: b, sameSlot: false});
    exits.push(q);
  }
  // A card that left one slot and a card that arrived in another with the
  // other face, in the same step, is most likely one physical card whose id
  // changed: say so, since only `was` makes it fly and flip.
  if (DEV) {
    for (const e of unpairedEnters) {
      const x = exits.find(
        q => q.slot !== e.slot && shownFaceUp(q.item) !== shownFaceUp(e.item),
      );
      if (x) {
        devWarn(
          `${x.slot}>${e.slot}`,
          `card '${e.id}' arrived in slot '${e.slot}' as '${x.id}' left slot '${x.slot}' with the other face; ` +
            `if they are one physical card, set was: '${x.id}' on '${e.id}' so it flies and flips instead of fading`,
        );
      }
    }
  }
  return out;
}
