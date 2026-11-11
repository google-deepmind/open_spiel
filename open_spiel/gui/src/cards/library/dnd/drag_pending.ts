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

// After a drag ends: the preview flies home, or a held drop keeps it until
// the items change, when the new layout FLIPs from the preview's pose.
import type {CardId, Point} from '../core/identity.ts';
import {isToken} from '../core/identity.ts';
import type {Placement} from '../core/scene.ts';
import type {SceneRuntime} from '../react/runtime.ts';
import {motionInternals} from '../motion/service.ts';
import {
  affineOf,
  motionDuration,
  toClient,
  tween,
  type DragPreview,
  type Tween,
} from './drag_preview.tsx';
import type {Pending} from './drag_util.ts';

/** What the pending steps need of the drag controller. */
export interface PendingHost {
  readonly rt: SceneRuntime;
  pending: Pending | null;
  writeTags(): void;
}

/** Ends `p` if it is still the pending one: removes its preview. */
export function endPending(h: PendingHost, p: Pending): void {
  if (h.pending !== p) return;
  h.pending = null;
  p.preview.remove();
  h.writeTags();
  p.settle();
}

/** Counts motion in flight for motion.settled(); call the result when it ends. */
function inFlight(rt: SceneRuntime): () => void {
  let end: () => void = () => {
    // Replaced by the promise's resolver below.
  };
  void motionInternals(rt.motion).track(
    new Promise<void>(r => {
      end = r;
    }),
  );
  return end;
}

/** The viewport centre a card's layout box has now (after any commit). */
function homeCentre(
  rt: SceneRuntime,
  id: CardId,
): {centre: Point; tableId: string; placement: Placement} | null {
  for (const t of rt.tables()) {
    const p =
      t.scene.byId(id) ??
      t.scene.placements.find(x => !isToken(x.item) && x.item.was === id);
    if (!p) continue;
    const af = affineOf(rt, t.tableId);
    return {
      centre: toClient(af, {
        x: p.box.x + p.box.width / 2,
        y: p.box.y + p.box.height / 2,
      }),
      tableId: t.tableId,
      placement: p,
    };
  }
  return null;
}

/** The preview flies back to where the lead card's layout box is now. */
export function returnHome(
  h: PendingHost,
  ids: readonly CardId[],
  preview: DragPreview,
  leadId: CardId,
): void {
  const {rt} = h;
  let tw: Tween | null = null;
  let frameHandle: number | null = null;
  const p: Pending = {
    kind: 'return',
    ids,
    preview,
    settle: inFlight(rt),
    finish: () => {
      if (frameHandle !== null) rt.ticker().cancel(frameHandle);
      frameHandle = null;
      if (tw) tw.cancel();
      endPending(h, p);
    },
  };
  h.pending = p;
  h.writeTags();
  const m = rt.style().motion;
  const duration = motionDuration(m);
  const go = (): void => {
    frameHandle = null;
    if (h.pending !== p) return;
    const lead =
      preview.cards.find(c => c.id === leadId) ?? preview.cards[0] ?? null;
    const home = lead ? homeCentre(rt, lead.id) : null;
    if (!lead || !home) {
      endPending(h, p);
      return;
    }
    const from = preview.offset();
    const to = {
      x: home.centre.x - lead.centre.x,
      y: home.centre.y - lead.centre.y,
    };
    tw = tween(
      rt.ticker(),
      duration,
      m.easing,
      k => {
        preview.setOffset({
          x: from.x + (to.x - from.x) * k,
          y: from.y + (to.y - from.y) * k,
        });
      },
      () => {
        endPending(h, p);
      },
    );
  };
  // Measure home after the game's response to the drop has committed.
  if (duration > 0) frameHandle = rt.ticker().request(go);
  else go();
}

/** A held drop: the preview stays (tagged 'held') until items change. */
export function holdDrop(
  h: PendingHost,
  ids: readonly CardId[],
  preview: DragPreview,
  leadId: CardId,
): Pending {
  const {rt} = h;
  const ms = rt.style().interaction.holdTimeoutMs;
  preview.setHeld(true);
  rt.motion.suppressNext(ids);
  const p: Pending = {
    kind: 'hold',
    ids,
    preview,
    settle: inFlight(rt),
    finish: () => {
      clearTimeout(p.timer);
      endPending(h, p);
    },
  };
  p.timer = setTimeout(() => {
    if (h.pending !== p) return;
    console.error(
      `[cards] drop of [${ids.join(', ')}] was held ${String(ms)} ms with no items change; animating home (EventResult {hold: true})`,
    );
    h.pending = null;
    returnHome(h, ids, preview, leadId);
    p.settle(); // after returnHome took over the preview's flight
  }, ms);
  h.pending = p;
  h.writeTags();
  return p;
}

interface HeldFlip {
  el: HTMLElement;
  orig: string;
  d: Point;
}

// Each held card's element and how far, in its slot's frame, it is from
// where the preview shows it.
function heldFlips(rt: SceneRuntime, p: Pending): HeldFlip[] {
  const flips: HeldFlip[] = [];
  for (const c of p.preview.cards) {
    const home = homeCentre(rt, c.id);
    if (!home) continue;
    const el = rt.element(home.placement.id);
    const now = p.preview.centreOf(c.id);
    if (!el || !now) continue;
    const af = affineOf(rt, home.tableId);
    // The element's transform is in its slot's rotated, scaled frame.
    const rot =
      (-Number(el.closest('[data-slot-id]')?.getAttribute('data-rotate') ?? 0) *
        Math.PI) /
      180;
    const dx = (now.x - home.centre.x) / af.scale;
    const dy = (now.y - home.centre.y) / af.scale;
    flips.push({
      el,
      orig: el.style.transform,
      d: {
        x: dx * Math.cos(rot) - dy * Math.sin(rot),
        y: dx * Math.sin(rot) + dy * Math.cos(rot),
      },
    });
  }
  return flips;
}

/** Items changed while a drop was held: FLIP the new layout from the preview's pose. */
export function completeHold(h: PendingHost, p: Pending): void {
  if (h.pending !== p || p.kind !== 'hold') return;
  const {rt} = h;
  clearTimeout(p.timer);
  const m = rt.style().motion;
  const duration = motionDuration(m);
  const flips = heldFlips(rt, p);
  endPending(h, p);
  if (!flips.length || duration <= 0) return;
  const write = (k: number): void => {
    for (const f of flips) {
      const x = f.d.x * (1 - k);
      const y = f.d.y * (1 - k);
      f.el.style.transform =
        k >= 1
          ? f.orig
          : `translate(${String(x)}px, ${String(y)}px)${f.orig ? ` ${f.orig}` : ''}`;
    }
  };
  write(0);
  void motionInternals(rt.motion).track(
    tween(rt.ticker(), duration, m.easing, write, () => {
      write(1);
    }).done,
  );
}
