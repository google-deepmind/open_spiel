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

// The trick collect's motion (trickCollect.ts has the geometry and timing):
// the finished trick's cards slide together toward the winner's seat, shrink
// and fade. Two drawers, one curve:
//  - collectTableCards: cards laid out by a shared <Table>, lifted into its
//    free layer (motion.free) so no re-render can move or unmount them
//    mid-collect, and driven on the scene's ticker. <Table collect> uses it.
//  - collectElements: any other DOM cards (tables that draw their own), by
//    the Web Animations API, sampled from the same collectFrame.
// Both leave the cards invisible at the end; the game removes them from its
// state when the hook's collect phase ends.
import type {CardId} from '../core/identity.ts';
import {defined} from '../core/defined.ts';
import {collectFrame, collectGather, type XY} from '../trick_collect.ts';
import type {FreeCard, MotionService, Run} from './service.ts';

/** A running collect (or arrive): when it ends, and how to stop it. */
export interface CollectRun {
  /** Resolves when the collect has finished (or was disposed). */
  done: Promise<void>;
  /** Stop, and remove what the collect put on screen. */
  dispose(): void;
}

/** Collect laid-out cards of one Table toward a point in that Table's px. */
export function collectTableCards(
  motion: MotionService,
  ids: readonly CardId[],
  toward: XY,
  o: {durationMs: number},
): CollectRun {
  if (!Number.isFinite(toward.x) || !Number.isFinite(toward.y)) {
    throw new Error(
      `collectTableCards: toward must be a finite point, got ${JSON.stringify(toward)}`,
    );
  }
  let disposed = false;
  let cards: FreeCard[] = [];
  let run: Run | null = null;
  const done = motion.free(ids, {frame: 'table'}).then(freed => {
    cards = freed;
    if (disposed) {
      for (const c of freed) c.dispose();
      return undefined;
    }
    const centres = freed.map(c => ({
      x: c.launch.x + c.launch.width / 2,
      y: c.launch.y + c.launch.height / 2,
    }));
    const gather = collectGather(centres, toward);
    const ms = o.durationMs;
    run = motion.run((_dt, t) => {
      const p = ms > 0 ? t / ms : 1;
      freed.forEach((c, i) => {
        const f = collectFrame(
          p,
          defined(centres[i], 'collect centre'),
          gather,
        );
        c.set({
          x: f.x - c.launch.width / 2,
          y: f.y - c.launch.height / 2,
          rotate: c.launch.rotate,
          scale: f.scale,
          opacity: f.opacity,
        });
      });
      return p < 1;
    });
    return run.done;
  });
  return {
    done,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      run?.cancel();
      for (const c of cards) c.dispose();
    },
  };
}

/** Frames sampled from collectFrame for the Web Animations keyframes. */
const SAMPLES = 12;

/** Collect DOM cards toward a point in viewport px. Without the Web
 *  Animations API (jsdom) or a layout box there is nothing to animate. */
export function collectElements(
  els: readonly HTMLElement[],
  toward: XY,
  o: {durationMs: number},
): CollectRun {
  const live = els.filter(
    el =>
      typeof el.animate === 'function' && el.getBoundingClientRect().width > 0,
  );
  if (live.length === 0 || o.durationMs <= 0) {
    return {
      done: Promise.resolve(),
      dispose: () => {
        // Nothing moved, so nothing to remove.
      },
    };
  }
  const boxes = live.map(el => el.getBoundingClientRect());
  const centres = boxes.map(r => ({
    x: r.left + r.width / 2,
    y: r.top + r.height / 2,
  }));
  const gather = collectGather(centres, toward);
  const anims = live
    .map((el, i) => {
      const box = defined(boxes[i], 'collect box');
      const centre = defined(centres[i], 'collect centre');
      // Viewport px -> the element's own px (an ancestor may scale it).
      const k = el.offsetWidth > 0 ? box.width / el.offsetWidth : 1;
      const moves: Keyframe[] = [];
      const fades: Keyframe[] = [];
      for (let s = 0; s <= SAMPLES; s++) {
        const p = s / SAMPLES;
        const f = collectFrame(p, centre, gather);
        moves.push({
          offset: p,
          transform: `translate(${String((f.x - centre.x) / k)}px, ${String((f.y - centre.y) / k)}px) scale(${String(f.scale)})`,
        });
        fades.push({offset: p, opacity: f.opacity});
      }
      const timing = {
        duration: o.durationMs,
        easing: 'linear',
        fill: 'forwards',
      } as const;
      // The move is added on top of the card's own transform (its place in
      // the trick); the fade replaces (an added opacity would stay at 1).
      return [
        el.animate(moves, {...timing, composite: 'add'}),
        el.animate(fades, timing),
      ];
    })
    .flat();
  return {
    done: Promise.all(
      anims.map(a =>
        a.finished.then(
          () => undefined,
          () => undefined,
        ),
      ),
    ).then(() => undefined),
    dispose: () => {
      for (const a of anims) a.cancel();
    },
  };
}
