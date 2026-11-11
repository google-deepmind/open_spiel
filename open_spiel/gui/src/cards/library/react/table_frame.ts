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

// A table's frame: the size its container gives it, and the check that what
// is drawn is where the scene computed it.
import {useLayoutEffect, useState, type RefObject} from 'react';
import {pixels} from '../../../styles/pixels.ts';
import {boxBounds, sceneInternals, type Scene} from '../core/scene.ts';
import type {SceneRuntimeImpl} from './scene_runtime.ts';

/**
 * The size the container gives the table frame, px. With `contain` the
 * height is read with the surface taken out of flow for the reading, so it
 * is the height the container gives, never the table's own: in a container
 * with no definite height the frame is as tall as what it draws, and fitting
 * that height would feed back (a table that grows is fitted smaller, which
 * shrinks the frame, ...). Such a container reads height 0, and the table
 * fits its width. The surface keeps its box in the layout (position, not
 * display), so nothing in it loses focus while it is read.
 */
export function useFrameBox(
  ref: RefObject<HTMLElement | null>,
  surfaceRef: RefObject<HTMLElement | null>,
  contain: boolean,
): {width: number; height: number} {
  const [size, setSize] = useState({width: 0, height: 0});
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // clientWidth and clientHeight round to the nearest px; the table
    // must not be fitted to the half px it does not have, so the used
    // (fractional, untransformed) size is read where there is one. The
    // frame has no padding or border: its used size is its content box.
    const used = (px: string, rounded: number): number => {
      const v = px.endsWith('px') ? pixels(px) : NaN;
      return Number.isFinite(v) && v > 0 ? Math.min(rounded, v) : rounded;
    };
    const read = (): void => {
      const width = used(getComputedStyle(el).width, el.clientWidth);
      const surface = surfaceRef.current;
      let height: number;
      if (contain && surface) {
        const was = surface.style.position;
        surface.style.position = 'absolute';
        height = used(getComputedStyle(el).height, el.clientHeight);
        surface.style.position = was;
      } else {
        height = used(getComputedStyle(el).height, el.clientHeight);
      }
      setSize(s =>
        s.width === width && s.height === height ? s : {width, height},
      );
    };
    read();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, [ref, surfaceRef, contain]);
  return size;
}

/** Check: a measured card box that disagrees with the computed box
 *  (mapped to the viewport) by more than 1 px means an ancestor transform the
 *  affine map cannot represent. */
export function checkMeasured(
  rt: SceneRuntimeImpl,
  tableId: string,
  scene: Scene,
): void {
  const root = rt.table(tableId).root.getBoundingClientRect();
  if (!(root.width > 0 && root.height > 0)) return; // no layout engine (jsdom)
  const {slots} = sceneInternals(scene);
  for (const l of slots) {
    if (l.overflow) continue; // scrolled piles are clipped by design
    for (const e of l.entries) {
      const el = rt.element(e.placement.id);
      // In flight (FLIP) or lifted into the free layer: its pose is motion's.
      if (
        !el ||
        el.getAnimations().length ||
        el.hasAttribute('data-motion') ||
        el.classList.contains('pcard--t-held')
      ) {
        continue;
      }
      // A library tag's paint-only lift (a hover peek, a scoped
      // translate) is the library's own offset. Unset reads 'none'
      // (and '' or undefined where the engine has no such property).
      const lifted = getComputedStyle(el).translate;
      if (lifted && lifted !== 'none') continue;
      const m = el.getBoundingClientRect();
      const b = boxBounds(e.placement.box);
      const tl = rt.tableToClient(tableId, {x: b.x, y: b.y});
      const br = rt.tableToClient(tableId, {
        x: b.x + b.width,
        y: b.y + b.height,
      });
      const d = Math.max(
        Math.abs(m.left - tl.x),
        Math.abs(m.top - tl.y),
        Math.abs(m.right - br.x),
        Math.abs(m.bottom - br.y),
      );
      if (d > 1) {
        console.error(
          `[cards] card '${e.placement.id}' measures ${d.toFixed(1)}px away from its computed box in table '${tableId}': an ancestor transform the table's affine map cannot represent`,
        );
      }
    }
  }
}
