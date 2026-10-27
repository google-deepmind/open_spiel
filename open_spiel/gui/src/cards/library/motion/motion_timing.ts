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

// Motion timing (the reduced-motion preference and the viewer's Animation
// speed applied to the style's durations) and the free layers.
import type {CardStyle} from '../style.ts';
import {animationScale, scaleMs} from '../../../motion/animation_speed.ts';
import {FLIGHT_Z} from './motion_util.ts';

/** z-index of the viewport free layer. */
export const VIEWPORT_LAYER_Z = '9000';

// ---------------------------------------------------------------------------
// Reduced motion

/** True when the user asks for reduced motion and the style respects it
 *  (CardStyle.motion.reducedMotion, default 'respect'). Games running their
 *  own frame loops (a win cascade) should check this too. */
export function prefersReducedMotion(
  m?: Pick<CardStyle['motion'], 'reducedMotion'>,
): boolean {
  if (m?.reducedMotion === 'ignore') return false;
  if (
    typeof window === 'undefined' ||
    typeof window.matchMedia !== 'function'
  ) {
    return false;
  }
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** The FLIP duration in effect: the style's base at the viewer's Animation
 *  speed (src/motion), and 0 under reduced motion (when respected) or at
 *  Instant. */
export function motionDuration(
  m: Pick<CardStyle['motion'], 'durationMs' | 'reducedMotion'>,
): number {
  if (!Number.isFinite(m.durationMs)) {
    throw new Error(
      `CardStyle.motion.durationMs must be finite, got ${String(m.durationMs)}`,
    );
  }
  return prefersReducedMotion(m)
    ? 0
    : scaleMs(
        Math.max(0, m.durationMs),
        animationScale({reducedMotion: 'ignore'}),
      );
}

/** The delay between cards of one plan, at the viewer's speed. */
export function motionStagger(
  m: Pick<CardStyle['motion'], 'staggerMs' | 'reducedMotion'>,
): number {
  if (!Number.isFinite(m.staggerMs)) {
    throw new Error(
      `CardStyle.motion.staggerMs must be finite, got ${String(m.staggerMs)}`,
    );
  }
  return prefersReducedMotion(m)
    ? 0
    : scaleMs(
        Math.max(0, m.staggerMs),
        animationScale({reducedMotion: 'ignore'}),
      );
}

// ---------------------------------------------------------------------------
// Layers

/** The one fixed, full-window free layer of a document (created on demand). */
export function viewportLayer(doc: Document = document): HTMLElement {
  let el = doc.body.querySelector<HTMLElement>(
    ':scope > [data-free-layer="viewport"]',
  );
  if (!el) {
    el = doc.createElement('div');
    el.setAttribute('data-free-layer', 'viewport');
    Object.assign(el.style, {
      position: 'fixed',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: VIEWPORT_LAYER_Z,
      overflow: 'visible',
    });
    doc.body.appendChild(el);
  }
  return el;
}

/** The free layer over one Table root (created on demand, absolutely over it). */
export function tableLayer(root: HTMLElement): HTMLElement {
  let el = [...root.children].find(
    (c): c is HTMLElement =>
      c instanceof HTMLElement && c.getAttribute('data-free-layer') === 'table',
  );
  if (!el) {
    el = root.ownerDocument.createElement('div');
    el.setAttribute('data-free-layer', 'table');
    Object.assign(el.style, {
      position: 'absolute',
      left: '0',
      top: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: FLIGHT_Z,
      overflow: 'visible',
    });
    root.appendChild(el);
  }
  return el;
}
