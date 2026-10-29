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

// <DropMarks>, owned by the dnd module: the legal drop targets
// of one Table that are not cards, drawn from the computed Scene while a drag
// is live — an outline over each legal slot (an EMPTY slot included: its
// EmptySlot wrapper also carries the 'drop-legal' tag), a caret at insertBox
// for each legal insert, an outline over each legal region.
//
// It renders only when the legal set changes (drag start and end). The
// 'drop-over' mark moves by direct attribute writes from the controller, so
// the per-move path makes no React render.
import {useSyncExternalStore, type CSSProperties, type JSX} from 'react';
import type {Box} from '../core/identity.ts';
import {sceneInternals, type Scene} from '../core/scene.ts';
import type {Target} from '../core/targets.ts';
import {targetKey} from '../core/targets.ts';
import {targetBox} from '../core/hit.ts';
import {tagStyles, useCardStyle} from '../style.ts';
import {dropMarksStore, type DragController} from './controller.ts';

const NONE: readonly Target[] = Object.freeze([]);
/** Insert carets are this wide at least, px (a gap box can be 0 wide). */
const MIN_CARET_PX = 4;

/** The mark's box, or null when the legal target is stale in this scene: the
 *  table re-rendered with new items before the commit that cancels the drag
 *  reached the controller. */
function markBox(scene: Scene, t: Target): Box | null {
  if (t.kind === 'region') {
    return scene.regions.some(r => r.id === t.region)
      ? targetBox(scene, t)
      : null;
  }
  if (t.kind !== 'slot' && t.kind !== 'insert') return null;
  const l = sceneInternals(scene).slotById.get(t.slot);
  if (!l) return null;
  if (t.kind === 'insert' && t.index > l.items.length) return null;
  return targetBox(scene, t);
}

/** The legal drop marks of one table while a drag is live. */
export function DropMarks({
  drag,
  tableId,
  scene,
}: {
  drag: DragController;
  tableId: string;
  scene: Scene;
}): JSX.Element | null {
  const store = dropMarksStore(drag);
  const legal = useSyncExternalStore(
    l => store.subscribe(l),
    () => store.legal(tableId),
    () => NONE,
  );
  const style = useCardStyle();
  const marks = legal.filter(
    t => t.kind === 'slot' || t.kind === 'insert' || t.kind === 'region',
  );
  if (marks.length === 0) return null;
  const tags = tagStyles(style);
  const vars: CSSProperties & Record<`--${string}`, string> = {
    position: 'absolute',
    left: 0,
    top: 0,
    width: 0,
    height: 0,
    overflow: 'visible',
    pointerEvents: 'none',
    zIndex: 900,
    '--dnd-legal': tags['drop-legal']?.ring ?? 'var(--color-success)',
    '--dnd-over': tags['drop-over']?.ring ?? 'var(--color-success)',
  };
  return (
    <div data-drop-marks={tableId} aria-hidden="true" style={vars}>
      {marks.map(t => {
        const b = markBox(scene, t);
        if (!b) return null;
        const key = targetKey(t);
        const caret = t.kind === 'insert';
        const w = caret ? Math.max(MIN_CARET_PX, b.width) : b.width;
        return (
          <div
            key={key}
            data-drop-mark={key}
            data-drop-kind={t.kind}
            data-insert={caret ? `${t.slot}:${String(t.index)}` : undefined}
            data-drop-legal="true"
            className="pcard--t-drop-legal"
            style={{
              position: 'absolute',
              pointerEvents: 'none',
              boxSizing: 'border-box',
              left: b.x + (b.width - w) / 2,
              top: b.y,
              width: w,
              height: b.height,
              transform: b.rotate
                ? `rotate(${String(b.rotate)}deg)`
                : undefined,
            }}
          />
        );
      })}
    </div>
  );
}
