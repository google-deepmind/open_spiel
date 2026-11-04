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

// <CardView>: a headless card with the full plumbing — registry,
// events, focus, a11y, tags — positioned by the caller (an SVG
// <foreignObject>, a 3D scene). It mounts as a one-card table over a 'free'
// slot named `slot`, so events, hit-testing and a11y are the Table's own.
import {useMemo, type CSSProperties, type JSX} from 'react';
import type {TableCard} from '../core/identity.ts';
import type {TableEvent} from '../core/events.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import type {ItemRef} from '../core/targets.ts';
import type {TableSpec} from '../table_spec.ts';
import {getDeck} from '../registry.ts';
import {useCardStyle} from '../style.ts';
import {Table} from './table.tsx';
import {useSceneRuntime} from './card_scene.tsx';

/** One card drawn outside a table, with its interaction. */
export interface CardViewProps {
  item: TableCard;
  /** A caller-chosen pseudo-slot name, reported in events. */
  slot: string;
  /** Reported as the ref's position (default 0). */
  position?: number | undefined;
  /** Card width, px. */
  width: number;
  style?: CSSProperties | undefined;
}

/** Headless card. Must be inside a CardScene (its policy answers the events). */
export function CardView({
  item,
  slot,
  position = 0,
  width,
  style,
}: CardViewProps): JSX.Element {
  const rt = useSceneRuntime();
  if (!rt) throw new Error('<CardView> must be inside a <CardScene>');
  if (!(width > 0)) {
    throw new Error(`<CardView> width must be > 0, got ${String(width)}`);
  }
  if (!Number.isInteger(position) || position < 0) {
    throw new Error(
      `<CardView> position must be a whole number >= 0, got ${String(position)}`,
    );
  }
  const cardStyle = useCardStyle();
  const deck = getDeck(cardStyle.deck);
  const spec = useMemo<TableSpec>(
    () => ({
      width: 1,
      height: 1 / deck.aspect,
      slots: [{id: slot, x: 0, y: 0, pile: 'free'}],
    }),
    [slot, deck.aspect],
  );
  const items = useMemo(
    () => ({[slot]: [{...item, at: {x: 0, y: 0}}]}),
    [slot, item],
  );
  // The ref's position is what the caller says it is.
  const scenePolicy = rt.scenePolicy;
  const policy = useMemo<InteractionPolicy | undefined>(() => {
    if (!scenePolicy) return undefined;
    const fix = (r: ItemRef): ItemRef => ({...r, position});
    const fixEvent = (e: TableEvent): TableEvent =>
      e.target.kind === 'card'
        ? {...e, target: {kind: 'card', ref: fix(e.target.ref)}}
        : e;
    const out: InteractionPolicy = {
      ...scenePolicy,
      onEvent: e => scenePolicy.onEvent(fixEvent(e)),
    };
    if (scenePolicy.canDrag) {
      out.canDrag = (r, e) =>
        scenePolicy.canDrag?.(fix(r), fixEvent(e)) ?? false;
    }
    if (scenePolicy.label) out.label = r => scenePolicy.label?.(fix(r));
    if (scenePolicy.tooltip) out.tooltip = r => scenePolicy.tooltip?.(fix(r));
    if (scenePolicy.describe) {
      out.describe = r => scenePolicy.describe?.(fix(r));
    }
    if (scenePolicy.announce) {
      out.announce = e => scenePolicy.announce?.(fixEvent(e));
    }
    return out;
  }, [scenePolicy, position]);
  return (
    <div data-card-view style={{width, ...style}}>
      <Table spec={spec} items={items} policy={policy} cardWidth={width} />
    </div>
  );
}
