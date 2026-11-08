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

// A table's slots and cards: each slot's element with its empty outline,
// badge and overflow note, and each card's wrapper (the a11y node, the
// motion and input target) around the card's drawing.
import type {CSSProperties, JSX, ReactNode} from 'react';
import type {DescriptionStore} from '../a11y/describe.ts';
import {describedById} from '../a11y/describe.ts';
import type {A11yNode} from '../core/a11y.ts';
import {isToken} from '../core/identity.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import {
  localCardGeometry,
  type Scene,
  type SlotEntry,
  type SlotLayout,
} from '../core/scene.ts';
import {resolveTagStyle} from '../core/tags.ts';
import {targetKey, type ItemRef} from '../core/targets.ts';
import {EmptySlot, PlayingCard, type CardA11yAttrs} from '../playing_card.tsx';
import {tagStyles, tagStyleVars, type CardStyle} from '../style.ts';
import type {Deck} from '../types.ts';
import {CardWidthContext} from './card_scene.tsx';
import type {SceneRuntimeImpl} from './scene_runtime.ts';
import type {CardRenderProps, TableProps} from './table.tsx';

/** What a table hands its slots and cards to draw them. */
export interface Renderer {
  rt: SceneRuntimeImpl;
  scene: Scene;
  policy: InteractionPolicy | undefined;
  a11y: Map<string, A11yNode>;
  deck: Deck;
  style: CardStyle;
  renderCard?: TableProps['renderCard'] | undefined;
  renderOverlay?: TableProps['renderOverlay'] | undefined;
  renderToken?: TableProps['renderToken'] | undefined;
  cardWidth: number;
  tableId: string;
  descStore: DescriptionStore;
  slotTags?: TableProps['slotTags'] | undefined;
}

/** The slots in spec order, consecutive slots of one group (policy.slotGroup)
 *  run together; a slot in no group is a run of its own. */
export function groupSlotRuns<L extends {spec: {id: string}}>(
  slots: readonly L[],
  policy: InteractionPolicy | undefined,
): Array<{group: string | undefined; slots: L[]}> {
  const runs: Array<{group: string | undefined; slots: L[]}> = [];
  for (const l of slots) {
    const group = policy?.slotGroup?.(l.spec.id);
    const last = runs.at(-1);
    if (group !== undefined && last?.group === group) last.slots.push(l);
    else runs.push({group, slots: [l]});
  }
  return runs;
}

// The scrolling frame of a slot that overflows its maxExtent.
function slotFrame(
  overflow: boolean,
  maxExtent: number | undefined,
  horizontal: boolean,
): CSSProperties {
  const frame: CSSProperties = {position: 'relative', flex: 'none'};
  if (overflow && maxExtent !== undefined) {
    if (horizontal) {
      Object.assign(frame, {
        maxWidth: maxExtent,
        overflowX: 'auto',
        overflowY: 'hidden',
      });
    } else {
      Object.assign(frame, {
        maxHeight: maxExtent,
        overflowY: 'auto',
        overflowX: 'hidden',
      });
    }
  }
  return frame;
}

// Whether any card of the slot is a visible button.
function hasButton(l: SlotLayout, r: Renderer): boolean {
  return l.entries.some(e => {
    const p = e.placement;
    const node = r.a11y.get(
      targetKey({
        kind: 'card',
        ref: {id: p.id, slot: p.slot, position: p.position, item: p.item},
      }),
    );
    return node?.role === 'button' && !node.hidden;
  });
}

// The slot element's a11y attributes and its kind's data attributes.
function slotAttrs(
  l: SlotLayout,
  slotNode: A11yNode | undefined,
): {
  slotA11y: Record<string, string | number | undefined>;
  kindAttrs: Record<string, string | number>;
} {
  const s = l.spec;
  const slotA11y = slotNode
    ? {
        role: slotNode.role,
        'aria-label': slotNode.label,
        tabIndex:
          slotNode.role === 'button' ? (slotNode.tabbable ? 0 : -1) : undefined,
      }
    : {};
  const kindAttrs =
    l.kind === 'hand'
      ? {
          'data-hand': '',
          'data-hand-mode': s.pile,
          'data-hand-rows': l.hand?.rows ?? 1,
        }
      : l.kind === 'free'
        ? {'data-free-slot': ''}
        : {'data-pile': '', 'data-pile-mode': s.pile};
  return {slotA11y, kindAttrs};
}

// An empty slot: its outline (styled by the slot's tags), or nothing.
function EmptyBody({
  l,
  r,
  tags,
  look,
  hidden,
}: {
  l: SlotLayout;
  r: Renderer;
  tags: readonly string[];
  look: ReturnType<typeof resolveTagStyle> | null;
  hidden: boolean;
}): JSX.Element {
  const s = l.spec;
  const slotNode = r.a11y.get(targetKey({kind: 'slot', slot: s.id}));
  return hidden ? (
    <div
      data-pile-empty="hidden"
      aria-hidden="true"
      style={{width: l.cw, height: l.ch}}
    />
  ) : (
    <div
      data-pile-empty
      data-tags={tags.length ? tags.join(' ') : undefined}
      className={
        look
          ? [...tags.map(t => `pcard--t-${t}`), look.className]
              .filter(Boolean)
              .join(' ')
          : undefined
      }
      style={{
        cursor:
          look?.cursor ?? (slotNode?.role === 'button' ? 'pointer' : undefined),
        ...(look
          ? {
              ...tagStyleVars(look),
              display: 'flex',
              borderRadius: `${String(Math.max(0, r.style.radius) * l.cw)}px`,
              boxShadow:
                '0 0 0 3px var(--pcard-ring, var(--color-none)), 0 0 14px 2px var(--pcard-glow, var(--color-none))',
              transition:
                'box-shadow var(--pcard-motion-ms, calc(120ms * var(--anim-scale, 1))) ease',
            }
          : {}),
      }}
    >
      {/* Decorative: the slot's own node (when the game names it) is the name. */}
      <EmptySlot deck={r.deck.id} width={l.cw} label={s.label} decorative />
    </div>
  );
}

// The pile's count badge and its overflow note.
function SlotNotes({
  l,
  horizontal,
}: {
  l: SlotLayout;
  horizontal: boolean;
}): JSX.Element {
  const n = l.items.length;
  return (
    <>
      {l.spec.countBadge && l.kind === 'pile' && n > 0 && (
        <span
          data-pile-count
          style={{
            // Inside the pile's own box, bottom-right: a badge that
            // overhangs lands on the neighbour's index.
            position: 'absolute',
            bottom: 3,
            right: 3,
            minWidth: 20,
            padding: '1px 5px',
            borderRadius: 10,
            background: 'var(--card-count-fill)',
            color: 'var(--card-count-ink)',
            fontSize: 12,
            fontWeight: 700,
            textAlign: 'center',
            zIndex: 1,
            pointerEvents: 'none',
          }}
        >
          {n}
        </span>
      )}
      {l.overflow && (
        <div
          data-pile-overflow
          style={{
            marginTop: 2,
            padding: '1px 5px',
            borderRadius: 4,
            background: 'var(--card-overflow-fill)',
            color: 'var(--card-overflow-ink)',
            fontSize: 12,
            fontWeight: 700,
            textAlign: 'center',
          }}
        >
          {horizontal ? 'scroll →' : 'scroll ↓'}
        </div>
      )}
    </>
  );
}

/** One slot: its cards (or empty outline), badge and overflow note. */
export function SlotView({l, r}: {l: SlotLayout; r: Renderer}): JSX.Element {
  const s = l.spec;
  const rot = s.rotate ?? 0;
  const slotKey = targetKey({kind: 'slot', slot: s.id});
  const slotNode = r.a11y.get(slotKey);
  const slotLib = r.rt.libraryTagsFor(slotKey);
  const n = l.items.length;
  const maxExtent =
    s.maxExtent !== undefined ? s.maxExtent * r.cardWidth : undefined;
  const horizontal =
    l.kind === 'hand' || s.pile === 'fan-right' || s.pile === 'fan-left';
  const frame = slotFrame(l.overflow, maxExtent, horizontal);
  // A scrolled pile with no button in it takes focus itself, so a keyboard
  // can scroll it (axe scrollable-region-focusable).
  const scrollPile = l.overflow && maxExtent !== undefined && !hasButton(l, r);
  const {slotA11y, kindAttrs} = slotAttrs(l, slotNode);
  // Slot tags (the spec's, then the render's) draw on the EMPTY outline: a
  // slot with cards shows its cards' own tags. TagStyle.empty 'hidden' draws
  // no outline at all (the box is kept: still a drop target).
  const tags = [...new Set([...(s.tags ?? []), ...(r.slotTags?.[s.id] ?? [])])];
  const look = tags.length ? resolveTagStyle(tags, tagStyles(r.style)) : null;
  const tooltip = r.policy?.slotTooltip?.(s.id);
  const hidden = n === 0 && look?.empty === 'hidden';
  const body =
    n === 0 ? (
      <EmptyBody l={l} r={r} tags={tags} look={look} hidden={hidden} />
    ) : (
      <div
        style={{
          position: 'relative',
          width: l.frame.width,
          height: l.frame.height,
        }}
      >
        {l.entries.map(e => (
          <CardSlotItem key={e.placement.id} e={e} r={r} />
        ))}
      </div>
    );
  return (
    <div
      data-slot={s.id}
      data-slot-id={s.id}
      data-rotate={rot || undefined}
      data-covered-by-design={s.coveredByDesign ? 'true' : undefined}
      data-slot-tags={tags.length ? tags.join(' ') : undefined}
      // A slot whose look hides its empty outline: a symbol-only card
      // in it draws none either (cards.css .pcard--symbol).
      data-empty-outline={look?.empty === 'hidden' ? 'hidden' : undefined}
      data-drop-legal={slotLib.includes('drop-legal') ? 'true' : undefined}
      className={slotLib.map(t => `pcard--t-${t}`).join(' ') || undefined}
      title={tooltip}
      {...slotA11y}
      style={{
        position: 'absolute',
        left: l.box.left,
        top: l.box.top,
        // An interactive slot (a stock you click to draw) points
        // wherever a click reaches it: its non-interactive cards
        // inherit this, interactive cards point anyway.
        cursor: slotNode?.role === 'button' ? 'pointer' : undefined,
        // A hidden empty slot nobody can activate draws nothing, so it
        // must not take the pointer from cards drawn under it (a slot
        // laid over another's pile). Drops resolve by geometry, not here.
        pointerEvents:
          hidden && slotNode?.role !== 'button' ? 'none' : undefined,
        transform: rot ? `rotate(${String(rot)}deg)` : undefined,
        transformOrigin: `${String(l.box.origin.x)}px ${String(l.box.origin.y)}px`,
      }}
    >
      <div
        {...kindAttrs}
        data-compressed={l.compressed ? 'true' : 'false'}
        data-overflow={l.overflow ? 'true' : 'false'}
        // Top-aligned: an inline-block on the baseline leaves the
        // line's descender under it, which makes the slot's element
        // taller than its computed box (a turned hand then reaches
        // past the table's edge by that much).
        style={{
          position: 'relative',
          display: 'inline-block',
          verticalAlign: 'top',
          flex: 'none',
        }}
      >
        <div style={frame} tabIndex={scrollPile ? 0 : undefined}>
          {body}
        </div>
        <SlotNotes l={l} horizontal={horizontal} />
      </div>
    </div>
  );
}

// The wrapper's a11y attributes, from the card's a11y node.
function cardA11y(
  node: A11yNode | undefined,
  r: Renderer,
  id: string,
): CardA11yAttrs {
  let wrapperA11y: CardA11yAttrs = {};
  if (node) {
    // role 'none' is decorative: hidden, and never named (a name on a
    // role-less element is a prohibited ARIA attribute).
    const named = node.role !== 'none';
    wrapperA11y = {
      role: named ? node.role : undefined,
      'aria-label': named ? node.label : undefined,
      'aria-hidden': node.hidden || undefined,
      'aria-pressed': node.role === 'button' ? node.pressed : undefined,
      'aria-disabled': node.role === 'button' ? node.disabled : undefined,
      title: node.tooltip,
      'aria-describedby':
        named && node.description !== undefined
          ? describedById(r.descStore, r.tableId, id)
          : undefined,
      tabIndex:
        node.role === 'button' && !node.hidden
          ? node.tabbable
            ? 0
            : -1
          : undefined,
    };
  }
  return wrapperA11y;
}

// The item's drawing: the game's token, or the card (turned for a rotated
// base).
function drawItem(
  e: SlotEntry,
  r: Renderer,
  ref: ItemRef,
  gameTags: readonly string[],
): ReactNode {
  const item = e.placement.item;
  const w = e.local.width;
  const h = e.local.height;
  if (isToken(item)) {
    if (!r.renderToken) {
      throw new Error('<Table> needs renderToken for a token');
    }
    return r.renderToken(ref, {width: w, height: h});
  }
  const cardW = e.rotatedBase ? h : w;
  const props: CardRenderProps = {
    ref,
    card: item,
    width: cardW,
    faceUp: e.drawnFaceUp,
    tags: gameTags,
    covered: e.covered,
  };
  const inner = r.renderCard ? (
    r.renderCard(props, LibraryCard)
  ) : (
    <LibraryCard {...props} />
  );
  return e.rotatedBase ? (
    // Turned about the slot's centre; the wrapper is the turned box
    // the legibility check measures.
    <div
      style={{
        position: 'absolute',
        left: (w - cardW) / 2,
        top: (h - w) / 2,
        width: cardW,
        height: w,
        transform: `rotate(${String(item.turn === 90 ? 90 : -90)}deg)`,
      }}
    >
      {inner}
    </div>
  ) : (
    inner
  );
}

// A layer over the card's face for renderOverlay.
function CardOverlay({
  e,
  r,
  itemRef,
}: {
  e: SlotEntry;
  r: Renderer;
  itemRef: ItemRef;
}): JSX.Element | null {
  if (!r.renderOverlay) return null;
  const g = localCardGeometry(e.placement);
  return (
    <div
      data-card-overlay
      style={{position: 'absolute', inset: 0, pointerEvents: 'none'}}
    >
      <CardWidthContext.Provider value={r.cardWidth}>
        {r.renderOverlay(itemRef, {
          width: e.local.width,
          height: e.local.height,
          badgeAnchor: g.badgeAnchor,
          indexBox: g.indexBox,
        })}
      </CardWidthContext.Provider>
    </div>
  );
}

/** The library renderer, also renderCard's `Default`. No a11y on the card
 *  element: the wrapper is the a11y node. */
/** The library's own card drawing. */
export function LibraryCard(p: CardRenderProps): JSX.Element {
  return (
    <PlayingCard
      card={p.card.card}
      faceUp={p.faceUp}
      width={p.width}
      tags={p.tags}
      a11y={null}
      noLift
    />
  );
}

function CardSlotItem({e, r}: {e: SlotEntry; r: Renderer}): JSX.Element {
  const p = e.placement;
  const item = p.item;
  const ref: ItemRef = {id: p.id, slot: p.slot, position: p.position, item};
  const node = r.a11y.get(targetKey({kind: 'card', ref}));
  const lib = r.rt.libraryTagsFor(p.id);
  const gameTags = [...new Set(item.tags ?? [])];
  const allTags = [...gameTags, ...lib];
  const token = isToken(item);
  const known = !token && item.card !== null && e.drawnFaceUp;
  const style: CSSProperties = {
    position: 'absolute',
    left: e.local.x,
    top: e.local.y,
    width: e.local.width,
    height: e.local.height,
    transform: e.localRotate
      ? `rotate(${String(e.localRotate)}deg)`
      : undefined,
    transformOrigin: '50% 50%',
    zIndex: e.zBump ? e.zBump : undefined,
  };
  // The a11y node is the wrapper.
  const wrapperA11y = cardA11y(node, r, p.id);
  const drawn = drawItem(e, r, ref, gameTags);
  const overlay = token ? null : <CardOverlay e={e} r={r} itemRef={ref} />;
  return (
    <div
      ref={el => {
        if (!el) return;
        r.rt.register(p.id, el);
        return () => {
          r.rt.unregister(p.id, el);
        };
      }}
      data-pile-card={token ? undefined : ''}
      data-card-id={p.id}
      data-slot={p.slot}
      data-position={p.position}
      data-tags={allTags.length ? allTags.join(' ') : undefined}
      data-game-tags={gameTags.length ? gameTags.join(' ') : undefined}
      data-interactive={
        p.interactive && !!r.policy && !r.policy.inert ? 'true' : 'false'
      }
      data-face={known ? 'known' : 'unknown'}
      data-token={token ? 'true' : undefined}
      data-deck={token ? undefined : r.deck.id}
      data-index-scale={token ? undefined : r.style.indexScale}
      data-index-corner={token ? undefined : e.corner}
      data-rotated={e.rotatedBase ? 'true' : undefined}
      data-face-up={token ? undefined : e.drawnFaceUp ? 'true' : 'false'}
      className={allTags.map(t => `pcard--t-${t}`).join(' ') || undefined}
      {...wrapperA11y}
      style={style}
    >
      {drawn}
      {overlay}
    </div>
  );
}
