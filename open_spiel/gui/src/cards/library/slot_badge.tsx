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

// Slot badges: information about a
// pile — '×4', '7 cards', a state word — drawn BESIDE the pile, never on a
// card. A badge is a table region attached to its slot (RegionSpec.attach),
// so the spec reserves its room, it turns with the slot or stays upright,
// and the legibility check counts it as cover (a badge that lands on a
// neighbour's index is reported, not hidden). The face stays the deck's art.
import type {CSSProperties, JSX, ReactNode} from 'react';
import type {RegionSpec} from './table_spec.ts';

/** Where a badge sits against its slot. */
export interface SlotBadgeOptions {
  /** Side of the slot, in its local frame. Default 'bottom'. */
  side?: 'top' | 'bottom' | 'left' | 'right' | undefined;
  /** Along that side. Default 'center'. */
  align?: 'start' | 'center' | 'end' | undefined;
  /** From the slot, card widths. Default 0.05. */
  gap?: number | undefined;
  /** Card widths. Default 1 × 0.28. */
  width?: number | undefined;
  height?: number | undefined;
  /** Drawn unrotated on a turned slot. Default true. */
  upright?: boolean | undefined;
  /** Default slotBadgeId(slot). */
  id?: string | undefined;
}

/** A badge region's width, × card width. */
export const SLOT_BADGE_WIDTH = 1;
/** A badge region's height, × card width. */
export const SLOT_BADGE_HEIGHT = 0.28;
/** The badge text's floor, px: it scales with --card-w above this. The
 *  text floor (12 px). */
export const SLOT_BADGE_MIN_FONT_PX = 12;

/** The pill's side padding: none at a 44 px card, about 0.06 card widths at 72 px. */
export const SLOT_BADGE_PADDING_CSS =
  'max(0px, calc((var(--card-w, 72px) - 44px) * 0.15))';
/** The same, px, at a card `cardWidth` px wide (for layout checks). */
export function slotBadgePaddingPx(cardWidth: number): number {
  return Math.max(0, (cardWidth - 44) * 0.15);
}

/** The region id a slot's badge uses by default. */
export function slotBadgeId(slot: string): string {
  if (!slot) throw new Error('slotBadgeId: slot id is empty');
  return `${slot}-badge`;
}

/** The RegionSpec for a badge beside `slot`: add it to TableSpec.regions and
 *  put a <SlotBadge> under its id in <Table regions> (slotBadges does both
 *  sides of that for a record of texts). */
export function slotBadgeRegion(
  slot: string,
  o: SlotBadgeOptions = {},
): RegionSpec {
  const width = o.width ?? SLOT_BADGE_WIDTH;
  const height = o.height ?? SLOT_BADGE_HEIGHT;
  if (!(width > 0 && height > 0)) {
    throw new Error(
      `slotBadgeRegion('${slot}'): size must be > 0, got ${String(width)} x ${String(height)}`,
    );
  }
  return {
    id: o.id ?? slotBadgeId(slot),
    width,
    height,
    attach: {
      slot,
      side: o.side ?? 'bottom',
      gap: o.gap ?? 0.05,
      align: o.align ?? 'center',
    },
    upright: o.upright ?? true,
    occludes: true,
  };
}

/** A badge's colours. */
export type SlotBadgeTone = 'neutral' | 'accent' | 'muted';

/** A badge: its text, tone and accessible name. */
export interface SlotBadgeProps {
  children: ReactNode;
  /** Default 'neutral'. */
  tone?: SlotBadgeTone | undefined;
  /** Accessible text when `children` is not plain words (an icon). */
  label?: string | undefined;
  testId?: string | undefined;
}

const TONES: Record<
  SlotBadgeTone,
  {background: string; color: string; border: string}
> = {
  neutral: {
    background: 'var(--card-badge-neutral-fill)',
    color: 'var(--card-badge-neutral-ink)',
    border: 'var(--color-none)',
  },
  accent: {
    background: 'var(--card-badge-accent-fill)',
    color: 'var(--card-badge-accent-ink)',
    border: 'var(--card-badge-accent-edge)',
  },
  muted: {
    background: 'var(--card-badge-muted-fill)',
    color: 'var(--card-badge-muted-ink)',
    border: 'var(--card-badge-muted-edge)',
  },
};

/** A pill that fills its region; text scales with the card width (--card-w,
 *  which <Table> sets on every region), never below SLOT_BADGE_MIN_FONT_PX. */
export function SlotBadge({
  children,
  tone = 'neutral',
  label,
  testId,
}: SlotBadgeProps): JSX.Element {
  if (!Object.hasOwn(TONES, tone)) {
    throw new Error(`SlotBadge: unknown tone ${JSON.stringify(tone)}`);
  }
  const t = TONES[tone];
  const style: CSSProperties = {
    boxSizing: 'border-box',
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    // Side padding shrinks to none at a 44 px card: there, at the 12 px
    // floor, '10 cards' needs the whole of a 1.3-card pill.
    padding: `0 ${SLOT_BADGE_PADDING_CSS}`,
    borderRadius: 'calc(var(--card-w, 72px) * 0.14)',
    border: `1px solid ${t.border}`,
    background: t.background,
    color: t.color,
    // Scales with the card, never below a legible floor (small cards on a phone).
    fontSize: `max(${String(SLOT_BADGE_MIN_FONT_PX)}px, calc(var(--card-w, 72px) * 0.17))`,
    fontWeight: 700,
    lineHeight: 1,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    pointerEvents: 'none',
  };
  return (
    <div
      data-slot-badge={tone}
      data-testid={testId}
      role={label ? 'img' : undefined}
      aria-label={label}
      style={style}
    >
      {children}
    </div>
  );
}

/**
 * Badges for several slots at once: the regions to add to the spec and the
 * content for <Table regions>. A slot with a null/undefined text keeps its
 * region (the room stays reserved, so nothing moves) and draws nothing.
 */
export function slotBadges(
  texts: Readonly<Record<string, ReactNode | null | undefined>>,
  o: SlotBadgeOptions & {tone?: SlotBadgeTone | undefined} = {},
): {regions: RegionSpec[]; content: Record<string, ReactNode>} {
  if (o.id !== undefined) {
    throw new Error(
      'slotBadges: per-slot ids come from slotBadgeId; do not pass id',
    );
  }
  const regions: RegionSpec[] = [];
  const content: Record<string, ReactNode> = {};
  for (const [slot, text] of Object.entries(texts)) {
    const r = slotBadgeRegion(slot, o);
    regions.push(r);
    if (text !== null && text !== undefined) {
      content[r.id] = <SlotBadge tone={o.tone}>{text}</SlotBadge>;
    }
  }
  return {regions, content};
}
