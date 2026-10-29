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

// A gin card's display state, and how it maps onto the shared card library:
// every state is a tag (styled by GIN_CARD_STYLE.tags), and the words a screen
// reader and a tooltip give for it come from here too.

import {
  cardLabel,
  type TableCard,
  type TableItem,
} from '../../../cards/library/index.ts';
import type {DeckConfig} from '../deck_context.tsx';
import {
  GRID_PREFIX,
  ginCard,
  gridSlotId,
  type GinZone,
} from '../gin_rummy_layout.ts';
import type {MeldType} from '../types.ts';

export type {MeldType} from '../types.ts';

/** A card's highlight: known to the other player, laying off, or staged. */
export type CardHighlight = 'alert' | 'layoff' | 'selected' | null;

/**
 * What a card's double-click does (the words its tooltip uses). 'discard'
 * also lets the card be dragged onto the discard pile.
 */
export type CardAction = 'discard' | 'compose' | 'take-back' | 'lay-off';

/**
 * A card a grid names without holding it, drawn as its symbol alone (the
 * shared TagStyle.symbol; GIN_CARD_STYLE.tags): 'dead' is discarded, grey;
 * 'known' is known to be in the other hand, orange.
 */
export type CardMark = 'dead' | 'known';

/** One grid cell's state: a card (with its marks) or an empty slot. */
export interface CardCellState {
  type: 'card' | 'slot';
  mark?: CardMark | undefined;
  isGhost?: boolean | undefined;
  /**
   * A staged lay-off on the knocker's meld: not in the hand, but drawn at
   * full strength with the 'staged-layoff' mark, never the ghost wash (a
   * washed-out card reads as dimmed).
   */
  isStagedLayoff?: boolean | undefined;
  isMelded?: boolean | undefined;
  meldType?: MeldType | null | undefined;
  highlight?: CardHighlight | undefined;
  isActionable?: boolean | undefined;
  onDoubleClick?: (() => void) | undefined;
  action?: CardAction | undefined;
}

interface TagOptions {
  ghost?: boolean;
  picked?: boolean;
}

/**
 * The tags a card in `state` carries. None dims a card: the staged discard
 * is its 'selected' ring at full contrast.
 */
export function ginTags(
  state: CardCellState | undefined,
  opts: TagOptions = {},
): string[] {
  // A marked card is its symbol and nothing else: no ghost wash, no ring.
  if (state?.mark !== undefined) return [state.mark];
  const ghost = opts.ghost === true || state?.isGhost === true;
  const staged = state?.isStagedLayoff === true;
  return [
    ghost && !staged && 'ghost',
    staged && 'staged-layoff',
    state?.meldType === 'run' && 'meld-run',
    state?.meldType === 'set' && 'meld-set',
    state?.highlight ?? false,
    opts.picked === true && 'picked',
  ].filter((t): t is string => typeof t === 'string');
}

/**
 * A face-up gin card on the table. Interactive only when it does something
 * (`interactive`), so a plain card is an image to a screen reader.
 */
export function ginItem(
  id: string,
  code: string,
  state: CardCellState | undefined,
  opts: TagOptions & {interactive?: boolean; was?: string} = {},
): TableCard {
  return {
    id,
    card: ginCard(code),
    tags: ginTags(state, opts),
    interactive:
      opts.interactive ??
      (state?.onDoubleClick !== undefined || state?.isActionable === true),
    ...(opts.was === undefined || opts.was === '' ? {} : {was: opts.was}),
  };
}

/**
 * A zone's suit x rank grid as table items: one per card with a 'card'
 * state, keyed by its code (a ghost by `${zone}ghost-${code}`, so the same
 * face can show as a card in one zone and a ghost in the other).
 */
export function gridItems(
  zone: GinZone,
  states: ReadonlyMap<string, CardCellState>,
  deck: DeckConfig,
): Record<string, TableItem[]> {
  const out: Record<string, TableItem[]> = {};
  for (const suit of deck.suitRows) {
    for (const rank of deck.rankCodes) {
      const code = `${rank}${suit.code}`;
      const state = states.get(code);
      if (state?.type !== 'card') continue;
      const id =
        state.isGhost === true ? `${GRID_PREFIX[zone]}ghost-${code}` : code;
      out[gridSlotId(zone, code)] = [ginItem(id, code, state)];
    }
  }
  return out;
}

const ACTION_WORDS: Readonly<Record<CardAction, string>> = {
  discard: 'discard it',
  compose: 'stage or cycle its meld',
  'take-back': 'take it back',
  'lay-off': "lay it off, or drag it onto the knocker's meld",
};

/** The state in words, for aria-describedby: 'In a run. Staged discard.' */
export function describeState(
  state: CardCellState | undefined,
  opts: TagOptions = {},
): string | undefined {
  const staged = state?.isStagedLayoff === true;
  const ghost = opts.ghost === true || state?.isGhost === true;
  const words = [
    state?.meldType === 'run' && 'In a run.',
    state?.meldType === 'set' && 'In a set.',
    staged ? 'Staged lay-off.' : ghost && 'Not in this hand.',
    state?.highlight === 'layoff' && !staged && 'Lays off.',
    state?.highlight === 'alert' && 'Known to the other player.',
    state?.highlight === 'selected' && 'Staged discard.',
    opts.picked === true && 'Picked to move.',
    state?.action !== undefined &&
      `Double-click to ${ACTION_WORDS[state.action]}.`,
  ].filter((w): w is string => typeof w === 'string');
  return words.length > 0 ? words.join(' ') : undefined;
}

/** The tooltip: the card and what a double-click does, when it does something. */
export function stateTooltip(
  code: string,
  state: CardCellState | undefined,
): string | undefined {
  if (state?.action === undefined) return undefined;
  return `${cardLabel(ginCard(code))}: double-click to ${ACTION_WORDS[state.action]}`;
}
