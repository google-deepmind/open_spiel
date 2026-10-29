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

import type {JSX} from 'react';
import {
  CARD_STYLES,
  TrickViewSection,
  useGameCardStyle,
  type CardStyle,
} from '../shared/trick_view_section.tsx';

export {CARD_STYLES, type CardStyle};

/** The card style setting's key. */
export const CARD_STYLE_KEY = 'euchre.cardStyle';

/** The Show all cards setting's key. */
export const SHOW_ALL_CARDS_KEY = 'euchre.showAllCards';

/**
 * Hook to get/set euchre's card style: corner index + large pip by default.
 */
export function useCardStyle(): [
  CardStyle,
  (next: CardStyle | ((current: CardStyle) => CardStyle)) => void,
] {
  return useGameCardStyle(CARD_STYLE_KEY);
}

/**
 * The View panel's Euchre section: card style, trick hold, show all cards.
 * Hearts draws the same section (trick_view_section.tsx).
 */
export function EuchreGameplayConfigExtras(): JSX.Element {
  return (
    <TrickViewSection
      title="Euchre"
      cardStyleKey={CARD_STYLE_KEY}
      showAllCardsKey={SHOW_ALL_CARDS_KEY}
    />
  );
}
