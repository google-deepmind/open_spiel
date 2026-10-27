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
import {getDeck} from '../../cards/library/index.ts';
import {useIsAnalysisMode} from '../../game/bottom_seat.ts';
import {useViewSetting} from '../shared/view_settings.ts';
import {TrickHoldField} from '../shared/trick_hold_field.tsx';

/**
 * The deck the cards are drawn with, by registry id: corner index (which a
 * fanned hand shows), large centred rank, or corner index + large pip.
 */
export type CardStyle = 'simple-corner' | 'simple' | 'basic';
/** The card styles in the menu's order; each is named by its deck's label. */
export const CARD_STYLES: readonly CardStyle[] = [
  'simple-corner',
  'simple',
  'basic',
];

/** The card style setting's key. */
export const CARD_STYLE_KEY = 'euchre.cardStyle';

/**
 * Hook to get/set the card style: the simple deck, indices in the corner, by default.
 */
export function useCardStyle(): [
  CardStyle,
  (next: CardStyle | ((current: CardStyle) => CardStyle)) => void,
] {
  return useViewSetting<CardStyle>(CARD_STYLE_KEY, 'simple-corner');
}

/** The View panel's Euchre section: card style, trick hold, show all cards. */
export function EuchreGameplayConfigExtras(): JSX.Element {
  const isAnalysisMode = useIsAnalysisMode();
  const [showAllCards, setShowAllCards] = useViewSetting<boolean>(
    'euchre.showAllCards',
    false,
  );
  const [cardStyle, setCardStyle] = useCardStyle();

  return (
    <section className="euchre-config-section" aria-label="Euchre">
      <h3>Euchre</h3>
      <label className="ogs-config-field">
        <span>Card style</span>
        <select
          value={cardStyle}
          onChange={e => {
            setCardStyle(e.target.value as CardStyle);
          }}
        >
          {CARD_STYLES.map(id => {
            const deck = getDeck(id);
            return (
              <option key={id} value={id} title={deck.description}>
                {deck.label}
              </option>
            );
          })}
        </select>
        <span className="ogs-config-note" data-testid="deck-description">
          {getDeck(cardStyle).description}
        </span>
      </label>
      <TrickHoldField />
      <label className="ogs-config-toggle">
        <input
          type="checkbox"
          checked={showAllCards}
          disabled={!isAnalysisMode}
          onChange={event => {
            setShowAllCards(event.target.checked);
          }}
        />
        <span>Show all cards</span>
      </label>
      {!isAnalysisMode && (
        <p className="ogs-config-note">
          Cards can only be revealed in analysis mode.
        </p>
      )}
    </section>
  );
}
