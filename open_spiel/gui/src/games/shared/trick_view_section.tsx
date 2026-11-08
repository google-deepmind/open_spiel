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

// A trick-taking game's section of the View panel: Card style, Trick hold
// and Show all cards, in that order. Euchre and hearts both draw exactly
// this section, so their options match: the same fields, labels, choices,
// order and defaults.
import type {JSX} from 'react';
import {getDeck} from '../../cards/library/index.ts';
import {useIsAnalysisMode} from '../../game/bottom_seat.ts';
import {TrickHoldField} from './trick_hold_field.tsx';
import {useViewSetting} from './view_settings.ts';
import '../../view/view.css';
import './trick_view_section.css';

/**
 * The deck the cards are drawn with, by registry id: corner index + large
 * pip, large corner index, or large centred rank and suit.
 */
export type CardStyle = 'basic' | 'simple-corner' | 'simple';

/** The card styles in the menu's order; each is named by its deck's label. */
export const CARD_STYLES: readonly CardStyle[] = [
  'basic',
  'simple-corner',
  'simple',
];

/** The card style every game starts with: corner index + large pip. */
export const DEFAULT_CARD_STYLE_ID: CardStyle = 'basic';

/** Hook to get/set the card style a game keeps under `key`. */
export function useGameCardStyle(
  key: string,
): [
  CardStyle,
  (next: CardStyle | ((current: CardStyle) => CardStyle)) => void,
] {
  return useViewSetting<CardStyle>(key, DEFAULT_CARD_STYLE_ID);
}

/** The "Card style" field, with the chosen deck's "best for" line. */
export function CardStyleField({
  settingKey,
}: {
  /** The game's setting key ('euchre.cardStyle', 'hearts.cardStyle'). */
  readonly settingKey: string;
}): JSX.Element {
  const [cardStyle, setCardStyle] = useGameCardStyle(settingKey);
  return (
    <label className="ogs-config-field">
      <span>Card style</span>
      <select
        value={cardStyle}
        onChange={e => {
          setCardStyle(e.target.value as CardStyle);
        }}
        data-testid="card-style-select"
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
  );
}

/** The "Show all cards" toggle: analysis only, as the oracle's view. */
export function ShowAllCardsField({
  settingKey,
  isAnalysisMode,
}: {
  /** The game's setting key ('euchre.showAllCards', 'hearts.showAllCards'). */
  readonly settingKey: string;
  readonly isAnalysisMode: boolean;
}): JSX.Element {
  const [showAllCards, setShowAllCards] = useViewSetting<boolean>(
    settingKey,
    false,
  );
  return (
    <>
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
    </>
  );
}

/**
 * A trick-taking game's View section: a card like its sibling tabs'
 * (Perspective), named for the game but with no heading of its own (the
 * tab already says the game's name).
 */
export function TrickViewSection({
  title,
  cardStyleKey,
  showAllCardsKey,
}: {
  /** The game's name, the section's accessible name ("Euchre", "Hearts"). */
  readonly title: string;
  readonly cardStyleKey: string;
  readonly showAllCardsKey: string;
}): JSX.Element {
  const isAnalysisMode = useIsAnalysisMode();
  return (
    <section className="view-card trick-view-section" aria-label={title}>
      <CardStyleField settingKey={cardStyleKey} />
      <TrickHoldField />
      <ShowAllCardsField
        settingKey={showAllCardsKey}
        isAnalysisMode={isAnalysisMode}
      />
    </section>
  );
}
