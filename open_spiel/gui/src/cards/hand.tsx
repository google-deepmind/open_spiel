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

import {CardFace} from './card_face.tsx';
import {actionForCard} from './card_hints.ts';
import {type Skin, STANDARD_SKIN} from './skins.ts';

import './cards.css';

/** What a row of cards is given. */
export interface HandProps {
  /** What the row is: "Your hand", "Player 1". */
  readonly label: string;
  /** The cards' codes, null for each face-down card. */
  readonly cards: ReadonlyArray<string | null>;
  /** The action each playable card plays, from `cardActions`. */
  readonly playable?: ReadonlyMap<string, number>;
  /** Plays an action; null when the viewer cannot play. */
  readonly play?: ((action: number) => void) | null;
  /**
   * Called instead of `play` when a card is pressed, for a renderer that
   * gathers several cards before it acts.
   */
  readonly onCard?: (code: string, action: number | undefined) => void;
  /** Codes drawn raised, as chosen. */
  readonly selected?: ReadonlySet<string>;
  readonly skin?: Skin;
  /** Overlaps the cards, as a held hand is fanned. */
  readonly fanned?: boolean;
}

/**
 * A labelled row of cards. A card is a button when pressing it does
 * something: plays its action, or calls `onCard`.
 */
export function Hand({
  label,
  cards,
  playable = new Map(),
  play = null,
  onCard,
  selected,
  skin = STANDARD_SKIN,
  fanned = false,
}: HandProps): JSX.Element {
  return (
    <section className="card-row" aria-label={label}>
      <h4 className="card-row__label">{label}</h4>
      <div
        className="card-row__cards"
        data-fanned={fanned ? 'true' : undefined}
      >
        {cards.map((code, index) => {
          const action = actionForCard(playable, code);
          const press = pressHandler(code, action, play, onCard);
          return (
            <CardFace
              key={`${code ?? 'hidden'}-${String(index)}`}
              code={code}
              skin={skin}
              selected={code !== null && (selected?.has(code) ?? false)}
              {...(press === undefined ? {} : {onSelect: press})}
              {...(action === undefined ? {} : {action: 'play'})}
            />
          );
        })}
        {cards.length === 0 && (
          <span className="card-row__empty">No cards</span>
        )}
      </div>
    </section>
  );
}

// What pressing a card does, or undefined when it does nothing.
function pressHandler(
  code: string | null,
  action: number | undefined,
  play: ((action: number) => void) | null,
  onCard: ((code: string, action: number | undefined) => void) | undefined,
): (() => void) | undefined {
  if (code !== null && onCard !== undefined) {
    return () => {
      onCard(code, action);
    };
  }
  if (action === undefined || play === null) return undefined;
  return () => {
    play(action);
  };
}
