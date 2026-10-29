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

/** What a pile is given. */
export interface PileProps {
  /** What the pile is: "Stock", "Discard". */
  readonly label: string;
  /** The pile's cards, bottom first; null for each face-down card. */
  readonly cards: ReadonlyArray<string | null>;
  /** The action each playable card plays, from `cardActions`. */
  readonly playable?: ReadonlyMap<string, number>;
  readonly play?: ((action: number) => void) | null;
  /** Plays this action when the top card is pressed, as drawing does. */
  readonly topAction?: number;
  readonly skin?: Skin;
}

/**
 * A pile shown by its top card and its size. The top card is a button when
 * it plays an action: its own hinted action, or `topAction`.
 */
export function Pile({
  label,
  cards,
  playable = new Map(),
  play = null,
  topAction,
  skin = STANDARD_SKIN,
}: PileProps): JSX.Element {
  const top = cards.length === 0 ? undefined : cards[cards.length - 1];
  const action =
    topAction ?? (top === undefined ? undefined : actionForCard(playable, top));
  const press =
    action === undefined || play === null
      ? undefined
      : () => {
          play(action);
        };
  return (
    <section className="card-pile" aria-label={label}>
      <h4 className="card-row__label">{label}</h4>
      {top === undefined ? (
        <span
          className="playing-card playing-card--empty"
          role="img"
          aria-label="Empty"
        />
      ) : (
        <CardFace
          code={top}
          skin={skin}
          {...(press === undefined ? {} : {onSelect: press, action: 'play'})}
        />
      )}
      <span className="card-pile__count">{`${String(cards.length)} cards`}</span>
    </section>
  );
}

/** One card played to a trick. */
export interface TrickPlay {
  /** Who played it: "Player 0", "North". */
  readonly label: string;
  readonly card: string | null;
}

/** The cards of the current trick, each under who played it. */
export function Trick({
  plays,
  label = 'Trick',
  skin = STANDARD_SKIN,
}: {
  readonly plays: readonly TrickPlay[];
  readonly label?: string;
  readonly skin?: Skin;
}): JSX.Element {
  return (
    <section className="card-trick" aria-label={label}>
      <h4 className="card-row__label">{label}</h4>
      <div className="card-trick__plays">
        {plays.map(({label: player, card}, index) => (
          <figure
            className="card-trick__play"
            key={`${player}-${String(index)}`}
          >
            <CardFace code={card} skin={skin} />
            <figcaption>{player}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
