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

import {readHints} from '../renderers/hints.ts';
import {Lines} from '../renderers/lines.tsx';
import type {RendererProps} from '../renderers/renderer.ts';
import {
  array,
  boolean,
  field,
  nullable,
  optionalField,
  record,
  string,
} from '../renderers/struct_reader.ts';
import {cardActions} from './card_hints.ts';
import {TableCardStyle} from './table_cards/index.ts';
import {Hand} from './hand.tsx';
import {Pile, Trick, type TrickPlay} from './pile.tsx';
import {skinFor} from './skins.ts';

import './cards.css';

interface Row {
  readonly label: string;
  readonly cards: Array<string | null>;
  readonly fanned: boolean;
}

function readCards(value: unknown, path: string): Array<string | null> {
  return array(value, path, (card, cardPath) =>
    nullable(card, cardPath, string),
  );
}

function readRow(value: unknown, path: string): Row {
  const fanned = optionalField(value, path, 'fanned', boolean, false);
  return {
    label: field(value, path, 'label', string),
    cards: field(value, path, 'cards', readCards),
    fanned,
  };
}

function readTrick(value: unknown, path: string): readonly TrickPlay[] {
  return array(value, path, (play, playPath) => ({
    label: field(play, playPath, 'label', string),
    card: field(play, playPath, 'card', (card, cardPath) =>
      nullable(card, cardPath, string),
    ),
  }));
}

// The optional field `name` of the view, read by `read`, or `fallback`.
function optional<T>(
  view: unknown,
  name: string,
  read: (value: unknown, path: string) => T,
  fallback: T,
): T {
  return optionalField(view, 'view', name, read, fallback);
}

/**
 * The `cards` renderer, for any adapter whose view is a card table:
 *
 *   {"hands": [{"label": "You", "cards": ["As", null], "fanned": true}],
 *    "piles": [{"label": "Discard", "cards": ["Td"]}],
 *    "trick": [{"label": "Player 0", "card": "Kh"}],
 *    "lines": ["Hearts are broken."], "skin": "standard"}
 *
 * Every field is optional. A card hinted {"card": code} plays its action
 * when pressed, wherever it lies; an action hinted {"button": text} is a
 * button under the table.
 */
export function CardTableRenderer({
  presentation,
  play,
}: RendererProps): JSX.Element {
  const {view} = presentation;
  const hands = optional(
    view,
    'hands',
    (value, path) => array(value, path, readRow),
    [],
  );
  const piles = optional(
    view,
    'piles',
    (value, path) => array(value, path, readRow),
    [],
  );
  const trick = optional(view, 'trick', readTrick, null);
  const skin = skinFor(optional(view, 'skin', string, null));
  const playable = cardActions(presentation);
  const buttons = readHints(presentation, (hint, path) =>
    hint !== null && 'button' in record(hint, path)
      ? field(hint, path, 'button', string)
      : null,
  );
  // The app-wide deck and back (the table-card settings), so face-down
  // cards draw the back the Card back picker chose.
  return (
    <TableCardStyle>
      <div className="renderer">
        <Lines view={view} />
        {piles.length > 0 && (
          <div className="renderer__piles">
            {piles.map(pile => (
              <Pile
                key={pile.label}
                label={pile.label}
                cards={pile.cards}
                playable={playable}
                play={play}
                skin={skin}
              />
            ))}
          </div>
        )}
        {trick !== null && <Trick plays={trick} skin={skin} />}
        {hands.map(hand => (
          <Hand
            key={hand.label}
            label={hand.label}
            cards={hand.cards}
            fanned={hand.fanned}
            playable={playable}
            play={play}
            skin={skin}
          />
        ))}
        {buttons.length > 0 && (
          <div className="renderer__buttons" role="group" aria-label="Moves">
            {buttons.map(({action, hint: text}) => (
              <button
                type="button"
                key={action}
                disabled={play === null}
                onClick={() => {
                  play?.(action);
                }}
              >
                {text}
              </button>
            ))}
          </div>
        )}
      </div>
    </TableCardStyle>
  );
}
