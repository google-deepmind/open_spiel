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

// Mapping cards to the legal actions that play them. An adapter hints a card
// action as {"card": "<code>"}; any other hint is not a card's.

import type {Presentation} from '../contract/contract.ts';
import {readHints} from '../renderers/hints.ts';
import {field, record, string} from '../renderers/struct_reader.ts';
import {cardCode, parseCard} from './card.ts';

/** The action each hinted card plays, keyed by the card's canonical code. */
export function cardActions(
  presentation: Presentation,
): ReadonlyMap<string, number> {
  const hinted = readHints(presentation, (hint, path) =>
    hint !== null && 'card' in record(hint, path)
      ? cardCode(parseCard(field(hint, path, 'card', string)))
      : null,
  );
  const actions = new Map<string, number>();
  for (const {action, hint: code} of hinted) {
    if (actions.has(code)) {
      throw new Error(`Two actions play the card ${code}.`);
    }
    actions.set(code, action);
  }
  return actions;
}

/**
 * The action that plays the card `code`, or undefined when it is not
 * playable or hidden.
 */
export function actionForCard(
  actions: ReadonlyMap<string, number>,
  code: string | null,
): number | undefined {
  return code === null ? undefined : actions.get(cardCode(parseCard(code)));
}
