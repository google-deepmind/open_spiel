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

import type {JSX, ReactNode} from 'react';
import './decks/index.ts';
import {getDeck} from './registry.ts';
import {CardStyleProvider, useCardStyle} from './style.ts';
import type {ColorScheme, Deck} from './types.ts';

/** A deck and colour scheme in effect. */
export interface CardDeckContextValue {
  deck: Deck;
  scheme: ColorScheme;
}

/**
 * App-wide default deck and colour scheme; nest another provider to override
 * per game. Either prop left out inherits from the enclosing provider. A thin
 * alias for <CardStyleProvider value={{ deck, scheme }}>.
 */
export function CardDeckProvider({
  deck,
  scheme,
  children,
}: {
  deck?: string | undefined;
  scheme?: ColorScheme | undefined;
  children: ReactNode;
}): JSX.Element {
  return (
    <CardStyleProvider value={{deck, scheme}}>{children}</CardStyleProvider>
  );
}

/** The deck and scheme in effect; outside any provider, basic / two-colour. */
export function useCardDeck(): CardDeckContextValue {
  const {deck, scheme} = useCardStyle();
  return {deck: getDeck(deck), scheme};
}
