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

import {type JSX, type ReactNode, useMemo} from 'react';

import {CardStyleProvider} from '../../cards/library/index.ts';
import {
  TableCardSettingsFields,
  TableCardStyle,
  type TableCardPatch,
} from '../../cards/table_cards/index.ts';
import {useCardFaceStyle} from './gin_rummy_gameplay_config_extras.tsx';
import {GIN_TABLE_CARDS_GAME} from './gin_table_cards.ts';
import {
  GIN_CARD_STYLE,
  GIN_FACE_STYLES,
  type GinCardFace,
} from './gin_rummy_layout.ts';

/**
 * The card face's pin: 'app' reads the shared deck; gin's own 'modern' face
 * fixes its deck.
 */
function facePin(face: GinCardFace): TableCardPatch | undefined {
  const {deck} = GIN_FACE_STYLES[face];
  return deck === undefined ? undefined : {cards: {deck}};
}

/**
 * Gin's CardStyle (gin_rummy_layout.ts) over the app's, with the shared deck,
 * back and suit symbols. Everything that draws gin cards sits inside one.
 */
export function GinCardStyleProvider({
  children,
}: {
  readonly children: ReactNode;
}): JSX.Element {
  const [face] = useCardFaceStyle();
  const faceStyle = useMemo(() => {
    const rest = {...GIN_FACE_STYLES[face]};
    delete rest.deck;
    return rest;
  }, [face]);
  const pin = facePin(face);
  return (
    <CardStyleProvider value={GIN_CARD_STYLE}>
      <TableCardStyle
        game={GIN_TABLE_CARDS_GAME}
        {...(pin === undefined ? {} : {pin})}
      >
        <CardStyleProvider value={faceStyle}>{children}</CardStyleProvider>
      </TableCardStyle>
    </CardStyleProvider>
  );
}

/** The shared table and card settings as Gin shows them (its face's pin applied). */
export function GinTableCardFields(): JSX.Element {
  const [face] = useCardFaceStyle();
  const pin = facePin(face);
  return (
    <TableCardSettingsFields
      game={GIN_TABLE_CARDS_GAME}
      {...(pin === undefined ? {} : {pin})}
    />
  );
}
