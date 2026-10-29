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

// A card side as static markup (internal to motion/): the side a flip turns
// away from, and the other side of a free card. Rendered with the library's
// own <PlayingCard> under the scene's CardStyle, so a flipped or freed card
// looks exactly like the one in the layout.
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {PlayingCard} from '../playing_card.tsx';
import {CardStyleProvider, type CardStyle} from '../style.ts';
import type {Card} from '../types.ts';

/** The HTML of one card side at `width` px: a face when `card` is given and
 *  `faceUp`, else a back. */
export function cardSideMarkup(
  style: CardStyle,
  card: Card | null,
  faceUp: boolean,
  width: number,
): string {
  if (!(width > 0)) {
    throw new Error(`cardSideMarkup: width must be > 0, got ${String(width)}`);
  }
  if (faceUp && card === null) {
    throw new Error(
      'cardSideMarkup: faceUp with card: null (an unknown face is always a back)',
    );
  }
  return renderToStaticMarkup(
    createElement(CardStyleProvider, {
      value: style,
      children: createElement(PlayingCard, {card, faceUp, width, a11y: null}),
    }),
  );
}

/** A detached element holding one card side, absolutely filling its parent. */
export function cardSideElement(
  doc: Document,
  style: CardStyle,
  card: Card | null,
  faceUp: boolean,
  width: number,
): HTMLElement {
  const el = doc.createElement('div');
  el.setAttribute('aria-hidden', 'true');
  el.style.position = 'absolute';
  el.style.left = '0';
  el.style.top = '0';
  el.innerHTML = cardSideMarkup(style, card, faceUp, width);
  return el;
}
