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

import {act, type JSX, useEffect} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, describe, expect, it} from 'vitest';

import {
  CardScene,
  useCardScene,
  type CardSceneApi,
} from '../../../cards/library/index.ts';
import {only} from '../../../testing/render.tsx';
import {
  GIN_STATUS,
  GIN_STATUS_JOIN,
  ginLaydownPrompt,
} from '../gin_turn_line.ts';
import {TurnLineAnnouncer} from './turn_line_announcer.tsx';

let root: Root | null = null;
let container: HTMLElement | null = null;
/** The rendered scene's API, for a drop announcement as the policy makes. */
const grabbed: {scene: CardSceneApi | null} = {scene: null};

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  container?.remove();
  root = null;
  container = null;
  grabbed.scene = null;
});

function Grab(): null {
  const scene = useCardScene();
  useEffect(() => {
    grabbed.scene = scene;
  });
  return null;
}

function Board({text}: {text: string}): JSX.Element {
  return (
    <CardScene>
      <Grab />
      <TurnLineAnnouncer text={text} />
    </CardScene>
  );
}

/** Renders the board with turn line `text`, into the same root each time. */
function show(text: string): void {
  if (root === null) {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
  }
  const r = root;
  act(() => {
    r.render(<Board text={text} />);
  });
}

/** The scene's live region's words, without the repeat marker. */
function live(): string {
  const region = only(document, '[data-card-live]');
  return region.textContent.replace(/\u00a0$/, '');
}

function status(...phrases: string[]): string {
  return phrases.join(GIN_STATUS_JOIN);
}

describe('TurnLineAnnouncer', () => {
  it('replaces a drop announcement with the knock and lay-down status', () => {
    show(status(GIN_STATUS.yourTurn, GIN_STATUS.discard));
    // A drag onto the discard pile: the policy's drop announcement.
    act(() => {
      grabbed.scene?.announce('Discarding Queen of Spades');
    });
    expect(live()).toBe('Discarding Queen of Spades');
    // The next move: the region reads the turn line, not the old drop.
    show(status(GIN_STATUS.opponentTurn));
    expect(live()).toBe(status(GIN_STATUS.opponentTurn));
    // The knock, then each lay-down prompt, as the stats region shows them.
    const knock = status(
      GIN_STATUS.yourTurn,
      GIN_STATUS.knock,
      GIN_STATUS.pressDone,
    );
    show(knock);
    expect(live()).toBe(knock);
    show(ginLaydownPrompt('discard'));
    expect(live()).toBe(ginLaydownPrompt('discard'));
    show(ginLaydownPrompt('discard', 'Ten of Hearts'));
    expect(live()).toBe('Discarding Ten of Hearts. Stage melds, then Done.');
    show(ginLaydownPrompt('melds'));
    expect(live()).toBe('Stage your melds, then Done.');
    show(GIN_STATUS.gameOver);
    expect(live()).toBe(GIN_STATUS.gameOver);
  });

  it('says a turn line once, not on every render', () => {
    show(status(GIN_STATUS.yourTurn, GIN_STATUS.discard));
    act(() => {
      grabbed.scene?.announce('Laying off Five of Clubs');
    });
    // A render with the same turn line leaves the newer drop's words.
    show(status(GIN_STATUS.yourTurn, GIN_STATUS.discard));
    expect(live()).toBe('Laying off Five of Clubs');
  });

  it('says nothing for an empty turn line', () => {
    show('');
    expect(live()).toBe('');
  });
});
