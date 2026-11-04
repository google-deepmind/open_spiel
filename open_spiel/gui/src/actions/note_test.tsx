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

import {describe, expect, it} from 'vitest';

import {only, render} from '../testing/render.tsx';
import {NoFrame, Note} from './note.tsx';

describe('Note', () => {
  it('is a status line', () => {
    const {container, unmount} = render(<Note>Spectating.</Note>);
    const note = only(container, 'p.actions-note');
    expect(note.getAttribute('role')).toBe('status');
    expect(note.textContent).toBe('Spectating.');
    unmount();
  });
});

describe('NoFrame', () => {
  it('claims no more than that there is no position to show', () => {
    const {container, unmount} = render(<NoFrame />);
    expect(only(container, '[role="status"]').textContent).toBe(
      'No position to show.',
    );
    unmount();
  });
});
