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

import {type JSX, useState} from 'react';

import type {HistoryEntry} from '../contract/contract.ts';
import {CopyableText} from '../controls/copy_button.tsx';
import {Segmented, type SegmentedOption} from '../controls/segmented.tsx';
import type {Session} from '../session/use_session.ts';
import {NoFrame, Note} from './note.tsx';

import './actions.css';

/** Whether the history is written as action labels or action ids. */
export type HistoryValues = 'strings' | 'ints';

/** Whether the history is written one action per line or on one line. */
export type HistoryLayout = 'multiline' | 'oneline';

const VALUES: ReadonlyArray<SegmentedOption<HistoryValues>> = [
  {value: 'strings', label: 'Strings'},
  {value: 'ints', label: 'Ints'},
];

const LAYOUTS: ReadonlyArray<SegmentedOption<HistoryLayout>> = [
  {value: 'multiline', label: 'Multi-line'},
  {value: 'oneline', label: 'One line'},
];

/**
 * The history as a JSON list, which is also a Python list literal, or null
 * when an entry is withheld from the viewer: a list with gaps replays
 * nothing.
 */
export function historyJson(
  history: readonly HistoryEntry[],
  values: HistoryValues,
  layout: HistoryLayout,
): string | null {
  const written: Array<number | string> = [];
  for (const {action, label} of history) {
    const value = values === 'ints' ? action : label;
    if (value === null) return null;
    written.push(value);
  }
  return layout === 'oneline'
    ? JSON.stringify(written)
    : JSON.stringify(written, null, 2);
}

/**
 * The history as text to copy, as labels or ids, on one line or many: two
 * pills, then the text in a card with its copy icon.
 */
export function HistoryText({session}: {session: Session}): JSX.Element {
  const [values, setValues] = useState<HistoryValues>('strings');
  const [layout, setLayout] = useState<HistoryLayout>('multiline');
  const {frame} = session;
  if (frame === null) return <NoFrame />;
  const text = historyJson(frame.history, values, layout);
  return (
    <div className="history-text">
      <div className="history-text__options">
        <Segmented
          label="Values"
          value={values}
          options={VALUES}
          onChange={setValues}
        />
        <Segmented
          label="Layout"
          value={layout}
          options={LAYOUTS}
          onChange={setLayout}
        />
      </div>
      {frame.history.length === 0 ? (
        <Note>No actions yet.</Note>
      ) : text === null ? (
        <Note>
          Some actions are withheld from this viewer, so the history cannot be
          written out.
        </Note>
      ) : (
        <CopyableText text={text} label="History" />
      )}
    </div>
  );
}
