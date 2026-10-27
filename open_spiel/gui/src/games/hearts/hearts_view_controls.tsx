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

// Hearts view settings as controls: your hand's shape, the trick layout and
// the trick hold, beside the shared table and card settings. They sit in the
// View panel's Hearts section (hearts_gameplay_config_extras.tsx); the board
// draws no View menu of its own, as the View panel holds every one.
import type {JSX} from 'react';
import {TableCardSettingsFields} from '../../cards/table_cards/index.ts';
import {TrickHoldField} from '../shared/trick_hold_field.tsx';
import {TrickLayoutField} from '../shared/trick_layout_field.tsx';
import {useHeartsTableCardScope} from './hearts_deck.ts';
import {HEARTS_HAND_SHAPES, type HeartsHandShape} from './hearts_layout.ts';
import {
  HAND_SHAPE_LABELS,
  useHeartsHandShape,
  useHeartsTrickLayout,
} from './hearts_view_settings.ts';

/** The hand shape, the trick layout and the trick hold, as config fields. */
export function HeartsViewFields(): JSX.Element {
  const [shape, setShape] = useHeartsHandShape();
  const [trick, setTrick] = useHeartsTrickLayout();
  return (
    <>
      <label className="ogs-config-field">
        <span>Hand shape</span>
        <select
          value={shape}
          onChange={e => {
            setShape(e.target.value as HeartsHandShape);
          }}
        >
          {HEARTS_HAND_SHAPES.map(s => (
            <option key={s} value={s}>
              {HAND_SHAPE_LABELS[s]}
            </option>
          ))}
        </select>
      </label>
      <TrickLayoutField value={trick} onChange={setTrick} />
      <TrickHoldField />
    </>
  );
}

/** The shared table and card settings, as Hearts shows them. */
export function HeartsTableCardFields(): JSX.Element {
  return <TableCardSettingsFields {...useHeartsTableCardScope()} />;
}
