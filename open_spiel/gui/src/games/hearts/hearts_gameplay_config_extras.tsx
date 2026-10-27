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

// The View panel's Hearts section: the shared table and card settings,
// Hearts' own view fields, and Show all cards (analysis only, which is the
// oracle's view).
import type {JSX} from 'react';
import {useIsAnalysisMode} from '../../game/bottom_seat.ts';
import {useViewSetting} from '../shared/view_settings.ts';
import {
  HeartsTableCardFields,
  HeartsViewFields,
} from './hearts_view_controls.tsx';
import './hearts_settings.css';

/** The Show all cards setting's key. */
export const SHOW_ALL_CARDS_KEY = 'hearts.showAllCards';

/** The View panel's Hearts section. */
export function HeartsGameplayConfigExtras(): JSX.Element {
  const isAnalysisMode = useIsAnalysisMode();
  const [showAllCards, setShowAllCards] = useViewSetting<boolean>(
    SHOW_ALL_CARDS_KEY,
    false,
  );

  return (
    <section className="hearts-config-section">
      <h3>Hearts</h3>
      <HeartsTableCardFields />
      <HeartsViewFields />
      <label className="ogs-config-toggle">
        <input
          type="checkbox"
          checked={showAllCards}
          disabled={!isAnalysisMode}
          onChange={event => {
            setShowAllCards(event.target.checked);
          }}
        />
        <span>Show all cards</span>
      </label>
      {!isAnalysisMode && (
        <p className="ogs-config-note">
          Cards can only be revealed in analysis mode.
        </p>
      )}
    </section>
  );
}
