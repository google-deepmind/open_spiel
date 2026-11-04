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

import {useState} from 'react';

import type {Controller, SeatConfig, SeatView} from '../contract/contract.ts';
import type {SeatDraft, SettingsErrors} from './seat_draft.ts';
import {
  chooseController,
  isChanged,
  readSeatDraft,
  sameConfig,
  savedConfig,
  savedDraft,
} from './seat_draft.ts';

/** A seat's settings as edited, and what can be done with them. */
export interface SeatDraftState {
  readonly draft: SeatDraft;
  /** Whether the draft differs from what the server holds. */
  readonly changed: boolean;
  /** Why the draft cannot be saved, once a save has been tried; or null. */
  readonly errors: SettingsErrors | null;
  readonly edit: (draft: SeatDraft) => void;
  readonly chooseController: (name: string) => void;
  /** Sends the draft, or shows why it cannot be sent. */
  readonly save: () => void;
  /** Goes back to what the server holds. */
  readonly reset: () => void;
}

// An edit in progress, and the server's settings it started from.
interface Edit {
  readonly base: SeatConfig;
  // The game and move it was made at.
  readonly moment: string;
  readonly draft: SeatDraft;
  // Whether a save was tried, after which the draft's errors are shown.
  readonly tried: boolean;
}

/**
 * Holds a draft of `seat`'s settings until it is saved through `onSave`.
 * The draft is dropped as soon as the server's settings for the seat change,
 * as they do when a save succeeds or another client changes the seat, and
 * as soon as `moment` changes, as it does with each move and each new game:
 * an unsaved draft does not outlive the position it was made in. A refused
 * save keeps it.
 */
export function useSeatDraft(
  seat: SeatView,
  controllers: readonly Controller[],
  moment: string,
  onSave: (config: SeatConfig) => void,
): SeatDraftState {
  const [edit, setEdit] = useState<Edit | null>(null);
  const stale =
    edit !== null && (edit.moment !== moment || !sameConfig(edit.base, seat));
  // Dropped during the render that finds it stale, so it is never shown and
  // cannot come back should the server return to the settings it edited.
  if (stale) setEdit(null);
  const current = stale ? null : edit;
  const draft = current?.draft ?? savedDraft(seat, controllers);
  const reading = readSeatDraft(draft);
  const tried = current?.tried ?? false;
  const keep = (next: SeatDraft, triedNext: boolean): void => {
    setEdit({base: seat, moment, draft: next, tried: triedNext});
  };
  return {
    draft,
    changed: isChanged(reading, savedConfig(seat, controllers)),
    errors: tried && !reading.ok ? reading.errors : null,
    edit: next => {
      keep(next, tried);
    },
    chooseController: name => {
      keep(chooseController(draft, name, seat, controllers), tried);
    },
    save: () => {
      if (reading.ok) onSave(reading.config);
      else keep(draft, true);
    },
    reset: () => {
      setEdit(null);
    },
  };
}
