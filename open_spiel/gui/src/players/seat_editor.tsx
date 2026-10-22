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

import type {JSX} from 'react';
import {useId} from 'react';

import type {Controller} from '../contract/contract.ts';
import {FieldForm} from '../form/field_form.tsx';
import type {SeatDraft, SettingsErrors} from './seat_draft.ts';
import {HUMAN, controllerNamed, editField, editJson} from './seat_draft.ts';
import type {SeatDraftState} from './use_seat_draft.ts';

/** The properties of a SeatEditor. */
export interface SeatEditorProps {
  /** The seat's name, such as "Player 0". */
  readonly name: string;
  readonly controllers: readonly Controller[];
  readonly state: SeatDraftState;
  /** Whether a request is in flight, during which nothing is saved. */
  readonly busy: boolean;
}

/**
 * The controls that choose who plays a seat and with what settings, between
 * two Save and Reset bars: one above the controller, saying whether there
 * are unsaved changes, and one after the last setting. One of the two sticks
 * to the scrolling seat list, so a tall form never scrolls them away.
 */
export function SeatEditor({
  name,
  controllers,
  state,
  busy,
}: SeatEditorProps): JSX.Element {
  const {draft, edit} = state;
  const id = useId();
  const chosen = controllerNamed(controllers, draft.controller);
  return (
    <div className="seat-editor">
      <SaveBar placement="top" name={name} state={state} busy={busy} />
      <div className="seat-editor__field">
        <label
          className="seat-editor__field-label"
          htmlFor={`${id}-controller`}
        >
          Controller
        </label>
        <div className="seat-editor__controller-row">
          <select
            id={`${id}-controller`}
            className="control"
            value={draft.controller}
            disabled={busy}
            onChange={event => {
              state.chooseController(event.currentTarget.value);
            }}
          >
            {controllers.map(option => (
              <option
                key={option.name}
                value={option.name}
                title={option.unavailable_reason ?? undefined}
              >
                {option.unavailable_reason === null
                  ? option.label
                  : `${option.label} — not available here`}
              </option>
            ))}
          </select>
          {draft.controller !== HUMAN && (
            <label
              className="seat-editor__autoplay"
              title="An autoplaying seat moves as soon as it is its turn; any other waits to be stepped."
            >
              <input
                type="checkbox"
                checked={draft.autoplay}
                disabled={busy}
                onChange={event => {
                  edit({...draft, autoplay: event.currentTarget.checked});
                }}
              />
              Autoplay
            </label>
          )}
        </div>
        {chosen.unavailable_reason !== null && (
          <p className="seat-editor__unavailable" role="note">
            {chosen.unavailable_reason}
          </p>
        )}
      </div>
      <SettingsEditor
        id={`${id}-settings`}
        draft={draft}
        errors={state.errors}
        onChange={edit}
      />
      <SaveBar placement="bottom" name={name} state={state} busy={busy} />
    </div>
  );
}

interface SettingsEditorProps {
  readonly id: string;
  readonly draft: SeatDraft;
  readonly errors: SettingsErrors | null;
  readonly onChange: (draft: SeatDraft) => void;
}

// A form for settings the schema describes, nothing when there are none,
// and a JSON box for any other.
function SettingsEditor({
  id,
  draft,
  errors,
  onChange,
}: SettingsEditorProps): JSX.Element | null {
  const {settings} = draft;
  if (settings.kind === 'form') {
    if (settings.fields.length === 0) return null;
    return (
      <FieldForm
        fields={settings.fields}
        drafts={settings.drafts}
        errors={errors?.kind === 'form' ? errors.errors : {}}
        onChange={(name, value) => {
          onChange(editField(draft, name, value));
        }}
      />
    );
  }
  const error = errors?.kind === 'json' ? errors.error : null;
  return (
    <div className="seat-editor__json">
      <label htmlFor={id}>Configuration (JSON)</label>
      <textarea
        id={id}
        className="control"
        value={settings.text}
        spellCheck={false}
        aria-invalid={error !== null}
        aria-describedby={error === null ? undefined : `${id}-error`}
        onChange={event => {
          onChange(editJson(draft, event.currentTarget.value));
        }}
      />
      {error !== null && (
        <p id={`${id}-error`} className="seat-editor__error">
          {error}
        </p>
      )}
    </div>
  );
}

interface SaveBarProps {
  /** The top bar says whether anything is unsaved; the bottom one does not. */
  readonly placement: 'top' | 'bottom';
  readonly name: string;
  readonly state: SeatDraftState;
  readonly busy: boolean;
}

// Saves or discards the draft. Both buttons wait only for a request in
// flight; a dot on Save marks unsaved changes.
function SaveBar({placement, name, state, busy}: SaveBarProps): JSX.Element {
  const {changed} = state;
  const top = placement === 'top';
  const dot = changed && (
    <span className="seat-editor__unsaved-dot" aria-hidden="true" />
  );
  return (
    <div
      className={`seat-editor__save-bar seat-editor__save-bar--${placement}`}
    >
      {top && (
        <span
          className="seat-editor__state"
          data-changed={changed}
          role="status"
        >
          {changed ? 'Unsaved changes' : 'No changes'}
        </span>
      )}
      <button
        type="button"
        className={top ? 'button button--compact' : 'button'}
        disabled={busy}
        aria-label={top ? `Save settings for ${name}` : undefined}
        title={changed ? 'Save unsaved changes' : 'Save settings'}
        onClick={state.save}
      >
        {top ? 'Save' : 'Save settings'}
        {dot}
      </button>
      <button
        type="button"
        className={
          top ? 'button button--ghost button--compact' : 'button button--ghost'
        }
        disabled={busy}
        aria-label={top ? `Reset changes for ${name}` : undefined}
        onClick={state.reset}
      >
        {top ? 'Reset' : 'Reset changes'}
      </button>
    </div>
  );
}
