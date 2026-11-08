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
import {useState} from 'react';

import type {
  CatalogueEntry,
  CreateSession,
  Frame,
  ParameterSpec,
} from '../contract/contract.ts';
import type {RequestError} from '../session/errors.ts';
import {setupRefusal} from './access.ts';
import {ParameterField} from './parameter_field.tsx';
import {
  type Draft,
  type ParameterValues,
  choiceLabel,
  isCheckbox,
  formatValue,
  newGame,
  parameterLabel,
  parseDrafts,
  requireDraft,
  requireSameGame,
  sessionDrafts,
  sessionParameters,
  sessionValue,
  setupParameters,
} from './parameters.ts';

import './setup.css';

/** The properties of GameOptions. */
export interface GameOptionsProps {
  /** The catalogue's entry for the session's game. */
  entry: CatalogueEntry;
  frame: Frame;
  busy: boolean;
  /** Why the latest request failed, or null. */
  error: RequestError | null;
  /** Starts the new game the options describe. */
  onCreate: (body: CreateSession) => void;
}

/**
 * The game's parameters: one typed control each for a viewer who may set up
 * the game, and their values for a spectator. A valid change starts a new
 * game with it at once; a change the server refuses is explained under its
 * control.
 *
 * @throws Error when the game offers no parameters: the Setup panel leaves
 *     the section out instead.
 */
export function GameOptions(props: GameOptionsProps): JSX.Element {
  const {entry, frame} = props;
  requireSameGame(entry, frame);
  const specs = setupParameters(entry);
  if (specs.length === 0) {
    throw new Error(`${entry.name} has no options to show.`);
  }
  const readOnly = setupRefusal(frame) !== null;
  return (
    <section className="setup-card" aria-label="Game Options">
      <div className="setup-card__header">
        <h3 className="setup-card__title">Game Options</h3>
        <p className="setup-card__description">
          {readOnly
            ? 'The parameters this game is played with.'
            : 'OpenSpiel game parameters. Changing one starts a new game.'}
        </p>
      </div>
      {readOnly ? (
        <OptionValues
          specs={specs}
          parameters={sessionParameters(entry, frame)}
        />
      ) : (
        // A new session starts the form again from its values.
        <OptionsForm key={frame.session.id} specs={specs} {...props} />
      )}
    </section>
  );
}

function OptionValues({
  specs,
  parameters,
}: {
  specs: readonly ParameterSpec[];
  parameters: ParameterValues;
}): JSX.Element {
  return (
    <dl className="setup-values">
      {specs.map(spec => {
        const value = sessionValue(spec, parameters);
        return (
          <div key={spec.name} className="setup-field">
            <dt className="setup-field__label">{parameterLabel(spec.name)}</dt>
            <dd className="setup-values__value">
              {spec.kind === 'choice'
                ? choiceLabel(spec, value)
                : formatValue(spec, value)}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

// A refused change: the parameter changed and the server's reason.
interface Refusal {
  readonly name: string;
  readonly message: string;
}

function OptionsForm({
  specs,
  entry,
  frame,
  busy,
  error,
  onCreate,
}: GameOptionsProps & {specs: readonly ParameterSpec[]}): JSX.Element {
  const parameters = sessionParameters(entry, frame);
  const [drafts, setDrafts] = useState(() => sessionDrafts(specs, parameters));
  // Why the drafts cannot be sent, by parameter name.
  const [problems, setProblems] = useState<ReadonlyMap<string, string>>(
    new Map(),
  );
  // The parameter whose change was sent last, until its answer arrives.
  const [sent, setSent] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  // A new game remounts the form, so an answer that leaves this one in place
  // with an error is the refusal of the change sent.
  if (sent !== null && !busy) {
    setSent(null);
    if (error !== null) {
      setRefusal({name: sent, message: `Not applied: ${error.message}`});
    }
  }
  const apply = (spec: ParameterSpec, next: Record<string, Draft>): void => {
    const parsed = parseDrafts(specs, next);
    setProblems(parsed.problems);
    const {values} = parsed;
    if (values === null) return;
    if (values[spec.name] === sessionValue(spec, parameters)) {
      return;
    }
    setSent(spec.name);
    onCreate(newGame(entry, frame, values));
  };
  return (
    <form
      className="setup-form"
      noValidate
      onSubmit={event => {
        event.preventDefault();
      }}
    >
      {specs.map(spec => (
        <ParameterField
          key={spec.name}
          spec={spec}
          draft={requireDraft(spec, drafts)}
          invalid={problems.has(spec.name)}
          notice={refusal?.name === spec.name ? refusal.message : null}
          onChange={draft => {
            const next = {...drafts, [spec.name]: draft};
            setDrafts(next);
            setRefusal(null);
            // A typed number or text applies once it is committed, not at
            // every keystroke, which would start a game with "1" of "13".
            if (appliesAtOnce(spec)) apply(spec, next);
            else setProblems(parseDrafts(specs, next).problems);
          }}
          onCommit={() => {
            if (!appliesAtOnce(spec)) apply(spec, drafts);
          }}
        />
      ))}
      {problems.size > 0 && (
        <p className="setup-form__problem" role="alert">
          {`Invalid game options: ${[...problems]
            .map(([name, problem]) => `${name}: ${problem}`)
            .join('; ')}`}
        </p>
      )}
    </form>
  );
}

// Whether a change to `spec` applies as soon as it is made: a checkbox or a
// choice from a list, where every change is a whole value.
function appliesAtOnce(spec: ParameterSpec): boolean {
  return isCheckbox(spec) || spec.kind === 'choice';
}
