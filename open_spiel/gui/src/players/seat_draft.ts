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

// A seat's settings as they are being edited, and how they compare with the
// settings the server holds.

import type {Controller, SeatConfig, SeatView} from '../contract/contract.ts';
import type {Field, FieldDraft, FieldValue} from '../form/field.ts';
import {draftOf, readDraft} from '../form/field.ts';
import {fieldsOf} from '../form/json_schema.ts';

/** The controller that a person plays. */
export const HUMAN = 'human';

/** A controller's settings as edited: in a form, or as JSON text. */
export type SettingsDraft =
  | {
      readonly kind: 'form';
      readonly fields: readonly Field[];
      readonly drafts: Readonly<Record<string, FieldDraft>>;
    }
  | {readonly kind: 'json'; readonly text: string};

/** Who plays a seat, as edited. */
export interface SeatDraft {
  readonly controller: string;
  readonly settings: SettingsDraft;
  readonly autoplay: boolean;
}

/** Why a draft's settings cannot be sent, in the shape of its editor. */
export type SettingsErrors =
  | {readonly kind: 'form'; readonly errors: Readonly<Record<string, string>>}
  | {readonly kind: 'json'; readonly error: string};

/** A draft read as a seat's configuration, or why it cannot be. */
export type DraftReading =
  | {readonly ok: true; readonly config: SeatConfig}
  | {readonly ok: false; readonly errors: SettingsErrors};

type Config = Readonly<Record<string, FieldValue>>;

/**
 * Returns the controller named `name`.
 *
 * @throws Error when `controllers` has none: the contract lists every
 *     seat's controller among the session's.
 */
export function controllerNamed(
  controllers: readonly Controller[],
  name: string,
): Controller {
  const controller = controllers.find(candidate => candidate.name === name);
  if (controller === undefined) {
    throw new Error(`The session lists no controller ${name}.`);
  }
  return controller;
}

/**
 * Returns what the server holds for `seat`, each setting the seat was given
 * without in its default, which is what the controller plays with. Settings
 * a form shows come in the form's order.
 */
export function savedConfig(
  seat: SeatView,
  controllers: readonly Controller[],
): SeatConfig {
  const fields = fieldsOf(
    controllerNamed(controllers, seat.controller).config_schema,
  );
  const defaults = Object.fromEntries(
    (fields ?? []).map(field => [field.name, field.default]),
  );
  return {
    controller: seat.controller,
    config: {...defaults, ...seat.config},
    autoplay: seat.autoplay,
  };
}

/** Returns the draft of the settings the server holds for `seat`. */
export function savedDraft(
  seat: SeatView,
  controllers: readonly Controller[],
): SeatDraft {
  const controller = controllerNamed(controllers, seat.controller);
  return {
    controller: seat.controller,
    settings: settingsDraft(controller, seat.config),
    autoplay: seat.autoplay,
  };
}

/**
 * Returns `draft` with `name` chosen to play the seat. Choosing the saved
 * controller again brings back the saved settings; any other starts from its
 * defaults. A human never autoplays, and a controller chosen in a human's
 * place starts out autoplaying, since otherwise it waits to be advanced.
 */
export function chooseController(
  draft: SeatDraft,
  name: string,
  seat: SeatView,
  controllers: readonly Controller[],
): SeatDraft {
  if (name === seat.controller) return savedDraft(seat, controllers);
  return {
    controller: name,
    settings: settingsDraft(controllerNamed(controllers, name), {}),
    autoplay: name !== HUMAN && (draft.controller === HUMAN || draft.autoplay),
  };
}

/** Returns `draft` with the form field `name` set to `value`. */
export function editField(
  draft: SeatDraft,
  name: string,
  value: FieldDraft,
): SeatDraft {
  if (draft.settings.kind !== 'form') {
    throw new Error('Settings edited as JSON have no fields.');
  }
  const drafts = {...draft.settings.drafts, [name]: value};
  return {...draft, settings: {...draft.settings, drafts}};
}

/** Returns `draft` with its JSON settings replaced by `text`. */
export function editJson(draft: SeatDraft, text: string): SeatDraft {
  if (draft.settings.kind !== 'json') {
    throw new Error('Settings edited in a form have no JSON text.');
  }
  return {...draft, settings: {kind: 'json', text}};
}

/** Reads `draft` as the configuration to send for its seat. */
export function readSeatDraft(draft: SeatDraft): DraftReading {
  const settings = readSettings(draft.settings);
  if (!settings.ok) return settings;
  return {
    ok: true,
    config: {
      controller: draft.controller,
      config: settings.config,
      autoplay: draft.controller !== HUMAN && draft.autoplay,
    },
  };
}

/**
 * Whether `reading` would change `saved`, the seat's settings as
 * `savedConfig` gives them. Settings compare by value, whatever the order of
 * their keys; a draft that cannot be read is a change.
 */
export function isChanged(reading: DraftReading, saved: SeatConfig): boolean {
  return !reading.ok || !sameConfig(reading.config, saved);
}

/** Whether two seat configurations are the same, settings compared by value. */
export function sameConfig(first: SeatConfig, second: SeatConfig): boolean {
  return (
    first.controller === second.controller &&
    first.autoplay === second.autoplay &&
    sameSettings(first.config, second.config)
  );
}

function sameSettings(first: Config, second: Config): boolean {
  const names = Object.keys(first);
  return (
    names.length === Object.keys(second).length &&
    names.every(
      name => Object.hasOwn(second, name) && first[name] === second[name],
    )
  );
}

function settingsDraft(controller: Controller, config: Config): SettingsDraft {
  const fields = fieldsOf(controller.config_schema);
  if (fields === null) {
    return {kind: 'json', text: JSON.stringify(config, null, 2)};
  }
  const drafts = Object.fromEntries(
    fields.map(field => [field.name, draftOf(field, config[field.name])]),
  );
  return {kind: 'form', fields, drafts};
}

type SettingsReading =
  | {readonly ok: true; readonly config: Config}
  | {readonly ok: false; readonly errors: SettingsErrors};

function readSettings(settings: SettingsDraft): SettingsReading {
  if (settings.kind === 'json') return readJson(settings.text);
  const config: Record<string, FieldValue> = {};
  const errors: Record<string, string> = {};
  for (const field of settings.fields) {
    const draft = settings.drafts[field.name];
    if (draft === undefined) {
      throw new Error(`The field ${field.name} has no draft.`);
    }
    const reading = readDraft(field, draft);
    if (reading.ok) config[field.name] = reading.value;
    else errors[field.name] = reading.error;
  }
  return Object.keys(errors).length === 0
    ? {ok: true, config}
    : {ok: false, errors: {kind: 'form', errors}};
}

// Reads the JSON box. An empty box holds no settings, as `{}` does.
function readJson(text: string): SettingsReading {
  const refuse = (error: string): SettingsReading => ({
    ok: false,
    errors: {kind: 'json', error},
  });
  let parsed: unknown;
  try {
    parsed = text.trim() === '' ? {} : JSON.parse(text);
  } catch (e) {
    if (!(e instanceof SyntaxError)) throw e;
    return refuse(`The settings are not JSON: ${e.message}`);
  }
  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    Array.isArray(parsed) ||
    !Object.values(parsed).every(isFieldValue)
  ) {
    return refuse(
      'The settings are a JSON object of booleans, finite numbers and strings.',
    );
  }
  return {ok: true, config: parsed as Config};
}

// JSON.parse reads a number too large for a double, such as 1e400, as
// Infinity, which JSON.stringify would send as null.
function isFieldValue(value: unknown): value is FieldValue {
  return (
    typeof value === 'boolean' ||
    typeof value === 'string' ||
    (typeof value === 'number' && Number.isFinite(value))
  );
}
