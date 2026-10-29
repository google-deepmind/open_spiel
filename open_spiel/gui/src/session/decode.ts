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

import type {
  Catalogue,
  Error as ErrorBody,
  ErrorCode,
  Frame,
  Health,
} from '../contract/contract.ts';
import {ContractError, TransportError} from './errors.ts';

/** The contract version this client speaks. */
export const CONTRACT_VERSION: Frame['contract_version'] = 8;

/** The success bodies, by the name a route's caller expects. */
export interface Bodies {
  frame: Frame;
  catalogue: Catalogue;
  health: Health;
}

// Each body's contract type, and a field only that body has, so a route
// answering with the wrong kind of body fails here rather than deep in a
// component.
const BODY = {
  frame: {type: 'Frame', field: 'session'},
  catalogue: {type: 'Catalogue', field: 'games'},
  health: {type: 'Health', field: 'capabilities'},
} as const satisfies Record<keyof Bodies, {type: string; field: string}>;

const ERROR_CODES = {
  bad_request: true,
  not_found: true,
  method_not_allowed: true,
  game_unavailable: true,
  viewer_unavailable: true,
  not_your_turn: true,
  illegal_action: true,
  controller_unavailable: true,
  controller_failed: true,
  nothing_to_undo: true,
} as const satisfies Record<ErrorCode, true>;

// The most of a body that is not JSON a message quotes, since it may be a
// whole error page.
const EXCERPT_LENGTH = 200;

/**
 * Decodes a response as the body `kind`.
 *
 * @throws ContractError when the server refused the request.
 * @throws TransportError when the response is not a body of the contract.
 */
export async function decode<K extends keyof Bodies>(
  response: Response,
  kind: K,
): Promise<Bodies[K]> {
  const status = String(response.status);
  let text: string;
  try {
    text = await response.text();
  } catch (e) {
    throw new TransportError(
      `The server's ${status} answer was cut off.`,
      response.status,
      {cause: e},
    );
  }
  if (!isJson(response)) {
    throw new TransportError(
      `The server answered ${status} without JSON: ${excerpt(text)}`,
      response.status,
    );
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch (e) {
    throw new TransportError(
      'The server sent malformed JSON.',
      response.status,
      {
        cause: e,
      },
    );
  }
  if (!isObject(body) || body['contract_version'] !== CONTRACT_VERSION) {
    throw new TransportError(
      `The server does not speak contract version ${String(CONTRACT_VERSION)}.`,
      response.status,
    );
  }
  if (!response.ok) {
    if (isErrorBody(body)) throw new ContractError(body, response.status);
    throw new TransportError(
      `The server answered ${status} without an error body.`,
      response.status,
    );
  }
  const expected = BODY[kind];
  if (!(expected.field in body)) {
    throw new TransportError(
      `The server answered ${status} without a ${expected.type} body.`,
      response.status,
    );
  }
  return body as unknown as Bodies[K];
}

function isJson(response: Response): boolean {
  const type = response.headers.get('Content-Type') ?? '';
  return type.split(';')[0]?.trim().toLowerCase() === 'application/json';
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isErrorBody(
  body: Record<string, unknown>,
): body is ErrorBody & Record<string, unknown> {
  const {code, message, seat, ply} = body;
  return (
    typeof code === 'string' &&
    Object.hasOwn(ERROR_CODES, code) &&
    typeof message === 'string' &&
    isIndexOrNull(seat) &&
    isIndexOrNull(ply)
  );
}

function isIndexOrNull(value: unknown): boolean {
  return (
    value === null ||
    (typeof value === 'number' && Number.isInteger(value) && value >= 0)
  );
}

function excerpt(text: string): string {
  const trimmed = text.trim();
  return trimmed.length <= EXCERPT_LENGTH
    ? trimmed
    : `${trimmed.slice(0, EXCERPT_LENGTH)}…`;
}
