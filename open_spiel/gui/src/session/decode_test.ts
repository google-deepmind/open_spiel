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

import {decode} from './decode.ts';
import {ContractError, TransportError} from './errors.ts';

function response(
  body: string,
  status = 200,
  type = 'application/json',
): Response {
  return new Response(body, {status, headers: {'Content-Type': type}});
}

const HEALTH = JSON.stringify({
  contract_version: 8,
  version: '1.0',
  revision: null,
  capabilities: {
    controllers: [],
    chance_policies: [],
    cpp_mcts_limits: 'simulations_only',
  },
});

describe('decode bodies', () => {
  it('returns a body of the kind expected', async () => {
    const health = await decode(response(HEALTH), 'health');
    expect(health.version).toBe('1.0');
  });

  it('accepts a JSON content type with parameters', async () => {
    const body = response(HEALTH, 200, 'Application/JSON; charset=utf-8');
    await expect(decode(body, 'health')).resolves.toBeDefined();
  });

  it('turns an error body into a ContractError with its code', async () => {
    const body = JSON.stringify({
      contract_version: 8,
      code: 'illegal_action',
      message: 'Action 9 is not legal.',
      seat: 1,
      ply: 4,
    });
    const error = await decode(response(body, 409), 'frame').catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ContractError);
    expect(error).toMatchObject({
      name: 'ContractError',
      code: 'illegal_action',
      message: 'Action 9 is not legal.',
      seat: 1,
      ply: 4,
      status: 409,
    });
  });

  it('refuses a body of another kind', async () => {
    await expect(decode(response(HEALTH), 'frame')).rejects.toThrow(
      'The server answered 200 without a Frame body.',
    );
  });
});

describe('decode failures', () => {
  it('refuses another contract version', async () => {
    const body = HEALTH.replace('"contract_version":8', '"contract_version":6');
    await expect(decode(response(body), 'health')).rejects.toThrow(
      'does not speak contract version 8',
    );
  });

  it('refuses a JSON value that is not an object', async () => {
    await expect(decode(response('[1]'), 'health')).rejects.toThrow(
      'does not speak contract version 8',
    );
  });

  it('refuses a failure status without an error body', async () => {
    await expect(decode(response(HEALTH, 500), 'health')).rejects.toThrow(
      'The server answered 500 without an error body.',
    );
  });

  it('refuses an error body outside the contract', async () => {
    const body = JSON.stringify({
      contract_version: 8,
      code: 'teapot',
      message: 'Refused.',
      seat: null,
      ply: null,
    });
    const error = await decode(response(body, 409), 'frame').catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(TransportError);
    expect(error).toMatchObject({
      message: 'The server answered 409 without an error body.',
    });
  });

  it('reports a response without JSON with its status and text', async () => {
    const body = response('The server failed.\n', 500, 'text/plain');
    const error = await decode(body, 'frame').catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TransportError);
    expect(error).toMatchObject({
      message: 'The server answered 500 without JSON: The server failed.',
      status: 500,
    });
  });

  it('quotes at most the start of a long response without JSON', async () => {
    const page = `<html>${'x'.repeat(1000)}</html>`;
    const error = await decode(response(page, 502, 'text/html'), 'frame').catch(
      (e: unknown) => e,
    );
    expect(error).toMatchObject({
      message: `The server answered 502 without JSON: ${page.slice(0, 200)}…`,
    });
  });

  it('reports an answer cut off before its end', async () => {
    const cause = new TypeError('terminated');
    const body = response(HEALTH);
    body.text = () => Promise.reject(cause);
    const error = await decode(body, 'health').catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TransportError);
    expect(error).toMatchObject({
      message: "The server's 200 answer was cut off.",
      status: 200,
      cause,
    });
  });

  it('reports malformed JSON', async () => {
    await expect(decode(response('{'), 'frame')).rejects.toThrow(
      'malformed JSON',
    );
  });
});
