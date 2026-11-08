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

// Frames recorded from the server, in frames.json, keyed
// "<game>/<viewer>/<ply>": the viewer is oracle, spectator or a seat index,
// and the ply is a number, "live", or "new" for a session just created.
//
// Each session was created with every seat human, chance_policy "auto" and
// seed 1. kuhn_poker deals two cards, then the seats play Pass, Bet, Bet to
// the end; tic_tac_toe plays 4, 0, 8. Session ids are replaced by readable
// names.

/// <reference types="node" />

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

import type {Frame} from '../../contract/contract.ts';
import {decode} from '../../session/decode.ts';

const RECORDED = JSON.parse(
  readFileSync(resolve(import.meta.dirname, 'frames.json'), 'utf8'),
) as Record<string, unknown>;

/**
 * The frame recorded under `name`, decoded as the client decodes a response,
 * so that a recording another contract version made is refused.
 */
export function recordedFrame(name: string): Promise<Frame> {
  if (!(name in RECORDED)) throw new Error(`No frame is recorded as ${name}.`);
  const response = new Response(JSON.stringify(RECORDED[name]), {
    headers: {'Content-Type': 'application/json'},
  });
  return decode(response, 'frame');
}
