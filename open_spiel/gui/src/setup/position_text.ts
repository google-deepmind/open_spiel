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

import type {Frame} from '../contract/contract.ts';

/** Typed text read as a history, or why it cannot be read as one. */
export type ParsedHistory =
  | {readonly ok: true; readonly history: readonly number[]}
  | {readonly ok: false; readonly problem: string};

/**
 * Reads typed action ids as a history: ids separated by spaces or commas,
 * optionally inside square brackets, so a copied history reads back. Empty
 * brackets are the empty history. A JSON object whose `history`, `actions`
 * or `sequence` is a list of ids, as an exported episode holds them, reads
 * as that list.
 */
export function parseHistory(text: string): ParsedHistory {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) return parseEpisode(trimmed);
  const bracketed = trimmed.startsWith('[');
  if (bracketed !== trimmed.endsWith(']')) {
    return {ok: false, problem: 'Close the brackets around the actions.'};
  }
  const inner = bracketed ? trimmed.slice(1, -1) : trimmed;
  const tokens = inner.split(/[\s,]+/).filter(token => token !== '');
  if (tokens.length === 0) {
    return bracketed && inner.trim() === ''
      ? {ok: true, history: []}
      : {ok: false, problem: 'Enter the action ids to replay.'};
  }
  const history: number[] = [];
  for (const token of tokens) {
    const action = Number(token);
    if (!/^\d+$/.test(token) || !Number.isSafeInteger(action)) {
      return {ok: false, problem: `${token} is not an action id.`};
    }
    history.push(action);
  }
  return {ok: true, history};
}

// The keys under which an episode's object may hold its action ids.
const EPISODE_KEYS = ['history', 'actions', 'sequence'] as const;

function parseEpisode(text: string): ParsedHistory {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return {ok: false, problem: 'The object is not valid JSON.'};
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return {ok: false, problem: 'The object is not valid JSON.'};
  }
  const record = parsed as Record<string, unknown>;
  const key = EPISODE_KEYS.find(name => name in record);
  const list = key === undefined ? undefined : record[key];
  if (!Array.isArray(list)) {
    return {
      ok: false,
      problem: 'The object holds no history, actions or sequence list.',
    };
  }
  const history: number[] = [];
  for (const action of list) {
    if (
      typeof action !== 'number' ||
      !Number.isSafeInteger(action) ||
      action < 0
    ) {
      return {
        ok: false,
        problem: `${JSON.stringify(action)} is not an action id.`,
      };
    }
    history.push(action);
  }
  return {ok: true, history};
}

/** `history` as text that `parseHistory` reads back. */
export function formatHistory(history: readonly number[]): string {
  return `[${history.join(', ')}]`;
}

/**
 * The action ids of `frame`'s history, which an omniscient viewer is shown
 * in full.
 *
 * @throws Error when the history withholds an action.
 */
export function historyActions(frame: Frame): number[] {
  return frame.history.map(({action}, ply) => {
    if (action === null) {
      throw new Error(`The history withholds ply ${String(ply)}.`);
    }
    return action;
  });
}
