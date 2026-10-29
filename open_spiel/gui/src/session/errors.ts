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

import type {Error as ErrorBody, ErrorCode} from '../contract/contract.ts';

/** A request the server refused, with the contract's error body. */
export class ContractError extends Error {
  override readonly name = 'ContractError';
  readonly code: ErrorCode;
  /** The seat the refusal concerns, or null. */
  readonly seat: number | null;
  /** The history index the refusal concerns, or null. */
  readonly ply: number | null;
  /** The HTTP status the refusal came with. */
  readonly status: number;

  constructor(body: ErrorBody, status: number) {
    super(body.message);
    this.code = body.code;
    this.seat = body.seat;
    this.ply = body.ply;
    this.status = status;
  }
}

/**
 * A request that got no answer in the contract: the server was unreachable,
 * failed, or sent a body the contract does not define.
 */
export class TransportError extends Error {
  override readonly name = 'TransportError';
  /** The HTTP status, or null when no response arrived. */
  readonly status: number | null;

  constructor(message: string, status: number | null, options?: ErrorOptions) {
    super(message, options);
    this.status = status;
  }
}

/** Why a request failed. */
export type RequestError = ContractError | TransportError;
