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

import {chessSession, chessViewer} from './chess_session.ts';
import {
  AFTER_E4_D5,
  chessFrame,
  chessHistory,
  chessPresentation,
} from './chess_test_fixtures.ts';

describe('chessSession', () => {
  it("reads the position, the moves and the seats under the board's names", () => {
    const shown = chessFrame();
    const session = chessSession(shown, chessPresentation());
    expect(session.mode).toBe('analysis');
    expect(session.state.text).toBe(AFTER_E4_D5);
    expect(session.state.currentPlayer).toBe(1);
    expect(session.state.turnNumber).toBe(2);
    expect(session.state.historyActionStrings).toEqual(['e4', 'd5']);
    expect(session.state.lastMove).toEqual({from: 'd7', to: 'd5'});
    expect(session.state.check).toBeNull();
    expect(session.state.legalActions[0]).toEqual({action: 10, label: 'exd5'});
    expect(session.state.moveHints[2]).toEqual({
      action: 12,
      from: 'g1',
      to: 'f3',
      promotion: null,
    });
    expect(session.players.map(player => player.controller)).toEqual([
      'human',
      'human',
    ]);
    expect(session.availableControllers[1]).toEqual({
      id: 'random',
      label: 'Uniform random',
    });
  });

  it("reads a promotion's piece", () => {
    const presentation = chessPresentation({}, [
      {id: 5, from: 'b7', to: 'b8', promotion: 'q'},
    ]);
    expect(
      chessSession(chessFrame({}, presentation), presentation).state.moveHints,
    ).toEqual([{action: 5, from: 'b7', to: 'b8', promotion: 'q'}]);
  });

  it('lists the moves up to the position shown on a replay', () => {
    const shown = chessFrame({
      history: chessHistory(['e4', 'd5', 'exd5']),
      ply: 2,
    });
    expect(
      chessSession(shown, chessPresentation()).state.historyActionStrings,
    ).toEqual(['e4', 'd5']);
  });

  it("refuses a view without the board's fields", () => {
    const presentation = chessPresentation();
    const broken = {...presentation, view: {fen: AFTER_E4_D5}};
    expect(() => chessSession(chessFrame({}, broken), broken)).toThrow(
      'view.chessCheck is missing',
    );
  });

  it('names a seat viewer and whether it is to move', () => {
    const shown = chessFrame({viewer: {kind: 'seat', seat: 1}});
    expect(chessViewer(shown)).toEqual({
      seat: 1,
      isParticipant: true,
      isActiveSeat: true,
    });
    expect(chessSession(shown, chessPresentation()).mode).toBe('game');
    expect(
      chessViewer(chessFrame({viewer: {kind: 'spectator', seat: null}})),
    ).toEqual({
      seat: null,
      isParticipant: false,
      isActiveSeat: false,
    });
  });
});
