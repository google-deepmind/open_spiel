# Drawing a game: server adapters and client renderers

A game is drawn from two self-contained parts:

- a **server adapter** (`open_spiel/python/gui/adapters/<game>_adapter.py`)
  that turns a state into a JSON view for one viewer, and hints which legal
  action each thing on the board plays; and
- a **client renderer** that draws that view and turns a press into an
  action id.

The frame carries the adapter's output as `presentation` (CONTRACT.md). A
game without an adapter has `presentation: null`, and the board shows the
game's text. Neither part touches any other game, so either can later be
replaced, for example by a renderer that reads the game's own structs,
without changing anything else.

## 1. The server adapter

Add one module, `adapters/<game>_adapter.py`. It defines `GAME`, the game's
short name, and `ADAPTER`, an instance of an `adapter.Adapter` subclass, and
is listed with one line in `adapters/__init__.py`, which `presenters`
registers. The module starts, after the licence header, with a one-line
comment naming what will replace it:

```python
# Replaced by the <game> ObservationStruct when available.
"""The <game> adapter: ..."""

from collections.abc import Mapping, Sequence

import pyspiel

from open_spiel.python.gui import adapter
from open_spiel.python.gui import contract

GAME = "<game>"


class MyGameAdapter(adapter.Adapter):
  renderer = "cards"  # A shared renderer, or the game's own.

  def present(
      self, state: pyspiel.State, viewpoint: adapter.Viewpoint
  ) -> dict[str, contract.JsonValue]:
    ...

  def describe_actions(
      self,
      state: pyspiel.State,
      actions: Sequence[int],
      viewpoint: adapter.Viewpoint,
  ) -> Mapping[int, contract.JsonValue]:
    ...


ADAPTER = MyGameAdapter()
```

- **`present`** returns a JSON object. It runs on the server for one viewer,
  so it filters: `viewpoint.sees_private(seat)` is true for the oracle, for
  every viewer of a game that hides nothing, and for the seat itself. A
  card another seat holds is sent as `null`, never as its code, unless the
  rules have made it public (a gin rummy knock lays the hands open, so every
  viewer may see them then).
- **`describe_actions`** is optional. It is called only when the viewer is
  given the legal actions (`actions`, the acting seat's), and returns a hint
  for each action the renderer can place. An action without a hint is still
  offered in Actions › Legal.
- Read the state however is quickest: its text, its observation or
  information-state strings, `to_struct`, the game's Python attributes.
  Raise on anything unexpected; a broken adapter fails the frame loudly.
- Test it beside the module (`adapters/<game>_adapter_test.py`): the view
  for each kind of viewer, and that no seat sees another's hidden cards.

## 2. The client renderer

A game whose view fits a shared renderer needs no client code: set
`renderer = "grid"` or `renderer = "cards"` and send the shapes below.
Otherwise add `src/games/<game>/<game>_renderer.tsx` exporting a component
of type `Renderer` (`src/renderers/renderer.ts`), and register it with one
line in `src/renderers/registry.ts`:

```ts
{name: '<game>', render: MyGameRenderer},
```

A renderer receives `{frame, presentation, play, playSequence, busy}`.
`play(actionId)` plays an action; it is `null` when the viewer cannot play at
this position (another seat's turn, a spectator, a replay, a finished game),
so draw nothing pressable then. `playSequence(actionIds)` plays several
actions in order as one move and resolves with the frame after the last (a
gin lay-down: a discard, melds, then pass); it is null whenever `play` is.
`busy` is true while a request is in flight.

A game page opens in analysis: the frame is the oracle's, every hand
included, and View › Perspective chooses the seat to draw at the bottom
without a request. Until one is chosen it names the viewer's seat, else
Seat 0. A board that rotates to a seat reads
`useBottomSeat(session, viewer)` (`src/game/bottom_seat.ts`): the seat
chosen, else the viewer's, else 0. A board that should change only on an
explicit choice reads `useAnalysisViewSeat()`, null until a seat is chosen;
chess does, so a fresh chess page rests on
Seat 0 drawn from White. A board section that reveals hidden cards
enables its toggle only when `useIsAnalysisMode()` is true. In a game (a seat
or a spectator), the frame holds only what that viewer may see.

A game drawn by a renderer is badged Board UI in the catalogue: the server
names the adapter's renderer in `CatalogueEntry.renderer`, and the
catalogue checks the client has it.

A registry entry may also set:

- `keepsState: true`: the renderer is not drawn afresh at each ply, so its
  component state (a hand's order, a move being composed, cards in flight)
  carries from one position to the next;
- `drawsStatus: true`: the board says whose turn it is on its own table, so
  the Gameplay panel shows no status line above it;
- `viewSections`: the board's own settings, shown as sections of the View
  panel while that board draws the game;
- `History`: the board's own list in Actions › History, in place of the
  rows (chess pairs its moves a turn to a row); it is given the live
  game's actions and labels, the position shown and `goToStep`
  (`src/actions/history_renderer.ts`).

Read `presentation.view` with the checked readers in
`src/renderers/struct_reader.ts` (`field`, `optionalField`, `array`,
`string`, ...), so a view of the wrong shape throws a `StructError` naming
the field. A renderer that throws is shown as an error over the game's
text; the game stays playable from Actions.

Build from the shared parts rather than drawing from scratch:

| Part                            | Where                              | What                                                                                          |
| ------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `Lines`                         | `renderers/lines.tsx`              | the view's optional `lines`, short texts shown as sent                                        |
| `readHints`, `actionLabels`     | `renderers/hints.ts`               | the hints a renderer understands; the legal labels by id                                      |
| `parseCard`, `Card`             | `cards/card.ts`                    | card codes: `"As"`, `"Td"`, `"10h"`; `null` is face down                                      |
| `CardFace`                      | `cards/card_face.tsx`              | one card in CSS, a button when given `onSelect`                                               |
| `Hand`                          | `cards/hand.tsx`                   | a labelled row of cards; playable cards play their action, or call `onCard` to gather several |
| `Pile`, `Trick`                 | `cards/pile.tsx`                   | a pile by its top card and size (`topAction` to draw); a trick's cards under their players    |
| `cardActions`                   | `cards/card_hints.ts`              | the action each `{"card": code}` hint plays                                                   |
| skins                           | `cards/skins.ts`                   | `registerSkin`, `skinFor`: glyphs and suit colours (`standard`, `four_colour`)                |
| `GridBoard`                     | `grid/grid_board.tsx`              | a board of cells: highlights, claim, drop and select-then-move                                |
| `readGridView`, `readGridMoves` | `grid/grid_view.ts`                | the grid view and its move hints                                                              |
| grid piece sets                 | `grid/piece_sets.ts`               | `registerPieceSet`, `pieceSetFor`: `chess`, `checkers`, `marks`, `stones`                     |
| chess piece sets                | `games/chess/pieces/piece_sets.ts` | `registerPieceSet(id, module)`, `getPieceSet`: `staunton`; `cburnett` once fetched            |

The client ships no images: cards and pieces are CSS, Unicode glyphs and
SVG drawn in code, chosen through the skin, deck and piece-set registries so
that art can be added later by registering a skin, a deck
(`cards/library/registry.ts`, `registerDeck`) or a piece set. Colours come only from
`src/styles/tokens.css` (the card and board tokens). Art the client does not
ship is fetched by the user ("Chess piece sets", below).

### Chess piece sets

The chess board, the players' strips and the promotion choice draw their
pieces with the set the viewer picks in View › Board › Pieces. The default,
`staunton`, is drawn in code for this client
(`games/chess/pieces/staunton.tsx`). A set is one module, registered under
an id (`games/chess/pieces/piece_sets.ts`):

```tsx
api.registerPieceSet('my_set', {
  label: 'My set',
  // Fills its box, square; carries the class chess-piece-art; aria-hidden.
  Piece: ({role, color}) => <MyPieceArt role={role} color={color} />,
  // Optional: the piece as an image URL, for a board that draws images
  // (chessground). Without it, such a board refuses the set.
  imageUrl: ({role, color}) => `/my_set/${color}-${role}.svg`,
});
```

`role` is `'king' | 'queen' | 'rook' | 'bishop' | 'knight' | 'pawn'` and
`color` is `'white' | 'black'`. `getPieceSet(id)` returns a set and throws,
naming the registered ids, for an id that is not one; `registerPieceSet`
throws for an id already taken and returns a function that removes the set
(for a test). `staunton` and a fetched set both give `imageUrl`; Staunton's
images are SVG data URLs in the page's piece colours. Register from an
add-on (below). The
cburnett pieces are a set the user fetches ("Using the cburnett pieces").

### Client add-ons

An add-on changes what the client draws without editing it: a module in
`open_spiel/gui/addons/` (a directory git ignores) that exports
`registerInto`. The entry point (`src/main.tsx`) calls each add-on's
`registerInto` once, in path order, before the app is drawn, with the
client's extension points (`src/addons.ts`):

| Extension point                 | What it does                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `replaceRenderer(name, render)` | draws renderer `name`'s presentations with `render`; returns the one replaced |
| `chessRenderer(board)`          | the `chess` renderer drawing with `board` ("Replacing the chess board")       |
| `registerPieceSet(id, module)`  | a chess piece set, offered in View › Board › Pieces; returns its remover      |
| `registerDeck(deck)`            | a card deck, offered as a Card deck in the card games' View settings          |
| `getDeck(id)`                   | a registered deck (`simple`, `simple-corner`, `basic`), to build one from     |

An add-on fails loud: a module without `registerInto`, or whose
`registerInto` throws (replacing a renderer that does not exist, an id
already taken), stops the client before it draws, naming the module.
`npm run typecheck` checks `addons/`; `npm run lint` leaves it alone. Restart
`npm run dev`, or rebuild, after adding or removing an add-on.
`npm run test:addons` loads every add-on in `addons/` without a browser and
draws each card deck one registers, its faces and every back; its test
names list the add-ons it loaded and each new deck with its backs.

The three recipes below are complete: each is the commands to run, the file
to create with its whole content, and how to check the result. chessground
and the cburnett pieces are not bundled because of licensing; they come with
their own licences.

#### Before a recipe

1. **Where**: every command runs in `open_spiel/gui`, the client's
   directory under the repository root, and every path a recipe names
   (`addons/`, `piece_sets/`) is in it.
2. **Node.js**: 22.22.2 or later (`engines` in `package.json`); check with
   `node --version`.
3. **Dependencies**: `npm ci` once, in `open_spiel/gui`, before the first
   recipe.
4. **The `addons/` directory**: a fresh checkout has none, and git ignores
   it; `mkdir -p addons` makes it.
5. **The server**: the client only draws what the server sends. In a second
   terminal, from the repository root, with OpenSpiel installed from it
   (`pip install -e .`):

   ```sh
   python -m open_spiel.python.gui   # serves http://127.0.0.1:8080/
   ```

   Then see the client either way:

   - **Built**: `npm run build`, then open `http://127.0.0.1:8080/`; the
     server serves `dist/`. Rebuild after each change.
   - **Dev server**: `npm run dev`, then open `http://localhost:5173/`.
     It sends `/api` requests to `http://127.0.0.1:8080`;
     `OPENSPIEL_GUI_SERVER=http://127.0.0.1:9000 npm run dev` names
     another server. Without a server, the page says "The games could not
     be loaded".

A game is at `/game/<name>`, such as `/game/chess` or `/game/gin_rummy`. A
setting written View › Visuals › Card deck is in the **View** panel (the
tab beside Setup, at the lower right of a game page): its **Visuals**
section (a tab inside the panel), the **Card deck** control. Chess's
settings are in View › Board.

### Replacing the chess board

To draw chess with chessground instead of the built-in board:

1. Install chessground into `addons/` (git ignores the directory; the
   client's own `package.json`, lockfile and `node_modules/` are untouched,
   so `npm ci` keeps it):

   ```sh
   npm install --prefix addons chessground
   ```

   The client ships none of chessground and depends on none of it. A build
   (`dist/`) made while this add-on is in `addons/` contains chessground.

2. Create `addons/chessground_board.tsx` with exactly this content:

   ```tsx
   // open_spiel/gui/addons/chessground_board.tsx
   import {Chessground} from 'chessground';
   import type {Api} from 'chessground/api';
   import type {Key} from 'chessground/types';
   import 'chessground/assets/chessground.base.css';
   import {type JSX, useEffect, useMemo, useRef, useState} from 'react';

   import type {AddOnApi} from '../src/addons.ts';
   import {
     ChessBoardFrame,
     PromotionPicker,
     type PromotionLetter,
     pieceSetCss,
     squareColors,
     useChessBoardSettings,
   } from '../src/games/chess/chess_board_kit.tsx';
   import {
     type ChessBoardProps,
     actionFor,
   } from '../src/games/chess/chess_board_slot.ts';

   function ChessgroundBoard({
     frame,
     viewer,
     act,
   }: ChessBoardProps): JSX.Element {
     const settings = useChessBoardSettings();
     const element = useRef<HTMLDivElement>(null);
     const board = useRef<Api | null>(null);
     const [promoting, setPromoting] = useState<{from: Key; to: Key} | null>(
       null,
     );

     // The viewer's piece set and board colours, as chessground's styles.
     const css = useMemo(() => {
       const {light, dark} = squareColors(settings.theme);
       return [
         pieceSetCss(settings.pieceSet, '.cg-addon'),
         `.cg-addon cg-board{background:repeating-conic-gradient(` +
           `${dark} 0 25%,${light} 0 50%) 0 0/25% 25%}`,
       ].join('\n');
     }, [settings.pieceSet, settings.theme]);

     // Plays `action`; a refusal puts the pieces back where the frame has them.
     function play(action: number | null): void {
       const undo = () => board.current?.set({fen: frame.fen});
       if (action === null) undo();
       else act(action).catch(undo);
     }

     useEffect(() => {
       if (element.current === null) return;
       board.current = Chessground(element.current, {});
       return () => {
         board.current?.destroy();
         board.current = null;
       };
     }, []);

     useEffect(() => {
       // Each square's targets, once each (a promotion is four moves, one per
       // piece, to the same square); none while the viewer cannot move.
       const targets = new Map<Key, Set<Key>>();
       for (const move of viewer.canMove ? frame.moves : []) {
         const from = move.from as Key;
         targets.set(
           from,
           (targets.get(from) ?? new Set()).add(move.to as Key),
         );
       }
       board.current?.set({
         fen: frame.fen,
         orientation: viewer.orientation,
         turnColor: frame.turn,
         check: frame.check ?? false,
         lastMove: frame.lastMove
           ? [frame.lastMove.from as Key, frame.lastMove.to as Key]
           : [],
         coordinates: settings.showCoordinates,
         animation: {
           enabled: settings.animationScale > 0,
           duration: 200 * settings.animationScale,
         },
         movable: {
           free: false,
           color: frame.turn,
           dests: new Map([...targets].map(([from, to]) => [from, [...to]])),
           events: {
             after: (from, to) => {
               if (actionFor(frame, from, to, 'q') !== null) {
                 setPromoting({from, to});
               } else {
                 play(actionFor(frame, from, to));
               }
             },
           },
         },
       });
     }, [frame, viewer, act, settings]);

     // Chessground draws the coordinates when it builds the board.
     useEffect(() => {
       board.current?.redrawAll();
     }, [settings.showCoordinates]);

     function promote(letter: PromotionLetter | null): void {
       if (promoting === null) return;
       setPromoting(null);
       play(
         letter === null
           ? null
           : actionFor(frame, promoting.from, promoting.to, letter),
       );
     }

     return (
       <ChessBoardFrame frame={frame}>
         {size => (
           <>
             <style>{css}</style>
             <div
               ref={element}
               className="cg-addon"
               style={{width: size, height: size}}
             />
             {promoting !== null && (
               <PromotionPicker
                 color={frame.turn}
                 onSelect={promote}
                 onCancel={() => {
                   promote(null);
                 }}
               />
             )}
           </>
         )}
       </ChessBoardFrame>
     );
   }

   export function registerInto(api: AddOnApi): void {
     api.replaceRenderer('chess', api.chessRenderer(ChessgroundBoard));
   }
   ```

   Its last line is the one registry call:
   `api.replaceRenderer('chess', api.chessRenderer(ChessgroundBoard))`.
   Everything else chessground draws comes from the client
   (`src/games/chess/chess_board_kit.tsx`), and no chessground theme or
   piece stylesheet is loaded:

   - **pieces**: the set picked in View › Board › Pieces (`staunton` by
     default, `cburnett` once fetched), given to chessground as images by
     `pieceSetCss`;
   - **squares, coordinates, animation**: View › Board's Theme and
     Coordinates, and the Animation speed (`useChessBoardSettings`,
     `squareColors`);
   - **orientation**: `viewer.orientation`, View › Board › View from for the
     oracle and a spectator;
   - **status**: whose move it is, check, and the result once the game is
     over (`ChessBoardFrame` draws `ChessStatus` above the board);
   - **size**: the largest square that fits the Gameplay panel, resized with
     it (`ChessBoardFrame`);
   - **promotion**: `PromotionPicker`, the four pieces in the viewer's set;
     a press outside it takes the pawn back.

3. Check it:

   ```sh
   npm run typecheck   # the add-on type-checks against chessground
   npm run test:addons # the add-on loads
   npm run build       # it bundles
   ```

   Then start the server and open the client ("Before a recipe", 5), at
   `/game/chess`.

   In the chess game, `document.querySelector('cg-container')` is an element
   and `document.querySelector('.chess-panel')` is null: chessground draws
   the board. Moving a piece plays the move; moving a pawn to the last rank
   offers the four pieces.

If chessground is missing (a fresh checkout with the add-on but not
`addons/node_modules/`), `npm run typecheck`, `npm run build` and
`npm run dev` fail naming `chessground` and `addons/chessground_board.tsx`:
run step 1 again, or delete the add-on. An add-on's imports are bundled with
the client, so the client cannot leave out one add-on whose package is
missing and run on.

To go back, delete `addons/chessground_board.tsx` and `addons/node_modules/`
(and `addons/package*.json`).

Any other board plugs in the same way: a component that takes the props
below, passed to `api.chessRenderer`, with whatever of the kit it wants.
Outside an add-on (a test, or an app built on the client), make the same
call directly, before the board is drawn: `replaceRenderer` from
`src/renderers/registry.ts` and `chessRenderer` from
`src/games/chess/chess_renderer.tsx`. `replaceRenderer` returns the
component it replaced, so a test can put it back. The `chess` entry's
options stay: View › Board's settings, the chess move list in Actions ›
History, `keepsState` (the board stays mounted from one ply to the next and
is given new props) and `drawsStatus` (the Gameplay panel shows no status
line, so the board says whose turn it is: `ChessStatus`). The built-in
board is `KernelChessBoard` (`games/chess/chess_renderer.tsx`).

The kit (`games/chess/chess_board_kit.tsx`), for any board:

| Export                           | What it gives                                                                |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `ChessBoardFrame`                | the status line over the largest square that fits the panel                  |
| `ChessStatus`, `chessStatusText` | "Move 12 · Black to move · Check", "1-0 · White wins by checkmate"           |
| `PromotionPicker`                | the choice of queen, rook, bishop or knight, over the board                  |
| `useChessBoardSettings()`        | View › Board's `theme`, `showCoordinates`, `pieceSet`; `animationScale`      |
| `squareColors(theme)`            | the light and dark squares' CSS colours                                      |
| `pieceImageUrl`, `pieceSetCss`   | the picked set's pieces as image URLs, or as CSS for `piece.<colour>.<role>` |

The props (`games/chess/chess_board_slot.ts`):

```ts
interface ChessBoardProps {
  readonly frame: ChessFrame;
  readonly viewer: ChessBoardViewer;
  readonly act: (action: number) => Promise<void>;
}

interface ChessFrame {
  readonly fen: string; // the position
  readonly turn: 'white' | 'black'; // the side to move
  readonly moves: readonly MoveHint[]; // the legal moves, with their actions
  readonly lastMove: {from: string; to: string} | null;
  readonly check: 'white' | 'black' | null; // the side in check
  readonly isTerminal: boolean;
  readonly ply: number; // plies played to the position shown
  readonly session: SessionPayload; // seats, moves in SAN, the outcome
  readonly source: Frame; // the frame as the server sent it
}

interface MoveHint {
  readonly action: number; // the action act() plays
  readonly from: string; // "e2"
  readonly to: string; // "e4"
  readonly promotion: string | null; // "q", "r", "b" or "n"; else null
}

interface ChessBoardViewer {
  readonly seat: number | null; // the viewer's seat; null: oracle, spectator
  readonly side: 'white' | 'black' | null; // seat 0 black, seat 1 white
  readonly orientation: 'white' | 'black'; // the side drawn at the bottom
  readonly isOracle: boolean; // may move for either side
  readonly isParticipant: boolean; // holds a seat
  readonly isActiveSeat: boolean; // holds the seat to move
  readonly canMove: boolean; // whether act() plays now
}
```

- **The position**: `frame.fen`. Set the pieces from it whenever `frame` or
  `viewer` changes; that also puts back a move the server refuses (below).
- **Legal moves**: `frame.moves`, one per move, with the action that plays
  it; empty when the viewer is not given the legal actions (another seat's
  turn, a spectator).
- **Promotion**: one move per piece, alike but for `promotion`, the piece's
  FEN letter. `actionFor(frame, from, to, promotion = null)` returns the
  action, or null when no legal move matches.
- **Last move and check**: `frame.lastMove` (null before the first move) and
  `frame.check` (the side whose king is in check, or null).
- **Orientation**: `viewer.orientation`: the viewer's own side for a seat;
  for the oracle (analysis) and a spectator, the side View › Board › View
  from picks (white until one is picked), which turns to the seat chosen in
  View › Perspective in analysis, seat 0 being Black.
- **Whose turn, and whether to take input**: `frame.turn` and
  `viewer.canMove`, false on another seat's turn, for a spectator, in a
  replay, in a finished game and while a move is in flight.
- **The end**: `frame.isTerminal`; `frame.session.state.outcome`
  (`checkmate`, `stalemate`, `threefold`) and `frame.session.state.returns`.

`act(action)` submits a move: pass the `action` of one of `frame.moves`.

- It resolves once the move is handed to the session, not once the server
  takes it. While the move is in flight `viewer.canMove` is false.
- It rejects, sending nothing, when `viewer.canMove` is false.
- A move the server takes arrives as a new `frame`.
- A move the server refuses shows the session's error in the Gameplay
  panel and leaves `frame` as it was. `viewer` is a new object when the
  move goes in flight and again when it settles, so a board that sets its
  pieces from `frame.fen` whenever `frame` or `viewer` changes shows the
  position again; a board that moved a piece on its own must do so (the
  recipe's `play` puts the pieces back when `act` rejects).

`games/chess/chess_board_slot_test.tsx` registers a fake board this way,
draws a chess frame through `GameplayBoard`, checks that a press plays its
move, and that `act` rejects while a move is in flight.

### Using the cburnett pieces

To draw the chess pieces with Colin M.L. Burnett's set:

1. Fetch it. Run this to download the cburnett pieces from Wikimedia
   Commons:

   ```sh
   npm run fetch-pieces -- cburnett
   ```

   `scripts/fetch_piece_set.ts` downloads the twelve SVG files from their
   Wikimedia Commons originals, with each file's Commons page and the
   licence text, from pinned URLs. Every file must match its pinned sha256;
   on any mismatch, or a download that fails or takes over 30 seconds, it
   names the file and writes nothing. The files go to `piece_sets/cburnett/`, which git
   ignores. The SVGs are each file's current Commons upload (Commons keeps
   only superseded uploads under `/archive/` URLs); the pinned sha256 is
   what fixes them, so a re-upload is refused rather than taken.

2. Restart `npm run dev`, or rebuild, with the server running ("Before a
   recipe", 5), open `/game/chess` and pick View › Board › Pieces ›
   cburnett.

3. Check it: `ls piece_sets/cburnett` lists 25 files (12 `.svg`, 12
   `.source.wikitext`, `LICENSE-BSD.wikitext`), and the pieces on the board
   are `img.chess-piece-art` elements.

The client offers `cburnett` exactly when that directory holds all 25 files
as the client is built or the dev server starts
(`games/chess/pieces/fetched_sets.tsx`). An incomplete directory leaves only
`cburnett` out of View › Board › Pieces, with an error on the browser
console naming the missing files; remove the directory and fetch again. The
client ships none of these files. A build made while the directory is there
carries the twelve pieces in `dist/`, and the build copies
`LICENSE-BSD.wikitext` and the twelve `.source.wikitext` pages into
`dist/piece_sets/cburnett/` beside them. To go back, delete
`piece_sets/cburnett/`.

### Adding a card deck or back

To offer a new card back (or a whole deck) in the card games, from
`open_spiel/gui` ("Before a recipe", above):

1. Make the add-on directory: `mkdir -p addons`.

2. Create `addons/green_back.tsx` with exactly this content. It registers
   the `basic` deck's faces and its five backs (Blue lattice, the default,
   Red lattice, Red diamonds, Black diamonds and Navy stripes) with one more
   back, Green, as a deck of its own, "Corner index + large pip, green back";
   replace `GreenBack` (and `renderFace`) to draw your own art:

   ```tsx
   // open_spiel/gui/addons/green_back.tsx
   import type {JSX} from 'react';

   import type {AddOnApi} from '../src/addons.ts';

   function GreenBack(): JSX.Element {
     return (
       <svg
         viewBox="0 0 250 350"
         width="100%"
         height="100%"
         preserveAspectRatio="none"
         style={{display: 'block'}}
       >
         <rect width={250} height={350} fill="#1d6b45" />
         <rect
           x={14}
           y={14}
           width={222}
           height={322}
           rx={8}
           fill="none"
           stroke="#f2d27a"
           strokeWidth={10}
         />
         <rect
           x={32}
           y={32}
           width={186}
           height={286}
           rx={4}
           fill="none"
           stroke="#ffffff"
           strokeOpacity={0.6}
           strokeWidth={3}
         />
       </svg>
     );
   }

   export function registerInto(api: AddOnApi): void {
     const basic = api.getDeck('basic');
     api.registerDeck({
       ...basic,
       id: 'basic_green',
       label: 'Corner index + large pip, green back',
       provenance: 'Drawn in code for this add-on',
       backs: [...(basic.backs ?? []), 'green'],
       renderBack: opts =>
         opts.variant === 'green' ? <GreenBack /> : basic.renderBack(opts),
     });
   }
   ```

   The gold frame keeps the green back apart from the green table.

3. Check it without a browser:

   ```sh
   npm run typecheck     # no output: the add-on type-checks
   npm run test:addons   # passes; lists the add-on and its deck
   npm run build         # ends "built in"
   ```

   `npm run test:addons` passes and its tests read
   `load: ../addons/green_back.tsx` and
   `deck basic_green ("Corner index + large pip, green back") draws its
faces and the backs: default, blue, red, diamond-red, diamond-black,
stripe-navy, green`.

4. Check it in the browser: start the server and open the client
   ("Before a recipe", 5), open `/game/gin_rummy`, and in View › Visuals
   the Card deck control lists "Corner index + large pip, green back". Pick
   it: Card back offers Deck default, Blue lattice, Red lattice, Red
   diamonds, Black diamonds, Navy stripes and Green, and Green draws the
   stock and the opponent's hand with the green back.

A deck is a `Deck` (`cards/library/types.ts`): its `id`, `label`, a
`provenance` line, `aspect` (width / height), the corner `index` box as
fractions of the card, `renderFace(card, opts)` and `renderBack(opts)`, both
filling their box, and `backs`, the back ids the Card back control offers
(`renderBack` gets the one picked as `opts.variant`, or none for Deck
default). `registerDeck` refuses an id already taken, a deck without
provenance and an index box outside the card. `src/addons_test.tsx`
registers this deck through `loadAddOns` and draws its backs. To go back,
delete `addons/green_back.tsx`.

A deck's `id` is the stored and wire name and stays fixed; its `label` names
the face style in the Card deck picker by what it draws, and its optional
`description` is the one "best for" line the picker shows under it (Large
corner index: fanned hands; Corner index + large pip: table piles and tricks;
Centred rank and suit: cards seen one at a time). Every card game starts on
Corner index + large pip (`basic`): it is `DEFAULT_CARD_STYLE.deck` and the
table-card settings' default, and euchre's and hearts' Card style default.
The built-in decks share
the card backs of `cards/library/decks/backs.tsx`, drawn in code (lattice,
diamonds, stripes; blue lattice by default), named in the Card back picker by
`Deck.backLabels`; the generic `cards` renderer's face-down cards
(`cards/card_face.tsx`) draw the same back, from the table-card settings.

## 3. The shared views and hints

### `grid`

```json
{
  "grid": {
    "rows": 8,
    "columns": 8,
    "cells": [["r", "n", null, "..."], "..."],
    "piece_set": "chess",
    "mode": "move",
    "highlights": [
      [6, 4],
      [4, 4]
    ],
    "alerts": [[0, 4]],
    "checkered": true,
    "flipped": false,
    "row_labels": ["8", "7", "6", "5", "4", "3", "2", "1"],
    "column_labels": ["a", "b", "c", "d", "e", "f", "g", "h"]
  },
  "lines": ["White to move."]
}
```

`cells` runs row by row from the top; each is a piece code of the piece set,
or `null`. `alerts` (cells drawn as a warning, such as a king in check),
`checkered`, `flipped`, `row_labels`, `column_labels` and `lines` are
optional. The mode says how a press becomes an action, and each mode
reads its own hint:

| Mode    | Hint                                           | A press                                                                                                                                                |
| ------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `claim` | `{"cell": [row, column]}`                      | on the cell plays it (tic-tac-toe)                                                                                                                     |
| `drop`  | `{"column": column}`                           | anywhere in the column plays it (connect four)                                                                                                         |
| `move`  | `{"from": [row, column], "to": [row, column]}` | on a piece selects it and marks its targets; on a target plays the move. Several moves between the same cells (promotions) are offered by their labels |
| `none`  |                                                | does nothing                                                                                                                                           |

Piece codes: `chess` takes FEN letters (upper case white); `checkers` takes
`o`, `O` (light man, king) and `x`, `X` (dark); `marks` and `stones` take
`x` and `o`.

### `cards`

```json
{
  "hands": [
    {"label": "Your hand", "cards": ["As", "Td"], "fanned": true},
    {"label": "Player 1", "cards": [null, null]}
  ],
  "piles": [{"label": "Discard", "cards": ["7c"]}],
  "trick": [{"label": "Player 0", "card": "Kh"}],
  "lines": ["Player 1 knocked."],
  "skin": "standard"
}
```

Every field is optional. A card hinted `{"card": "As"}` plays its action
when pressed, wherever it is drawn; an action hinted `{"button": "Knock"}`
is a button under the table.

## 4. Controllers

A bot is a controller. Register it in the server before serving, with
`controllers.register(name, controllers.Kind(label, settings,
reads_full_state=..., build=...))`; `build(game, settings, seed)` returns an
object whose `choose_action(state, seat)` returns a legal action. A
controller that reads more than its seat may see sets
`reads_full_state=True`, and is then refused for games that hide
information.

`settings` is a tuple of `controllers.Setting(name, kind, default,
description, ...)`, where `kind` is `bool`, `int`, `float` or `str`. A
number may be bounded (`minimum`, `maximum`); a string may be bounded in
length (`min_length`, `max_length`) and held to a `pattern`; a string or a
number may be held to `choices`. A setting whose `default` is
`controllers.REQUIRED` must be given: a seat without it is refused, and the
Players form marks it required and will not save without it. A bot that
plays a checkpoint declares:

```python
(
    controllers.Setting(
        "checkpoint", str, controllers.REQUIRED, "The checkpoint to play.",
        min_length=1,
    ),
    controllers.Setting(
        "deterministic", bool, True, "Whether it plays its most likely move."
    ),
    controllers.Setting(
        "temperature", float, 1.0, "The temperature it samples at.",
        minimum=0.0,
    ),
)
```

`build` receives every setting, the defaults filled in.

A controller may set a `last_notice` attribute in `choose_action`: a line of
plain text about the move, at most 200 characters, or None. The seat shows it
to its own viewer and to omniscient viewers, after the controller's name in
Setup › Players; it has no effect on play.

A bot that runs in its own process, or on another machine, needs no plugin:
the built-in `remote` controller plays any bot server, and
`open_spiel/python/gui/REMOTE_BOTS.md` gives the protocol
and a reference server that wraps any bot.

A controller has 10 seconds to choose each move (`--move-budget-seconds`
changes it). The server asks it on a thread of its own, so a slow controller
holds up only the request that asked it, and the server answers others
meanwhile. One that has not returned in time fails as one that raises does:
`choose` raises `ControllerError` with the error `Timeout`, the seat pauses
and the request is refused with `controller_failed`. What it returns later is
discarded. Python cannot stop a thread, so a controller that may take long
calls `move_budget.check()` between units of its work and stops when it
raises; one that cannot must bound its own work to fit the budget. The
built-in `mcts` controller does one or the other: MCTS = pyspiel C++ MCTSBot
(memory cap 1000 MB) for C++ games, bounded by a capability check at
startup: a wall-clock backstop inside the move budget and up to 10000
simulations where the served pyspiel stops at whichever limit comes first,
otherwise no clock and simulations capped at 250 to fit the budget
(`mcts_controller`); OpenSpiel Python MCTS for Python-implemented games,
which checks the budget between simulations and holds at most 1,000,000 tree
nodes (`mcts_search.NODE_BUDGET`). The limits and the measurements they rest
on are in `open_spiel/python/gui/CONTRACT.md` (Controllers).

`build` may take its time: a bot that loads and warms up a trained model can
take tens of seconds. The server builds each seat's controller on a thread of
its own, and the request that seats it answers at once, with the seat
`preparing`; it plays once built, and autoplay goes on from there. A build has
120 seconds (`--build-budget-seconds` changes it). One that raises
`UnavailableError`, raises anything else, or overruns leaves its seat
`unavailable`, with a `failure` and a `message` that says why, and the session
paused until the seat is given a controller again. The settings are checked
before any build, so a seat given invalid ones is still refused at once with
`controller_unavailable`, as is a build that fails within a quarter of a
second.

### Controller plugins

A bot from another package registers itself through a plugin: a module with
a function `register_into(controllers)`, which the server calls with its
`controllers` module before it starts serving.

```python
# my_bots/gui_plugin.py
def register_into(controllers):
  controllers.register(
      "my_bot",
      controllers.Kind(
          "My bot", (), reads_full_state=False, build=build_my_bot
      ),
  )
```

`open_spiel/python/gui/fake_plugin.py` is a complete example.

No plugin is loaded unless asked for. Name the modules with `--controllers`,
comma separated or with the flag repeated:

```sh
python -m open_spiel.python.gui --controllers my_bots.gui_plugin
```

A distribution may instead declare its plugin in the entry-point group
`open_spiel.gui.controllers`, naming the module or its `register_into`:

```toml
[project.entry-points."open_spiel.gui.controllers"]
my_bots = "my_bots.gui_plugin"
```

Such plugins load only with `--controllers auto`, which loads every one
installed, in name order; `auto` may stand beside module names. A bare
`python -m open_spiel.python.gui` imports no plugin.

A plugin fails loud. A module that cannot be imported, one without
`register_into`, an entry point that names neither a module nor a callable,
and a `register_into` that raises, for instance by registering a name already
taken, each stop the server before it starts, with a `PluginError` naming the
plugin.
