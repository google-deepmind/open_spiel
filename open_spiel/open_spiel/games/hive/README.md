# Hive

Implements the base game of [Hive](https://www.gen42.com/games/hive) and its
three expansion pieces: Mosquito, Ladybug, and Pillbug.

![Picture of playing on the console:](https://imgur.com/mkEObfx.png)

*<center>Example game state viewed on the command-line (left) and with an
external viewer "Mzinga" (right)</center>*

This implementation follows the rules outlined by the Universal Hive Protocol
([UHP](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)), which
means states can be serialized to and deserialized from valid UHP game strings.
With a bit of I/O handling, this can also be used as a UHP-compliant Hive
Engine, making interactions with other engines straightforward.

## State

### Observation Tensor

First, the hexagonal grid needs to be represented as a rectangular one for 2D
convolution:

![Storing a hexagonal grid in a 2D array](https://imgur.com/CIy5ctM.png)

*<center>Example transformation - taken from
[RedBlobGames](https://www.redblobgames.com/grids/hexagons/#map-storage)</center>*

The observation tensor then takes the form of multiple 2D feature planes
describing the board and turn state, similar to what was done for AlphaZero
chess.

However, since Hive's "board" is a repeating hexagonal tiling, the size is
bounded only by the maximum number of tiles that can be laid in a straight line
(28 total tiles for all expansions). Yet, a grid of size 28x28 is far too large
to be computationally practical.

To help offset the complications this would bring for training in AlphaZero, the
board can be paramaterized with `board_size` to reduce the tensor's overall
sparsity. Using a `board_size` smaller than `kMaxBoardSize` means that some
outlier games cannot be perfectly represented and are instead forced to a Draw.
In practice, games that would approach that board length are extremely rare, so
the trade-off feels acceptable.

The 2D feature planes are one-hot encodings that indicate:

-   the presence of a particular bug type, for each player
-   which bugs are pinned
-   which bugs are covered
-   the available positions that each player can place a new bug tile
-   all 1's or all 0's to distinguish the current player's turn

### Action Space

**An action in Hive is described as:** 1) choosing which tile to move 2)
choosing which tile it moves next to (or on top of) 3) the relative direction of
the tile it moves next to

*e.g.* "wA2 bL/" - *White moves their 2nd Ant to the top right edge of Black's
Ladybug*

With there being 28 unique tiles and 7 directions (the 6 hexagonal edges and
"above"), the action space can be thought of as entries into a 3D matrix with
dimensions **7 x 28 x 28** = **5488** total actions.

This is not a *perfect* action space representation as there are a handful of
unused actions (e.g. moving a tile next to itself?), but it does capture every
legal move. Unfortunately, with the introduction of the Pillbug, each player is
able to move their own piece *or* the enemy's, meaning we can't implicitly
expect the tile being moved to be the colour of the current player. This ends up
doubling the action space size from 7x14x28 to 7x28x28

## To-do

Below are some concrete features and fixes I intend to implement to either help
speed up training or improve the interoperability between other Hive software
(e.g. displaying games directly to
[MzingaViewer](https://github.com/jonthysell/Mzinga)):

-   [ ] Address the efficiency of code that uses the most compute time
    (`HiveState::GenerateValidSlides()` and `HiveState::IsGated()` from recent
    perf tests)
-   [ ] Implement zobrist hashing to handle a "3-repeated moves" forced draw
    (unofficial community rule)
-   [ ] Undo()
-   [ ] Perft()
-   [ ] Make it possible to load many UHP gamestrings from a file for training,
    or to collect interesting game statistics
-   [ ] Create a separate binary that handles I/O and behaves as a proper
    UHP-compliant engine
-   [ ] Provide a simplified action space for games that do not use expansion
    pieces

## Future Improvements / Thoughts

While developing this engine, I came across many interesting ideas that have the
potential for serious progress towards a viable expert-level AZ-bot for Hive.
And as of this submission, no such Hive AI exists, making the prospect of any
improvements much more appealing.

Below is a record of those miscellaneous thoughts, in approximate order of the
potential I think it has:

-   **Design a more exact action space**. There are a handful of other suggested
    notations from the Hive community, each with their own advantages and
    drawbacks, that may be useful to look into for an alternative action space.
    One that looks very promising is
    [Direction-Based Notation](https://psoukenik.medium.com/direction-based-notation-for-hive-dd7fd234d4d),
    as it implicitly covers all rotations and reflections by design.

-   **Use a Hexagonal CNN model or filter**. One problem that has been
    conveniently unaddressed is the fact that 2D convolution is performed on
    Hexagonal data that has be refitted onto a square. The typical 3x3 filter
    then doesn't accurately describe the 6 neighbours of a hex, as 2 extra
    values are contained in the filter. One option would be to use a custom 3x3
    filter that zeroes-out the two values along the diagonal, or to attempt
    using a more advanced implementation like
    [HexCNN](https://arxiv.org/pdf/2101.10897) or
    [Rotational-Invariant CNN](https://www.jstage.jst.go.jp/article/transinf/E107.D/2/E107.D_2023EDP7023/_pdf/-char/en).
    The first option would be much easier to implement into the existing
    AlphaZero framework.

-   **Attempt a graph/node-based representation**. With how a game of Hive is
    structed like a graph itself, I think there is potential in using Graph
    Neural Networks (GNN) for learning. Some recent research has been done by
    applying
    [GNNs to AlphaZero for board game AI](https://arxiv.org/pdf/2107.08387),
    which indicates there is at least some proven success already.
